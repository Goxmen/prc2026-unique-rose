"""Relaxed twin stages A/B/C with reference tables fitted on training data only."""

import duckdb
import pandas as pd

from recovery import PLAN_FIELDS, PUBLIC_FIELDS, HIDDEN


def recover_relaxed(dep, arr, claims, reference_dep, reference_arr, prior_audit):
    for frame in [dep, arr, claims, reference_dep, reference_arr]:
        if HIDDEN.intersection(frame.columns):
            raise ValueError('Recovery must receive observable columns only')
    exact = prior_audit.loc[prior_audit.stage.eq('exact')]
    used_ids = set(exact.mvt_id)
    probes = dep.loc[dep[PLAN_FIELDS].isna().any(axis=1) & ~dep.mvt_id.isin(used_ids)]
    out, audit = dep.copy(), prior_audit.copy().set_index('mvt_id')
    if probes.empty or arr.empty:
        return out, audit.reset_index().set_axis(prior_audit.index)
    # Reserve exact-stage plans even where their movement-side ID was missing.
    reserved = claims[['mvt_id', 'FLIGHT_ID_mvt']].copy()
    exact_ids = exact.set_index('mvt_id').donor_flight_id
    reserved['FLIGHT_ID_mvt'] = reserved.FLIGHT_ID_mvt.fillna(reserved.mvt_id.map(exact_ids))
    reference = pd.concat([reference_dep, reference_arr], ignore_index=True)
    with duckdb.connect(config={'threads': 2}) as con:
        for name, frame in [('dep', reference_dep), ('refarr', reference_arr), ('reference', reference),
                            ('probes', probes), ('arrivals', arr), ('claims', reserved)]:
            con.register(name, frame)
        con.execute("""
            CREATE MACRO id_style(s) AS CASE WHEN regexp_matches(s, '^[A-Z]{3}[0-9]') THEN 'icao'
                WHEN regexp_matches(s, '^([A-Z][A-Z]|[A-Z][0-9]|[0-9][A-Z])[0-9]{1,4}[A-Z]?$')
                THEN 'iata' ELSE NULL END;
            CREATE MACRO id_prefix(s) AS CASE id_style(s) WHEN 'icao' THEN s[1:3] WHEN 'iata' THEN s[1:2] END;
            CREATE MACRO id_rest(s) AS CASE id_style(s) WHEN 'icao' THEN s[4:] WHEN 'iata' THEN s[3:] END;
            CREATE MACRO id_tail(s) AS CASE WHEN regexp_matches(id_rest(s), '^[0-9]+[A-Z]{0,3}$')
                THEN nullif(regexp_replace(id_rest(s), '^0+', ''), '') END;
            CREATE TABLE airline_map AS
                WITH counts AS (SELECT id_prefix(FLIGHT_mvt) iata, id_prefix(CALLSIGN_flt) icao, count(*) n
                    FROM reference WHERE FLIGHT_ID_mvt IS NOT NULL AND id_style(FLIGHT_mvt) = 'iata'
                    AND id_style(CALLSIGN_flt) = 'icao' AND id_tail(FLIGHT_mvt) = id_tail(CALLSIGN_flt)
                    GROUP BY 1,2), ranks AS (SELECT *, sum(n) OVER (PARTITION BY iata) total,
                    row_number() OVER (PARTITION BY iata ORDER BY n DESC, icao) rank FROM counts)
                SELECT iata, icao FROM ranks WHERE rank=1 AND n>=30 AND n/total>=0.9;
            CREATE MACRO normalized(s) AS
                CASE WHEN id_tail(s) IS NOT NULL THEN
                    coalesce(CASE id_style(s) WHEN 'icao' THEN id_prefix(s) END,
                        (SELECT icao FROM airline_map WHERE iata=id_prefix(s))) || ':' || id_tail(s) END;
            CREATE TABLE route_duration AS
                SELECT d.ADEP_mvt, d.ADES_mvt, median(epoch(a.MVT_TIME_UTC_mvt-d.MVT_TIME_UTC_mvt)) duration_sec
                FROM dep d JOIN refarr a ON a.FLIGHT_ID_mvt=d.FLIGHT_ID_mvt
                 AND a.ADEP_mvt=d.ADEP_mvt AND a.ADES_mvt=d.ADES_mvt
                 AND a.MVT_TIME_UTC_mvt>d.MVT_TIME_UTC_mvt
                 AND a.MVT_TIME_UTC_mvt<d.MVT_TIME_UTC_mvt + INTERVAL 16 HOUR
                WHERE d.FLIGHT_ID_mvt IS NOT NULL
                GROUP BY 1,2 HAVING count(DISTINCT d.mvt_id)>=20;
            CREATE TABLE p AS SELECT *, normalized(FLIGHT_mvt) norm FROM probes;
            CREATE TABLE a AS SELECT *, normalized(FLIGHT_mvt) norm_f, normalized(CALLSIGN_flt) norm_c
                FROM arrivals WHERE FLIGHT_ID_mvt IS NOT NULL AND CALLSIGN_flt IS NOT NULL;
        """)
        frames = []
        for stage, ident, time_window in [
            ('relaxed_A', '(a.CALLSIGN_flt=p.FLIGHT_mvt OR a.FLIGHT_mvt=p.FLIGHT_mvt)', 'dt>900 AND dt<28800'),
            ('relaxed_B', '(p.norm IS NOT NULL AND (p.norm=a.norm_f OR p.norm=a.norm_c))', 'dt>900 AND dt<28800'),
            ('relaxed_C', 'true', 'dt>0.8*rd.duration_sec AND dt<1.3*rd.duration_sec+900'),
        ]:
            route_join = 'JOIN route_duration rd USING (ADEP_mvt, ADES_mvt)' if stage == 'relaxed_C' else ''
            result = con.sql(f"""
                SELECT p.mvt_id, a.mvt_id donor_mvt_id, a.FLIGHT_ID_mvt donor_flight_id,
                       '{stage}' stage
                FROM p {route_join} JOIN a ON a.ADEP_mvt=p.ADEP_mvt AND a.ADES_mvt=p.ADES_mvt
                 AND a.AIRCRAFT_TYPE_mvt=p.AIRCRAFT_TYPE_mvt
                 AND a.ADEP_flt=p.ADEP_mvt AND a.ADES_flt=p.ADES_mvt
                 AND (p.FLIGHT_ID_mvt IS NULL OR p.FLIGHT_ID_mvt=a.FLIGHT_ID_mvt)
                 AND {ident}
                CROSS JOIN LATERAL (SELECT epoch(a.MVT_TIME_UTC_mvt-p.MVT_TIME_UTC_mvt) dt)
                WHERE {time_window}
                 AND NOT EXISTS (SELECT 1 FROM claims c WHERE c.FLIGHT_ID_mvt=a.FLIGHT_ID_mvt
                                 AND c.mvt_id!=p.mvt_id)
                QUALIFY count(*) OVER (PARTITION BY p.mvt_id)=1
                ORDER BY p.mvt_id, a.mvt_id
            """).df()
            frames.append(result)
    choices = pd.concat(frames, ignore_index=True).drop_duplicates('mvt_id', keep='first')
    collisions = choices.donor_mvt_id.duplicated(False) | choices.donor_flight_id.duplicated(False)
    audit.loc[choices.loc[collisions, 'mvt_id'], 'reason'] = 'relaxed_donor_collision'
    chosen = choices.loc[~collisions].set_index('mvt_id')
    if not chosen.empty:
        donors = arr.set_index('mvt_id').loc[chosen.donor_mvt_id, PLAN_FIELDS].set_axis(chosen.index)
        filled = out.set_index('mvt_id')
        changes = filled.loc[chosen.index, PLAN_FIELDS].isna() & donors.notna()
        filled.loc[chosen.index, PLAN_FIELDS] = filled.loc[chosen.index, PLAN_FIELDS].combine_first(donors)
        out[PLAN_FIELDS] = filled.loc[out.mvt_id, PLAN_FIELDS].set_axis(out.index)
        ids = chosen.index[changes.any(axis=1)]
        audit.loc[ids, 'stage'] = chosen.loc[ids, 'stage']
        audit.loc[ids, 'confidence'] = 'unique_relaxed'
        audit.loc[ids, 'reason'] = 'recovered'
        audit.loc[ids, ['donor_mvt_id', 'donor_flight_id']] = chosen.loc[ids, ['donor_mvt_id', 'donor_flight_id']]
        audit.loc[ids, 'filled_fields'] = changes.loc[ids].apply(lambda r: ','.join(r.index[r]), axis=1)
        audit.loc[ids, 'anchor_recovered'] = changes.loc[ids, 'AOBT_3_flt']
        audit.loc[ids, 'plan_available_after'] = filled.loc[ids, 'ADEP_flt'].notna()
        audit.loc[ids, 'aobt3_available_after'] = filled.loc[ids, 'AOBT_3_flt'].notna()
    return out, audit.reset_index().set_axis(prior_audit.index)
