"""Fold-scoped flight-plan recovery from supplied arrival movements.

Callers pass observable columns only. No target, movement block timestamp,
whole-dataset cache, or fitted information from evaluation enters matching.
"""

import duckdb
import numpy as np
import pandas as pd

PLAN_FIELDS = [
    'LOBT_flt', 'CALLSIGN_flt', 'ADEP_flt', 'ADES_flt', 'ADES_FILED_flt',
    'MARKET_SEGMENT_flt', 'IOBT_flt', 'FLIGHT_RULE_flt', 'FLIGHT_TYPE_flt',
    'AIRCRAFT_TYPE_flt', 'WK_TBL_CAT_flt', 'AIRCRAFT_OPERATOR_flt',
    'EOBT_1_flt', 'ARVT_1_flt', 'AOBT_3_flt', 'ARVT_3_flt',
]
MATCH_FIELDS = ['mvt_id', 'FLIGHT_ID_mvt', 'FLIGHT_mvt', 'ADEP_mvt',
                'ADES_mvt', 'AIRCRAFT_TYPE_mvt', 'MVT_TIME_UTC_mvt']
PUBLIC_FIELDS = MATCH_FIELDS + PLAN_FIELDS
HIDDEN = {'taxi_sec', 'TAXITIME_SEC_mvt', 'BLOCK_TIME_UTC_mvt'}


def public(frame):
    """Physical allowlist boundary shared by training, validation and submission."""
    return frame.loc[:, PUBLIC_FIELDS]


def recover_exact(departures, arrivals, claims=None):
    """Return filled public fields and an audit of every eligible departure.

    Exact flight number + movement route, landing 0–8 h after takeoff,
    compatible aircraft and consistent plan route. Unknown aircraft is allowed
    as in notebook §2b. Existing IDs must agree; plan ownership and competing
    movement/plan claims are checked before any fill. Raw values always win.
    Confidence is a rule label, not an invented calibrated probability.
    """
    claims = departures if claims is None else claims
    for frame in [departures, arrivals, claims]:
        if HIDDEN.intersection(frame.columns):
            raise ValueError('Recovery must receive observable columns only')
        if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
            raise ValueError('Recovery requires unique nonmissing movement IDs')
    dep = departures
    eligible = dep[PLAN_FIELDS].isna().any(axis=1)
    probes = dep.loc[eligible, PUBLIC_FIELDS]
    audit = probes[['mvt_id', 'ADEP_mvt']].copy()
    audit['raw_flight_id_missing'] = probes.FLIGHT_ID_mvt.isna()
    audit['raw_plan_block_missing'] = probes.ADEP_flt.isna()
    audit['raw_aobt3_missing'] = probes.AOBT_3_flt.isna()
    audit['stage'] = 'unrecovered'
    audit['confidence'] = 'none'
    audit['reason'] = 'no_candidate'
    audit['donor_mvt_id'] = np.nan
    audit['donor_flight_id'] = np.nan
    audit['candidate_count'] = 0
    audit['filled_fields'] = ''
    audit['anchor_recovered'] = False
    audit['static_imputed'] = False
    audit['plan_available_after'] = probes.ADEP_flt.notna()
    audit['aobt3_available_after'] = probes.AOBT_3_flt.notna()
    out = dep.copy()
    if probes.empty or arrivals.empty:
        return out, audit
    # Use a bounded local connection: recovery must not take every laptop CPU.
    with duckdb.connect(config={'threads': 2}) as con:
        con.register('probes', probes)
        con.register('arrivals', arrivals)
        con.register('claims', claims[['mvt_id', 'FLIGHT_ID_mvt']])
        pairs = con.sql("""
            SELECT p.mvt_id, a.mvt_id donor_mvt_id,
                   a.FLIGHT_ID_mvt donor_flight_id,
                   count(*) OVER (PARTITION BY p.mvt_id) candidate_count,
                   (p.AIRCRAFT_TYPE_mvt IS NULL OR a.AIRCRAFT_TYPE_mvt IS NULL
                    OR p.AIRCRAFT_TYPE_mvt = a.AIRCRAFT_TYPE_mvt) aircraft_ok,
                   coalesce(a.ADEP_flt = p.ADEP_mvt AND a.ADES_flt = p.ADES_mvt, false) route_ok,
                   (p.FLIGHT_ID_mvt IS NULL OR p.FLIGHT_ID_mvt = a.FLIGHT_ID_mvt) id_ok,
                   EXISTS (SELECT 1 FROM claims c WHERE c.FLIGHT_ID_mvt = a.FLIGHT_ID_mvt
                           AND c.mvt_id != p.mvt_id) already_owned
            FROM probes p JOIN arrivals a ON a.FLIGHT_mvt = p.FLIGHT_mvt
             AND a.ADEP_mvt = p.ADEP_mvt AND a.ADES_mvt = p.ADES_mvt
             AND a.MVT_TIME_UTC_mvt BETWEEN p.MVT_TIME_UTC_mvt
                                      AND p.MVT_TIME_UTC_mvt + INTERVAL 8 HOUR
            WHERE a.FLIGHT_ID_mvt IS NOT NULL AND a.CALLSIGN_flt IS NOT NULL
            ORDER BY p.mvt_id, a.mvt_id
        """).df()
    if pairs.empty:
        return out, audit
    by_id = audit.set_index('mvt_id')
    counts = pairs.groupby('mvt_id', sort=False).candidate_count.first()
    by_id.loc[counts.index, 'candidate_count'] = counts
    by_id.loc[counts.index, 'reason'] = 'ambiguous'
    single = pairs.loc[pairs.candidate_count.eq(1)].copy()
    for flag, reason in [('aircraft_ok', 'incompatible_aircraft'), ('route_ok', 'incompatible_route'),
                         ('id_ok', 'incompatible_plan_id')]:
        by_id.loc[single.loc[~single[flag], 'mvt_id'], 'reason'] = reason
    by_id.loc[single.loc[single.already_owned, 'mvt_id'], 'reason'] = 'plan_already_owned'
    valid = single.loc[single.aircraft_ok & single.route_ok & single.id_ok & ~single.already_owned]
    # Reject all competing claims, including duplicate arrival rows for one plan.
    collision = valid.donor_mvt_id.duplicated(False) | valid.donor_flight_id.duplicated(False)
    by_id.loc[valid.loc[collision, 'mvt_id'], 'reason'] = 'donor_collision'
    chosen = valid.loc[~collision].set_index('mvt_id')
    if not chosen.empty:
        donors = arrivals.set_index('mvt_id').loc[chosen.donor_mvt_id, PLAN_FIELDS]
        donors.index = chosen.index
        filled = out.set_index('mvt_id')
        changes = filled.loc[chosen.index, PLAN_FIELDS].isna() & donors.notna()
        filled.loc[chosen.index, PLAN_FIELDS] = filled.loc[chosen.index, PLAN_FIELDS].combine_first(donors)
        out[PLAN_FIELDS] = filled.loc[out.mvt_id, PLAN_FIELDS].set_axis(out.index)
        used = changes.any(axis=1)
        ids = chosen.index[used]
        by_id.loc[chosen.index, 'reason'] = 'donor_has_no_new_fields'
        by_id.loc[ids, 'reason'] = 'recovered'
        by_id.loc[ids, 'stage'] = 'exact'
        by_id.loc[ids, 'confidence'] = 'unique_exact'
        by_id.loc[ids, ['donor_mvt_id', 'donor_flight_id']] = chosen.loc[ids, ['donor_mvt_id', 'donor_flight_id']]
        by_id.loc[ids, 'filled_fields'] = changes.loc[ids].apply(
            lambda row: ','.join(row.index[row]), axis=1)
        by_id.loc[ids, 'anchor_recovered'] = changes.loc[ids, 'AOBT_3_flt']
        by_id.loc[ids, 'plan_available_after'] = filled.loc[ids, 'ADEP_flt'].notna()
        by_id.loc[ids, 'aobt3_available_after'] = filled.loc[ids, 'AOBT_3_flt'].notna()
    audit = by_id.loc[probes.mvt_id].reset_index()
    audit.index = probes.index
    return out, audit


def recover_frame(dep, arr, claims=None, mode='exact', reference_dep=None, reference_arr=None):
    """Attach only recovered plan columns to the original model frame."""
    if mode not in ['exact', 'relaxed', 'history']:
        raise ValueError(f'Unsupported recovery mode: {mode}')
    recovered, audit = recover_exact(public(dep), public(arr),
                                      public(claims) if claims is not None else None)
    if mode in ['relaxed', 'history']:
        from recovery_relaxed import recover_relaxed
        recovered, audit = recover_relaxed(recovered, public(arr), public(dep if claims is None else claims),
            public(dep if reference_dep is None else reference_dep),
            public(arr if reference_arr is None else reference_arr), audit)
    if mode == 'history':
        from recovery_history import impute_static
        recovered, audit = impute_static(recovered, public(dep if reference_dep is None else reference_dep),
                                         audit, causal=reference_dep is None)
    result = dep.copy()
    result[PLAN_FIELDS] = recovered[PLAN_FIELDS]
    return result, audit


def summarize_audit(audit):
    recovered = audit.stage.ne('unrecovered')
    groups = ['ADEP_mvt', 'raw_flight_id_missing', 'raw_plan_block_missing', 'stage', 'reason']
    return {'eligible': len(audit), 'recovered': int(recovered.sum()),
            'anchors_recovered': int(audit.anchor_recovered.sum()),
            'static_imputed': int(audit.static_imputed.sum()),
            'groups': audit.groupby(groups, dropna=False).size().rename('n').reset_index().to_dict('records')}
