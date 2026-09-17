"""Static-category imputation, causal by day for training and frozen for evaluation."""

import json

import duckdb
import pandas as pd

from recovery import HIDDEN

KEYS = {'L1': ['FLIGHT_mvt', 'ADEP_mvt'], 'L2': ['prefix', 'ADEP_mvt'],
        'L3': ['prefix'], 'LT': ['AIRCRAFT_TYPE_mvt']}
# field: (level, observed support, modal agreement, low-confidence flag)
POLICY = {
    'AIRCRAFT_OPERATOR_flt': [('L1', 3, .8, False), ('L2', 3, .8, False), ('L3', 3, .8, False)],
    'MARKET_SEGMENT_flt': [('L1', 3, .8, False), ('L2', 10, .99, False), ('L3', 10, .99, False)],
    # Notebook's low-confidence prefix flight-type assignments are excluded.
    'FLIGHT_TYPE_flt': [('L1', 3, .8, False)],
    'WK_TBL_CAT_flt': [('LT', 3, .8, False), ('L1', 3, .8, False), ('L2', 3, .9, False)],
    'AIRCRAFT_TYPE_flt': [('LT', 3, .8, False), ('L1', 3, .8, True)],
}


def impute_static(dep, reference, prior_audit, *, causal):
    if HIDDEN.intersection(dep.columns) or HIDDEN.intersection(reference.columns):
        raise ValueError('History imputation must receive observable columns only')
    out = dep.copy().set_index('mvt_id', drop=False)
    audit = prior_audit.copy().set_index('mvt_id')
    audit['static_details'] = ''
    reference = reference.loc[reference.FLIGHT_ID_mvt.notna() & reference.ADEP_flt.notna()]
    details = {}
    with duckdb.connect(config={'threads': 2}) as con:
        con.register('reference', reference)
        con.execute("""
            CREATE MACRO prefix_of(s) AS CASE
                WHEN regexp_matches(s, '^[A-Z]{3}[0-9]') THEN s[1:3]
                WHEN regexp_matches(s, '^([A-Z][A-Z]|[A-Z][0-9]|[0-9][A-Z])[0-9]{1,4}[A-Z]?$') THEN s[1:2] END;
            CREATE TABLE hist AS SELECT *, prefix_of(FLIGHT_mvt) prefix,
                date_trunc('day', MVT_TIME_UTC_mvt AT TIME ZONE 'UTC') AS event_day FROM reference;
        """)
        for field, steps in POLICY.items():
            for level, support, agreement, low_confidence in steps:
                missing = out.loc[out[field].isna()]
                if missing.empty:
                    break
                con.register('missing', missing.reset_index(drop=True))
                keys = KEYS[level]
                keycols = ', '.join(keys)
                equal_keys = ' AND '.join(f'q.{k}=h.{k}' for k in keys)
                temporal = 'AND h.event_day < q.event_day' if causal else ''
                # Aggregate by day before joining: prefix-level history stays small.
                chosen = con.sql(f"""
                    WITH targets AS (SELECT mvt_id, {keycols},
                        date_trunc('day', MVT_TIME_UTC_mvt AT TIME ZONE 'UTC') AS event_day
                        FROM (SELECT *, prefix_of(FLIGHT_mvt) prefix FROM missing)),
                    queries AS (SELECT DISTINCT {keycols}, event_day FROM targets),
                    h AS (SELECT {keycols}, event_day, {field} val, count(*) n FROM hist
                        WHERE {field} IS NOT NULL GROUP BY ALL),
                    counts AS (SELECT {', '.join('q.'+k for k in keys)}, q.event_day, h.val, sum(h.n) n
                        FROM queries q JOIN h ON {equal_keys} {temporal} GROUP BY ALL),
                    ranked AS (SELECT *, sum(n) OVER (PARTITION BY {keycols}, event_day) total,
                        row_number() OVER (PARTITION BY {keycols}, event_day ORDER BY n DESC, val) rank FROM counts),
                    modes AS (SELECT * FROM ranked WHERE rank=1 AND total>={support} AND n/total>={agreement})
                    SELECT t.mvt_id, m.val, m.total support, m.n/m.total agreement
                    FROM targets t JOIN modes m ON t.event_day=m.event_day
                        AND {' AND '.join(f't.{k}=m.{k}' for k in keys)} ORDER BY t.mvt_id
                """).df()
                if chosen.empty:
                    continue
                chosen = chosen.set_index('mvt_id')
                out.loc[chosen.index, field] = chosen.val
                for ident, row in chosen.iterrows():
                    details.setdefault(ident, []).append(dict(field=field, level=level,
                        value=row.val, support=int(row.support), agreement=float(row.agreement),
                        low_confidence=low_confidence))
    if details:
        ids = pd.Index(details)
        audit.loc[ids, 'static_imputed'] = True
        bare = ids[audit.loc[ids, 'stage'].eq('unrecovered')]
        audit.loc[bare, 'stage'] = 'static'
        audit.loc[bare, 'reason'] = 'static_categories_only'
        audit.loc[bare, 'confidence'] = 'gated_history'
        for ident, values in details.items():
            audit.loc[ident, 'static_details'] = json.dumps(values, sort_keys=True)
            fields = ','.join(v['field'] for v in values)
            audit.loc[ident, 'filled_fields'] = ','.join(filter(None, [audit.loc[ident, 'filled_fields'], fields]))
    return out.reset_index(drop=True).set_axis(dep.index), audit.reset_index().set_axis(prior_audit.index)
