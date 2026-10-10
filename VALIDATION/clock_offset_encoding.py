"""Whole-month-excluded target encodings of departure recorder clock offsets.

Reference labels come only from supplied, allowed training departures. Every
training query excludes its entire UTC movement month from all group medians
and airport fallbacks. Held queries use the outer fold's training months only.
All four outputs are absent when the query's original actual AOBT_3 is absent.
No schedule replacement, departure-label-dependent query gate, or inner model
stopping selection is allowed. The NN runner uses its parent fixed epochs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'data/derived/clock_offset_encoding_20261010'
FEATURE_NAMES = ['te_dlt_stand', 'te_dlt_op', 'te_dlt_rwy', 'imp_minus_dlt_stand']
GROUP_COLUMNS = dict(te_dlt_stand='stand', te_dlt_op='operator', te_dlt_rwy='runway')
FOLDS = {'2025-01': [f'2025-{m:02d}' for m in range(3,12)],
         '2025-07': ['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
SETTINGS = dict(feature_names=FEATURE_NAMES, groups=GROUP_COLUMNS, minimum_group_support=30,
    max_absolute_offset_sec_exclusive=3600, schedule_copy_tolerance_sec=60,
    offset='BLOCK_TIME_UTC_mvt minus original AOBT_3_flt',
    clean_reference='finite BLOCK,AOBT_3,SCHED; supplied taxi-out>=0; abs(BLOCK-AOBT_3)<3600; abs(BLOCK-SCHED)>60',
    reference_population='all supplied allowed-training departures, including rule-handled rows',
    training_exclusion='entire own UTC movement month, including airport fallback',
    held_reference='all outer-fold allowed training months only',
    fallback='airport clean median from same reference months; missing airport prior gives NaN',
    missing_group='missing/empty and unseen grouping keys use airport fallback',
    query_gate='all four NaN when original actual AOBT_3 absent; no target-dependent applicability',
    corrected_gap='MVT_TIME_UTC_mvt minus original AOBT_3_flt minus stand clock-offset median',
    model_stopping='parent fixed full-fit epochs; no inner stopping search')
QUERY_COLUMNS = ['mvt_id','airport','stand','operator','runway','t','aobt','month']


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def connection():
    con=duckdb.connect(config={'threads':2})
    con.execute("SET TimeZone='UTC'; SET memory_limit='3GB'")
    return con


def read_sources(con, paths):
    """Explicit original fields; no feature cache filled AOBT is used."""
    con.execute('''CREATE OR REPLACE TABLE events AS SELECT
        MVT_ID_mvt::BIGINT mvt_id, ADEP_mvt airport,
        nullif(STAND_mvt,'') stand,nullif(AIRCRAFT_OPERATOR_flt,'') AS "operator",
        nullif(RUNWAY_mvt,'') runway,epoch(MVT_TIME_UTC_mvt) t,
        epoch(AOBT_3_flt) aobt,epoch(BLOCK_TIME_UTC_mvt) block,
        epoch(SCHED_TIME_UTC_mvt) schedule,TAXITIME_SEC_mvt::DOUBLE y,
        strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS "month"
        FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''',[[str(p) for p in paths]])
    n,unique,bad,max_error=con.execute('''SELECT count(*),count(DISTINCT mvt_id),
        count(*) FILTER(WHERE isfinite(t) AND isfinite(block) AND isfinite(y) AND abs(t-block-y)>0.0001),
        max(abs(t-block-y)) FROM events''').fetchone()
    if n!=unique or bad:
        raise ValueError(f'Invalid source labels or IDs: rows={n},unique={unique},bad={bad},max={max_error}')
    missing=con.execute('''SELECT count(*) FILTER(WHERE block IS NULL OR y IS NULL OR t IS NULL),
        count(*) FILTER(WHERE aobt IS NULL),count(*) FILTER(WHERE schedule IS NULL) FROM events''').fetchone()
    return dict(rows=n,unique_ids=unique,label_relationship_max_abs_error_sec=max_error,
                missing_target_or_block_or_takeoff=missing[0],missing_original_aobt=missing[1],missing_schedule=missing[2])


def fit_lookups(events, reference_months, con=None):
    """Fit finite median/support tables on the exact explicit reference months."""
    con=con or connection()
    con.register('encoder_events',events)
    con.register('reference_months',pd.DataFrame({'month':list(reference_months)}))
    con.execute('''CREATE OR REPLACE TEMP TABLE clean AS SELECT *,block-aobt offset_sec
      FROM encoder_events WHERE month IN(SELECT month FROM reference_months)
      AND isfinite(block) AND isfinite(aobt) AND isfinite(schedule) AND isfinite(y)
      AND y>=0 AND abs(block-aobt)<3600 AND abs(block-schedule)>60''')
    pieces=[con.execute('''SELECT 'airport' feature,airport,NULL::VARCHAR AS "key",
        median(offset_sec)::DOUBLE offset_sec,count(*)::BIGINT n
        FROM clean GROUP BY airport ORDER BY airport''').df()]
    for feature,column in GROUP_COLUMNS.items():
        pieces.append(con.execute(f'''SELECT '{feature}' feature,airport,{column}::VARCHAR AS "key",
          median(offset_sec)::DOUBLE offset_sec,count(*)::BIGINT n
          FROM clean WHERE {column} IS NOT NULL AND {column}::VARCHAR<>''
          GROUP BY airport,{column} HAVING count(*)>=30 ORDER BY airport,{column}''').df())
    return pd.concat(pieces,ignore_index=True)


def apply_lookups(queries,lookup):
    """Queries read only identity/grouping fields and original observed clocks."""
    q=queries[QUERY_COLUMNS].copy()
    if q.mvt_id.duplicated().any():raise ValueError('Query movement IDs must be unique')
    prior=lookup.loc[lookup.feature.eq('airport')].set_index('airport').offset_sec
    if not prior.index.is_unique:raise ValueError('Duplicate airport prior')
    out=pd.DataFrame(index=pd.Index(q.mvt_id.to_numpy(),name='mvt_id'))
    fallback=q.airport.map(prior).to_numpy(float)
    actual=np.isfinite(q.aobt.to_numpy(float))
    for feature,column in GROUP_COLUMNS.items():
        table=lookup.loc[lookup.feature.eq(feature)].set_index(['airport','key']).offset_sec
        if not table.index.is_unique:raise ValueError('Duplicate encoder grouping key')
        keys=pd.MultiIndex.from_arrays([q.airport,q[column]])
        v=table.reindex(keys).to_numpy(dtype=float,copy=True)
        missing_key=q[column].isna().to_numpy() | q[column].eq('').to_numpy()
        v[~np.isfinite(v) | missing_key]=fallback[~np.isfinite(v) | missing_key]
        v[~actual]=np.nan
        out[feature]=v.astype(np.float32)
    gap=q.t.to_numpy(float)-q.aobt.to_numpy(float)
    out['imp_minus_dlt_stand']=(gap-out.te_dlt_stand.to_numpy(float)).astype(np.float32)
    out.loc[~actual,:]=np.nan
    return out[FEATURE_NAMES]


def build_folds(out=DEFAULT):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'folds.json').exists():raise FileExistsError('Completed encoding exists; reuse its audited manifest')
    sources=sorted((ROOT/'data').glob('training_2025-*.parquet'))
    con=connection(); source_audit=read_sources(con,sources)
    events=con.execute('SELECT * FROM events').df()
    source_manifest=[dict(path=str(p),sha256=sha256(p)) for p in sources]
    reports={};lookups={};started=time.monotonic()
    for held,train in FOLDS.items():
        reports[held]={}
        lookup_dir=out/'lookups'/held;lookup_dir.mkdir(parents=True,exist_ok=True)
        views={'outer':train,**{month:[m for m in train if m!=month] for month in train}}
        for view,refs in views.items():
            table=fit_lookups(events,refs,con)
            path=lookup_dir/f'{view}.parquet'
            if path.exists():raise FileExistsError(path)
            table.to_parquet(path,index=False)
            lookups[(held,view)]=dict(path=str(path),sha256=sha256(path),reference_months=refs,
                rows=len(table),clean_reference_rows=int(table.loc[table.feature.eq('airport'),'n'].sum()))
        for part,query_months in [('train',train),('held',[held])]:
            q=events.loc[events.month.isin(query_months),QUERY_COLUMNS]
            pieces=[]
            for query_month,query in q.groupby('month',sort=False):
                view=query_month if part=='train' else 'outer'
                lookup=pd.read_parquet(lookups[(held,view)]['path'])
                pieces.append(apply_lookups(query,lookup))
            frame=pd.concat(pieces).reindex(q.mvt_id.to_numpy())
            assert len(frame)==len(q) and frame.index.is_unique
            path=out/f'{held}_{part}.parquet'
            if path.exists():raise FileExistsError(path)
            frame.reset_index().to_parquet(path,index=False)
            record=dict(path=str(path),sha256=sha256(path),rows=len(frame),reference_months=train,
                nonmissing={c:int(frame[c].notna().sum()) for c in FEATURE_NAMES},
                lookup_tables={v:lookups[(held,v)] for v in (train if part=='train' else ['outer'])})
            if part=='train':record['per_query_month_reference_months']={m:views[m] for m in train}
            reports[held][part]=record
            print(f'{held} {part}: {len(frame):,} rows, {time.monotonic()-started:.1f}s',flush=True)
    write_json(out/'folds.json',dict(settings=SETTINGS,folds=reports,source_sha256=sha256(Path(__file__)),
        movement_sources=source_manifest,source_audit=source_audit,elapsed_sec=time.monotonic()-started))
    con.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['build'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    args=parser.parse_args();build_folds(args.out)


if __name__=='__main__':main()
