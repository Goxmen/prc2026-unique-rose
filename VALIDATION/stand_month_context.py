"""Input-only monthly stand context, including stands absent from model training.

Departure clocks use supplied AOBT_3, never departure taxi labels. The query's
own clock is removed. Arrival taxi-in labels are supplied prediction inputs.
This encodes typical stand geometry; no arrival-to-departure drift correction.
"""
import argparse
from pathlib import Path
import statistics

import duckdb
import numpy as np
import pandas as pd

import public_daily_taxi_lgb as base
from records import write_json

ROOT=base.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_stand_month_context'
FEATURES=['stand_month_peer_clock_median','stand_month_peer_clock_log_n',
          'stand_month_arrival_median','stand_month_arrival_log_n']
art,read,verify=base.art,base.read,base.verify
DESIGN=dict(features=FEATURES,keys=['view','airport','stand','UTCmonth'],
    departure_clock='takeoff-AOBT_3 between1 and7200 seconds; exclude query movement from own group',
    arrivals='supplied arrival taxi-in between1 and7200 seconds; airport=ADES',
    aggregate='exact median and log1p(count); no minimum-count gate or airport selection',
    missing='null median and log_n0 when no eligible peer; null/blank stand gets no group',
    source_views='original2025 files separately from original2026 ranking; final files unused',
    availability='full batch supplied movement/plan fields and arrival labels; not an online forecast',
    novelty='same-month stand context for previously unseen/rare stand categories',
    distinction='no coefficient or temporal drift adaptation from the failed arrival-transfer pilot',
    credit='Related within-frame stand context is described by gracious-tornado fs8; independent implementation.',
    reference='https://github.com/il-miscusi/prc-2026-gracious-tornado/blob/main/docs/METHOD.md',
    no_departure_target_or_block=True,no_fit=True,no_upload=True)


def connection():
    con=duckdb.connect(config={'threads':1,'memory_limit':'512MB'})
    con.execute("SET TimeZone='UTC'")
    return con


def raw(con):
    clauses=[];parameters=[]
    for view,path in [('training',str(ROOT/'data/training_2025-*.parquet')),('old',str(ROOT/'data/ranking.parquet'))]:
        clauses.append(f'''SELECT '{view}' AS view,MVT_ID_mvt::BIGINT AS mvt_id,PHASE_mvt AS phase,
            CASE WHEN PHASE_mvt='DEP' THEN ADEP_mvt ELSE ADES_mvt END AS airport,
            nullif(trim(STAND_mvt),'') AS stand,strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month,
            CASE WHEN PHASE_mvt='DEP' THEN epoch(MVT_TIME_UTC_mvt)-epoch(AOBT_3_flt) END AS clock,
            CASE WHEN PHASE_mvt='ARR' THEN TAXITIME_SEC_mvt::DOUBLE END AS taxiin
            FROM read_parquet(?) WHERE PHASE_mvt IN ('ARR','DEP')''')
        parameters.append(path)
    con.execute('CREATE TEMP TABLE r AS '+' UNION ALL '.join(clauses),parameters)


def compute(con):
    return con.execute('''WITH grouped AS (
        SELECT view,airport,stand,month,count(*) AS n,list_sort(list(clock)) AS sorted_values
        FROM r WHERE phase='DEP' AND stand IS NOT NULL AND clock BETWEEN 1 AND 7200 GROUP BY ALL),
      clocks AS (SELECT *,sorted_values[n//2] AS left_value,sorted_values[n//2+1] AS middle_value,
        sorted_values[n//2+2] AS right_value FROM grouped),
      arrivals AS (SELECT view,airport,stand,month,count(*) AS n,median(taxiin) AS value
        FROM r WHERE phase='ARR' AND stand IS NOT NULL AND taxiin BETWEEN 1 AND 7200 GROUP BY ALL),
      joined AS (SELECT q.view,q.mvt_id,q.airport,q.stand,q.month,q.clock,
        c.n,c.left_value,c.middle_value,c.right_value,
        coalesce(q.clock BETWEEN 1 AND 7200,FALSE) AS own,
        a.n AS arrival_n,a.value AS arrival_value
        FROM r q LEFT JOIN clocks c USING(view,airport,stand,month)
        LEFT JOIN arrivals a USING(view,airport,stand,month) WHERE q.phase='DEP')
      SELECT view,mvt_id,airport,stand,month,
        CASE WHEN n IS NULL OR (own AND n=1) THEN NULL
             WHEN NOT own AND n%2=1 THEN middle_value
             WHEN NOT own THEN (left_value+middle_value)/2
             WHEN n%2=0 THEN CASE WHEN clock<=left_value THEN middle_value ELSE left_value END
             WHEN clock<=left_value THEN (middle_value+right_value)/2
             WHEN clock<=middle_value THEN (left_value+right_value)/2
             ELSE (left_value+middle_value)/2 END AS stand_month_peer_clock_median,
        ln(1+greatest(0,coalesce(n,0)-CASE WHEN own THEN 1 ELSE 0 END)) AS stand_month_peer_clock_log_n,
        arrival_value AS stand_month_arrival_median,
        ln(1+coalesce(arrival_n,0)) AS stand_month_arrival_log_n
      FROM joined ORDER BY view,mvt_id''').df()


def checks():
    rows=[];at=1
    for key,values in enumerate([[5],[5,8],[5,8,10],[5,5,8,10],[5,8,8,8,10],[None,None],[0,5,9000,None]]):
        for value in values+[None]:
            rows.append(dict(view='training',mvt_id=at,phase='DEP',airport='X',stand=str(key),month='2025-03',clock=value,taxiin=None));at+=1
        for value in [10,20,30,9000]:
            rows.append(dict(view='training',mvt_id=at,phase='ARR',airport='X',stand=str(key),month='2025-03',clock=None,taxiin=value));at+=1
    frame=pd.DataFrame(rows)
    with connection() as con:
        con.register('r',frame);result=compute(con)
    for row in result.itertuples():
        peers=frame.loc[frame.phase.eq('DEP') & frame.stand.eq(row.stand) & frame.mvt_id.ne(row.mvt_id),'clock']
        valid=[float(v) for v in peers if pd.notna(v) and 1<=v<=7200]
        expected=statistics.median(valid) if valid else np.nan
        np.testing.assert_allclose(row.stand_month_peer_clock_median,expected,equal_nan=True,atol=0,rtol=0)
        np.testing.assert_allclose(row.stand_month_peer_clock_log_n,np.log1p(len(valid)),atol=1e-15,rtol=0)
        assert row.stand_month_arrival_median==20
    return dict(queries=len(result),exact_odd_even_duplicate_single_missing_and_invalid_peer_medians=True)


def freeze():
    assert not (OUT/'manifest.json').exists()
    paths=[Path(__file__),ROOT/'data/ranking.parquet',*sorted(ROOT.glob('data/training_2025-*.parquet'))]
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'manifest.json',dict(design=DESIGN,source=art(__file__),sources=[art(p) for p in paths],checks=checks()))


def guard():
    value=read(OUT/'manifest.json');assert value['design']==DESIGN and value['source']==art(__file__)
    for record in value['sources']:verify(record)
    return value


def build():
    guard();assert not (OUT/'result.json').exists()
    with connection() as con:
        raw(con)
        assert con.execute("SELECT count(*) FROM r WHERE phase='DEP' AND taxiin IS NOT NULL").fetchone()[0]==0
        frame=compute(con)
        counts=dict(con.execute("SELECT view,count(*) FROM r WHERE phase='DEP' GROUP BY view").fetchall())
    assert frame.mvt_id.is_unique
    outputs={};coverage={}
    for view,q in frame.groupby('view'):
        assert len(q)==counts[view]
        if view=='training':assert len(q)==2085047
        else:assert len(q)==344841
        path=OUT/f'features_{view}.parquet';q[['mvt_id',*FEATURES]].to_parquet(path,index=False)
        outputs[view]=art(path)
        coverage[view]={name:int(q[name].notna().sum()) for name in [FEATURES[0],FEATURES[2]]}
    write_json(OUT/'result.json',dict(manifest=art(OUT/'manifest.json'),rows=counts,outputs=outputs,coverage=coverage))
    print(dict(rows=counts,coverage=coverage),flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists();result=read(OUT/'result.json')
    for record in result['outputs'].values():verify(record)
    with connection() as con:
        raw(con)
        for view,record in result['outputs'].items():
            path=verify(record);con.read_parquet(str(path)).create_view('f')
            n,u,bad=con.execute('''SELECT count(*),count(distinct q.mvt_id),sum(CASE WHEN f.mvt_id IS NULL
                OR f.stand_month_peer_clock_log_n<0 OR f.stand_month_arrival_log_n<0
                OR (f.stand_month_peer_clock_log_n=0 AND f.stand_month_peer_clock_median IS NOT NULL)
                OR (f.stand_month_arrival_log_n=0 AND f.stand_month_arrival_median IS NOT NULL)
                THEN 1 ELSE 0 END) FROM (SELECT * FROM r WHERE phase='DEP' AND view=?) q
                FULL JOIN f USING(mvt_id)''',[view]).fetchone()
            assert n==u==result['rows'][view] and bad==0
            # Fixed ID-hash sample, enriched with single-peer and missing-clock cases.
            ids=con.execute('''SELECT f.mvt_id FROM f JOIN r q USING(mvt_id)
                WHERE q.view=? AND q.phase='DEP' ORDER BY hash(f.mvt_id) LIMIT 600''',[view]).fetchnumpy()['mvt_id']
            sample=con.execute('SELECT * FROM r WHERE view=? AND mvt_id IN (SELECT unnest(?))',[view,ids.tolist()]).df()
            con.register('queries',sample)
            linked=con.execute('''SELECT q.mvt_id AS query,p.phase,p.mvt_id,p.clock,p.taxiin FROM queries q
                JOIN r p ON q.view=p.view AND q.airport=p.airport AND q.stand=p.stand AND q.month=p.month''').df()
            output=pd.read_parquet(path).set_index('mvt_id')
            grouped={key:g for key,g in linked.groupby('query')}
            for row in sample.itertuples():
                group=grouped.get(row.mvt_id,linked.iloc[:0])
                values=[]
                for phase,column in [('DEP','clock'),('ARR','taxiin')]:
                    q=group.loc[group.phase.eq(phase) & (group.mvt_id.ne(row.mvt_id) if phase=='DEP' else True),column]
                    valid=[float(v) for v in q if pd.notna(v) and 1<=v<=7200]
                    values.extend([statistics.median(valid) if valid else np.nan,np.log1p(len(valid))])
                np.testing.assert_allclose(output.loc[row.mvt_id,FEATURES].to_numpy(float),values,atol=1e-12,rtol=0,equal_nan=True)
            con.unregister('queries');con.execute('DROP VIEW f')
    write_json(OUT/'audit.json',dict(status='pass',manifest=art(OUT/'manifest.json'),result=art(OUT/'result.json'),
        source=art(__file__),rows=result['rows'],independent_scalar_queries=1200,
        exact_raw_source_query_exclusion_medians_counts=True,no_departure_target_carried=True))
    print('All IDs and1200 independent raw-source median/count reconstructions pass',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','build','audit'])
    globals()[p.parse_args().stage]()
