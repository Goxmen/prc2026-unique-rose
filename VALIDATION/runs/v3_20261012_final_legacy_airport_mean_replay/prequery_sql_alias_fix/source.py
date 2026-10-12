"""Recover v1 exactly from its specified airport means; no fitted model."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'VALIDATION/runs/v3_20261012_final_legacy_airport_mean_replay'


def art(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))


def main():
    assert not (OUT/'result.json').exists(),'Preserve completed inference'
    OUT.mkdir(parents=True,exist_ok=True)
    training=sorted((ROOT/'data').glob('training_2025-*.parquet'))
    assert len(training)==12
    reference=ROOT/'data/derived/scored_submission_cache_20261011/manifest.json'
    cache=json.loads(reference.read_text())['1']
    original=ROOT/cache['file'];assert sha256(original)==cache['sha256']
    paths=[Path(__file__),reference,original,*training,ROOT/'data/ranking.parquet',
           ROOT/'data/final_ranking.parquet',ROOT/'data/submitting.parquet',ROOT/'data/final_submitting.parquet']
    identity=dict(recipe='Full2025 departure label mean at each airport, labels1..7200 inclusive; global mean fallback',
        gate='Original344841 integer predictions exactly equal before final prediction',
        sources=[art(p) for p in paths],no_model_fit=True,no_upload=True)
    if (OUT/'identity.json').exists():assert json.loads((OUT/'identity.json').read_text())==identity
    else:write_json(OUT/'identity.json',identity)
    # SQL aggregation and a separate pandas summation both use integers. Their
    # exact sums/counts agree before mean division and nearest-even rounding.
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        clean=con.execute("""SELECT ADEP_mvt airport,TAXITIME_SEC_mvt label FROM read_parquet(?)
            WHERE PHASE_mvt='DEP' AND TAXITIME_SEC_mvt BETWEEN 1 AND 7200""",
            [[str(p) for p in training]]).df()
        con.register('clean',clean)
        cells=con.execute('SELECT airport,count(*) n,sum(label)::BIGINT total FROM clean GROUP BY airport ORDER BY airport').df()
        independently=clean.groupby('airport').label.agg(['count','sum']).sort_index()
        assert list(cells.airport)==list(independently.index)
        np.testing.assert_array_equal(cells.n,independently['count'])
        np.testing.assert_array_equal(cells.total,independently['sum'])
        assert np.isfinite(clean.label).all() and (clean.label==np.rint(clean.label)).all()
        cells['mean']=cells.total/cells.n
        fallback=float(clean.label.sum()/len(clean))
        means=cells.set_index('airport')['mean']
        records={}
        for view in ['old','final']:
            source=ROOT/('data/ranking.parquet' if view=='old' else 'data/final_ranking.parquet')
            template=ROOT/('data/submitting.parquet' if view=='old' else 'data/final_submitting.parquet')
            ids=pq.read_table(template,columns=['MVT_ID_mvt'])['MVT_ID_mvt'].to_numpy()
            rows=con.execute("SELECT MVT_ID_mvt mvt_id,ADEP_mvt airport,MVT_TIME_UTC_mvt FROM read_parquet(?) WHERE PHASE_mvt='DEP'",[str(source)]).df().set_index('mvt_id')
            assert rows.index.is_unique and set(rows.index)==set(ids)
            rows=rows.loc[ids].reset_index()
            pred=rows.airport.map(means).fillna(fallback).to_numpy(float)
            submitted=np.rint(np.maximum(pred,0))
            if view=='old':
                target=pd.read_parquet(original).set_index('MVT_ID_mvt').loc[ids,'TAXITIME_SEC_mvt'].to_numpy()
                np.testing.assert_array_equal(submitted,target)
            frame=pd.DataFrame(dict(mvt_id=ids,pred=pred,submitted=submitted))
            path=OUT/f'{view}_v1.parquet';frame.to_parquet(path,index=False)
            con.register('rows',rows);con.register('means',cells);con.register('predictions',frame)
            n,u,bad=con.execute('''SELECT count(*),count(distinct p.mvt_id),sum(CASE
                WHEN p.pred IS DISTINCT FROM coalesce(m.total::DOUBLE/m.n,?)
                OR p.submitted IS DISTINCT FROM round_even(greatest(0,coalesce(m.total::DOUBLE/m.n,?)),0)
                THEN 1 ELSE 0 END) FROM predictions p JOIN rows r USING(mvt_id) LEFT JOIN means m USING(airport)''',[fallback,fallback]).fetchone()
            assert n==u==len(ids) and bad==0
            months=pd.to_datetime(rows.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m').value_counts().sort_index().to_dict()
            expected={'2026-01':152719,'2026-07':192122} if view=='old' else {'2026-01':152719,'2026-02':144158,'2026-06':181791,'2026-07':192122}
            assert months==expected
            records[view]=dict(rows=len(ids),prediction=art(path),months=months,all_row_SQL_values_exact=True,
                template_ID_set_order_exact=True,old_integer_reference_exact=(view=='old'))
    write_json(OUT/'result.json',dict(status='pass',version=1,identity=art(OUT/'identity.json'),
        training_rows=len(clean),airport_means=cells.to_dict(orient='records'),fallback=fallback,
        views=records,independent_integer_training_sums_exact=True,no_model_fit=True,no_upload=True))
    print(json.dumps(dict(version=1,old_rows=records['old']['rows'],final_rows=records['final']['rows'],status='pass')))


if __name__=='__main__':
    with threadpool_limits(1):main()
