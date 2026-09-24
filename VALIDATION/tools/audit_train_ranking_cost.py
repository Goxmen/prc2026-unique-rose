"""Independent checks of step-3 aggregates against saved losses and neighbours."""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VALIDATION/reports/train_ranking_step3'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    for filename,digest in audit['outputs'].items():
        assert sha(OUT/filename)==digest
    for filename,digest in audit['cache'].items():
        assert sha(ROOT/filename)==digest
    actual=pd.read_csv(OUT/'actual_cost.csv')
    feat=pd.read_parquet(ROOT/'tmp/train_ranking_step1/2025-07.parquet')
    sim=pd.read_parquet(ROOT/'tmp/train_ranking_step2/similarity_rows.parquet').set_index('mvt_id')
    paths=[str(ROOT/f'VALIDATION/runs/v3_20260919_artifact_experiments/oof_A1_seed{s}.parquet') for s in [42,142,242]]
    with duckdb.connect() as con:
        independent=con.execute('''SELECT mvt_id, any_value(airport) AS airport,
            avg(pow(pred_combined-y,2)) AS sq FROM read_parquet(?) GROUP BY mvt_id''',[paths]).df().set_index('mvt_id')
    f=feat.set_index('mvt_id').join(independent[['sq']],validate='one_to_one')
    assert f.sq.notna().all()
    checks=0
    for airport,g in f.groupby('airport'):
        for name,mask in [('all',np.ones(len(g),bool)),('missing_flight_id',g.no_flight_id.eq(1)),
                          ('schedule_gap_over_4h',g.takeoff_minus_sched_min.gt(240)),
                          ('snow',g.wx_snow.eq(1)),('raw_far5',sim.loc[g.index,'raw_far5'])]:
            row=actual[(actual.model=='v38')&(actual.taxi_fold=='A1')&(actual.airport==airport)&(actual.group==name)].iloc[0]
            x=g.loc[mask,'sq']
            assert row.n==len(x)
            np.testing.assert_allclose(row.sse,x.sum(),rtol=1e-12,atol=1e-5)
            np.testing.assert_allclose(row.sse_pct,100*x.sum()/g.sq.sum(),rtol=1e-12,atol=1e-8)
            if len(x): np.testing.assert_allclose(row.rmse,np.sqrt(x.mean()),rtol=1e-12)
            checks+=1
    # Check actual nearest-donor costs on selected source AND ranking rows,
    # independently of the proxy implementation and exported summary tables.
    risk=pd.read_parquet(ROOT/'tmp/train_ranking_step3/v38_neighbour_risk.parquet')
    q=sim.loc[risk.index]
    chosen=[]
    for _,g in q.groupby(['label','airport']):
        chosen.extend(g.sort_index().index[::max(1,len(g)//5)][:5])
    for idx in chosen:
        for view in ['raw','rich']:
            ids=q.loc[idx,[f'{view}_neighbour{i}' for i in range(1,6)]].to_numpy(float)
            expected=sum(float(independent.loc[i,'sq']) for i in ids)/5
            np.testing.assert_allclose(risk.loc[idx,f'{view}_neighbour_proxy_sq5'],expected,rtol=1e-12)
            assert all(independent.loc[i,'airport']==q.loc[idx,'airport'] for i in ids)
    # Novelty is independently recomputed from eligible Jan-Jun training rows.
    train=pd.concat([pd.read_parquet(ROOT/f'tmp/train_ranking_step1/2025-{m:02d}.parquet',columns=['mvt_id','airport','stand']) for m in range(1,7)])
    with duckdb.connect() as con:
        labels=con.execute("SELECT MVT_ID_mvt AS mvt_id, TAXITIME_SEC_mvt AS y FROM read_parquet(?) WHERE PHASE_mvt='DEP'",[[str(p) for p in (ROOT/'data').glob('training_*.parquet')]]).df()
    train=train.merge(labels,on='mvt_id',validate='one_to_one')
    train=train[train.y.between(1,200000)&train.stand.notna()]
    known=set(zip(train.airport,train.stand))
    for ap in ['EDDF','EDDM']:
        g=f[f.airport.eq(ap)]
        absent=g.stand.notna()&pd.Series([(a,s) not in known for a,s in zip(g.airport,g.stand)],index=g.index)
        r=actual[(actual.model=='v38')&(actual.taxi_fold=='A1')&(actual.airport==ap)&(actual.group=='stand_unseen_in_fold_training')].iloc[0]
        assert r.n==absent.sum()
        np.testing.assert_allclose(r.sse,g.loc[absent,'sq'].sum(),rtol=1e-12)
    result=dict(passed=True,independent_sql_slice_checks=checks,
        independent_neighbour_checks=2*len(chosen),independent_stand_support_checks=2,
        output_hashes_checked=True,source_sha256=sha(Path(__file__)))
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
