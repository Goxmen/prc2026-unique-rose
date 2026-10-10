"""Frozen saved auxiliary-copy probability replacement/half blend in v58 Rome only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import open_meteo_jan_jul as baseline
import hpo_jan_jul as hpo
import tuning_seasonal as ts
import rome_hybrid as rh

OUT=ROOT/'tmp/research_20261010/error_mechanisms/auxiliary_rome'
SOURCE=ROOT/'VALIDATION/runs/v3_20261010_nn_auxiliary_modes_jan_jul'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def diagnostic(y,p):
    if not len(y):return {'rows':0}
    eps=np.clip(p,1e-15,1-1e-15)
    return dict(rows=len(y),copies=int(y.sum()),copy_rate=float(y.mean()),mean_p=float(p.mean()),
        p_quantiles=np.quantile(p,[0,.25,.5,.75,.95,1]).tolist(),
        predicted_copy_rate_p05=float((p>=.5).mean()),above_stack_gate_005=float((p>.05).mean()),
        brier=float(np.mean((y-p)**2)),logloss=float(-np.mean(y*np.log(eps)+(1-y)*np.log(1-eps))))

def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    sc,rome,pool,trees=baseline.baseline()
    assert hpo.W_NN==.71 and rh.MATCH_SEC==60
    errors={name:{} for name in ['v58','auxiliary_probability','half_auxiliary_half_R20']}
    records={};frames=[]
    for month in ts.SPLITS:
        d=sc.d[month];raw=d['held']['raw'];V=d['held']['V']
        path=SOURCE/'auxiliary'/f'{month}_s42_k0.parquet'
        record=json.loads((SOURCE/'networks'/f'{month}_s42.json').read_text())
        assert sha(path)==record['networks']['0']['auxiliary_sha256']
        aux=pd.read_parquet(path)
        np.testing.assert_array_equal(aux.mvt_id,d['ids'])
        p=aux.p_schedule_copy.to_numpy(float)
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        is_rome=raw.ADEP_mvt.eq('LIRF').to_numpy()
        p=np.where(is_rome,p,0.)
        assert np.all(rome[month][~is_rome]==0)
        probs=dict(v58=rome[month],auxiliary_probability=p,half_auxiliary_half_R20=.5*p+.5*rome[month])
        target=rh.substituted(V,d['y'],raw).astype(float)
        raw_gap=(pd.to_datetime(raw.MVT_TIME_UTC_mvt,utc=True)-pd.to_datetime(raw.SCHED_TIME_UTC_mvt,utc=True)).dt.total_seconds().to_numpy()
        np.testing.assert_array_equal(target,is_rome&(np.abs(raw_gap-d['y'])<=60))
        noid=raw.raw_flight_id_missing.to_numpy().astype(bool)
        masks={'all_rome':is_rome,'no_id':is_rome&noid,'no_id_gap_above1h':is_rome&noid&(raw_gap>3600),'id_present':is_rome&~noid}
        diagnostics={name:{arm:diagnostic(target[mask],pr[mask]) for arm,pr in probs.items()} for name,mask in masks.items()}
        for arm,pr in probs.items():
            errors[arm][month]=np.mean([sc.se(month,seed,nn=pool[month],p=pr,
                **{hpo.MEMBER_OF[part]:trees[part][seed][month] for part in trees}) for seed in hpo.SEEDS],axis=0)
        coverage_source=ROOT/'VALIDATION/runs/v3_20261010_rome_adsblol_jan_jul/inputs'/f'{month}.pkl'
        coverage_cache=pd.read_pickle(coverage_source)
        np.testing.assert_array_equal(coverage_cache['ids'],d['ids'])
        matched=np.zeros(len(d['ids']),bool)
        matched[coverage_cache['use']]=pd.to_numeric(coverage_cache['B'].ab_matched,errors='coerce').fillna(0).gt(0).to_numpy()
        coverage={name:float(matched[mask].mean()) for name,mask in masks.items()}
        del coverage_cache
        records[month]=dict(auxiliary_sha256=sha(path),diagnostics=diagnostics,adsb_own_match_share=coverage)
        frames.append(pd.DataFrame(dict(mvt_id=d['ids'],month=month,day=d['day'],p_R20=rome[month],p_auxiliary=p,
            target=target,**{f'se_{arm}':errors[arm][month] for arm in errors})))
    scored=pd.concat(frames,ignore_index=True)
    scores={name:sc.cv(parts) for name,parts in errors.items()}
    assert abs(scores['v58']['cv']-295.7414415741871)<1e-7
    comparisons={name:ts.paired_bootstrap(scored.day,scored.month,scored.se_v58,scored[f'se_{name}']) for name in errors if name!='v58'}
    # Count the actual auxiliary population under its preserved ordinary-row mask.
    con=duckdb.connect(config={'threads':2});con.execute("SET TimeZone='UTC'")
    r=con.execute('''SELECT ADEP_mvt airport,TAXITIME_SEC_mvt::DOUBLE y,FLIGHT_ID_mvt IS NULL noid,
       epoch(MVT_TIME_UTC_mvt)-epoch(SCHED_TIME_UTC_mvt) gap,strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month
       FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''',[str(ROOT/'data/training_2025-*.parquet')]).df()
    con.close();training={}
    for month in ts.SPLITS:
        q=r[r.month.isin(ts.FOLDS[month])&r.y.between(1,200000)]
        lirf=q.airport.eq('LIRF');copy=(q.gap-q.y).abs().le(60)
        rule=lirf&q.noid&q.gap.ge(14400)
        ordinary=q.loc[~rule&~(lirf&copy)]
        pre=json.loads((SOURCE/'preprocessing'/f'{month}.json').read_text())
        assert len(ordinary)==pre['train_rows']
        ordinary_copy=(ordinary.gap-ordinary.y).abs().le(60)
        assert int(ordinary_copy.sum())==pre['encoding']['auxiliary_schedule_copies']
        original=json.loads((ROOT/'VALIDATION/runs/v3_20261010_rome_catboost_jan_jul/predictions'/f'{month}_s42.json').read_text())
        assert int((lirf&~rule).sum())==original['training_rows']
        assert int((lirf&~rule&copy).sum())==original['positives']
        training[month]=dict(auxiliary_all_rows=len(ordinary),auxiliary_all_copy_positives=int(ordinary_copy.sum()),
            auxiliary_rome_rows=int(ordinary.airport.eq('LIRF').sum()),
            auxiliary_rome_copy_positives=int((ordinary.airport.eq('LIRF')&ordinary_copy).sum()),
            R20_rome_rows=original['training_rows'],R20_rome_copy_positives=original['positives'])
    report=dict(status='passed',scores=scores,comparisons=comparisons,months=records,training=training,
        recipe='v58 ordinary15-network pool,3 unchanged tree seeds,.71 NN share; only Rome probabilities replaced or fixed50/50 mean; nonRome p=0; thresholds/calibration unchanged',
        target='Rome abs((takeoff-original schedule)-raw uncapped taxi)<=60 exactly matches R20',
        caveats=['Exploratory post-hoc use on reused Jan/Jul development labels, not independent confirmation.',
                 'Auxiliary head sees156 ordinary features including22 ADS-B inputs; R20 uses68 features and lacks those22.',
                 'Auxiliary ordinary training population excludes all identified Rome schedule-copy labels; R20 includes both classes.',
                 'ADSB coverage and the training population differ; this is not an isolated architecture comparison.'],
        source_sha256=sha(Path(__file__)),elapsed_sec=time.monotonic()-started)
    scored.to_parquet(OUT/'scored.parquet',index=False)
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','scores','comparisons','training','elapsed_sec']},indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
