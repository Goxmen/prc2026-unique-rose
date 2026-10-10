"""Replay saved current-fold predictions for a new leaderboard-selected blend.

No fits. Tree seed42, historical neural pools and the original correction order
are retained. The 2026-only probes (v52/v55) and nearest-row recipe (v59) are
excluded. Each output is a genuine prediction, never reconstructed from SSE.
"""
import os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import numpy as np,pandas as pd,duckdb
from threadpoolctl import threadpool_limits
import open_meteo_jan_jul as baseline,open_meteo_all as wx
import sparse_track_combo_score as combo
import nn_improve_jan_jul as ple
import rome_cost_jan_jul as rc
from registered_rules import apply_rule
from records import sha256,write_json
OUT=ROOT/'VALIDATION/runs/v3_20261011_modern_prediction_replay'
SOURCES={}

def values(path,ids,column='base'):
    path=Path(path)
    SOURCES[str(path.relative_to(ROOT))]=sha256(path)
    f=pd.read_parquet(path).set_index('mvt_id')
    assert f.index.is_unique and np.isin(ids,f.index).all(),str(path)
    return f.loc[ids,column].to_numpy(float)

def seen_fn():
    # Only the membership/count needed for v56's unseen-number fallback.
    with duckdb.connect(config={'threads':2}) as con:
        con.execute("SET TimeZone='UTC'")
        f=con.execute('''SELECT MVT_ID_mvt mvt_id, ADEP_mvt, FLIGHT_ID_mvt,
            FLIGHT_mvt, MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt
            FROM read_parquet(?) WHERE PHASE_mvt='DEP' AND ADEP_mvt='LIRF'
            AND TAXITIME_SEC_mvt BETWEEN 1 AND 200000''',
            [str(ROOT/'data/training_2025-*.parquet')]).df()
    _,rule=apply_rule(np.zeros(len(f)),f,'rome_4h')
    f=f.loc[~rule.rule_triggered.to_numpy()]
    f['month']=pd.to_datetime(f.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m')
    for path in sorted((ROOT/'data').glob('training_2025-*.parquet')):
        SOURCES[str(path.relative_to(ROOT))]=sha256(path)
    return {m:set(f.loc[f.month.isin(baseline.ts.FOLDS[m]),'FLIGHT_mvt'].fillna('').astype(str))
            for m in baseline.ts.SPLITS}

def main():
    assert not (OUT/'complete.json').exists(),'Completed replay is retained; do not repeat'
    OUT.mkdir(parents=True,exist_ok=True)
    sc,rome,pool,trees=baseline.baseline()
    seen=seen_fn();scores={};parts=[]
    for m in baseline.ts.SPLITS:
        d=sc.d[m];ids=d['ids'];held=d['held'];pred={}
        sp=baseline.ts.split_dir(baseline.hpo.SPLITS_FROM,m)
        for path in [sp/'heldout.pkl',sp/'meta.json']:
            SOURCES[str(path.relative_to(ROOT))]=sha256(path)
        old=ROOT/'VALIDATION/runs/v3_20261002_adsblol_jan_jul/predictions'
        for v,arm in [(44,'control'),(45,'cov'),(46,'stand')]:
            b=values(old/f'{arm}_seed42.parquet',ids,'r5000')
            # adsblol_jan_jul stores the final corrected prediction here.
            pred[v]=b
        p47=ROOT/'VALIDATION/runs/v3_20261002_nn_v46_jan_jul/predictions/nn_seeds42_142_242.parquet'
        tree46=values(p47,ids,'lgb_base_42')
        np.testing.assert_allclose(sc.final(m,tree46),pred[46],atol=.05,rtol=0)
        pred[47]=sc.final(m,.7*tree46+.3*values(p47,ids,'nn_base_42'))
        oldnets={}
        for s in [42,142,242]:
            for k in range(5):
                oldnets[s,k]=values(ple.OUT/'networks'/f'ple_{m}_s{s}_k{k}.parquet',ids)
        n5=np.mean([oldnets[42,k] for k in range(5)],axis=0)
        n15=np.mean(list(oldnets.values()),axis=0)
        for v,w,neural in [(48,.6,n5),(49,.3,n5),(50,.6,n15)]:
            pred[v]=sc.final(m,(1-w)*tree46+w*neural)
        original_tree=np.mean([d['members'][a,42] for a in baseline.hpo.MEMBERS],axis=0)
        base51=.6*d['pool']+.4*original_tree
        pred[51]=sc.final(m,base51)
        pred[53]=sc.final(m,base51,rome[m])
        hpotree=np.mean([trees[p][42][m] for p in trees],axis=0)
        for v,w in [(54,.6),(57,.45),(58,.71)]:
            pred[v]=sc.final(m,w*pool[m]+(1-w)*hpotree,rome[m])
        pfn=values(rc.OUT/'predictions/F_s42.parquet',ids,'c1700').copy()
        raw=held['raw'];lh=(raw.ADEP_mvt.to_numpy()=='LIRF')&np.isfinite(held['V'][rc.rome_hybrid.GAP].to_numpy(float))
        unseen=lh&~raw.FLIGHT_mvt.fillna('').astype(str).isin(seen[m]).to_numpy()
        pfn[unseen]=rome[m][unseen]
        pred[56]=sc.final(m,.6*pool[m]+.4*hpotree,pfn)
        pweather=values(wx.pred_path('rome',m,42),ids,'pred')
        pred[60]=sc.final(m,.71*pool[m]+(1-.71)*hpotree,pweather)
        weather_tree=np.mean([values((wx.OUT/'predictions'/f'{m}_s42.parquet') if p=='lgb'
            else wx.pred_path(p,m,42),ids,'pred') for p in ['lgb','airport','catboost']],axis=0)
        weather_nn=np.mean([values(wx.OUT/'networks'/f'{m}_s{s}_k{k}.parquet',ids)
            for s in [42,142,242] for k in range(5)],axis=0)
        pred[61]=sc.final(m,.71*weather_nn+(1-.71)*weather_tree,pweather)
        calls=values(combo.sequence.CALLS_PRED,ids,'candidate_nn')
        sparse=values(combo.sequence.OUT/'scored_pool3.parquet',ids,'candidate_nn')
        main=values(combo.parent.control.candidate.OUT/'predictions'/f'{m}_s42.parquet',ids,'pred')
        pc=.5*combo.parent.cat.probabilities(m,ids)+.5*rome[m]
        tc=np.mean([main,trees['airport'][42][m],trees['catboost'][42][m]],axis=0)
        pred[64]=sc.final(m,.71*(.5*(calls+sparse))+(1-.71)*tc,pc)
        expected=values(combo.OUT/'scored_errors.parquet',ids,'half_sparse3_combination')
        np.testing.assert_array_equal((pred[64]-d['y'])**2,expected)
        f=pd.DataFrame(dict(mvt_id=ids,month=m,day=d['day'],airport=raw.ADEP_mvt.to_numpy(),y=d['y'],
            **{f'v{v}':a for v,a in pred.items()}))
        assert np.isfinite(f[[f'v{v}' for v in pred]]).all().all()
        path=OUT/f'{m}.parquet';assert not path.exists();f.to_parquet(path,index=False)
        for v,a in pred.items():scores.setdefault(str(v),{})[m]=float(np.sqrt(np.mean((a-d['y'])**2)))
        parts.append(dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),rows=len(f)))
        print(m,{v:round(scores[str(v)][m],5) for v in pred},flush=True)
    for row in scores.values():row['cv']=float(np.mean(list(row.values())))
    assert abs(scores['64']['cv']-293.6566180991573)<1e-8
    report=dict(source_sha256=sha256(__file__),scores=scores,parts=parts,inputs=SOURCES,
        folds=baseline.ts.FOLDS,tree_seed=42,note='Existing predictions replayed; reused development months; no fitting.')
    write_json(OUT/'complete.json',report)
    print(scores,flush=True)

if __name__=='__main__':
    with threadpool_limits(2):main()
