"""Read-only v58 residual study from existing Jan/Jul predictions. No fits/uploads."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline')]
import open_meteo_jan_jul as om
import tuning_seasonal as ts

OUT = ROOT/'tmp/research_20261010/error_mechanisms'
OUT.mkdir(parents=True, exist_ok=True)

def main():
    sc, rome, nn, trees = om.baseline()
    frames, summaries = [], {}
    for month in ts.SPLITS:
        d = sc.d[month]
        h, meta = d['held'], d['meta']
        raw, V, y, p = h['raw'], h['V'], d['y'], rome[month]
        base = .71*nn[month]+.29*np.mean([trees[k][42][month] for k in trees], axis=0)
        pred = sc.final(month, base, p=p)
        saved = pd.read_parquet(om.OUT/'all_scored_rows.parquet')
        saved = saved[(saved.month == month) & (saved.seed == 42)]
        assert np.array_equal(saved.mvt_id, d['ids'])
        assert np.allclose(saved.se_control, (pred-y)**2, atol=1e-5)
        f = raw.copy().reset_index(drop=True)
        f['month'], f['y'], f['pred'], f['base'], f['p'] = month, y, pred, base, p
        f['error'], f['se'] = pred-y, (pred-y)**2
        f['gap'] = (pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)-pd.to_datetime(raw.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().to_numpy()
        extra = [c for c in V if c not in f]
        f = pd.concat([f, V[extra].reset_index(drop=True)], axis=1)
        noid = raw.FLIGHT_ID_mvt.isna().to_numpy()
        isrome = raw.ADEP_mvt.eq('LIRF').to_numpy()
        gap = f.gap.to_numpy()
        f['group'] = np.select([isrome & noid & (gap>=50400), isrome & noid & (gap>=14400),
            isrome & noid & (gap>3600),isrome & noid,noid],
            ['rome_14h','rome_4_14h','rome_1_4h','rome_other_noid','other_noid'],'other_id')
        f['airport'] = raw.ADEP_mvt.to_numpy()
        f['day'] = d['day']
        total = f.se.sum()
        def summarize(key):
            rows=[]
            for name, g in f.groupby(key, observed=True):
                rows.append(dict(group=str(name), n=len(g), sse_share=float(g.se.sum()/total),
                                 rmse=float(np.sqrt(g.se.mean())),bias=float(g.error.mean()),
                                 oracle_global_gain=float(np.sqrt(total/len(f))-np.sqrt((total-g.se.sum())/len(f)))))
            return sorted(rows,key=lambda x:x['sse_share'],reverse=True)
        summaries[month] = dict(n=len(f),rmse=float(np.sqrt(f.se.mean())),groups=summarize('group'),airports=summarize('airport'),
            top_days=summarize('day')[:10],top_error_share={str(n):float(f.se.nlargest(n).sum()/total) for n in [1,5,10,25,100,1000]})
        # Fixed, descriptive counterfactuals, never a selected model.
        band=isrome & noid & (gap>=14400) & (gap<43200)
        late=np.minimum(meta['parameters']['late_mean'],np.maximum(gap-3600,60))
        mix=p*gap+(1-p)*late
        effects={}
        for name, weight in [('no_confidence',np.ones(len(f))),('soft_probability',p),
                             ('ramp_0_10',np.clip(p/.1,0,1)),('ramp_0_20',np.clip(p/.2,0,1)),
                             ('hard_10',(p>=.1).astype(float)),('hard_20',(p>=.2).astype(float))]:
            a=pred.copy()
            a[band]=weight[band]*gap[band]+(1-weight[band])*late[band]
            effects[name]=dict(rmse=float(np.sqrt(np.mean((a-y)**2))),changed=int(np.sum(np.abs(a-pred)>1e-6)))
        summaries[month]['rule_ablations']=effects
        summaries[month]['classifier_columns']=list(ts.CLASSIFIER)
        frames.append(f)
    f=pd.concat(frames,ignore_index=True)
    f.to_parquet(OUT/'diagnostic_rows.parquet',index=False)
    keep=['month','mvt_id','ADEP_mvt','FLIGHT_mvt','FLIGHT_ID_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','y','pred','base','p','error','gap','group']
    f.nlargest(150,'se')[keep].to_csv(OUT/'top_errors.csv',index=False)
    f[f.group.isin(['rome_14h','rome_4_14h'])].to_parquet(OUT/'rome_extreme_rows.parquet',index=False)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    with threadpool_limits(2):
        main()
