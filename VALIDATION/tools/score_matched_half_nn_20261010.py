"""Saved-prediction audit: compare candidate half blends to a matched N09 half blend."""
import argparse
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
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import open_meteo_jan_jul as baseline
import hpo_jan_jul as hpo
import tuning_seasonal as ts

OUT=ROOT/'tmp/research_20261010/error_mechanisms/matched_half_nn'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main(spread=False,nets=(0,)):
    global OUT
    nets=list(nets)
    if len(set(nets))!=len(nets):raise ValueError('Duplicate network members')
    tag='g42_k'+'_'.join(map(str,nets))
    if nets!=[0]:OUT=OUT.with_name(OUT.name+'_'+tag)
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    sc,rome,pool,trees=baseline.baseline()
    assert hpo.W_NN==.71
    member_frames={}
    for seed in hpo.SEEDS:
        member_frames[seed]=sc.frame('nn','N09',seed)
    extra=sc.frame('nn','N09_k34',42)
    def individual(month,seed,k):
        frame=extra if seed==42 and k>=3 else member_frames[seed]
        return frame[month][f'n{k}'].to_numpy(float)
    def squared(month,pred):
        return np.mean([sc.se(month,seed,nn=pred,p=rome[month],
            **{hpo.MEMBER_OF[part]:trees[part][seed][month] for part in trees})
            for seed in hpo.SEEDS],axis=0)
    controls={};pool_se={};rows=[]
    for month in ts.SPLITS:
        d=sc.d[month];single=np.mean([individual(month,42,k) for k in nets],axis=0)
        # The actual unchanged final stack is applied to each tree seed separately.
        controls[month]=squared(month,.5*single+.5*pool[month])
        pool_se[month]=squared(month,pool[month])
        rows.append(pd.DataFrame(dict(mvt_id=d['ids'],month=month,day=d['day'],
            direct_nn=single,pool_nn=pool[month],se_matched_half=controls[month],se_v58=pool_se[month])))
    control=pd.concat(rows,ignore_index=True)
    assert abs(sc.cv(pool_se)['cv']-295.7414415741871)<1e-7
    path=OUT/'control.parquet';control.to_parquet(path,index=False)
    report=dict(status='passed',interpretation='Same reused development months; paired-day intervals are exploratory, not independent confirmation.',
        nn_share=.71,tree_seeds=hpo.SEEDS,matched_half=f'.5 historical N09 seed42 mean({nets}) + .5 original 15-network v58 pool',
        control=sc.cv(controls),v58=sc.cv(pool_se),
        matched_half_vs_v58=ts.paired_bootstrap(control.day,control.month,control.se_v58,control.se_matched_half),arms={})
    paths=sorted((ROOT/'VALIDATION/runs').glob(f'v3_20261010_nn_*/scored_{tag}.parquet'))
    for source in paths:
        frame=pd.read_parquet(source)
        assert frame[['mvt_id','month','day']].equals(control[['mvt_id','month','day']])
        np.testing.assert_array_equal(frame.direct_nn,control.direct_nn)
        np.testing.assert_allclose(frame.se_v58,control.se_v58,atol=1e-5,rtol=1e-12)
        halfs=[c for c in frame if c.startswith('se_half_')]
        assert len(halfs)==1
        half=halfs[0]
        cn='candidate_nn' if 'candidate_nn' in frame else 'residual_nn'
        checked={}
        for month in ts.SPLITS:
            f=frame[frame.month.eq(month)]
            expect=squared(month,.5*f[cn].to_numpy(float)+.5*pool[month])
            np.testing.assert_allclose(expect,f[half],atol=1e-5,rtol=1e-12)
            checked[month]=expect
        name=source.parent.name.removeprefix('v3_20261010_nn_').removesuffix('_jan_jul')
        report['arms'][name]=dict(scores=sc.cv(checked),
            vs_matched_half=ts.paired_bootstrap(control.day,control.month,control.se_matched_half,frame[half]),
            vs_v58=ts.paired_bootstrap(control.day,control.month,control.se_v58,frame[half]),
            source=str(source.relative_to(ROOT)),source_sha256=sha(source),
            delta_by_month={month:float(np.sqrt(checked[month].mean())-np.sqrt(controls[month].mean())) for month in ts.SPLITS})
        print(name,report['arms'][name]['vs_matched_half'],flush=True)
    report['elapsed_sec']=time.monotonic()-started
    (OUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='arms'},indent=2),flush=True)
    if spread:
        entries=[]
        for seed in hpo.SEEDS:
            for k in range(5):
                full={};half={}
                for month in ts.SPLITS:
                    single=individual(month,seed,k)
                    full[month]=squared(month,single)
                    half[month]=squared(month,.5*single+.5*pool[month])
                entries.append(dict(seed=seed,k=k,single=sc.cv(full),half=sc.cv(half)))
                print('spread',seed,k,entries[-1],flush=True)
        summary={arm:{metric:{'min':float(min(e[arm][metric] for e in entries)),
                    'max':float(max(e[arm][metric] for e in entries)),
                    'mean':float(np.mean([e[arm][metric] for e in entries])),
                    'std':float(np.std([e[arm][metric] for e in entries]))}
                    for metric in [*ts.SPLITS,'cv']} for arm in ['single','half']}
        (OUT/'individual_spread.json').write_text(json.dumps(dict(entries=entries,summary=summary,
            elapsed_sec=time.monotonic()-started,interpretation=report['interpretation']),indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spread',action='store_true')
    parser.add_argument('--nets',type=int,nargs='+',choices=range(5),default=[0]);args=parser.parse_args()
    with threadpool_limits(limits=2):main(args.spread,args.nets)
