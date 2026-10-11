"""Unexplored ensemble use of retained original TabM predictions; no new fits.

Pool all3 retained seeds, then allocate a fixed25% of v70's neural component
to this family, leaving the original physical/callsign mix in the other75%.
Old TabM has134 inputs and48-bin encoding; source provenance is retained.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import rome_public_delay_jan_jul as shared
from records import sha256,write_json
ROOT=shared.ROOT;OUT=ROOT/'VALIDATION/runs/v3_20261011_tabm_retained_blend'
ARCHIVE=ROOT/'VALIDATION/runs/v3_20261003_nn_improve_jan_jul'
DESIGN=dict(parent='v70',change='fixed25% neural share to mean of3 original TabM seeds',seeds=[42,142,242],
    tabm_inputs=134,tabm_bins=48,no_new_fits=True,no_weight_search=True,
    old_recipe='same Jan/Jul folds; original ordinary labels/population; original48-bin input preprocessing',
    primary='existing physical3/calls15 neural mixture retains75% share',secondary='fixed final delta to v74')

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'result.json').exists()
    source=shared.shared.ts.read(ARCHIVE/'result.json')
    assert source['source_sha256']['runner']==sha256(ROOT/'VALIDATION/nn_improve_jan_jul.py')
    assert source['source_sha256']['v47_runner']==sha256(ROOT/'VALIDATION/nn_jan_jul.py')
    ph=shared.physical;replay=ph.replay;sc,rome,_,trees=replay.baseline.baseline();frames=[]
    parent_cv=pd.read_parquet(ph.OUT/'pool3/cv/predictions.parquet').set_index('mvt_id')
    best=pd.read_parquet(shared.constraints.OUT/'submission_v74/cv.parquet').set_index('mvt_id')
    raw=pd.read_parquet(shared.features.base.OUT/'rows2025.parquet').set_index('mvt_id');matches=pd.read_parquet(shared.constraints.exclusion.OUT/'matches2025.parquet')
    identity=shared.shared.ts.read(ph.production.cv.MANIFEST);sources={str((ARCHIVE/'result.json').relative_to(ROOT)):sha256(ARCHIVE/'result.json')}
    for month in shared.shared.ts.SPLITS:
        d=sc.d[month];ids=d['ids'];tab=[]
        for seed in DESIGN['seeds']:
            path=ARCHIVE/'networks'/f'tabm_{month}_s{seed}_k0.parquet';f=pd.read_parquet(path)
            np.testing.assert_array_equal(f.mvt_id,ids);assert f.base.notna().all()
            tab.append(f.base.to_numpy(float));sources[str(path.relative_to(ROOT))]=sha256(path)
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        old_nn=.5*(calls+np.mean([replay.values(ph.verify(ph.pool.completed(month,k,identity)['prediction']),ids) for k in [0,1,2]],axis=0))
        new_nn=.75*old_nn+.25*np.mean(tab,axis=0)
        main=replay.values(replay.combo.parent.control.candidate.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        tree=np.mean([main,trees['airport'][42][month],trees['catboost'][42][month]],axis=0)
        p=.5*replay.combo.parent.cat.probabilities(month,ids)+.5*rome[month]
        control=sc.final(month,.71*old_nn+.29*tree,p);np.testing.assert_allclose(control,parent_cv.loc[ids,'pred'],rtol=0,atol=1e-9)
        pred=sc.final(month,.71*new_nn+.29*tree,p)
        v74=best.loc[ids,'pred'].to_numpy(float);q=shared.constraints.matched_bound(raw.loc[ids].reset_index(),v74+pred-control)
        permitted=matches[matches.donor_month.isin(shared.features.base.FOLDS[month]+[month])]
        q,_=shared.constraints.exclusion.apply(ids,np.maximum(0,q),permitted)
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],airport=raw.loc[ids,'airport'].to_numpy(),y=d['y'],control=control,pred=pred,v74=v74,transferred=q,tabm=np.mean(tab,axis=0))))
    f=pd.concat(frames,ignore_index=True);f.to_parquet(OUT/'scored.parquet',index=False)
    columns=['control','pred','v74','transferred'];scores={m:{n:float(np.sqrt(np.mean((q[n]-q.y)**2))) for n in columns} for m,q in f.groupby('month')}
    paired={f'{b}_vs_{a}':shared.shared.ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2) for a,b in [('control','pred'),('v74','transferred')]}
    result=dict(design=DESIGN,scores=scores,cv={n:float(np.mean([q[n] for q in scores.values()])) for n in columns},paired=paired,sources=sources,source=shared.features.base.artifact(__file__))
    write_json(OUT/'result.json',result);print(scores,paired,flush=True)
if __name__=='__main__':
    with threadpool_limits(1):main()
