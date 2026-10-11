"""Fixed CatBoost pool expansion using the same12 public daily features.

All3 original seeds, original fold-specific rounds and training rows unchanged.
Compare Cat3 replacement alone and its fixed half mixture with new R20.
No new early-stopping or hyperparameter selection.
"""
import argparse,gc,time
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import rome_public_delay_jan_jul as r20
import rome_catboost_bagging_jan_jul as parent
from records import write_json,sha256
ROOT=r20.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_rome_public_delay_cat'
DESIGN=dict(parent='Cat3/R20 mixture in v70',change='same12 public daily aggregate inputs in CatBoost',features=r20.features.FEATURES,
    seeds=parent.SEEDS,rounds=parent.ROUNDS,no_stopping_search=True,rows='exact retained original Rome population',
    arms=['newCat3_oldR20','newCat3_newR20'])

def guard():
    OUT.mkdir(parents=True,exist_ok=True)
    identity=dict(design=DESIGN,source=r20.features.base.artifact(__file__),
        helpers=[r20.features.base.artifact(m.__file__) for m in [r20,parent,parent.parent,parent.parent.cbj]],
        inputs={m:r20.features.base.artifact(r20.OUT/'inputs'/f'{m}.pkl') for m in parent.ROUNDS})
    path=OUT/'identity.json'
    if path.exists():assert parent.shared.ts.read(path)==identity
    else:write_json(path,identity)

def fit():
    from catboost import CatBoostClassifier,Pool
    (OUT/'models').mkdir(exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    for month in parent.ROUNDS:
        original,old=parent.validated_parent(month)
        d=pd.read_pickle(r20.OUT/'inputs'/f'{month}.pkl');names=list(d['A']);cats=original['categoricals']
        for field in ['target','train_ids','ids','use']:np.testing.assert_array_equal(d[field],old[field])
        for part in ['A','B']:pd.testing.assert_frame_equal(d[part][original['columns']],old[part][original['columns']])
        X=parent.parent.cbj.frame(d['A'],cats);V=parent.parent.cbj.frame(d['B'],cats)
        for seed in parent.SEEDS:
            pred=OUT/'predictions'/f'{month}_s{seed}.parquet';mp=OUT/'models'/f'{month}_s{seed}.cbm'
            if pred.exists():
                rec=parent.shared.ts.read(pred.with_suffix('.json'));assert rec['identity']==r20.features.base.artifact(OUT/'identity.json')
                assert rec['prediction']==r20.features.base.artifact(pred) and rec['model']==r20.features.base.artifact(mp);continue
            params=dict(original['params'],random_seed=seed)
            start=time.monotonic();model=CatBoostClassifier(iterations=parent.ROUNDS[month],**params)
            model.fit(Pool(X,d['target'],cat_features=cats));p=np.zeros(len(d['ids']))
            p[d['use']]=model.predict_proba(Pool(V,cat_features=cats))[:,1]
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            model.save_model(str(mp));pd.DataFrame(dict(mvt_id=d['ids'],p=p)).to_parquet(pred,index=False)
            write_json(pred.with_suffix('.json'),dict(identity=r20.features.base.artifact(OUT/'identity.json'),
                model=r20.features.base.artifact(mp),prediction=r20.features.base.artifact(pred),seed=seed,month=month,
                rounds=parent.ROUNDS[month],params=params,minutes=(time.monotonic()-start)/60,columns=names,categoricals=cats))
            del model;gc.collect()
        del X,V,d,old;gc.collect()

def score():
    ph=r20.physical;replay=ph.replay;sc,rome,_,trees=replay.baseline.baseline();frames=[]
    parent_cv=pd.read_parquet(ph.OUT/'pool3/cv/predictions.parquet').set_index('mvt_id')
    best=pd.read_parquet(r20.constraints.OUT/'submission_v74/cv.parquet').set_index('mvt_id')
    raw=pd.read_parquet(r20.features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    matches=pd.read_parquet(r20.constraints.exclusion.OUT/'matches2025.parquet')
    identity=parent.shared.ts.read(ph.production.cv.MANIFEST)
    for month in parent.ROUNDS:
        d=sc.d[month];ids=d['ids'];new=[]
        for seed in parent.SEEDS:
            path=OUT/'predictions'/f'{month}_s{seed}.parquet';rec=parent.shared.ts.read(path.with_suffix('.json'))
            assert rec['prediction']==r20.features.base.artifact(path);f=pd.read_parquet(path);np.testing.assert_array_equal(f.mvt_id,ids);new.append(f.p.to_numpy(float))
        newcat=np.mean(new,axis=0);newr20=pd.read_parquet(r20.OUT/'predictions'/f'{month}_s42.parquet')
        np.testing.assert_array_equal(newr20.mvt_id,ids)
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        neural=[replay.values(ph.verify(ph.pool.completed(month,k,identity)['prediction']),ids) for k in [0,1,2]]
        main=replay.values(replay.combo.parent.control.candidate.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        tree=np.mean([main,trees['airport'][42][month],trees['catboost'][42][month]],axis=0)
        ordinary=.71*.5*(calls+np.mean(neural,axis=0))+.29*tree
        oldcat=parent.probabilities(month,ids);control=sc.final(month,ordinary,.5*oldcat+.5*rome[month])
        np.testing.assert_allclose(control,parent_cv.loc[ids,'pred'],atol=1e-9,rtol=0)
        v74=best.loc[ids,'pred'].to_numpy(float);out={}
        for name,rp in [('cat_only',rome[month]),('both',newr20.p.to_numpy(float))]:
            candidate=sc.final(month,ordinary,.5*newcat+.5*rp);out[name]=candidate
            q=r20.constraints.matched_bound(raw.loc[ids].reset_index(),v74+candidate-control)
            permitted=matches[matches.donor_month.isin(r20.features.base.FOLDS[month]+[month])]
            out[name+'_transfer'],_=r20.constraints.exclusion.apply(ids,np.maximum(0,q),permitted)
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],airport=raw.loc[ids,'airport'].to_numpy(),y=d['y'],control=control,v74=v74,newcat=newcat,**out)))
    f=pd.concat(frames,ignore_index=True);f.to_parquet(OUT/'scored.parquet',index=False)
    columns=['control','v74','cat_only','both','cat_only_transfer','both_transfer']
    scores={m:{n:float(np.sqrt(np.mean((q[n]-q.y)**2))) for n in columns} for m,q in f.groupby('month')}
    pairs=[('control','cat_only'),('control','both'),('v74','cat_only_transfer'),('v74','both_transfer')]
    paired={f'{b}_vs_{a}':parent.shared.ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2) for a,b in pairs}
    write_json(OUT/'result.json',dict(design=DESIGN,scores=scores,cv={n:float(np.mean([q[n] for q in scores.values()])) for n in columns},paired=paired))
    print(scores,paired,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','score']);a=p.parse_args()
    with threadpool_limits(2):guard();globals()[a.stage]()
