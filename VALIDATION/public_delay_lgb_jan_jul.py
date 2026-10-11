"""Add the audited12 public daily inputs to the selected main LightGBM.

Retain all162 original/callsign inputs, labels, LB1 settings,10000 rounds,seed42.
Sequential two-fold desktop run; saved main controls were produced on laptop,
so tiny numerical differences cannot be attributed confidently to the features.
"""
import argparse,gc,time,platform
from pathlib import Path
import lightgbm as lgb,numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
import callsign_lgb_jan_jul as parent
import public_delay_features as features
import rome_public_delay_jan_jul as selected
from records import write_json,sha256
ROOT=parent.ROOT;OUT=ROOT/'VALIDATION/runs/v3_20261011_public_delay_lgb'
DESIGN=dict(parent='selected callsign main LightGBM inside v70',change='12 public IFR daily aggregate inputs',
    original_columns=162,new_features=features.FEATURES,rounds=10000,seed=42,trial='LB1',threads=4,
    primary='same selected ordinary/Cat3-R20 stack; only main tree changes',
    caveat='Saved main control on laptop, candidate on desktop; reused Jan/Jul development')

def guard():
    OUT.mkdir(parents=True,exist_ok=True);parent.guard()
    current=dict(design=DESIGN,source=features.base.artifact(__file__),
        helpers=[features.base.artifact(m.__file__) for m in [parent,features,selected]],
        feature_manifest=features.base.artifact(features.OUT/'manifest.json'),audit=features.base.artifact(features.OUT/'audit.json'))
    p=OUT/'identity.json'
    if p.exists():assert parent.ts.read(p)==current
    else:write_json(p,current)

def fit():
    spec=parent.hpo.trial_spec('lgb','LB1');params=parent.ts.lgb_params(dict(parent.hpo.V44,**spec['overrides']),42,spec['learning_rate']);params['num_threads']=4
    table=pd.read_parquet(features.OUT/'features2025.parquet').set_index('mvt_id')
    (OUT/'models').mkdir(exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    for month in parent.ts.SPLITS:
        path=OUT/'predictions'/f'{month}_s42.parquet';mp=OUT/'models'/f'{month}_s42.txt'
        if path.exists():
            record=parent.ts.read(path.with_suffix('.json'));assert record['identity']==features.base.artifact(OUT/'identity.json')
            assert record['prediction']==features.base.artifact(path) and record['model']==features.base.artifact(mp);continue
        source=parent.input_cache.input_path(month);d=pd.read_pickle(source);X,V=parent.append_pair(d,month)
        assert len(X.columns)==162
        for name,ids in [('X',d['train_ids']),('V',d['ids'])]:
            old=X if name=='X' else V
            extra=table.loc[ids,features.FEATURES].set_axis(old.index)
            value=pd.concat([old,extra],axis=1)
            assert list(value)==list(old)+features.FEATURES
            if name=='X':X=value
            else:V=value
        cats=parent.ideas.signals._categorical(X.columns)+[c for c in parent.ideas.ef.CATEGORICAL if c in X]
        start=time.monotonic();parent.log(f'{month}: {len(X):,} rows,174 inputs,10000 fixed rounds')
        dataset=lgb.Dataset(X,d['y'],categorical_feature=cats)
        model=lgb.train(params,dataset,num_boost_round=10000)
        pred=parent.modeling.predict(model,V,threads=4);assert np.isfinite(pred).all()
        model.save_model(str(mp));pd.DataFrame(dict(mvt_id=d['ids'],pred=pred.astype('float32'))).to_parquet(path,index=False)
        write_json(path.with_suffix('.json'),dict(identity=features.base.artifact(OUT/'identity.json'),input=features.base.artifact(source),
            features=features.base.artifact(features.OUT/'features2025.parquet'),model=features.base.artifact(mp),prediction=features.base.artifact(path),
            params=params,columns=list(X),categoricals=cats,train_rows=len(X),held_rows=len(V),minutes=(time.monotonic()-start)/60,host=platform.node()))
        parent.log(f'{month}: completed in {(time.monotonic()-start)/60:.1f}min')
        del d,X,V,value,old,extra,model,dataset;gc.collect()

def score():
    ph=selected.physical;replay=ph.replay;sc,rome,_,trees=replay.baseline.baseline();frames=[]
    parent_cv=pd.read_parquet(ph.OUT/'pool3/cv/predictions.parquet').set_index('mvt_id')
    best=pd.read_parquet(selected.constraints.OUT/'submission_v74/cv.parquet').set_index('mvt_id')
    raw=pd.read_parquet(features.base.OUT/'rows2025.parquet').set_index('mvt_id');matches=pd.read_parquet(selected.constraints.exclusion.OUT/'matches2025.parquet')
    identity=parent.ts.read(ph.production.cv.MANIFEST)
    for month in parent.ts.SPLITS:
        d=sc.d[month];ids=d['ids'];path=OUT/'predictions'/f'{month}_s42.parquet';rec=parent.ts.read(path.with_suffix('.json'))
        assert rec['prediction']==features.base.artifact(path);new=pd.read_parquet(path);np.testing.assert_array_equal(new.mvt_id,ids)
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        nn=.5*(calls+np.mean([replay.values(ph.verify(ph.pool.completed(month,k,identity)['prediction']),ids) for k in [0,1,2]],axis=0))
        old=replay.values(replay.combo.parent.control.candidate.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        p=.5*replay.combo.parent.cat.probabilities(month,ids)+.5*rome[month]
        control=sc.final(month,.71*nn+.29*np.mean([old,trees['airport'][42][month],trees['catboost'][42][month]],axis=0),p)
        np.testing.assert_allclose(control,parent_cv.loc[ids,'pred'],rtol=0,atol=1e-9)
        pred=sc.final(month,.71*nn+.29*np.mean([new.pred.to_numpy(float),trees['airport'][42][month],trees['catboost'][42][month]],axis=0),p)
        v74=best.loc[ids,'pred'].to_numpy(float);q=selected.constraints.matched_bound(raw.loc[ids].reset_index(),v74+pred-control)
        permitted=matches[matches.donor_month.isin(features.base.FOLDS[month]+[month])]
        q,_=selected.constraints.exclusion.apply(ids,np.maximum(0,q),permitted)
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],airport=raw.loc[ids,'airport'].to_numpy(),y=d['y'],control=control,pred=pred,v74=v74,transferred=q)))
    f=pd.concat(frames,ignore_index=True);f.to_parquet(OUT/'scored.parquet',index=False)
    columns=['control','pred','v74','transferred'];scores={m:{n:float(np.sqrt(np.mean((q[n]-q.y)**2))) for n in columns} for m,q in f.groupby('month')}
    paired={f'{b}_vs_{a}':parent.ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2) for a,b in [('control','pred'),('v74','transferred')]}
    write_json(OUT/'result.json',dict(design=DESIGN,scores=scores,cv={n:float(np.mean([q[n] for q in scores.values()])) for n in columns},paired=paired))
    print(scores,paired,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','score']);a=p.parse_args()
    with threadpool_limits(4):guard();globals()[a.stage]()
