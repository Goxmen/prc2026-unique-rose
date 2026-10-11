"""Refit audited public-day Rome classifiers and prepare a v74 delta candidate.

No uploads here. Only classifier fits are new; retain all ordinary models and
v74 predictions. Fixed CV transfer recipe, source/template/readback checks.
"""
import argparse,gc,time
from pathlib import Path
import numpy as np,pandas as pd,pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import rome_public_delay_jan_jul as r20
import rome_public_delay_cat as cat
import rome_catboost_submit as parent
from records import write_json,sha256
from submit import fill_template
ROOT=r20.ROOT;features=r20.features;physical=r20.physical
OUT=ROOT/'VALIDATION/runs/v3_20261011_rome_public_delay_submit'

def artifact(path):return features.base.artifact(path)
def read(path):return r20.shared.ts.read(path)
def verify(record):
    p=ROOT/record['path'];assert sha256(p)==record['sha256'];return p

def guard():
    OUT.mkdir(parents=True,exist_ok=True)
    identity=dict(source=artifact(__file__),helpers=[artifact(m.__file__) for m in [r20,cat,parent,features,physical,r20.constraints]],
        feature_audit=artifact(features.OUT/'audit.json'),r20_cv=artifact(r20.OUT/'result.json'),r20_audit=artifact(r20.OUT/'audit.json'),
        cat_cv=artifact(cat.OUT/'result.json'),parent=artifact(r20.constraints.OUT/'submission_v74/manifest.json'))
    p=OUT/'identity.json'
    if p.exists():assert read(p)==identity
    else:write_json(p,identity)

def prepare():
    target=OUT/'prepared.json'
    if target.exists():
        v=read(target)
        for name in ['train','old','ranking_features']:verify(v[name])
        return
    recipe=parent.validate_prepared();train=pd.read_pickle(verify(recipe['train']))
    old=pd.read_pickle(verify(recipe['views']['old']['packet']))
    original=list(train['X']);assert len(original)==68
    table=pd.read_parquet(features.OUT/'features2025.parquet').set_index('mvt_id')
    train['X']=pd.concat([train['X'],table.loc[train['ids'],features.FEATURES].set_axis(train['X'].index)],axis=1)
    ranking=features.build(features.load([ROOT/'data/ranking.parquet']),features.base.public(2026)).set_index('mvt_id')
    old['X']=pd.concat([old['X'],ranking.loc[old['ids'][old['use']],features.FEATURES].set_axis(old['X'].index)],axis=1)
    assert list(train['X'])==list(old['X'])==original+features.FEATURES
    assert len(train['X'])==160577 and train['y'].sum()==33205 and len(old['ids'])==344841
    (OUT/'inputs').mkdir(exist_ok=True);pd.to_pickle(train,OUT/'inputs/train.pkl');pd.to_pickle(old,OUT/'inputs/old.pkl')
    ranking.reset_index().to_parquet(OUT/'ranking_features.parquet',index=False)
    write_json(target,dict(identity=artifact(OUT/'identity.json'),train=artifact(OUT/'inputs/train.pkl'),old=artifact(OUT/'inputs/old.pkl'),
        ranking_features=artifact(OUT/'ranking_features.parquet'),parent_prepared=artifact(parent.OUT/'prepared.json'),
        public=artifact(features.base.SOURCE/'all_pre_departure_delays_2026.csv'),rows=160577,positives=33205,columns=list(train['X'])))
    print('Production classifier packets prepared',flush=True)

def fit(arm):
    prepared=read(OUT/'prepared.json');train=pd.read_pickle(verify(prepared['train']));old=pd.read_pickle(verify(prepared['old']))
    (OUT/'models').mkdir(exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
    out=OUT/'predictions/r20.parquet';mp=OUT/'models/r20.txt'
    if arm in ['r20','both'] and not out.exists():
        spec=r20.shared.hpo.trial_spec('rome','R20')
        params=dict(r20.shared.rome_hybrid.CLASSIFIER_PARAMS,**spec['overrides'],seed=42,bagging_seed=43,feature_fraction_seed=44,num_threads=2)
        start=time.monotonic();model=r20.shared.signals._train(params,train['X'],train['y'],1700,'Public-day full-year R20')
        p=np.zeros(len(old['ids']));p[old['use']]=model.predict(old['X'],num_threads=2)
        model.save_model(str(mp));pd.DataFrame(dict(mvt_id=old['ids'],p=p)).to_parquet(out,index=False)
        write_json(out.with_suffix('.json'),dict(identity=artifact(OUT/'identity.json'),prepared=artifact(OUT/'prepared.json'),model=artifact(mp),prediction=artifact(out),params=params,rounds=1700,minutes=(time.monotonic()-start)/60))
        del model;gc.collect()
    elif arm in ['r20','both']:
        rec=read(out.with_suffix('.json'));verify(rec['model']);verify(rec['prediction']);assert rec['identity']==artifact(OUT/'identity.json')
    if arm=='r20':return
    from catboost import CatBoostClassifier,Pool
    _,cats,params=parent.parent_spec();rounds=read(parent.OUT/'selection.json')['rounds'];assert rounds==2017
    X=parent.cbj.frame(train['X'],cats);V=parent.cbj.frame(old['X'],cats)
    for seed in parent.SEEDS:
        out=OUT/'predictions'/f'cat_s{seed}.parquet';mp=OUT/'models'/f'cat_s{seed}.cbm'
        if out.exists():
            rec=read(out.with_suffix('.json'));verify(rec['model']);verify(rec['prediction']);assert rec['identity']==artifact(OUT/'identity.json');continue
        start=time.monotonic();spec=dict(params,random_seed=seed);model=CatBoostClassifier(iterations=rounds,**spec)
        model.fit(Pool(X,train['y'],cat_features=cats));p=np.zeros(len(old['ids']));p[old['use']]=model.predict_proba(Pool(V,cat_features=cats))[:,1]
        model.save_model(str(mp));pd.DataFrame(dict(mvt_id=old['ids'],p=p)).to_parquet(out,index=False)
        write_json(out.with_suffix('.json'),dict(identity=artifact(OUT/'identity.json'),prepared=artifact(OUT/'prepared.json'),model=artifact(mp),prediction=artifact(out),params=spec,rounds=rounds,minutes=(time.monotonic()-start)/60))
        del model;gc.collect()

def probabilities(name,ids):
    p=OUT/'predictions'/name;r=read(p.with_suffix('.json'));assert r['identity']==artifact(OUT/'identity.json')
    f=pd.read_parquet(verify(r['prediction']));verify(r['model']);np.testing.assert_array_equal(f.mvt_id,ids)
    assert f.p.between(0,1).all();return f.p.to_numpy(float)

def build(version,arm):
    dest=OUT/f'submission_v{version}';output=ROOT/f'submissions/unique-rose_v{version}.parquet'
    assert not dest.exists() and not output.exists()
    source=r20.OUT if arm=='r20' else cat.OUT;col={'r20':'transferred','cat':'cat_only_transfer','both':'both_transfer'}[arm]
    result=read(source/'result.json');key={'r20':'transferred_vs_v74','cat':'cat_only_transfer_vs_v74','both':'both_transfer_vs_v74'}[arm]
    assert result['paired'][key]['decision']=='improvement'
    f=pd.read_parquet(source/'scored.parquet');cv=f[['mvt_id','month','day','airport','y','v74']].copy();cv['pred']=f[col];cv['submitted']=np.rint(np.maximum(cv.pred,0))
    data=physical.parent.ranking_data('old');ids=data['rank_ids']
    manifest=read(physical.OUT/'submission_v70/manifest.json');reference=pd.read_parquet(verify(manifest['parts']))
    np.testing.assert_array_equal(reference.mvt_id,ids)
    control,masks=physical.parent.final_frame(data,reference.nn_base.to_numpy(),reference.gbm_base.to_numpy(),reference.p.to_numpy())
    np.testing.assert_array_equal(control.pred,reference.pred)
    assert masks==manifest['correction_masks']
    oldcat_path=verify(read(parent.OUT/'old.json')['prediction'])
    c=pd.read_parquet(oldcat_path);np.testing.assert_array_equal(c.mvt_id,ids);oldcat=c.p.to_numpy(float)
    if arm in ['cat','both']:cats=np.mean([probabilities(f'cat_s{s}.parquet',ids) for s in parent.SEEDS],axis=0)
    else:cats=oldcat
    if arm=='cat':
        r20p=2*reference.p.to_numpy(float)-oldcat
        assert r20p.min()>-1e-15 and r20p.max()<1+1e-15
        np.testing.assert_allclose(.5*(oldcat+r20p),reference.p,rtol=0,atol=1e-15)
        r20p=np.clip(r20p,0,1)
    else:r20p=probabilities('r20.parquet',ids)
    new,masks=physical.parent.final_frame(data,reference.nn_base.to_numpy(),reference.gbm_base.to_numpy(),.5*(cats+r20p))
    old=pd.read_parquet(ROOT/'submissions/unique-rose_v74.parquet');np.testing.assert_array_equal(old.MVT_ID_mvt,ids)
    raw=pd.read_parquet(features.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    pred=r20.constraints.matched_bound(raw,old.TAXITIME_SEC_mvt.to_numpy(float)+new.pred.to_numpy()-control.pred.to_numpy())
    matched=pd.read_parquet(r20.constraints.exclusion.OUT/'ranking_matches.parquet')
    pred,audit=r20.constraints.exclusion.apply(ids,np.maximum(0,pred),matched)
    template_path=ROOT/'data/submitting.parquet';template=pq.read_table(template_path);table=fill_template(template,ids,pred)
    assert table.schema==template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    expected=np.rint(np.maximum(pred,0)).astype('int32');np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),expected)
    outside=raw.airport.ne('LIRF').to_numpy();np.testing.assert_array_equal(expected[outside],old.TAXITIME_SEC_mvt.to_numpy()[outside])
    dest.mkdir(parents=True);cv.to_parquet(dest/'cv.parquet',index=False)
    scores={m:float(np.sqrt(np.mean((g.pred-g.y)**2))) for m,g in cv.groupby('month')};score=float(np.mean(list(scores.values())))
    write_json(dest/'cv.json',dict(cv=score,months=scores,paired=result['paired'][key],prediction=artifact(dest/'cv.parquet'),source=artifact(source/'result.json'),caveat='Reused development; fixed delta transfer from v70 to leaderboard-conditioned v74'))
    pd.DataFrame(dict(mvt_id=ids,parent=old.TAXITIME_SEC_mvt,control=control.pred,candidate=new.pred,p=.5*(cats+r20p),pred=pred,submitted=expected)).to_parquet(dest/'predictions.parquet',index=False)
    pq.write_table(table,output);assert pq.read_table(output).equals(table)
    paths=[Path(__file__),Path(r20.__file__),Path(cat.__file__),Path(features.__file__),Path(parent.__file__),Path(physical.__file__),
        OUT/'identity.json',OUT/'prepared.json',OUT/'inputs/train.pkl',OUT/'inputs/old.pkl',OUT/'ranking_features.parquet',
        source/'result.json',source/'scored.parquet',r20.constraints.OUT/'submission_v74/manifest.json',ROOT/'submissions/unique-rose_v74.parquet',
        physical.OUT/'submission_v70/manifest.json',verify(manifest['parts']),features.base.OUT/'ranking_inputs.parquet',
        r20.constraints.exclusion.OUT/'ranking_matches.parquet',template_path,features.base.SOURCE/'all_pre_departure_delays_2026.csv',
        dest/'cv.json',dest/'cv.parquet',dest/'predictions.parquet']
    paths+=[oldcat_path,parent.OUT/'old.json',parent.OUT/'selection.json']
    paths+=list((OUT/'predictions').glob('*'))+list((OUT/'models').glob('*'))
    sources={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    write_json(dest/'manifest.json',dict(version=version,state='prepared',parent='v74',change=f'public daily aggregate inputs in Rome {arm}',arm=arm,
        file=str(output.relative_to(ROOT)),sha256=sha256(output),rows=len(ids),jan_jul_cv=score,cv_record=artifact(dest/'cv.json'),sources=sources,
        authorization='October11 user authorized autonomous uploads',audit=dict(parent_v70_exact=True,template_ids_and_schema_exact=True,readback_exact=True,only_rome_changes=True),
        changed_rows=int((expected!=old.TAXITIME_SEC_mvt).sum()),max_abs_change=float(np.max(np.abs(expected-old.TAXITIME_SEC_mvt)))))
    print(dict(version=version,arm=arm,cv=score,changes=int((expected!=old.TAXITIME_SEC_mvt).sum())),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','fit','build']);p.add_argument('--arm',choices=['r20','cat','both'],default='cat');p.add_argument('--version',type=int,default=75);a=p.parse_args()
    with threadpool_limits(2):
        guard()
        if a.stage=='prepare':prepare()
        elif a.stage=='fit':fit(a.arm)
        else:build(a.version,a.arm)
