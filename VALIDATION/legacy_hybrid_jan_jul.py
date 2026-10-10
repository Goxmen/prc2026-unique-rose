"""Reconstruct v26-v38 recipes and reuse v43 on the current Jan/Jul folds for a new blend.

Reuses audited frozen feature matrices and training-only correction fits.
One CPU fit at a time; completed predictions and saved component models are
hash-guarded. Original full-year submissions are not refitted or modified.
"""
import gc,json,os,sys,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','6')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import duckdb,lightgbm as lgb,numpy as np,pandas as pd
import tuning_seasonal as ts,hpo_jan_jul as hpo
import rome_signals as signals,rome_hybrid as rh
import rome_hybrid_submit as recipes,stand_day_run as stand
import artifact_submit,dayshift_hedge as hedge
from records import sha256,write_json
OUT=ROOT/'VALIDATION/runs/v3_20261011_legacy_hybrid_jan_jul'
VERSIONS=[26,27,28,29,30,33,34,37,38,43]
THREADS=6

def paths(m,v):return OUT/'predictions'/f'{m}_v{v}.parquet'

def artifact(path):
    return dict(path=str(Path(path).relative_to(ROOT)),sha256=sha256(path))

def history():
    with duckdb.connect(config={'threads':2}) as c:
        c.execute("SET TimeZone='UTC'")
        f=c.execute('''SELECT ADEP_mvt,FLIGHT_ID_mvt,FLIGHT_mvt,MVT_TIME_UTC_mvt,
            SCHED_TIME_UTC_mvt,TAXITIME_SEC_mvt y FROM read_parquet(?)
            WHERE PHASE_mvt='DEP' AND ADEP_mvt='LIRF' ''',
            [str(ROOT/'data/training_2025-*.parquet')]).df()
    months=pd.to_datetime(f.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m')
    return {m:hedge.history(f.loc[months.isin(ts.FOLDS[m])],f.loc[months.isin(ts.FOLDS[m]),'y'].to_numpy(float)) for m in ts.SPLITS}

def fitpart(month,name,X,y,raw,extra,part,identity):
    path=OUT/'models'/f'{month}_{name}_{part}.txt';rp=path.with_suffix('.json')
    config=dict(month=month,name=name,part=part,extra=list(extra),training_rows=len(y),
        columns=list(X),identity_sha256=sha256(OUT/'identity.json'),threads=THREADS,seed=42,
        rounds=800 if part=='ordinary' else rh.CLASSIFIER_ROUNDS)
    if rp.exists():
        r=ts.read(rp);assert r['config']==config and sha256(path)==r['model']['sha256']
        return lgb.Booster(model_file=str(path))
    assert not path.exists(),'Unrecorded model: inspect before retrying'
    started=time.monotonic()
    fn=signals.fit_ordinary if part=='ordinary' else signals.fit_classifier
    kwargs=dict(threads=THREADS,extra=extra)
    if part=='ordinary':kwargs['rounds']=800
    b=fn(X,y,raw,'clock',42,**kwargs)
    b.save_model(str(path))
    write_json(rp,dict(config=config,seconds=time.monotonic()-started,model=artifact(path)))
    return b

def save(month,version,ids,pred,identity,**details):
    path=paths(month,version);rp=path.with_suffix('.json')
    if rp.exists():
        r=ts.read(rp);assert r['identity_sha256']==sha256(OUT/'identity.json')
        assert sha256(path)==r['prediction']['sha256'];return
    assert not path.exists(),'Unrecorded predictions: inspect before retrying'
    assert np.isfinite(pred).all() and len(ids)==len(pred)
    pd.DataFrame(dict(mvt_id=ids,month=month,pred=pred)).to_parquet(path,index=False)
    write_json(rp,dict(identity_sha256=sha256(OUT/'identity.json'),version=version,month=month,
        prediction=artifact(path),**details))
    print(f'{month} v{version} complete',flush=True)

def complete(month,version):
    path=paths(month,version);rp=path.with_suffix('.json')
    if not rp.exists():return False
    r=ts.read(rp);assert r['identity_sha256']==sha256(OUT/'identity.json') and r['prediction']['sha256']==sha256(path)
    return True

def main():
    for name in ['models','predictions']:(OUT/name).mkdir(parents=True,exist_ok=True)
    sources={str(Path(mod.__file__).relative_to(ROOT)):sha256(mod.__file__)
        for mod in [ts,hpo,signals,rh,recipes,stand,artifact_submit,hedge]}
    inputs=[artifact(ts.split_dir(hpo.SPLITS_FROM,m)/name) for m in ts.SPLITS
        for name in ['before_drop.pkl','heldout.pkl','meta.json','control.parquet']]
    ident=dict(source=artifact(__file__),helpers=sources,inputs=inputs,folds=ts.FOLDS,
        versions=VERSIONS,seed=42,threads=THREADS,host=os.uname().nodename)
    ip=OUT/'identity.json'
    if ip.exists():assert ts.read(ip)==ident
    else:write_json(ip,ident)
    hist=history()
    for m in ts.SPLITS:
        pending=[v for v in VERSIONS if not complete(m,v)]
        if not pending:continue
        sp=ts.split_dir(hpo.SPLITS_FROM,m);meta=ts.read(sp/'meta.json');held=pd.read_pickle(sp/'heldout.pkl')
        assert meta['train_months']==ts.FOLDS[m] and meta['freeze']==ts.fold_freeze(m)
        V,vr=held['V'],held['raw'];ids=vr.mvt_id.to_numpy()
        before=pd.read_pickle(sp/'before_drop.pkl');X,y=before['X'],np.asarray(before['y'])
        raw=ts.minimal_raw(before['airport'],X.index)
        # v26-v28 share precisely the same 45-input classifier.
        if any(v in pending for v in [26,27,28]):
            clf=fitpart(m,'v26',X,y,raw,(),'classifier',ident)
            p=signals.substitution_probability(clf,V,vr,'clock',THREADS,());del clf
            for v in [26,27,28]:
                if v not in pending:continue
                extra=recipes.RECIPES[v]['extra']
                reg=fitpart(m,f'v{v}',X,y,raw,extra,'ordinary',ident)
                base=signals.ordinary_prediction(reg,V,'clock',THREADS,extra);del reg
                pred=signals.combine(base,p,V,vr)['final']
                save(m,v,ids,pred,ident)
        for v,extra,ce in [(29,stand.REGRESSOR,stand.CLASSIFIER),
                           (30,stand.REGRESSOR+stand.STAND,stand.CLASSIFIER+stand.STAND)]:
            needed=[29] if v==29 else [30,33,34,37,38]
            if not any(n in pending for n in needed):continue
            reg=fitpart(m,f'v{v}',X,y,raw,extra,'ordinary',ident)
            base=signals.ordinary_prediction(reg,V,'clock',THREADS,extra);del reg
            clf=fitpart(m,f'v{v}',X,y,raw,ce,'classifier',ident)
            p=signals.substitution_probability(clf,V,vr,'clock',THREADS,ce);del clf
            pred=signals.combine(base,p,V,vr,late_mean=meta['parameters']['late_mean'])['final']
            if v in pending:save(m,v,ids,pred,ident)
            if v==30:
                frame,_=artifact_submit.apply_corrections(pred,vr,V,meta['corrections'],p)
                for vv,col in [(33,'pred_window'),(34,'pred_control'),(38,'pred_combined')]:
                    if vv in pending:save(m,vv,ids,frame[col].to_numpy(float),ident)
                if 37 in pending:save(m,37,ids,hedge.hedge(frame.pred_control.to_numpy(float),vr,hist[m]),ident,history=hist[m])
        del before,X,y,raw;gc.collect()
        if 43 in pending:
            control=pd.read_parquet(sp/'control.parquet');np.testing.assert_array_equal(control.mvt_id,ids)
            save(m,43,ids,control.pred.to_numpy(float),ident,reused=artifact(sp/'control.parquet'))
        del held,V,vr;gc.collect()
    write_json(OUT/'complete.json',dict(identity_sha256=sha256(ip),versions=VERSIONS))

if __name__=='__main__':main()
