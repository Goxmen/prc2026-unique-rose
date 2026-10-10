"""Cost-weighted Brier calibration of R20 using honest inner time holdouts.

Classifier training stays unweighted. The last three allowed training months
are calibration data; the inner classifier and every departure-label history
exclude them. Only two calibration parameters per flight-ID group are fitted.
The outer held-out January/July labels enter final scoring only.
"""
import argparse,gc,os,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from threadpoolctl import threadpool_limits
import tuning_seasonal as ts
import overnight_run as shared
import modeling,rome_hybrid,rome_signals as signals,stand_day,opdi
import modern_prediction_replay as replay
from records import sha256,write_json

ROOT=ts.ROOT;OUT=ts.RUNS/'v3_20261011_rome_cost_calibration'
DESIGN=dict(parent='v64',changed_part='R20 half of Rome probability mixture',
    inner='last three allowed training months held out together; departure histories rebuilt and frozen at inner training end',
    classifier='original68 inputs, unweighted R20,1700 rounds,seed42',
    calibrator='cost-weighted Brier; two parameters per supplied flight-ID group',
    cost='clip(abs(schedule_gap-training_noncopy_median),300,10800)^2; mean-normalized within ID group',
    prior=20.,minimum_rows=50,minimum_effective_rows=30,inverse_temperature_bounds=[.5,2.],bias_bounds=[-2.,2.],
    no_epoch_or_hyperparameter_search=True,ordinary_and_catboost='v64 unchanged',
    validation='reused Jan/Jul development; held targets only in scoring',threads=8)

def art(p):return dict(path=str(Path(p).relative_to(ROOT)),sha256=sha256(p))
def verify(a):
    p=ROOT/a['path'];assert sha256(p)==a['sha256'],p;return p
def log(s):print(time.strftime('[%H:%M:%S] ')+s,flush=True)

def identity():
    config=ts.read(ts.DEFAULT/'config.json')['config']
    inputs={str(p.relative_to(ROOT)):sha256(p) for p in sorted((ROOT/'data').glob('training_2025-*.parquet'))}
    inputs[str(opdi.FEATURES_PATH.relative_to(ROOT))]=sha256(opdi.FEATURES_PATH)
    modules=[ts,shared,modeling,rome_hybrid,signals,stand_day,opdi]
    value=dict(design=DESIGN,config=config,source=art(__file__),inputs=inputs,
        helpers={str(Path(m.__file__).relative_to(ROOT)):sha256(m.__file__) for m in modules})
    p=OUT/'identity.json';OUT.mkdir(exist_ok=True)
    if p.exists():assert ts.read(p)==value
    else:write_json(p,value)
    return value

def extras(X,raw,dep,arr,table):
    X=signals.add_signals(X,raw,arr)
    X=stand_day.add_stand_day(X,raw,dep,arr,features=stand_day.STAND_FEATURES)
    return opdi.add_opdi(X,raw,table)

def objective(theta,z,label,weight):
    q=expit(theta[0]*z+theta[1]);res=q-label;diff=theta-np.array([1.,0.])
    loss=np.sum(weight*res*res)+.5*DESIGN['prior']*(diff@diff)
    d=2*weight*res*q*(1-q)
    grad=np.array([d@z,d.sum()])+DESIGN['prior']*diff
    return float(loss),grad

def calibrator(frame,typical):
    groups={}
    for name,mask in [('id_present',~frame.noid.to_numpy()),('id_missing',frame.noid.to_numpy())]:
        f=frame.loc[mask];z=logit(np.clip(f.p.to_numpy(float),1e-7,1-1e-7));label=f.copy_label.to_numpy(float)
        w=np.clip(np.abs(f.gap.to_numpy(float)-typical),300,10800)**2
        ess=float(w.sum()**2/np.sum(w*w)) if len(w) else 0
        theta=np.array([1.,0.]);status='identity: insufficient support'
        if len(w):w=w/w.mean()
        before=objective(theta,z,label,w)[0]
        if len(f)>=DESIGN['minimum_rows'] and ess>=DESIGN['minimum_effective_rows']:
            result=minimize(objective,theta,args=(z,label,w),jac=True,method='L-BFGS-B',
                bounds=[DESIGN['inverse_temperature_bounds'],DESIGN['bias_bounds']],
                options=dict(maxiter=500,ftol=1e-12,gtol=1e-8))
            assert result.success and result.fun<=before+1e-7,result.message
            theta=result.x;status=str(result.message)
        groups[name]=dict(theta=theta.tolist(),rows=len(f),effective_rows=ess,
            positive=int(label.sum()),objective_before=before,objective_after=objective(theta,z,label,w)[0],status=status)
    return dict(groups=groups,typical_sec=typical,design=DESIGN)

def apply(p,noid,fitted):
    assert fitted['design']==DESIGN
    p=np.asarray(p,float);out=p.copy();z=logit(np.clip(p,1e-7,1-1e-7))
    for name,mask in [('id_present',~noid),('id_missing',noid)]:
        theta=fitted['groups'][name]['theta'];out[mask]=expit(theta[0]*z[mask]+theta[1])
    assert np.isfinite(out).all() and ((out>=0)&(out<=1)).all();return out

def fit_job(name,allowed,dep,arr,table,record):
    dest=OUT/name;done=dest/'calibrator.json'
    if done.exists():
        saved=ts.read(dest/'record.json');assert saved['identity']==art(OUT/'identity.json')
        for a in saved['outputs']:verify(a)
        return
    assert not dest.exists(),'Incomplete inner job needs inspection, not duplicate fits'
    inner_train=allowed[:-3];inner_hold=allowed[-3:]
    assert set(inner_train).isdisjoint(inner_hold) and len(inner_train)>=3
    td,ta,views=ts.seasonal.split_frames(dep,arr,dict(train=inner_train,val=inner_hold))
    freeze=pd.Period(max(inner_train),'M').end_time.strftime('%Y-%m-%d')
    config=dict(record['config'],threads=8,prediction_threads=8)
    log(f'{name}: prepare original features; train {inner_train}, calibrate {inner_hold}, freeze {freeze}')
    X,y,cats,raw=shared.prepare_training(td,ta,config)
    keep=~ts.recipe.rule_rows(raw,config['prediction_rule'])
    X,y,raw=X.loc[keep],y[keep],raw.loc[keep]
    X=extras(X,raw,td,ta,table)
    rome=raw.ADEP_mvt.eq('LIRF').to_numpy();rawL=raw.loc[rome].copy();yL=y[rome]
    A=signals.classifier_matrix(X.loc[rome],rawL,ts.KIND,ts.CLASSIFIER)
    assert A.shape[1]==68
    label=rome_hybrid.substituted(X.loc[rome],yL,rawL).astype(float)
    typical=float(np.median(yL[label==0]));del X,y,raw;gc.collect()
    params=dict(rome_hybrid.CLASSIFIER_PARAMS,num_leaves=127,feature_fraction=.6,min_data_in_leaf=50,
        learning_rate=.02,seed=42,bagging_seed=43,feature_fraction_seed=44,num_threads=8)
    model=signals._train(params,A,label,1700,f'cost calibration inner {name}')
    dest.mkdir();model.save_model(str(dest/'inner_model.txt'));cols=list(A)
    train_ids=rawL.mvt_id.to_numpy();pd.DataFrame(dict(mvt_id=train_ids,y=yL,copy_label=label)).to_parquet(dest/'train_ids.parquet',index=False)
    del A,rawL,yL;gc.collect();frames=[]
    for month in inner_hold:
        vd,va=views[month]
        raw=vd.loc[vd.ADEP_mvt.eq('LIRF') & vd.taxi_sec.between(1,200000)].copy()
        assert not np.intersect1d(raw.mvt_id,train_ids).size
        V=modeling.validation_matrix(td,ta,raw,va,freeze,cats,config)
        V=extras(V,raw,pd.concat([td,vd],ignore_index=True),pd.concat([ta,va],ignore_index=True),table)
        B=signals.classifier_matrix(V,raw,ts.KIND,ts.CLASSIFIER);assert list(B)==cols
        p=model.predict(B,num_threads=8);gap=V[rome_hybrid.GAP].to_numpy(float)
        copy=rome_hybrid.substituted(V,raw.taxi_sec.to_numpy(float),raw)
        # Calibration is used only on finite-clock rows not unconditionally
        # overwritten by the registered four-hour rule.
        eligible=np.isfinite(gap)&~ts.recipe.rule_rows(raw,config['prediction_rule'])
        f=pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(),month=month,y=raw.taxi_sec.to_numpy(float),
            p=p,gap=gap,noid=raw.FLIGHT_ID_mvt.isna().to_numpy(),copy_label=copy)).loc[eligible]
        frames.append(f);del V,B,raw;gc.collect()
    frame=pd.concat(frames,ignore_index=True);assert frame.mvt_id.is_unique
    assert set(frame.month)==set(inner_hold) and name not in set(frame.month)
    frame.to_parquet(dest/'inner_predictions.parquet',index=False)
    fitted=calibrator(frame,typical);write_json(done,fitted)
    write_json(dest/'record.json',dict(identity=art(OUT/'identity.json'),inner_train=inner_train,inner_hold=inner_hold,
        freeze=freeze,columns=cols,params=params,rounds=1700,train_rows=len(train_ids),calibration_rows=len(frame),
        outputs=[art(dest/f) for f in ['inner_model.txt','train_ids.parquet','inner_predictions.parquet','calibrator.json']]))
    log(f'{name}: calibration {fitted["groups"]}')

def score():
    sc,rome,_,trees=replay.baseline.baseline();rows=[]
    for month in ts.SPLITS:
        fitted=ts.read(OUT/month/'calibrator.json');d=sc.d[month];ids=d['ids'];raw=d['held']['raw'];V=d['held']['V']
        use=raw.ADEP_mvt.eq('LIRF').to_numpy()&np.isfinite(V[rome_hybrid.GAP].to_numpy(float))
        p=rome[month].copy();p[use]=apply(p[use],raw.FLIGHT_ID_mvt.isna().to_numpy()[use],fitted)
        cat=replay.combo.parent.cat.probabilities(month,ids)
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        sparse=replay.values(replay.combo.sequence.OUT/'scored_pool3.parquet',ids,'candidate_nn')
        main=replay.values(replay.combo.parent.control.candidate.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        tree=np.mean([main,trees['airport'][42][month],trees['catboost'][42][month]],axis=0)
        base=.71*.5*(calls+sparse)+(1-.71)*tree
        control=sc.final(month,base,.5*cat+.5*rome[month]);candidate=sc.final(month,base,.5*cat+.5*p)
        expected=replay.values(replay.combo.OUT/'scored_errors.parquet',ids,'half_sparse3_combination')
        np.testing.assert_array_equal((control-d['y'])**2,expected)
        rows.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],control=control,pred=candidate,p_r20=p)))
    f=pd.concat(rows,ignore_index=True);f.to_parquet(OUT/'scored.parquet',index=False)
    scores={arm:{m:float(np.sqrt(np.mean((x[arm]-x.y)**2))) for m,x in f.groupby('month')} for arm in ['control','pred']}
    for row in scores.values():row['cv']=float(np.mean(list(row.values())))
    result=dict(identity=art(OUT/'identity.json'),scores=scores,
        paired=ts.paired_bootstrap(f.day,f.month,(f.control-f.y)**2,(f.pred-f.y)**2),prediction=art(OUT/'scored.parquet'),
        caution='Reused development months; inner labels are training data and never cross outer scope')
    write_json(OUT/'result.json',result);log(str(result))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['screen','full']);a=ap.parse_args()
    with threadpool_limits(8):
        r=identity();dep,arr=shared.load_inputs(1.);table=opdi.load_features()
        if a.stage=='screen':
            for month in ts.SPLITS:fit_job(month,ts.FOLDS[month],dep,arr,table,r)
            del dep,arr,table;gc.collect();score()
        else:
            assert ts.read(OUT/'result.json')['paired']['decision']=='improvement'
            fit_job('full',[f'2025-{m:02d}' for m in range(1,13)],dep,arr,table,r)
