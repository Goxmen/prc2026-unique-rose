"""Independent nested-fit receipt, packet algebra, raw CV and score audit.

No model inference, source likelihood rerun, optimization or training. Original
all-row model reload and zero-posterior replays remain immutable laptop receipts.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json
import hashlib
from pathlib import Path
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import re
import time
import numpy as np
import pandas as pd
import duckdb
from scipy.special import softmax
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_posterior_aware_crossfit'
INPUT=OUT/'inputs'
IDENTITY='acab0bee4edd1ae7c5ba84493239e133b2bf029164a05a891efb8ed9427132c4'
RESULT='00613f1ff2966d709f33db4b89d534a0484b724aaa42688c6fea82187ec36426'
PREFIT='e7fc03053453e27c07aff37d4a2dbb814a0d45b8d25f5e3994394341e00777d2'
CATEGORIES=['airport','dest','aircraft_type','runway','stand','flight_rule','market_segment','flight_type','wake_cat','operator','airline']


def read(p):return json.loads(Path(p).read_text())
def art(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())
def verify(a):
    p=ROOT/a['path'];assert art(p)=={k:a[k] for k in ['path','sha256']},p
    return p

def receipt(path,arm=None,outer=None):
    x=read(path);assert x['identity']==art(OUT/'identity.json'),path
    if arm:assert x['arm']==arm
    if outer:assert x['outer']==outer
    for key in ['artifacts','optimizers','dependencies']:
        for a in x.get(key,[]):verify(a)
    return x

def close(a,b,atol=1e-8):np.testing.assert_allclose(a,b,rtol=0,atol=atol,equal_nan=True)

def moments(p,theta,kind):
    log=p['log_weight']+np.einsum('ij,j->i',p['features'],theta)[:,None]*p['statistic']
    if kind!='prior':log=log+p[kind+'_like']
    w=softmax(log,axis=1);mean=np.einsum('ij,ij->i',w,p['values'])
    variance=np.einsum('ij,ij->i',w,(p['values']-mean[:,None])**2)
    grad=variance[:,None]*p['features']/p['scale'][:,None]
    return mean,grad

def objective(p,y,theta,kind):
    if kind=='prior':pred,grad=moments(p,theta,'prior')
    else:
        jm,jg=moments(p,theta,'joint');dm,dg=moments(p,theta,'dep')
        pred=p['dep_zero']+.5*(jm-dm);grad=.5*(jg-dg)
    residual=(pred-y)/3600.
    return float(np.mean(residual**2)+.001*np.dot(theta,theta)),2*np.mean(residual[:,None]*grad/3600.,axis=0)+.002*theta

def packet(folder):
    with np.load(folder/'packet.npz',allow_pickle=False) as f:return {k:f[k] for k in f.files}

def concatenate(ps):
    width=max(p['values'].shape[1] for p in ps);out={}
    for k in ps[0]:
        arrays=[]
        for p in ps:
            a=p[k]
            if a.ndim==2 and k!='features' and a.shape[1]<width:
                a=np.pad(a,((0,0),(0,width-a.shape[1])),constant_values=-np.inf if k=='log_weight' else 0.)
            arrays.append(a)
        out[k]=np.concatenate(arrays)
    assert len(np.unique(out['mvt_id']))==len(out['mvt_id'])
    return out


def complete(day,held):
    start=datetime.fromisoformat(day).replace(tzinfo=ZoneInfo('Europe/Rome'))
    end=start+timedelta(days=1)-timedelta(microseconds=1)
    return start.astimezone(timezone.utc).strftime('%Y-%m') in held and end.astimezone(timezone.utc).strftime('%Y-%m') in held


def arm_checks(arm,identity,meta,features,labels):
    folder=OUT/'arms'/arm['key'];prior_receipt=receipt(folder/'prior.json',arm=arm)
    saved=receipt(folder/'packet.json',arm=arm)
    assert prior_receipt['all_row_reload_exact'] and saved['original_apply_zero_replay']
    assert prior_receipt['dependencies']==[art(OUT/'prefit_audit.json'),art(OUT/'preflight.json'),art(folder/'calibration.json')]
    assert saved['dependencies']==[art(folder/'prior.json'),art(folder/'calibration.json')]
    chosen=meta.month.isin(arm['train']);tr=meta[chosen].copy();tr['y']=labels.reindex(tr.mvt_id).y.to_numpy()
    tr=tr[np.isfinite(tr.y)&tr.y.between(1,200000)]
    training=pd.read_parquet(folder/'prior_training_ids.parquet')
    pd.testing.assert_frame_equal(training.reset_index(drop=True),tr[['mvt_id','month']].reset_index(drop=True),check_dtype=False)
    assert len(tr)==prior_receipt['training_rows'] and not set(tr.month)&set(arm['held']+[arm['outer']])
    categories=read(folder/'categories.json');assert set(categories)==set(CATEGORIES)
    x=features.set_index('mvt_id').loc[tr.mvt_id]
    for col in CATEGORIES:assert categories[col]==sorted(x[col].dropna().astype(str).unique().tolist())
    # Inspect every saved tree's numerical state without predictor execution.
    text=(folder/'model.txt').read_text();assert len(re.findall(r'^Tree=\d+$',text,re.M))==800
    names=re.search(r'^feature_names=(.*)$',text,re.M).group(1).split()
    assert names==identity['features'] and 'max_feature_idx=36\n' in text
    numeric=0
    for line in text.splitlines():
        if line.startswith(('leaf_value=','internal_value=','split_gain=','leaf_weight=','internal_weight=')):
            a=np.fromstring(line.split('=',1)[1],sep=' ');assert np.isfinite(a).all();numeric+=len(a)
    missing_splits=0
    for tree in text.split('\nTree=')[1:]:
        fields=dict(line.split('=',1) for line in tree.splitlines()[1:] if '=' in line and not line.startswith('['))
        threshold=np.fromstring(fields['threshold'],sep=' ')
        feature=np.fromstring(fields['split_feature'],sep=' ',dtype=int)
        decision=np.fromstring(fields['decision_type'],sep=' ',dtype=int)
        assert not np.isnan(threshold).any() and not np.isneginf(threshold).any()
        for ix in np.flatnonzero(np.isposinf(threshold)):
            assert decision[ix]&12==8 and decision[ix]&1==0
            assert x[identity['features'][feature[ix]]].isna().any()
            missing_splits+=1
        numeric+=int(np.isfinite(threshold).sum())
    for pattern in ['[objective: regression]','[learning_rate: 0.05]','[num_leaves: 255]','[min_data_in_leaf: 200]','[feature_fraction: 0.9]','[bagging_fraction: 0.8]','[bagging_freq: 1]','[seed: 42]']:
        assert pattern in text,pattern
    held=meta[meta.month.isin(arm['held'])].reset_index(drop=True)
    prior=pd.read_parquet(folder/'prior.parquet');rows=pd.read_parquet(folder/'posterior_inputs.parquet')
    assert list(rows)==['mvt_id','month','gap','prior','eligible','quality','runway','local_day','assigned_day','fltrule','delay_scope']
    for g in [prior,rows]:
        np.testing.assert_array_equal(g.mvt_id,held.mvt_id);np.testing.assert_array_equal(g.month,held.month)
    np.testing.assert_array_equal(prior.prior,rows.prior)
    assert len(held)==prior_receipt['predicted_raw_peers']==saved['complete_raw_peers']
    assert np.isfinite(prior.prior).all() and prior.prior.ge(0).all()
    for col in ['gap','eligible','runway','local_day','fltrule']:pd.testing.assert_series_equal(rows[col],held[col],check_names=False,check_dtype=False)
    assigned=pd.to_datetime(held.takeoff_epoch-rows.prior,unit='s',utc=True)
    np.testing.assert_array_equal(rows.assigned_day,assigned.dt.strftime('%Y-%m-%d'))
    scope=assigned.dt.strftime('%Y-%m').eq(held.month)&assigned.dt.day.gt(1)&assigned.dt.day.lt(assigned.dt.days_in_month)
    np.testing.assert_array_equal(rows.delay_scope,scope)
    keep=np.array([complete(day,arm['held']) for day in held.local_day])
    np.testing.assert_array_equal(rows.quality,held.quality&keep)
    p=packet(folder);positions=p['positions'];assert len(positions)==len(np.unique(positions))==saved['targets']
    np.testing.assert_array_equal(rows.mvt_id.iloc[positions],p['mvt_id']);np.testing.assert_array_equal(rows.prior.iloc[positions],p['prior'])
    assert len(positions)>0 and (rows.gap.iloc[positions]>3600).all()
    for k,v in p.items():
        if k=='log_weight':assert not np.isnan(v).any() and not np.isposinf(v).any()
        else:assert np.isfinite(v).all(),k
    params=read(folder/'parameters.json');tau=params['support']['tau'];gap=rows.gap.to_numpy()[positions]
    scale=np.maximum(gap-tau,60.);close(p['scale'],scale)
    prob=np.clip((p['prior']-tau)/(gap-tau),1e-6,1-1e-6);a=np.log(gap/3600.);b=np.log(prob)-np.log1p(-prob)
    close(p['features'],np.column_stack([np.ones(len(a)),a,b,a*b]))
    mask=np.isfinite(p['log_weight']);expected=(p['values']-p['prior'][:,None])/scale[:,None]
    close(p['statistic'][mask],expected[mask]);assert (p['statistic'][~mask]==0).all()
    prior_mean,_=moments(p,np.zeros(4),'prior');close(prior_mean,p['prior'],1e-4)
    dep,_=moments(p,np.zeros(4),'dep');joint,_=moments(p,np.zeros(4),'joint')
    close(dep,p['dep_zero']);close(joint,p['joint_zero'])
    zero=pd.read_parquet(folder/'zero.parquet');np.testing.assert_array_equal(zero.mvt_id,rows.mvt_id)
    touched=np.zeros(len(rows),bool);touched[positions]=True;np.testing.assert_array_equal(zero.touched,touched)
    for key,values in [('dep_only',dep),('joint',joint),('increment',joint-dep)]:
        expected=rows.prior.to_numpy().copy() if key!='increment' else np.zeros(len(rows))
        expected[positions]=values;close(expected,zero[key])
    diag=saved['diagnostics'];assert diag['incremental_derivative_rank']==4 and diag['target_only_fixed_messages']
    return p,dict(arm=arm['key'],training_rows=len(tr),complete_peer_rows=len(rows),targets=len(positions),finite_tree_values=numeric,positive_infinity_NaN_threshold_splits=missing_splits,trees=800,reload_receipt_verified=True)


def main():
    started=time.monotonic();assert not (OUT/'independent_result_audit.json').exists()
    identity=read(OUT/'identity.json');result=read(OUT/'result.json')
    assert art(OUT/'identity.json')['sha256']==IDENTITY and art(OUT/'result.json')['sha256']==RESULT
    assert art(OUT/'prefit_audit.json')['sha256']==PREFIT and read(OUT/'prefit_audit.json')['status']=='pass'
    for a in identity['sources']+identity['inputs']+identity['score_inputs']:verify(a)
    returned=read(OUT/'return_audit.json');assert returned['status']=='pass' and returned['result']==art(OUT/'result.json') and returned['identity']==art(OUT/'identity.json')
    remote=read(verify(returned['remote_manifest']))
    assert remote['host']=='glaptop-wsl' and len(remote['files'])==returned['files']==114
    for a in remote['files']:
        path=verify(a);assert path.stat().st_size==a['bytes']
    meta=pd.read_parquet(INPUT/'rows.parquet');features=pd.read_parquet(INPUT/'features.parquet')
    labels=pd.concat([pd.read_parquet(INPUT/f'labels_{m}.parquet') for m in sorted(meta.month.unique())]).set_index('mvt_id');assert labels.index.is_unique
    packets={};summaries=[];predictions=[];optimizer_checks=[]
    for arm in identity['arms']:
        packets[arm['key']],summary=arm_checks(arm,identity,meta,features,labels);summaries.append(summary)
    for outer in ['2025-01','2025-07']:
        folder=OUT/outer;learned=receipt(folder/'learned.json',outer=outer);inference=receipt(folder/'inference.json',outer=outer)
        inner=[a for a in identity['arms'] if a['outer']==outer and a['part']!='outer']
        assert learned['dependencies']==[art(OUT/'arms'/a['key']/'packet.json') for a in inner]
        assert inference['dependencies']==[art(folder/'learned.json'),art(OUT/'arms'/f'{outer}_outer/packet.json')]
        p=concatenate([packets[a['key']] for a in inner]);y=labels.loc[p['mvt_id'],'y'].to_numpy()
        assert len(y)==learned['targets'] and learned['outer_labels_not_read'] and learned['raw_target_no_clipping']
        heldmeta=meta.set_index('mvt_id').loc[p['mvt_id']];assert outer not in set(heldmeta.month)
        assert set(heldmeta.month).issubset(learned['allowed'])
        for kind in ['prior','posterior']:
            opt=read(folder/f'optimizer_{kind}.json');assert opt['success'] and opt['objective']==kind and opt['rows']==len(y)
            assert opt['theta']==learned['parameters'][kind] and opt['ridge']==.001 and opt['scale']==3600.
            assert opt['settings']==dict(maxiter=200,ftol=1e-12,gtol=1e-8)
            for theta,losskey,gradkey in [(np.zeros(4),'initial_loss','initial_gradient'),(np.array(opt['theta']),'final_loss','gradient')]:
                loss,gradient=objective(p,y,theta,kind);close(loss,opt[losskey],1e-12);close(gradient,opt[gradkey],1e-10)
            assert opt['final_loss']<=opt['initial_loss'] and np.isfinite(opt['theta']).all()
            optimizer_checks.append(dict(outer=outer,kind=kind,loss=opt['final_loss'],gradient_max=float(np.max(np.abs(opt['gradient'])))))
        rows=pd.read_parquet(OUT/'arms'/f'{outer}_outer/posterior_inputs.parquet');p=packets[f'{outer}_outer']
        pred=pd.read_parquet(folder/'predictions.parquet');np.testing.assert_array_equal(pred.mvt_id,rows.mvt_id)
        for kind in ['zero','prior','posterior']:
            theta=np.zeros(4) if kind=='zero' else np.array(learned['parameters'][kind])
            jm,_=moments(p,theta,'joint');dm,_=moments(p,theta,'dep');values=np.zeros(len(rows));values[p['positions']]=jm-dm
            close(pred[kind+'_increment'],values)
        predictions.append(pred)
    pred=pd.concat(predictions).set_index('mvt_id');assert pred.index.is_unique and len(pred)==result['raw_peer_prediction_rows']==26528
    f=pd.read_parquet(verify(result['scored']));old=pd.read_parquet(INPUT/'old_predictions.parquet').set_index('mvt_id')
    control=pd.read_parquet(verify(result['current'])).set_index('mvt_id').loc[f.mvt_id]
    np.testing.assert_array_equal(f.control,control.three_tabm_runway)
    for k in ['y','month','day']:np.testing.assert_array_equal(f[k],control[k])
    assert set(old.index).issubset(pred.index)
    con=duckdb.connect(config={'threads':1,'memory_limit':'512MB'});con.execute("SET TimeZone='UTC'");con.register('f',f)
    rawpaths=[]
    for m in ['2025-01','2025-07']:
        rawpaths.extend(str(verify(a)) for a in read(OUT/'prepared.json')['raw_sources'] if Path(a['path']).name.startswith('training_'+m+'-01_'))
    assert len(rawpaths)==2;con.read_parquet(rawpaths).create_view('raw')
    n,u,bad=con.execute('''SELECT count(*),count(distinct f.mvt_id),sum(CASE WHEN
        f.y IS DISTINCT FROM r.TAXITIME_SEC_mvt OR f.month IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m')
        OR f.day IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m-%d') OR r.PHASE_mvt!='DEP' THEN 1 ELSE 0 END)
        FROM f JOIN raw r ON f.mvt_id=r.MVT_ID_mvt''').fetchone();assert n==u==344419 and bad==0
    con.register('p',pred.reset_index());con.register('o',old.reset_index())
    for kind in ['zero','prior','posterior']:
        delta=f'.5*(coalesce(p.{kind}_increment,0.)-coalesce(p.zero_increment,0.))'
        family=f'.5*(coalesce(p.{kind}_increment,0.)-coalesce(o.increment,0.))'
        bad=con.execute(f'''SELECT count(*) FROM f LEFT JOIN p USING(mvt_id) LEFT JOIN o USING(mvt_id)
            WHERE f.{kind}_delta IS DISTINCT FROM ({delta}) OR f.{kind} IS DISTINCT FROM round_even(greatest(f.control+({delta}),0.),0)
            OR f.{kind}_family IS DISTINCT FROM round_even(greatest(f.control+({family}),0.),0)''').fetchone()[0]
        assert bad==0,(kind,bad)
    scores={m:{} for m in sorted(f.month.unique())}
    for col in result['cv']:
        for m,v in con.execute(f'SELECT month,sqrt(avg(pow(CAST("{col}" AS DOUBLE)-y,2))) FROM f GROUP BY month').fetchall():scores[m][col]=v;close(v,result['scores'][m][col])
        close(np.mean([r[col] for r in scores.values()]),result['cv'][col])
    con.close();paired={}
    for col in ['control','prior']:
        rng=np.random.default_rng(0);draws=np.zeros(2000)
        for m in sorted(f.month.unique()):
            g=f[f.month.eq(m)];codes,_=pd.factorize(g.day,sort=False);n=np.bincount(codes)
            a=np.bincount(codes,(g[col]-g.y)**2);b=np.bincount(codes,(g.posterior-g.y)**2)
            ix=rng.integers(0,len(n),size=(2000,len(n)));den=n[ix].sum(1)
            draws+=(np.sqrt(b[ix].sum(1)/den)-np.sqrt(a[ix].sum(1)/den))/2
        change=float(np.mean([v['posterior']-v[col] for v in scores.values()]));ci=np.percentile(draws,[2.5,97.5])
        close(change,result['paired'][col]['change']);close(ci,result['paired'][col]['ci95'])
        paired[col]=dict(change=change,ci95=ci.tolist())
    gate=all(v['ci95'][1]<0 for v in paired.values()) and all(s['posterior']<min(s['control'],s['prior']) for s in scores.values())
    assert gate==result['gate']==False
    report=dict(status='pass',source=art(Path(__file__)),identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),
        prefit_audit=art(OUT/'prefit_audit.json'),return_audit=art(OUT/'return_audit.json'),scored=result['scored'],rows=len(f),
        arms=summaries,optimizers=optimizer_checks,all_SQL_truth_control_deltas_family_serialization_scores_exact=True,
        paired=paired,gate=False,conclusion='Closed without tuning, production or upload; both months regress against exact v80.',
        source_parameter_refits=0,model_fits=0,model_inference=0,source_likelihood_reruns=0,
        limits='Original all-row saved-model reload and zero-source replay verified from frozen laptop receipts; independent audit rederives packet algebra, objective/gradient and scores, not original likelihood evaluation.',
        seconds=time.monotonic()-started)
    (OUT/'independent_result_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(audit=art(OUT/'independent_result_audit.json'),paired=paired,gate=gate,seconds=report['seconds'])))

if __name__=='__main__':
    with threadpool_limits(1):main()
