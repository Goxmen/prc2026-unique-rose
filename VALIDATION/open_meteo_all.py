"""User-requested extension: ERA5 inputs in every v58 model part.

Reuses completed main-LightGBM fits. All settings, rows and blend weights stay fixed.
Networks retain 15 members (three seed groups, five networks each) and select epochs
on the last training month. CatBoost retains its original feature set plus ERA5.
Rome uses its full classifier population, including substituted-label positives.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline'), str(ROOT/'VALIDATION/tools')]
import open_meteo_jan_jul as base
from open_meteo_jan_jul import OUT, wx, hpo, ideas, ts, modeling, log
from records import sha256, write_json


def pred_path(part, month, seed):
    return OUT/'predictions'/f'{part}_{month}_s{seed}.parquet'


def save_part(part, month, seed, ids, pred, info):
    path = pred_path(part, month, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert np.isfinite(pred).all() and len(ids) == len(pred)
    temp = path.with_suffix('.parquet.tmp')
    dtype = 'float64' if part == 'rome' else 'float32'
    pd.DataFrame(dict(mvt_id=ids, pred=np.asarray(pred, dtype=dtype))).to_parquet(temp, index=False)
    temp.replace(path)
    write_json(path.with_suffix('.json'), dict(part=part, month=month, seed=seed,
        prediction_sha256=sha256(path), source_sha256=sha256(Path(__file__)), **info))


def raw_lookup():
    import duckdb
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET TimeZone='UTC'")
    f = con.execute("SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt, MVT_TIME_UTC_mvt, TAXITIME_SEC_mvt AS taxi_sec "
                    "FROM read_parquet(?) WHERE PHASE_mvt='DEP'", [str(ROOT/'data/training_2025-*.parquet')]).df()
    con.close()
    assert not f.mvt_id.duplicated().any()
    return f.set_index('mvt_id')


def trees(part, seeds):
    import lightgbm as lgb
    import rome_signals as signals
    spec = hpo.trial_spec(part, {'airport':'P04','catboost':'K07'}[part])
    rounds = int(hpo.winner(part)['checkpoint'][1:])
    for month in ts.SPLITS:
        data = pd.read_pickle(OUT/'inputs'/f'{month}.pkl')
        for seed in seeds:
            if pred_path(part,month,seed).exists(): continue
            started = time.monotonic()
            folder = OUT/'models'/f'{part}_{month}_s{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            if part == 'airport':
                params = ts.lgb_params(dict(hpo.V44, **spec['overrides']),seed,spec['learning_rate'])
                X,V = data['X'],data['V']
                tap,vap = X.airport.astype(str).to_numpy(),V.airport.astype(str).to_numpy()
                pred = np.full(len(V),np.nan)
                for ap in sorted(set(tap)):
                    booster = ideas.train(X.loc[tap==ap],data['y'][tap==ap],params,rounds)
                    booster.save_model(str(folder/f'{ap}.txt'))
                    if (vap==ap).any():
                        pred[vap==ap] = modeling.predict(booster,V.loc[vap==ap],threads=ts.THREADS)
                    log(f'{part} {month} seed {seed}: {ap} done')
                    del booster
            else:
                from catboost import CatBoostRegressor, Pool
                from catboost_jan_jul import frame
                # K07 was trained without the 22 combo inputs; adding them would confound this test.
                cols = [c for c in data['baseline'] if c not in ideas.COMBO] + wx.WEATHER
                cats = signals._categorical(cols)
                X,V = frame(data['X'][cols],cats),frame(data['V'][cols],cats)
                params = dict(hpo.CB_BASE, **spec['overrides'])
                assert 'combo_inputs' not in params
                booster = CatBoostRegressor(loss_function='RMSE',iterations=rounds,random_seed=seed,
                    task_type='GPU',devices='0',thread_count=ts.THREADS,verbose=1000,allow_writing_files=False,**params)
                booster.fit(Pool(X,data['y'],cat_features=cats))
                booster.save_model(str(folder/'model.cbm'))
                pred = np.maximum(booster.predict(Pool(V,cat_features=cats)),0)
                del booster
            save_part(part,month,seed,data['ids'],pred,dict(rounds=rounds,params=params,
                columns=list(X.columns),minutes=(time.monotonic()-started)/60,
                inputs_sha256=sha256(OUT/'inputs'/f'{month}.pkl')))
            log(f'{part} {month} seed {seed} complete in {(time.monotonic()-started)/60:.1f} min')
            del X,V
            gc.collect()
        del data
        gc.collect()


def networks():
    import torch
    import nn_jan_jul as nnj
    import nn_improve_jan_jul as nni
    torch.set_num_threads(ts.THREADS)
    cfg = hpo.nn_config(hpo.trial_spec('nn','N09')['overrides'])
    hpo.apply_nn(cfg,nnj,nni)
    lookup = raw_lookup()
    for month in ts.SPLITS:
        data = pd.read_pickle(OUT/'inputs'/f'{month}.pkl')
        t = pd.to_datetime(lookup.loc[data['train_ids'],'MVT_TIME_UTC_mvt'],utc=True).dt.strftime('%Y-%m').to_numpy()
        assert set(t) == set(ts.FOLDS[month])
        raw = lookup.loc[data['ids']].reset_index()
        item = dict(X=data['X'],Vn=data['V'],y=data['y'],raw=raw,tmonth=t,meta=dict(month=month))
        d = nni.Data(item,'cuda')
        for seed in hpo.SEEDS:
            folder = OUT/'networks'
            folder.mkdir(exist_ok=True)
            record = folder/f'{month}_s{seed}.json'
            if record.exists():
                info = ts.read(record)
            else:
                started = time.monotonic()
                _,_,curve = nni.train('ple',d,d.inner_train,seed,cfg['epochs'],(d.subset(d.inner_val),d.y[d.inner_val]))
                stop = min(curve,key=lambda c:c['rmse'])['epoch']
                info = dict(config=cfg,seed=seed,month=month,stop_epoch=stop,inner_curve=curve,encoding=d.info,
                            stop_minutes=(time.monotonic()-started)/60,networks={})
                write_json(record,info)
            for k in range(5):
                path = folder/f'{month}_s{seed}_k{k}.parquet'
                if path.exists(): continue
                started = time.monotonic()
                model,pred,_ = nni.train('ple',d,np.arange(len(d.y)),seed+1000*k,info['stop_epoch'],(d.held,d.yv))
                model_path = OUT/'models'/f'nn_{month}_s{seed}_k{k}.pt'
                model_path.parent.mkdir(exist_ok=True)
                torch.save(dict(state_dict=model.state_dict(),bins=d.bins,cards=d.cards,mu=d.mu,sd=d.sd),model_path)
                nni.save_network(path,data['ids'],pred)
                info['networks'][str(k)] = dict(minutes=(time.monotonic()-started)/60,sha256=sha256(path),model_sha256=sha256(model_path))
                write_json(record,info)
                del model
                torch.cuda.empty_cache()
                log(f'network {month} seed {seed} k{k} saved')
        del data,item,d
        gc.collect()
        torch.cuda.empty_cache()


def rome(seeds):
    import overnight_run
    import rome_hybrid
    import rome_signals as signals
    dep,arr = overnight_run.load_inputs(1.)
    del arr
    table = wx.weather_table()
    spec = hpo.trial_spec('rome','R20')
    rounds = int(hpo.winner('rome')['checkpoint'][1:])
    for month in ts.SPLITS:
        source = ts.split_dir(hpo.SPLITS_FROM,month)
        train,held = pd.read_pickle(source/'after_drop.pkl'),pd.read_pickle(source/'heldout.pkl')
        mask = train['airport']=='LIRF'
        X = train['X'].loc[mask]
        y = np.asarray(train['y'])[mask]
        raw = dep.loc[X.index]
        assert raw.ADEP_mvt.eq('LIRF').all() and np.array_equal(raw.taxi_sec.to_numpy(),y)
        assert set(pd.to_datetime(raw.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m')).issubset(set(ts.FOLDS[month]))
        target = rome_hybrid.substituted(X,y,raw).astype(float)
        A = pd.concat([signals.classifier_matrix(X,raw,ts.KIND,ts.CLASSIFIER),wx.weather_join(raw,table)],axis=1)
        vr,V = held['raw'],held['V']
        use = (vr.ADEP_mvt.to_numpy()=='LIRF') & np.isfinite(V[rome_hybrid.GAP].to_numpy(float))
        B = pd.concat([signals.classifier_matrix(V.loc[use],vr.loc[use],ts.KIND,ts.CLASSIFIER),
                       wx.weather_join(vr.loc[use],table)],axis=1)
        assert not A[wx.WEATHER].isna().any().any() and not B[wx.WEATHER].isna().any().any()
        for seed in seeds:
            if pred_path('rome',month,seed).exists(): continue
            started = time.monotonic()
            params = dict(rome_hybrid.CLASSIFIER_PARAMS,**spec['overrides'],seed=seed,
                          bagging_seed=seed+1,feature_fraction_seed=seed+2,num_threads=ts.THREADS)
            model = signals._train(params,A,target,rounds,f'ERA5 Rome {month}')
            pred = np.zeros(len(V))
            pred[use] = model.predict(B,num_threads=ts.THREADS)
            path = OUT/'models'/f'rome_{month}_s{seed}.txt'
            path.parent.mkdir(exist_ok=True)
            model.save_model(str(path))
            save_part('rome',month,seed,vr.mvt_id.to_numpy(),pred,dict(rounds=rounds,params=params,
                columns=list(A.columns),training_rows=len(A),positives=int(target.sum()),
                minutes=(time.monotonic()-started)/60,model_sha256=sha256(path)))
        del train,held,X,A,B,V,vr
        gc.collect()


def load_prediction(part,month,seed,ids):
    path = (OUT/'predictions'/f'{month}_s{seed}.parquet') if part=='lgb' else pred_path(part,month,seed)
    f = pd.read_parquet(path)
    assert np.array_equal(f.mvt_id.to_numpy(),ids),str(path)
    return f.pred.to_numpy(float)


def score(seeds):
    sc,old_rome,old_nn,old_trees = base.baseline()
    new_nn = {}
    for m in ts.SPLITS:
        members = []
        for s in hpo.SEEDS:
            for k in range(5):
                f = pd.read_parquet(OUT/'networks'/f'{m}_s{s}_k{k}.parquet')
                assert np.array_equal(f.mvt_id.to_numpy(),sc.d[m]['ids'])
                members.append(f.base.to_numpy(float))
        new_nn[m] = np.mean(members,axis=0)
    arms = {p:[p] for p in ['lgb','airport','catboost','nn','rome']}
    arms['all'] = list(arms)
    all_errors = {a:{} for a in ['control']+list(arms)}
    by_seed,frames = {},[]
    for seed in seeds:
        by_seed[seed] = {}
        for month in ts.SPLITS:
            d = sc.d[month]
            old = {hpo.MEMBER_OF[p]:old_trees[p][seed][month] for p in old_trees}
            new = {p:load_prediction(p,month,seed,d['ids']) for p in ['lgb','airport','catboost','rome']}
            control = sc.se(month,seed,nn=old_nn[month],p=old_rome[month],**old)
            all_errors['control'].setdefault(month,[]).append(control)
            by_seed[seed].setdefault('control',{})[month] = float(np.sqrt(control.mean()))
            for arm,parts in arms.items():
                kwargs = dict(old)
                for p in parts:
                    if p in hpo.MEMBER_OF: kwargs[hpo.MEMBER_OF[p]] = new[p]
                e = sc.se(month,seed,nn=new_nn[month] if 'nn' in parts else old_nn[month],
                          p=new['rome'] if 'rome' in parts else old_rome[month],**kwargs)
                all_errors[arm].setdefault(month,[]).append(e)
                by_seed[seed].setdefault(arm,{})[month] = float(np.sqrt(e.mean()))
                if arm=='all':
                    frames.append(pd.DataFrame(dict(mvt_id=d['ids'],seed=seed,month=month,day=d['day'],
                        airport=d['held']['raw'].ADEP_mvt.to_numpy(),se_control=control,se_candidate=e)))
    errors = {arm:{m:np.mean(es,axis=0) for m,es in v.items()} for arm,v in all_errors.items()}
    days = np.concatenate([sc.d[m]['day'] for m in ts.SPLITS])
    months = np.concatenate([np.full(len(sc.d[m]['ids']),m) for m in ts.SPLITS])
    flat = lambda e:np.concatenate([e[m] for m in ts.SPLITS])
    result = dict(parent='v58',network_share=.71,network_members=15,seeds=seeds,by_seed=by_seed,
        control=sc.cv(errors['control']),parts={})
    for arm in arms:
        result['parts'][arm] = dict(**sc.cv(errors[arm]),
            change=ts.paired_bootstrap(days,months,flat(errors['control']),flat(errors[arm])))
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'all_scored_rows.parquet',index=False)
    write_json(OUT/'all_result.json',result)
    log(json.dumps(result,indent=2))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['nn','airport','catboost','rome','score'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    args = p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='nn': networks()
        elif args.stage=='rome': rome(args.seeds)
        elif args.stage=='score': score(args.seeds)
        else: trees(args.stage,args.seeds)
