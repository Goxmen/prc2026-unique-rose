"""Review the September 20 adversarial diagnostic without changing its results.

Run from the repository root with:
    .venv/bin/python VALIDATION/tools/review_adversarial_shift.py

Fits six original discriminators plus two with the correct B-long source.
Caches the feature row order, weights and models under tmp/; subsequent runs
reuse these artifacts. Bootstrap intervals condition on those fitted weights.
This is an audit, not a new validation score or submission recipe.
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION/tools'))
import analyze_adversarial_shift as a
from reporting import group_loss
OUT = ROOT / 'tmp/adversarial_review_20260920'
OUT.mkdir(parents=True, exist_ok=True)
if (OUT / 'review.json').exists():
    previous = json.loads((OUT / 'review.json').read_text())
    if previous['original_source_sha256'] != hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest():
        raise ValueError('Original diagnostic changed; preserve the old audit and choose a new output directory.')

def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda x: x.item()) + '\n')

def propensity(df, source, target, name):
    cache = OUT / (name + '.parquet')
    if cache.exists():
        return pd.read_parquet(cache).set_index('mvt_id')
    idx = df.index[source]
    half = np.random.default_rng(a.SEED).random(len(idx)) < .5
    out = pd.DataFrame(index=idx, columns=['p', 'ratio', 'half'], dtype=float)
    counts = []
    for k, keep in enumerate((half, ~half)):
        train = pd.concat([df.loc[idx[~keep]], df.loc[target]])
        y = np.r_[np.zeros((~keep).sum()), np.ones(target.sum())]
        start = time.time()
        model = a.fit(train, y)
        model.save_model(str(OUT / f'{name}_{k}.txt'))
        p = model.predict(df.loc[idx[keep], a.FEATURES]).clip(1e-6, 1-1e-6)
        out.loc[idx[keep], 'p'] = p
        out.loc[idx[keep], 'ratio'] = p/(1-p) * (~keep).sum()/target.sum()
        out.loc[idx[keep], 'half'] = k
        counts.append({'source_train': int((~keep).sum()), 'target_train': int(target.sum()), 'source_holdout': int(keep.sum())})
        print(name, k, 'seconds', round(time.time()-start, 2), flush=True)
        del train, model
        gc.collect()
    out['mvt_id'] = df.loc[idx, 'mvt_id']
    out['w'] = out.p/(1-out.p)
    out.to_parquet(cache, index=False)
    dump(name + '_counts.json', counts)
    return out.set_index('mvt_id')

def score(loss, weights):
    sq = loss.sq.mean(axis=0)
    w = weights.reindex(loss.meta.index).to_numpy()
    assert np.isfinite(w).all() and (w > 0).all()
    return float(np.sqrt(np.average(sq, weights=w)))

def bootstrap(groups, weights, mix, block, reps=2000):
    daily = []
    for name in ('R_A', 'R_B_long'):
        loss = groups[name]
        w = weights.reindex(loss.meta.index).to_numpy()
        daily.append(pd.DataFrame({'day':loss.meta.mvt_time.dt.floor('D'), 'n':1.,
                                   's':loss.sq.mean(axis=0), 'w':w, 'ws':w*loss.sq.mean(axis=0)}).groupby('day').sum())
    dates = daily[0].index
    arr = np.stack([x.reindex(dates, fill_value=0).to_numpy() for x in daily])
    rng = np.random.default_rng(20260920+block)
    month_ids = [np.flatnonzero(dates.month==m) for m in sorted(set(dates.month))]
    changes=[]
    for _ in range(reps):
        samples=[]
        for ids in month_ids:
            starts = rng.integers(len(ids), size=int(np.ceil(len(ids)/block)))
            positions = ((starts[:,None]+np.arange(block))%len(ids)).ravel()[:len(ids)]
            samples.extend(ids[positions])
        totals = arr[:, samples].sum(axis=1)
        plain = np.sqrt(mix*totals[0,1]/totals[0,0]+(1-mix)*totals[1,1]/totals[1,0])
        weighted = np.sqrt(mix*totals[0,3]/totals[0,2]+(1-mix)*totals[1,3]/totals[1,2])
        changes.append(weighted-plain)
    return {'block_days':block, 'reps':reps, 'delta_p025_p50_p975':np.quantile(changes,[.025,.5,.975]).tolist(), 'note':'conditional on fitted weights; paired calendar blocks shared across A/B; months stratified'}

start=time.time()
cache = OUT / 'features.parquet'
if cache.exists():
    df=pd.read_parquet(cache)
else:
    df=a.load()
    df.to_parquet(cache)
print('loaded', len(df), 'seconds', round(time.time()-start,2), flush=True)
assert df.mvt_id.notna().all() and not df.mvt_id.duplicated().any()
mask25=df.ym.isin(['2025-01','2025-07'])
rank=df.ym.isin(['2026-01','2026-07'])
long=df.ym.isin(['2025-11','2025-12'])
val=df.ym.isin(a.VAL_MONTHS)
result={'original_source_sha256':hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
        'row_order_sha256':hashlib.sha256(df.mvt_id.to_numpy().tobytes()).hexdigest(),
        'monthly_counts':{str(k):int(v) for k,v in df.groupby('ym').size().items()},
        'auc_replay':[]}
aucfile=OUT/'auc_replay.json'
if aucfile.exists():
    result['auc_replay']=json.loads(aucfile.read_text())
else:
    coin=pd.Series(np.random.default_rng(7).random(len(df))<.5,index=df.index)&mask25
    for name,neg,pos in [('D0',mask25&~coin,coin),('D1',df.year.eq('2025'),rank),('D2',mask25,rank),('D3',mask25,long)]:
        result['auc_replay'].append(a.discriminate(df,name,name,neg,pos))
        gc.collect()
    dump('auc_replay.json',result['auc_replay'])
weights=propensity(df,val,rank,'original_dw')
corrected=propensity(df,long,rank,'long_source_dw')
record=json.loads((a.RUN/'result.json').read_text())
record['prediction_column']='pred_combined'
# Only A and selected B folds are needed; original helper handles seed/origin averaging.
record['folds']={k:v for k,v in record['folds'].items() if k in ['A1','A2','A3','A4','A5','A6','B3','B4','B7']}
losses=a.load_losses(record,a.RUN)
groups={'R_A':group_loss(losses,['A1','A2','A3','A4','A5','A6']), 'R_B_long':group_loss(losses,['B3','B4','B7'])}
del losses
gc.collect()
mix=record['ranking_weights']['jan_fraction']
combine=lambda x: float(np.sqrt(mix*x['R_A']**2+(1-mix)*x['R_B_long']**2))
result['groups']={}
raw=df.set_index('mvt_id')
for name,loss in groups.items():
    sq=loss.sq.mean(axis=0)
    w=weights.w.reindex(loss.meta.index).to_numpy()
    ratio=weights.ratio.reindex(loss.meta.index)
    wc=np.minimum(w,np.quantile(w,.99))
    assert np.isfinite(w).all()
    own=corrected.w if name=='R_B_long' else weights.w
    contribution=sq*(w/w.sum()-1/len(sq))
    top=np.argsort(-contribution)
    weight_sq=w*sq
    pos=contribution.clip(min=0)
    features=raw.loc[loss.meta.index]
    balance=[]
    for col in ['no_flight_id','no_aobt3','no_stand','no_operator']:
        values=features[col].to_numpy()
        balance.append({'feature':col,'ranking_pct':float(100*df.loc[rank,col].mean()),
                        'source_pct':float(100*values.mean()),'weighted_pct':float(100*np.average(values,weights=w)),
                        'own_source_weighted_pct':float(100*np.average(values,weights=own.reindex(loss.meta.index)))})
    airport=[]
    for ap in sorted(df.airport.dropna().unique()):
        mask=features.airport.eq(ap).to_numpy()
        airport.append({'airport':ap, 'ranking_pct':float(100*df.loc[rank,'airport'].eq(ap).mean()),
                        'source_pct':float(100*mask.mean()),'weighted_pct':float(100*np.average(mask,weights=w)),
                        'mse_delta':float(contribution[mask].sum())})
    top_rows=[]
    for i in top[:15]:
        top_rows.append({'id':int(loss.meta.index[i]), 'airport':str(loss.meta.airport.iloc[i]),
                         'day':str(loss.meta.mvt_time.iloc[i]), 'sq':float(sq[i]),'w':float(w[i]), 'mse_delta':float(contribution[i])})
    result['groups'][name]={'n':len(sq),'plain':float(np.sqrt(sq.mean())),
        'weighted':score(loss,weights.w),'clipped':float(np.sqrt(np.average(sq,weights=wc))),
        'prior_corrected':score(loss,weights.ratio),'own_source_weighted':score(loss,own),
        'seed_plain':np.sqrt(loss.sq.mean(axis=1)).tolist(),
        'seed_weighted':np.sqrt(np.average(loss.sq,axis=1,weights=w)).tolist(),
        'ess_pct':float(100*w.sum()**2/np.sum(w**2)/len(w)),
        'balance':balance,'airports':airport,'top_positive_delta_rows':top_rows,
        'top10_weighted_sse_pct':float(100*np.sort(weight_sq)[-10:].sum()/weight_sq.sum()),
        'top10_positive_delta_pct':float(100*pos[top[:10]].sum()/pos.sum()),
        'mse_delta':float(contribution.sum()),'top10_mse_delta':float(contribution[top[:10]].sum())}
    print(name, {k:result['groups'][name][k] for k in ['plain','weighted','clipped','own_source_weighted','top10_mse_delta','mse_delta']},flush=True)
result['combined']={k:combine({n:r[k] for n,r in result['groups'].items()}) for k in ['plain','weighted','clipped','prior_corrected','own_source_weighted']}
result['combined']['seed_plain']=[combine({n:r['seed_plain'][i] for n,r in result['groups'].items()}) for i in range(3)]
result['combined']['seed_weighted']=[combine({n:r['seed_weighted'][i] for n,r in result['groups'].items()}) for i in range(3)]
result['bootstrap']=[bootstrap(groups,weights.w,mix,b) for b in (1,3)]
result['support']=[]
for label,mask in [('A',val),('B_long',long),('all2025',df.year.eq('2025'))]:
    for cols in [['stand'],['airport','stand'],['airport','ades'],['airport','stand','runway']]:
        src=pd.MultiIndex.from_frame(df.loc[mask,cols].astype('string').fillna('<NA>'))
        target=pd.MultiIndex.from_frame(df.loc[rank,cols].astype('string').fillna('<NA>'))
        unseen=~target.isin(src)
        result['support'].append({'source':label,'keys':cols,'ranking_unseen_n':int(unseen.sum()),'ranking_unseen_pct':float(100*unseen.mean())})
result['elapsed_seconds']=round(time.time()-start,2)
dump('review.json',result)
print('COMBINED',json.dumps(result['combined']),flush=True)
print('BOOTSTRAP',json.dumps(result['bootstrap']),flush=True)
print('SUPPORT',json.dumps(result['support']),flush=True)
