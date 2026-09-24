"""Step-2 weather-distance correction, preserving the first run for comparison.

An IQR and p90-p10 of zero previously fell back to one physical unit, even when
the weather variable varied outside its central 80%. One foot or one metre is
an arbitrary, often tiny scale. This uses the observed donor range next. Truly
constant donor variables use a binary changed/not-changed encoding; an entirely
missing donor variable contributes only its missing indicator. All quantities
come from the same permitted source donor bank. Other groups are unchanged.

This is a documented correction after inspecting the initial outputs, not an
independent confirmation. No models or previously selected weights are changed.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import train_ranking_similarity as sim
import check_train_ranking_similarity as base

ROOT = base.ROOT
CACHE = ROOT/'tmp/train_ranking_step2_weather_scale'
OUT = base.OUT


def corrected_encoding(reference, query):
    specs = sim.fit_distance(reference)
    r, q = sim.encode_distance(reference, specs), sim.encode_distance(query, specs)
    columns = specs['weather']['numeric']
    rv = reference[columns].to_numpy(dtype=float,na_value=np.nan)
    qv = query[columns].to_numpy(dtype=float,na_value=np.nan)
    rz, qz = np.empty_like(rv), np.empty_like(qv)
    changed=[]
    for j,column in enumerate(columns):
        a,b=rv[:,j],qv[:,j]
        finite=np.isfinite(a)
        if not finite.any():
            rz[:,j]=0.;qz[:,j]=0.
            changed.append(dict(feature=column, mode='all_donors_missing'))
            continue
        lo,p25,median,p75,hi=np.quantile(a[finite],[.1,.25,.5,.75,.9])
        scale=p75-p25 if p75-p25>1e-8 else hi-lo
        if scale<=1e-8:
            scale=float(np.ptp(a[finite]))
            changed.append(dict(feature=column, mode='observed_range' if scale>1e-8 else 'constant', scale=scale))
        if scale>1e-8:
            rz[:,j]=(np.where(finite,a,median)-median)/scale
            qz[:,j]=(np.where(np.isfinite(b),b,median)-median)/scale
        else:
            rz[:,j]=0.
            qz[:,j]=(np.isfinite(b)&~np.isclose(b,median,rtol=1e-10,atol=1e-10)).astype(float)
    r['weather']=(r['weather'][0],np.c_[rz,~np.isfinite(rv)]/np.sqrt(2*len(columns)))
    q['weather']=(q['weather'][0],np.c_[qz,~np.isfinite(qv)]/np.sqrt(2*len(columns)))
    assert np.isfinite(r['weather'][1]).all() and np.isfinite(q['weather'][1]).all()
    return r,q,changed


def corrected_nearest(reference,query):
    r,q,changed=corrected_encoding(reference,query)
    distances=np.empty((len(query),10)); indices=np.empty((len(query),10),dtype=np.int32)
    for start,stop,views in sim.distance_blocks(r,q):
        matrix=views['rich']
        thresholds=np.partition(matrix,9,axis=1)[:,9]
        for row in range(len(matrix)):
            candidates=np.flatnonzero(matrix[row]<=thresholds[row])
            chosen=candidates[np.argsort(matrix[row,candidates],kind='stable')[:10]]
            indices[start+row]=chosen
            distances[start+row]=np.sqrt(matrix[row,chosen])
    return distances,indices,changed


def calculate(frame,case,airport,fold,signature):
    checkpoint=CACHE/f'{case}_{airport}_fold{fold}.json'
    if checkpoint.exists():
        d=json.loads(checkpoint.read_text());assert d['design_sha256']==signature
        assert base.sha(CACHE/d['file'])==d['sha256']
        return
    started=time.time();role=sim.roles(frame.date,fold)
    f=frame.assign(role=role).loc[frame.airport.eq(airport)]
    reference=f[f.role.eq('fit')&f.label.eq(0)].sort_values('mvt_id').reset_index(drop=True)
    cal=f[f.role.eq('cal')&f.label.eq(0)];test=f[f.role.eq('test')]
    queries=pd.concat([cal,test],ignore_index=True)
    assert not set(reference.mvt_id)&set(queries.mvt_id)
    dist,idx,changes=corrected_nearest(reference,queries)
    mask=queries.role.eq('test').to_numpy()
    output=queries.loc[mask,['mvt_id','date','airport','label']].reset_index(drop=True)
    output['fold']=fold
    thresholds={}
    for k in [1,5,10]:
        thresholds[str(k)]=float(np.quantile(dist[~mask,k-1],.95))
        output[f'd{k}']=dist[mask,k-1]
        output[f'far{k}']=dist[mask,k-1]>thresholds[str(k)]
    for k in range(10):
        output[f'neighbour{k+1}']=reference.mvt_id.to_numpy()[idx[mask,k]]
    path=checkpoint.with_suffix('.parquet');output.to_parquet(path,index=False)
    # One exact-query check using a full stable sort, independent of partitioning.
    r,q,_=corrected_encoding(reference,queries.iloc[:1])
    _,_,views=next(sim.distance_blocks(r,q))
    expected=np.argsort(views['rich'][0],kind='stable')[:10]
    np.testing.assert_array_equal(idx[0],expected)
    base.write_json(checkpoint,dict(case=case,airport=airport,fold=fold,design_sha256=signature,
        file=path.name,sha256=base.sha(path),thresholds=thresholds,scale_changes=changes,
        n_reference=len(reference),n_cal=len(cal),n_test=len(test),exact_sort_check=True))
    print(f'weather scale {case} {airport} fold {fold}: {time.time()-started:.1f}s',flush=True)


def summarize(signature):
    results=[]
    for case in ['january_year','july_year']:
        original=base.frame_for(base.CASES[case])
        for airport in sorted(original.airport.unique()):
            pieces=[]
            for fold in range(4):
                p=CACHE/f'{case}_{airport}_fold{fold}.json';d=json.loads(p.read_text())
                assert d['design_sha256']==signature and base.sha(CACHE/d['file'])==d['sha256']
                pieces.append(pd.read_parquet(CACHE/d['file']))
            f=pd.concat(pieces,ignore_index=True)
            assert f.mvt_id.is_unique and set(f.mvt_id)==set(original.loc[original.airport.eq(airport),'mvt_id'])
            for k in [1,5,10]:
                rates=f.groupby('label')[f'far{k}'].mean()*100
                lo,hi=base.block_rate_difference(f,f'far{k}')
                results.append(dict(case=case,airport=airport,k=k,
                    source_far_pct=float(rates[0]),target_far_pct=float(rates[1]),
                    delta_pp=float(rates[1]-rates[0]),delta_p025=lo,delta_p975=hi))
    pd.DataFrame(results).to_csv(OUT/'knn_weather_scale_corrected.csv',index=False)
    base.write_json(OUT/'weather_scale_checks.json',dict(design_sha256=signature,checkpoints=80,
        exact_sort_checks=80,script_sha256=base.sha(Path(__file__)),
        output_sha256=base.sha(OUT/'knn_weather_scale_corrected.csv')))
    print('Corrected weather-distance summary complete',flush=True)


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    original=json.loads((OUT/'design.json').read_text())
    assert original['utility_sha256']==base.sha(Path(sim.__file__))
    for month,digest in original['input_sha256'].items():
        assert base.sha(base.SOURCE/f'{month}.parquet')==digest
    design=dict(reason=__doc__,original_design_sha256=base.sha(OUT/'design.json'),
        groups='Only weather numeric scaling changes; raw distances, classifiers, day splits and priors unchanged.',
        scale='IQR, then p90-p10, then observed donor range. Truly constant variables: binary deviation. All missing donors: missingness alone.',
        status='Correction after seeing initial results; preserves initial outputs and repeats all 20 airport/month comparisons with the same method.',
        utility_sha256=base.sha(Path(sim.__file__)),script_sha256=base.sha(Path(__file__)))
    signature=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    path=OUT/'weather_scale_design.json'
    if path.exists():assert json.loads(path.read_text())==design
    else:base.write_json(path,design)
    for case in ['january_year','july_year']:
        frame=base.frame_for(base.CASES[case])
        def airport_run(airport):
            for fold in range(4):calculate(frame,case,airport,fold,signature)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures=[executor.submit(airport_run,a) for a in sorted(frame.airport.unique())]
            for future in as_completed(futures):future.result()
    summarize(signature)


if __name__=='__main__':
    main()
