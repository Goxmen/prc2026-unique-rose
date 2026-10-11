"""Test mean-preserving delay-grid interpolation in the actual v76 stack.

All physical states, Gaussian ordinary prior, source noise, weights and later
corrections remain unchanged. No new fits or parameter search.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_heavy_tail as heavy
import daily_grid_interpolation as interpolation
import daily_delay_submit as daily
import public_constraints_candidate as constraints
import public_stack_cv as joint
from records import write_json

base=heavy.base
ROOT=base.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_daily_grid_interpolation'
V72=daily.OUT/'submission_v72'
V73=ROOT/'VALIDATION/runs/v3_20261011_expanded_affine/submission_v73'
V74=constraints.OUT/'submission_v74'
art,read,verify=joint.art,joint.read,joint.verify
DESIGN=dict(parent='selected v76 combined public recipe',
    change='preserve expected positive delay by interpolating each physical state between adjacent minute nodes',
    numerical=interpolation.SETTINGS,
    source_dates='original takeoff dates; original all-departure population',
    prior='original serialized v71, as actually used by v72',
    noise='reuse original v72 permitted-training parameters',
    daily_weight='unchanged v73 stored weight on v72',
    monthly='original v74 fitted composition coefficients; recompute shift from changed incoming predictions',
    retained='v73 serialization, v74 plan corrections, v76 public-model delta and plan corrections',
    no_new_taxi_fit=True,no_new_parameter_fit=True,no_weight_or_scale_search=True,
    evaluation='numerical approximation follow-up on reused development; exact actual-v76 control',
    production=False)


def guard():
    assert read(joint.OUT/'audit.json')['status']=='pass'
    sources=[Path(__file__),Path(interpolation.__file__),Path(heavy.__file__),Path(daily.__file__),Path(heavy.old.__file__),
        Path(constraints.__file__),Path(constraints.monthly.__file__),Path(constraints.exclusion.__file__),
        heavy.OUT/'identity.json',heavy.OUT/'result.json',heavy.OUT/'audit.json',
        heavy.old.OUT/'screen.json',V72/'cv.parquet',V73/'geometry.json',V73/'cv.parquet',
        V74/'cv.parquet',V74/'parameters.json',joint.OUT/'scored.parquet',joint.OUT/'audit.json',
        joint.neural.cat.OUT/'scored.parquet',base.OUT/'rows2025.parquet',
        constraints.monthly.OUT/'rows2025.parquet',constraints.exclusion.OUT/'matches2025.parquet',
        base.SOURCE/'all_pre_departure_delays_2025.csv',base.SOURCE/'taxi_out_additional_time_2025.csv']
    geometry=read(V73/'geometry.json')
    value=dict(design=DESIGN,daily_weight=geometry['weights']['72'],sources=[art(p) for p in sources],checks=interpolation.checks())
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'identity.json'
    if path.exists():assert read(path)==value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(path,value)
    return value


def screen():
    identity=guard()
    assert not (OUT/'result.json').exists(),'Preserve completed result'
    original=pd.read_parquet(V72/'cv.parquet')
    ids=original.mvt_id.to_numpy()
    raw=pd.read_parquet(base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    mrows=pd.read_parquet(constraints.monthly.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    v73=pd.read_parquet(V73/'cv.parquet').set_index('mvt_id').loc[ids]
    v74=pd.read_parquet(V74/'cv.parquet').set_index('mvt_id').loc[ids]
    selected=pd.read_parquet(joint.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    parts=pd.read_parquet(joint.neural.cat.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    for f in [raw,mrows,v73,v74,selected,parts]:
        np.testing.assert_array_equal(original.y,f.y)
    np.testing.assert_array_equal(np.rint(np.maximum(v73.raw_pred,0)),v73.pred)
    parameters=read(heavy.old.OUT/'screen.json')['parameters']
    monthly_parameters=read(V74/'parameters.json')['folds']
    source=base.public(2025).set_index(['airport','day'])
    matches=pd.read_parquet(constraints.exclusion.OUT/'matches2025.parquet')
    posterior=original.pred.to_numpy(float).copy()
    logs=[];max_replay=0.
    for month in base.FOLDS:
        ix=np.flatnonzero(original.month.eq(month))
        rows=raw.iloc[ix].reset_index(drop=True)
        for (airport,day),local in rows.groupby(['airport','day']).indices.items():
            at=ix[local];r=rows.iloc[local];pub=source.loc[(airport,day)]
            assert np.isfinite([pub.total,pub.n_public]).all()
            args=(r.gap.to_numpy(float),original.parent.iloc[at].to_numpy(float),
                r.has_id.to_numpy(bool),r.nm_gap.to_numpy(float),airport,parameters[month][airport],
                float(pub.total),float(pub.n_public-len(r)))
            control,_=heavy.condition(*args,logprior=heavy.GAUSSIAN)
            error=float(np.max(abs(control-original.pred.iloc[at].to_numpy())))
            max_replay=max(max_replay,error)
            np.testing.assert_allclose(control,original.pred.iloc[at],atol=1e-8,rtol=0)
            posterior[at],info=interpolation.condition(*args)
            logs.append(dict(month=month,airport=airport,day=day,rows=len(r),**info))
        print(f'{month}: actual v72 Gaussian replay passed, grid interpolation complete',flush=True)
    submitted=np.rint(np.maximum(posterior,0))
    new73=np.rint(np.maximum(v73.raw_pred.to_numpy()+identity['daily_weight']*(submitted-original.submitted.to_numpy()),0))
    # Recompute both arms through the actual retained later stages.
    results={}
    stages={}
    for name,start in [('control',v73.pred.to_numpy(float)),('candidate',new73)]:
        after_monthly=start.copy()
        for month in base.FOLDS:
            ix=np.flatnonzero(original.month.eq(month))
            after_monthly[ix],_=constraints.monthly.apply(mrows.iloc[ix].reset_index(drop=True),
                start[ix],constraints.monthly.public(2025),monthly_parameters[month])
        if name=='control':np.testing.assert_array_equal(after_monthly,v74.monthly)
        bound=constraints.matched_bound(raw,after_monthly)
        final74=bound.copy()
        for month,allowed in base.FOLDS.items():
            ix=np.flatnonzero(original.month.eq(month))
            permitted=matches[matches.donor_month.isin(allowed+[month])]
            final74[ix],_=constraints.exclusion.apply(ids[ix],final74[ix],permitted)
        if name=='control':np.testing.assert_array_equal(final74,v74.pred)
        # Preserve the exact operation order of public_stack_cv.score.
        final76=constraints.matched_bound(raw,final74+selected.combined.to_numpy()-parts.control.to_numpy())
        for month,allowed in base.FOLDS.items():
            ix=np.flatnonzero(original.month.eq(month))
            permitted=matches[matches.donor_month.isin(allowed+[month])]
            final76[ix],_=constraints.exclusion.apply(ids[ix],np.maximum(final76[ix],0),permitted)
        if name=='control':np.testing.assert_array_equal(final76,selected.combined_transfer)
        results[name]=final76
        stages.update({name+'_v73':start,name+'_monthly':after_monthly,name+'_v74':final74})
    f=original[['mvt_id','month','day','y']].copy()
    for name,value in dict(old_posterior=original.pred.to_numpy(),new_posterior=posterior,
        old_submitted=original.submitted.to_numpy(),new_submitted=submitted,**stages,**results).items():f[name]=value
    assert len(f)==344419 and f.mvt_id.is_unique and np.isfinite(f[['control','candidate']]).all().all()
    assert f[['control','candidate']].ge(0).all().all()
    columns=['old_posterior','new_posterior','control','candidate']
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in f.groupby('month')}
    import tuning_seasonal as ts
    paired={f'{b}_vs_{a}':ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2)
        for a,b in [('old_posterior','new_posterior'),('control','candidate')]}
    f.to_parquet(OUT/'scored.parquet',index=False)
    pd.DataFrame(logs).to_parquet(OUT/'days.parquet',index=False)
    value=dict(design=DESIGN,scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in columns},
        paired=paired,daily_weight=identity['daily_weight'],all_v73_v74_v76_controls_exact=True,
        gaussian_replay_max_error=max_replay,identity=art(OUT/'identity.json'),prediction=art(OUT/'scored.parquet'),
        interpretation='Only numerical grid allocation changes; physical prior states and Gaussian ordinary weights retained; no dates/weights/fitted parameters change; no production')
    write_json(OUT/'result.json',value)
    print(dict(cv=value['cv'],paired=paired,gaussian_replay_max_error=max_replay),flush=True)


def audit():
    identity=guard()
    assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json');verify(result['prediction'])
    assert result['identity']==art(OUT/'identity.json')
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('candidate',OUT/'scored.parquet'),('truth',base.OUT/'rows2025.parquet'),
            ('v72',V72/'cv.parquet'),('v73',V73/'cv.parquet'),('v74',V74/'cv.parquet'),('v76',joint.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from t.y or c.month is distinct from t.month or c.day is distinct from t.day
            or c.control is distinct from f.combined_transfer or c.old_posterior is distinct from a.pred
            or c.old_submitted is distinct from a.submitted or c.control_v73 is distinct from b.pred
            or c.control_monthly is distinct from d.monthly or c.control_v74 is distinct from d.pred
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join v72 a using(mvt_id)
            join v73 b using(mvt_id) join v74 d using(mvt_id) join v76 f using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        # DuckDB round_even agrees with numpy rint at exact half-integers.
        errors=con.execute('''select count(*) from candidate c join v73 b using(mvt_id)
            where c.new_submitted != round_even(greatest(c.new_posterior,0),0)
            or c.candidate_v73 != round_even(greatest(b.raw_pred+?*(c.new_submitted-c.old_submitted),0),0)''',
            [identity['daily_weight']]).fetchone()[0]
        assert errors==0
        for name in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) AS value from candidate group by month').fetchall()
            for month,value in values:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in values]),result['cv'][name],atol=1e-8,rtol=0)
    value=dict(status='pass',rows=n,independent_sql_labels_calendar_all_controls_scores=True,
        replacement_weight_and_serialization_exact=True,all_later_control_stages_replayed=True,
        result=art(OUT/'result.json'),identity=art(OUT/'identity.json'),source=art(__file__))
    write_json(OUT/'audit.json',value)
    print(value,flush=True)


def preflight():
    value=guard()
    print(value['checks'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','screen','audit'])
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
