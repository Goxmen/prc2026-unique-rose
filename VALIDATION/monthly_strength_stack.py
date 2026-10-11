"""One fixed full-strength monthly reconciliation on the actual v76 stack.

Use the retained training-only composition parameters and all existing source
population gates, the120-second raw-shift cap and final constraints. Change
only the monthly multiplier from0.5 to1. No refits or weight search.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_heavy_tail as heavy
import daily_delay_submit as daily
import public_constraints_candidate as constraints
import public_stack_cv as joint
from records import write_json

base=heavy.base
ROOT=base.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_monthly_strength'
V72=daily.OUT/'submission_v72'
V73=ROOT/'VALIDATION/runs/v3_20261011_expanded_affine/submission_v73'
V74=constraints.OUT/'submission_v74'
art,read,verify=joint.art,joint.read,joint.verify
DESIGN=dict(parent='selected v76 combined public recipe',
    change='monthly correction multiplier0.5 to1.0',
    unchanged='all earlier predictions, population gates, raw120-second cap, training coefficients and later constraints',
    no_new_taxi_fit=True,no_parameter_refit=True,no_airport_gate=True,
    no_weight_search=True,previous_weight=.5,candidate_weight=1.,
    evaluation='one natural full-reconciliation endpoint on reused Jan/Jul development',production=False)



def guard():
    assert read(joint.OUT/'audit.json')['status']=='pass'
    sources=[Path(__file__),Path(heavy.__file__),Path(daily.__file__),Path(heavy.old.__file__),
        Path(constraints.__file__),Path(constraints.monthly.__file__),Path(constraints.exclusion.__file__),
        heavy.OUT/'identity.json',heavy.OUT/'result.json',heavy.OUT/'audit.json',
        heavy.old.OUT/'screen.json',V72/'cv.parquet',V73/'geometry.json',V73/'cv.parquet',
        V74/'cv.parquet',V74/'parameters.json',joint.OUT/'scored.parquet',joint.OUT/'audit.json',
        joint.neural.cat.OUT/'scored.parquet',base.OUT/'rows2025.parquet',
        constraints.monthly.OUT/'rows2025.parquet',constraints.exclusion.OUT/'matches2025.parquet',
        base.SOURCE/'all_pre_departure_delays_2025.csv',base.SOURCE/'taxi_out_additional_time_2025.csv']
    geometry=read(V73/'geometry.json')
    value=dict(design=DESIGN,daily_weight=geometry['weights']['72'],sources=[art(p) for p in sources],checks=checks())
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
    monthly_parameters=read(V74/'parameters.json')['folds']
    matches=pd.read_parquet(constraints.exclusion.OUT/'matches2025.parquet')
    new73=v73.pred.to_numpy(float)
    # Recompute both arms through the actual retained later stages.
    results={}
    stages={}
    for name,start in [('control',v73.pred.to_numpy(float)),('candidate',new73)]:
        after_monthly=start.copy()
        for month in base.FOLDS:
            ix=np.flatnonzero(original.month.eq(month))
            after_monthly[ix],_=constraints.monthly.apply(mrows.iloc[ix].reset_index(drop=True),
                start[ix],constraints.monthly.public(2025),monthly_parameters[month])
        np.testing.assert_array_equal(after_monthly,v74.monthly)
        if name=='candidate':after_monthly=np.maximum(0.,2*after_monthly-start)
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
    for name,value in dict(**stages,**results).items():f[name]=value
    assert len(f)==344419 and f.mvt_id.is_unique and np.isfinite(f[['control','candidate']]).all().all()
    assert f[['control','candidate']].ge(0).all().all()
    columns=['control','candidate']
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in f.groupby('month')}
    import tuning_seasonal as ts
    paired={f'{b}_vs_{a}':ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2)
        for a,b in [('control','candidate')]}
    f.to_parquet(OUT/'scored.parquet',index=False)
    value=dict(design=DESIGN,scores=scores,cv={c:float(np.mean([v[c] for v in scores.values()])) for c in columns},
        paired=paired,daily_weight=identity['daily_weight'],all_v73_v74_v76_controls_exact=True,
        monthly_multiplier=1.,identity=art(OUT/'identity.json'),prediction=art(OUT/'scored.parquet'),
        interpretation='Only the monthly correction multiplier changes; no refits or production')
    write_json(OUT/'result.json',value)
    print(dict(cv=value['cv'],paired=paired),flush=True)


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
            or c.control is distinct from f.combined_transfer or c.control_v73 is distinct from b.pred
            or c.control_monthly is distinct from d.monthly or c.control_v74 is distinct from d.pred
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join v72 a using(mvt_id)
            join v73 b using(mvt_id) join v74 d using(mvt_id) join v76 f using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        errors=con.execute("""select count(*) from candidate c join v73 b using(mvt_id)
            where c.candidate_v73 is distinct from b.pred
            or abs(c.candidate_monthly-greatest(0,2*c.control_monthly-b.pred))>1e-8""").fetchone()[0]
        assert errors==0
        for name in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) AS value from candidate group by month').fetchall()
            for month,value in values:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in values]),result['cv'][name],atol=1e-8,rtol=0)
    value=dict(status='pass',rows=n,independent_sql_labels_calendar_all_controls_scores=True,
        monthly_multiplier_and_zero_floor_exact=True,all_later_control_stages_replayed=True,
        result=art(OUT/'result.json'),identity=art(OUT/'identity.json'),source=art(__file__))
    write_json(OUT/'audit.json',value)
    print(value,flush=True)


def checks():
    p=np.array([0.,1.,29.,59.,60.,61.,120.,500.,7200.,10000.])
    for shift in [-60.,-30.,0.,30.,60.]:
        half=np.maximum(0,p+shift)
        np.testing.assert_array_equal(np.maximum(0,2*half-p),np.maximum(0,p+2*shift))
    return dict(full_endpoint_arithmetic=True,zero_floor_edges=True,no_parameters_fitted=True)


def preflight():
    print(guard()['checks'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','screen','audit'])
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
