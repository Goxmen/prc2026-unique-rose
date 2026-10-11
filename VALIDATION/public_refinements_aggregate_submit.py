"""Prepare the validated t5/full-monthly adjustments on original ranking rows.

Reuse every fitted parameter and affine coefficient. Reproduce the actual
v72/v73/v74/v76 stages before substitution, retaining production serialization.
This reusable component has no submission version, model fit or upload.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_refinements_cv as cv
import public_neural_candidate as selected
from records import write_json

ROOT = cv.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_public_refinements_aggregate_submit'
MANIFEST = OUT/'identity.json'
tail, constraints = cv.tail, cv.constraints
base, heavy = tail.base, tail.heavy
V72, V73, V74 = tail.V72, tail.V73, tail.V74
V76 = selected.OUT/'submission_v76'
art, read, verify = cv.art, cv.read, cv.verify
DESIGN = dict(parent='v76', change='validated joint t5 daily and full monthly refinement',
    daily='original v71 prior, takeoff dates, population, full-year parameters and v73 weight',
    monthly='original full-year composition coefficients and120-second cap; multiplier1',
    controls=['v72 Gaussian','serialized v73 affine','v74 stages','v76 transfer'],
    serialization='retain rounding at v72, v73 and v74 production boundaries',
    final_constraints='original matched and recovered-plan constraints',
    no_new_parameters=True, no_model_fit=True, no_version=True, no_upload=True)


def guard():
    cv.guard()
    result = read(cv.OUT/'result.json')
    assert read(cv.OUT/'audit.json')['status']=='pass'
    assert result['paired']['combined_vs_control']['decision']=='improvement'
    assert read(V76/'audit.json')['status']=='pass'
    geometry=read(V73/'geometry.json'); manifest=read(V73/'manifest.json')
    versions=geometry['versions']
    assert set(map(str,versions))==set(geometry['weights'])
    sources=[Path(__file__),Path(cv.__file__),Path(tail.__file__),Path(heavy.__file__),
        Path(constraints.__file__),Path(constraints.monthly.__file__),Path(constraints.exclusion.__file__),
        Path(selected.main_candidate.__file__),cv.OUT/'result.json',cv.OUT/'audit.json',
        base.OUT/'ranking_inputs.parquet',ROOT/'data/ranking.parquet',ROOT/'data/submitting.parquet',
        base.SOURCE/'all_pre_departure_delays_2026.csv',base.SOURCE/'taxi_out_additional_time_2026.csv',
        V72/'parameters.json',V72/'predictions.parquet',V73/'geometry.json',V73/'manifest.json',
        V74/'parameters.json',V74/'predictions.parquet',V76/'predictions.parquet',V76/'audit.json',
        constraints.exclusion.OUT/'ranking_matches.parquet']
    basis={}
    for version in versions:
        matches=[p for p in manifest['sources'] if Path(p).name==f'unique-rose_v{version}.parquet']
        assert len(matches)==1,(version,matches)
        path=ROOT/matches[0]
        record=art(path)
        assert manifest['sources'][record['path']]==record['sha256']
        basis[str(version)]=record; sources.append(path)
    for version in [71,73,74,76]:sources.append(ROOT/f'submissions/unique-rose_v{version}.parquet')
    value=dict(design=DESIGN,sources=[art(p) for p in sources],basis=basis,
        weights=geometry['weights'],versions=versions,daily_weight=geometry['weights']['72'],
        cv=art(cv.OUT/'result.json'),no_ranking_labels=True)
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(MANIFEST,value)
    return value


def constrain(ids,raw,values,matches):
    values=constraints.matched_bound(raw,values)
    values,_=constraints.exclusion.apply(ids,np.maximum(values,0),matches)
    return values


def prepare():
    identity=guard();assert not (OUT/'result.json').exists(),'Preserve completed preparation'
    template=pd.read_parquet(ROOT/'data/submitting.parquet');ids=template.MVT_ID_mvt.to_numpy()
    assert len(ids)==344841 and pd.Index(ids).is_unique
    raw=pd.read_parquet(base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    mrows=constraints.monthly.load([ROOT/'data/ranking.parquet'],False).set_index('mvt_id').loc[ids].reset_index()
    assert 'y' not in raw and 'y' not in mrows and 'BLOCK_TIME_UTC_mvt' not in raw
    stages={}
    for name,path in [('v72',V72),('v74',V74),('v76',V76)]:
        frame=pd.read_parquet(path/'predictions.parquet')
        np.testing.assert_array_equal(frame.mvt_id,ids);stages[name]=frame
    actual={}
    for version in [71,73,74,76]:
        frame=pd.read_parquet(ROOT/f'submissions/unique-rose_v{version}.parquet')
        np.testing.assert_array_equal(frame.MVT_ID_mvt,ids)
        actual[version]=frame.TAXITIME_SEC_mvt.to_numpy(float)
    np.testing.assert_array_equal(stages['v72'].parent,actual[71])
    params=read(V72/'parameters.json')['parameters']
    source=base.public(2026).set_index(['airport','day'])
    posterior=np.full(len(ids),np.nan);gaussian=np.full(len(ids),np.nan);logs=[]
    for (airport,day),at in raw.groupby(['airport','day']).indices.items():
        rows=raw.iloc[at];pub=source.loc[(airport,day)]
        assert np.isfinite([pub.total,pub.n_public]).all()
        args=(rows.gap.to_numpy(float),actual[71][at],rows.has_id.to_numpy(bool),
            rows.nm_gap.to_numpy(float),airport,params[airport],float(pub.total),float(pub.n_public-len(rows)))
        gaussian[at],_=heavy.condition(*args,logprior=heavy.GAUSSIAN)
        posterior[at],info=heavy.condition(*args)
        logs.append(dict(airport=airport,day=day,rows=len(rows),**info))
    assert np.isfinite(posterior).all() and np.isfinite(gaussian).all()
    np.testing.assert_allclose(gaussian,stages['v72'].pred,atol=1e-8,rtol=0)
    np.testing.assert_array_equal(np.rint(gaussian),stages['v72'].submitted)
    P=[]
    for version in identity['versions']:
        frame=pd.read_parquet(verify(identity['basis'][str(version)]))
        np.testing.assert_array_equal(frame.MVT_ID_mvt,ids)
        P.append(frame.TAXITIME_SEC_mvt.to_numpy(float))
    P=np.column_stack(P);w=np.array([identity['weights'][str(v)] for v in identity['versions']])
    affine=P@w
    np.testing.assert_array_equal(np.rint(np.maximum(affine,0)),actual[73])
    daily=np.rint(np.maximum(posterior,0))
    new73=np.rint(np.maximum(affine+identity['daily_weight']*(daily-stages['v72'].submitted.to_numpy()),0))
    monthly_params=read(V74/'parameters.json')['full_year']
    old_monthly,_=constraints.monthly.apply(mrows,actual[73],constraints.monthly.public(2026),monthly_params)
    half_monthly,cells=constraints.monthly.apply(mrows,new73,constraints.monthly.public(2026),monthly_params)
    np.testing.assert_array_equal(old_monthly,stages['v74'].monthly)
    full_monthly=np.maximum(0,2*half_monthly-new73)
    matches=pd.read_parquet(constraints.exclusion.OUT/'ranking_matches.parquet')
    old74=constrain(ids,raw,old_monthly,matches)
    np.testing.assert_array_equal(old74,stages['v74'].pred)
    np.testing.assert_array_equal(np.rint(old74),actual[74])
    new74=constrain(ids,raw,full_monthly,matches)
    # Existing v76 production starts from the actual integer v74 file.
    control76=selected.main_candidate.constrained(ids,raw,actual[74],
        stages['v76'].candidate.to_numpy(),stages['v76'].control.to_numpy(),matches)
    np.testing.assert_array_equal(control76,stages['v76'].pred)
    np.testing.assert_array_equal(np.rint(control76),actual[76])
    new74_serialized=np.rint(np.maximum(new74,0))
    new76=selected.main_candidate.constrained(ids,raw,new74_serialized,
        stages['v76'].candidate.to_numpy(),stages['v76'].control.to_numpy(),matches)
    assert np.isfinite(new76).all() and (new76>=0).all()
    frame=pd.DataFrame(dict(mvt_id=ids,old71=actual[71],old72=gaussian,new72=posterior,
        old72_serialized=stages['v72'].submitted,new72_serialized=daily,
        raw73=affine,old73=actual[73],new73=new73,old_monthly=old_monthly,
        half_monthly=half_monthly,full_monthly=full_monthly,old74=old74,new74=new74,
        old74_serialized=actual[74],new74_serialized=new74_serialized,
        old76=control76,aggregate76=new76,submitted=np.rint(new76).astype(np.int32)))
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    pd.DataFrame(logs).to_parquet(OUT/'days.parquet',index=False)
    cells.to_parquet(OUT/'monthly_cells.parquet',index=False)
    report=dict(design=DESIGN,identity=art(MANIFEST),prediction=art(OUT/'predictions.parquet'),
        days=art(OUT/'days.parquet'),monthly_cells=art(OUT/'monthly_cells.parquet'),rows=len(ids),
        gaussian_max_error=float(abs(gaussian-stages['v72'].pred).max()),
        controls_replayed=['v72','v73','v74','v76'],all_control_serializations_exact=True,
        changed_rows=int((frame.submitted.to_numpy()!=actual[76]).sum()),
        rms_change=float(np.sqrt(np.mean((frame.submitted.to_numpy()-actual[76])**2))),
        cv_aggregate=read(cv.OUT/'result.json')['cv']['aggregate'],
        cv_caveat='CV retains unrounded v74 boundary, as inherited by v76; production retains its serialized boundary',
        no_version=True,no_upload=True)
    write_json(OUT/'result.json',report);print(report,flush=True)


def audit():
    identity=guard();assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json');verify(result['prediction'])
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('candidate',OUT/'predictions.parquet'),('raw',base.OUT/'ranking_inputs.parquet'),
            ('v72',V72/'predictions.parquet'),('v74',V74/'predictions.parquet'),('v76',V76/'predictions.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            abs(c.old72-a.pred)>1e-8 or c.old72_serialized is distinct from a.submitted
            or c.old_monthly is distinct from b.monthly or c.old74 is distinct from b.pred
            or c.old74_serialized is distinct from b.submitted or c.old76 is distinct from d.pred
            or c.new72_serialized != round_even(greatest(c.new72,0),0)
            or c.new73 != round_even(greatest(c.raw73+?*(c.new72_serialized-c.old72_serialized),0),0)
            or c.full_monthly is distinct from greatest(0,2*c.half_monthly-c.new73)
            or c.new74_serialized != round_even(greatest(c.new74,0),0)
            or c.submitted != round_even(greatest(c.aggregate76,0),0)
            then 1 else 0 end) from candidate c join raw r using(mvt_id)
            join v72 a using(mvt_id) join v74 b using(mvt_id) join v76 d using(mvt_id)''',
            [identity['daily_weight']]).fetchone()
        assert n==u==344841 and errors==0
    f=pd.read_parquet(verify(result['prediction']))
    ids=f.mvt_id.to_numpy()
    raw=pd.read_parquet(base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(constraints.exclusion.OUT/'ranking_matches.parquet')
    np.testing.assert_array_equal(constrain(ids,raw,f.full_monthly.to_numpy(),matches),f.new74)
    parts=pd.read_parquet(V76/'predictions.parquet');np.testing.assert_array_equal(parts.mvt_id,ids)
    expected=selected.main_candidate.constrained(ids,raw,f.new74_serialized.to_numpy(),
        parts.candidate.to_numpy(),parts.control.to_numpy(),matches)
    np.testing.assert_array_equal(expected,f.aggregate76)
    report=dict(status='pass',rows=n,identity=art(MANIFEST),result=art(OUT/'result.json'),
        all_controls_and_boundaries_checked_in_sql=True,all_later_constraints_replayed=True,
        source=art(__file__),no_version=True,no_upload=True)
    write_json(OUT/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['guard','prepare','audit'])
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
