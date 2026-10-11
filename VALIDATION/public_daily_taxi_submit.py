"""Prepare and audit the selected daily-taxi/refinement production component.

This needs the ten audited airport production models. It prepares a template-
compatible file without reserving a submission number or uploading anything.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import public_daily_taxi_quality as quality
import public_daily_taxi_features as features
import public_airport_submit as airport
import public_refinements_aggregate_submit as aggregate
from records import write_json,sha256
from submit import fill_template

ROOT=quality.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_public_daily_taxi_submit'
MANIFEST=OUT/'identity.json'
daily=quality.original
constraints=daily.constraints
selected=aggregate.selected
art,read,verify=quality.art,quality.read,quality.verify
DESIGN=dict(parent='v76',change='public airport trees, t5/full-monthly and quality-gated daily runway taxi totals',
    cv='retained production-rounded quality.refined_quality',
    production='reuse all selected models and aggregate parameters; daily calibration refit once on full2025',
    quality_gate='same monthly invalid share<0.05',rows=344841,
    no_new_taxi_model_fit=True,no_version=True,no_upload=True)


def guard():
    assert read(quality.OUT/'audit.json')['status']=='pass'
    result=read(quality.OUT/'result.json')
    assert result['paired']['refined_quality_vs_refined']['decision']=='improvement'
    assert all(x['refined_quality']<x['refined'] for x in result['scores'].values())
    assert read(airport.OUT/'audit.json')['status']=='pass'
    assert read(aggregate.OUT/'audit.json')['status']=='pass'
    airport.guard();aggregate.guard()
    original=read(aggregate.V76/'manifest.json')
    assert original['version']==76 and original['sha256']==sha256(ROOT/original['file'])
    for path,digest in original['sources'].items():assert sha256(ROOT/path)==digest,path
    paths=[Path(__file__),Path(quality.__file__),Path(daily.__file__),Path(features.__file__),
        Path(airport.__file__),Path(aggregate.__file__),Path(constraints.__file__),Path(constraints.exclusion.__file__),
        Path(selected.rome.physical.parent.__file__),Path(airport.main.old.__file__),
        quality.OUT/'identity.json',quality.OUT/'result.json',quality.OUT/'audit.json',quality.OUT/'scored.parquet',
        airport.MANIFEST,airport.OUT/'audit.json',airport.OUT/'predictions/old.json',airport.OUT/'predictions/old.parquet',
        aggregate.MANIFEST,aggregate.OUT/'result.json',aggregate.OUT/'audit.json',aggregate.OUT/'predictions.parquet',
        aggregate.V76/'predictions.parquet',ROOT/'submissions/unique-rose_v76.parquet',daily.CACHE,
        daily.SOURCE/'manifest.json',daily.SOURCE/'daily_runway.parquet',features.OUT/'audit.json',
        daily.base.SOURCE/'taxi_out_additional_time_2026.csv',ROOT/'data/ranking.parquet',ROOT/'data/submitting.parquet',
        daily.base.OUT/'ranking_inputs.parquet',constraints.exclusion.OUT/'ranking_matches.parquet',
        airport.main.OUT/'predictions/old.json',airport.main.OUT/'predictions/old.parquet']
    for name in daily.ZONES:paths.extend(airport.paths(name))
    # Parent sources are immutable; do not bind its mutable upload bookkeeping.
    sources=dict(original['sources'])
    sources.update({str(Path(p).relative_to(ROOT)):sha256(p) for p in paths})
    value=dict(design=DESIGN,sources=sources,parent_file=original['file'],parent_sha256=original['sha256'])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('*.parquet'));write_json(MANIFEST,value)
    return value


def prepare():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed preparation'
    data=selected.rome.physical.parent.ranking_data('old');ids=np.asarray(data['rank_ids'])
    prior=pd.read_parquet(aggregate.V76/'predictions.parquet')
    ag=pd.read_parquet(aggregate.OUT/'predictions.parquet')
    replacement=pd.read_parquet(airport.OUT/'predictions/old.parquet')
    public_main=pd.read_parquet(airport.main.OUT/'predictions/old.parquet')
    for f in [prior,ag,replacement,public_main]:np.testing.assert_array_equal(f.mvt_id,ids)
    components,component_sources=airport.main.old.parent_components('old',ids)
    base_components=components.copy();base_components['lgb']=public_main.pred.to_numpy(float)
    names=airport.main.old.parent.hs.GBM;assert names==['lgb','airport','catboost']
    old_tree=base_components[names].to_numpy(float).mean(axis=1)
    np.testing.assert_array_equal(old_tree,prior.tree)
    new_components=base_components.copy();new_components['airport']=replacement.pred.to_numpy(float)
    new_tree=new_components[names].to_numpy(float).mean(axis=1)
    neural=prior.neural_new.to_numpy();probability=prior.p.to_numpy()
    original,_=selected.rome.physical.parent.final_frame(data,neural,old_tree,probability)
    np.testing.assert_array_equal(original.pred,prior.candidate)
    updated,_=selected.rome.physical.parent.final_frame(data,neural,new_tree,probability)
    raw=pd.read_parquet(daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(constraints.exclusion.OUT/'ranking_matches.parquet')
    control=selected.main_candidate.constrained(ids,raw,ag.old74_serialized.to_numpy(),original.pred.to_numpy(),prior.control.to_numpy(),matches)
    np.testing.assert_array_equal(control,prior.pred)
    actual76=pd.read_parquet(ROOT/'submissions/unique-rose_v76.parquet')
    np.testing.assert_array_equal(actual76.MVT_ID_mvt,ids)
    np.testing.assert_array_equal(np.rint(control),actual76.TAXITIME_SEC_mvt)
    aggregate_control=selected.main_candidate.constrained(ids,raw,ag.new74_serialized.to_numpy(),original.pred.to_numpy(),prior.control.to_numpy(),matches)
    np.testing.assert_array_equal(aggregate_control,ag.aggregate76)
    refined_float=selected.main_candidate.constrained(ids,raw,ag.new74_serialized.to_numpy(),updated.pred.to_numpy(),prior.control.to_numpy(),matches)
    refined=np.rint(np.maximum(refined_float,0))
    train=daily.rows();allowed=[f'2025-{m:02d}' for m in range(1,13)]
    assert set(train.month)==set(allowed)
    params,calibration=daily.fit(train,daily.public(),allowed)
    inputs=features.load(ROOT/'data/ranking.parquet').set_index('mvt_id').loc[ids].reset_index()
    eligibility=constraints.monthly.load([ROOT/'data/ranking.parquet'],False).set_index('mvt_id').loc[ids]
    inputs['eligible']=eligibility.eligible.to_numpy();inputs['month']=inputs.takeoff.dt.strftime('%Y-%m')
    inputs=inputs.drop(columns='takeoff');assert 'y' not in inputs
    shifted,cells,shift=daily.apply(inputs,refined,daily.public(),params)
    q=inputs[['airport','local_day']].assign(local_month=inputs.local_day.str[:7]).merge(
        constraints.monthly.public(2026)[['airport','local_month','invalid']],
        on=['airport','local_month'],how='left',validate='many_to_one')
    quality_eligible=q.invalid.lt(.05).to_numpy()
    applied=np.where(quality_eligible,shift,0.)
    bounded=aggregate.constrain(ids,raw,refined+applied,matches)
    candidate=np.rint(np.maximum(bounded,0)).astype('int32')
    template=pq.read_table(ROOT/'data/submitting.parquet');table=fill_template(template,ids,candidate)
    assert len(ids)==344841 and pd.Index(ids).is_unique
    assert table.schema==template.schema
    for name in template.column_names:
        if name!='TAXITIME_SEC_mvt':assert table[name].equals(template[name])
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),candidate)
    pq.write_table(table,OUT/'candidate.parquet');assert pq.read_table(OUT/'candidate.parquet').equals(table)
    frame=inputs.copy()
    for name,value in dict(parent76=actual76.TAXITIME_SEC_mvt.to_numpy(),main=public_main.pred.to_numpy(),
        airport_old=components.airport.to_numpy(),airport_new=replacement.pred.to_numpy(),catboost=components.catboost.to_numpy(),
        tree_old=old_tree,tree_new=new_tree,direct_old=original.pred.to_numpy(),direct_new=updated.pred.to_numpy(),
        refined_float=refined_float,refined=refined,unrestricted_shift=shift,invalid_share=q.invalid.to_numpy(),
        quality_eligible=quality_eligible,applied_shift=applied,bounded=bounded,submitted=candidate).items():frame[name]=value
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    cells.to_parquet(OUT/'cells.parquet',index=False);calibration.to_parquet(OUT/'calibration.parquet',index=False)
    write_json(OUT/'parameters.json',params)
    cv=read(quality.OUT/'result.json')
    held=pd.read_parquet(quality.OUT/'scored.parquet')[['mvt_id','month','day','y','refined','refined_quality']]
    held=held.rename(columns={'refined':'control','refined_quality':'pred'});held.to_parquet(OUT/'cv.parquet',index=False)
    score={m:float(np.sqrt(np.mean((g.pred-g.y)**2))) for m,g in held.groupby('month')}
    np.testing.assert_allclose(np.mean(list(score.values())),cv['cv']['refined_quality'],atol=1e-10,rtol=0)
    result=dict(design=DESIGN,identity=art(MANIFEST),rows=len(ids),jan_jul_cv=cv['cv']['refined_quality'],
        cv_months=score,paired=cv['paired']['refined_quality_vs_refined'],
        artifacts=[art(OUT/name) for name in ['candidate.parquet','predictions.parquet','cells.parquet','calibration.parquet','parameters.json','cv.parquet']],
        component_sources=component_sources,controls_reproduced=['v76 direct','v76 final and serialization','aggregate production'],
        changed_rows=int((candidate!=actual76.TAXITIME_SEC_mvt.to_numpy()).sum()),
        prediction_rms_change=float(np.sqrt(np.mean((candidate-actual76.TAXITIME_SEC_mvt.to_numpy(float))**2))),
        quality_excluded_rows=int((~quality_eligible).sum()),no_version=True,no_upload=True)
    write_json(OUT/'result.json',result);print(result,flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json')
    for artifact in result['artifacts']:verify(artifact)
    for artifact in result['component_sources']:verify(artifact)
    with duckdb.connect(config={'threads':1}) as con:
        con.read_parquet(str(OUT/'predictions.parquet')).create_view('c')
        con.read_parquet(str(OUT/'candidate.parquet')).create_view('f')
        con.read_parquet(str(aggregate.V76/'predictions.parquet')).create_view('p')
        con.read_parquet(str(airport.OUT/'predictions/old.parquet')).create_view('a')
        con.read_csv(str(daily.base.SOURCE/'taxi_out_additional_time_2026.csv')).create_view('s')
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.direct_old is distinct from p.candidate or c.tree_old is distinct from p.tree
            or c.airport_new is distinct from a.pred or c.parent76 is distinct from p.submitted
            or c.refined != round_even(greatest(c.refined_float,0),0)
            or c.invalid_share is distinct from (s.TF-s.VALID_FL)/s.TF
            or c.quality_eligible != coalesce((s.TF-s.VALID_FL)/s.TF<0.05,false)
            or c.applied_shift != case when c.quality_eligible then c.unrestricted_shift else 0 end
            or c.submitted != round_even(greatest(c.bounded,0),0)
            or c.submitted is distinct from f.TAXITIME_SEC_mvt then 1 else 0 end)
            from c join p using(mvt_id) join a using(mvt_id) join f on c.mvt_id=f.MVT_ID_mvt
            left join s on c.airport=s.APT_ICAO and substr(c.local_day,1,7)=cast(s.YEAR as varchar)||'-'||lpad(cast(s.MONTH_NUM as varchar),2,'0')''').fetchone()
        assert n==u==344841 and errors==0,(n,u,errors)
        tree_error=con.execute('select max(abs(tree_new-(main+airport_new+catboost)/3.0)) from c').fetchone()[0]
        assert tree_error<1e-8
        con.read_parquet(str(OUT/'cv.parquet')).create_view('cv')
        scores=con.execute('select month,sqrt(avg(pow(pred-y,2))) from cv group by month').fetchall()
        for month,value in scores:np.testing.assert_allclose(value,result['cv_months'][month],atol=1e-8,rtol=0)
        np.testing.assert_allclose(np.mean([v for m,v in scores]),result['jan_jul_cv'],atol=1e-8,rtol=0)
    f=pd.read_parquet(OUT/'predictions.parquet');ids=f.mvt_id.to_numpy()
    inputs=f[['mvt_id','airport','local_day','runway','eligible','month']]
    _,cells,shift=daily.apply(inputs,f.refined.to_numpy(),daily.public(),read(OUT/'parameters.json'))
    np.testing.assert_array_equal(shift,f.unrestricted_shift)
    raw=pd.read_parquet(daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(constraints.exclusion.OUT/'ranking_matches.parquet')
    np.testing.assert_array_equal(aggregate.constrain(ids,raw,f.refined.to_numpy()+f.applied_shift.to_numpy(),matches),f.bounded)
    table=pq.read_table(OUT/'candidate.parquet');template=pq.read_table(ROOT/'data/submitting.parquet')
    assert table.schema==template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    report=dict(status='pass',rows=n,all_sql_controls_quality_serializations_cv=True,
        tree_mean_max_error=float(tree_error),all_daily_shifts_and_final_bounds_replayed=True,
        template_ids_schema_exact=True,identity=art(MANIFEST),result=art(OUT/'result.json'),source=art(__file__),
        no_version=True,no_upload=True)
    write_json(OUT/'audit.json',report);print(report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['guard','prepare','audit'])
    a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
