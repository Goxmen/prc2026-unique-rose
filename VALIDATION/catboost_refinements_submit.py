"""Combine selected full-input CatBoost with audited daily-quality refinements.

No refitting, submission version, or upload. The existing full-year source
calibration is reused exactly and recomputed shifts reflect the changed tree.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import catboost_full_inputs_submit as cat
import catboost_refinements_cv as cv
import public_daily_taxi_submit as parent
from records import write_json
from submit import fill_template

ROOT = parent.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_catboost_refinements_submit'
MANIFEST = OUT/'identity.json'
art,read,verify = parent.art,parent.read,parent.verify
DESIGN = dict(parent='selected public-airport/t5/full-monthly/daily-quality production component',
    change='ordinary CatBoost full174 inputs, original K07/10000/seed42',
    calibration='reuse original full2025 daily parameters; recompute shift from changed incoming prediction',
    cv='catboost_refinements_cv.candidate; tiny January gain and cross-machine control caveat retained',
    no_fits=True,no_version=True,no_upload=True)


def guard():
    for module in [cat,cv,parent]:
        assert read(module.OUT/'audit.json')['status']=='pass'
        module.guard()
    result = read(cv.OUT/'result.json')
    assert result['paired']['quality']['decision']=='improvement'
    assert all(x['candidate']<x['control'] for x in result['scores'].values())
    paths = [Path(__file__),Path(cat.__file__),Path(cv.__file__),Path(parent.__file__),
        Path(parent.daily.__file__),Path(parent.aggregate.__file__),
        Path(parent.selected.rome.physical.parent.__file__),Path(parent.constraints.__file__),
        Path(parent.constraints.exclusion.__file__),
        cat.MANIFEST,cat.OUT/'audit.json',cat.OUT/'predictions/old.json',cat.OUT/'predictions/old.parquet',
        cv.OUT/'identity.json',cv.OUT/'result.json',cv.OUT/'audit.json',cv.OUT/'scored.parquet',
        parent.MANIFEST,parent.OUT/'result.json',parent.OUT/'audit.json',parent.OUT/'predictions.parquet',
        parent.OUT/'candidate.parquet',parent.OUT/'parameters.json',parent.OUT/'cv.parquet',
        parent.aggregate.OUT/'predictions.parquet',parent.aggregate.V76/'predictions.parquet',
        parent.daily.SOURCE/'daily_runway.parquet',parent.daily.base.OUT/'ranking_inputs.parquet',
        parent.constraints.exclusion.OUT/'ranking_matches.parquet',ROOT/'data/submitting.parquet']
    value = dict(design=DESIGN,sources=[art(p) for p in paths])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists(): assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(MANIFEST,value)
    return value


def prepare():
    guard()
    assert not (OUT/'result.json').exists(),'Preserve completed component'
    data = parent.selected.rome.physical.parent.ranking_data('old')
    ids = np.asarray(data['rank_ids'])
    control = pd.read_parquet(parent.OUT/'predictions.parquet')
    new_cat = pd.read_parquet(cat.OUT/'predictions/old.parquet')
    original = pd.read_parquet(parent.aggregate.V76/'predictions.parquet')
    aggregate = pd.read_parquet(parent.aggregate.OUT/'predictions.parquet')
    for f in [control,new_cat,original,aggregate]: np.testing.assert_array_equal(f.mvt_id,ids)
    raw = pd.read_parquet(parent.daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(parent.constraints.exclusion.OUT/'ranking_matches.parquet')
    parameters = read(parent.OUT/'parameters.json')
    inputs = control[['mvt_id','airport','local_day','runway','eligible','month']]
    outputs = {}
    cells = []
    for arm,pred in [('control',control.catboost.to_numpy(float)),('candidate',new_cat.pred.to_numpy(float))]:
        tree = np.column_stack([control.main.to_numpy(float),control.airport_new.to_numpy(float),pred]).mean(axis=1)
        direct,_ = parent.selected.rome.physical.parent.final_frame(data,
            original.neural_new.to_numpy(),tree,original.p.to_numpy())
        if arm=='control':
            np.testing.assert_array_equal(tree,control.tree_new)
            np.testing.assert_array_equal(direct.pred,control.direct_new)
        refined_float = parent.selected.main_candidate.constrained(ids,raw,
            aggregate.new74_serialized.to_numpy(),direct.pred.to_numpy(),original.control.to_numpy(),matches)
        refined = np.rint(np.maximum(refined_float,0))
        _,c,shift = parent.daily.apply(inputs,refined,parent.daily.public(),parameters)
        applied = np.where(control.quality_eligible,shift,0.)
        bounded = parent.aggregate.constrain(ids,raw,refined+applied,matches)
        final = np.rint(np.maximum(bounded,0)).astype('int32')
        if arm=='control':
            for actual,key in [(refined_float,'refined_float'),(refined,'refined'),
                (shift,'unrestricted_shift'),(applied,'applied_shift'),(bounded,'bounded'),(final,'submitted')]:
                np.testing.assert_array_equal(actual,control[key])
        outputs[arm] = dict(catboost=pred,tree=tree,direct=direct.pred.to_numpy(),refined_float=refined_float,
            refined=refined,shift=shift,applied=applied,bounded=bounded,pred=final)
        cells.append(c.assign(arm=arm))
    f = inputs.copy()
    f['quality_eligible'] = control.quality_eligible.to_numpy()
    for arm,values in outputs.items():
        for key,value in values.items():f[arm+'_'+key] = value
    f.to_parquet(OUT/'predictions.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    template = pq.read_table(ROOT/'data/submitting.parquet')
    table = fill_template(template,ids,outputs['candidate']['pred'])
    assert table.schema==template.schema
    for name in template.column_names:
        if name!='TAXITIME_SEC_mvt':assert table[name].equals(template[name])
    pq.write_table(table,OUT/'candidate.parquet')
    assert pq.read_table(OUT/'candidate.parquet').equals(table)
    held = pd.read_parquet(cv.OUT/'scored.parquet')[['mvt_id','month','day','y','control','candidate']].rename(columns={'candidate':'pred'})
    held_parent = pd.read_parquet(parent.OUT/'cv.parquet').set_index('mvt_id')
    np.testing.assert_array_equal(held.control,held_parent.loc[held.mvt_id,'pred'])
    held.to_parquet(OUT/'cv.parquet',index=False)
    validation = read(cv.OUT/'result.json')
    result = dict(design=DESIGN,identity=art(MANIFEST),rows=len(ids),jan_jul_cv=validation['cv']['candidate'],
        cv_months={m:x['candidate'] for m,x in validation['scores'].items()},paired=validation['paired']['quality'],
        artifacts=[art(OUT/p) for p in ['candidate.parquet','predictions.parquet','cells.parquet','cv.parquet']],
        parent_control_all_stages_exact=True,full2025_calibration_reused=art(parent.OUT/'parameters.json'),
        changed_rows=int((f.candidate_pred!=f.control_pred).sum()),
        prediction_rms_change=float(np.sqrt(np.mean((f.candidate_pred-f.control_pred).to_numpy(float)**2))),
        no_version=True,no_upload=True)
    write_json(OUT/'result.json',result)
    print(result,flush=True)


def audit():
    guard()
    assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    result = read(OUT/'result.json')
    for a in result['artifacts']:verify(a)
    with duckdb.connect(config={'threads':1}) as con:
        for alias,path in [('c',OUT/'predictions.parquet'),('p',parent.OUT/'predictions.parquet'),
            ('k',cat.OUT/'predictions/old.parquet'),('s',OUT/'candidate.parquet'),
            ('v',OUT/'cv.parquet'),('r',cv.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(alias)
        n,u,errors,tree_error = con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.control_catboost is distinct from p.catboost or c.control_tree is distinct from p.tree_new
            or c.control_direct is distinct from p.direct_new or c.control_refined_float is distinct from p.refined_float
            or c.control_refined is distinct from p.refined or c.control_shift is distinct from p.unrestricted_shift
            or c.control_applied is distinct from p.applied_shift or c.control_bounded is distinct from p.bounded
            or c.control_pred is distinct from p.submitted or c.quality_eligible is distinct from p.quality_eligible
            or c.candidate_catboost is distinct from k.pred
            or c.candidate_refined != round_even(greatest(c.candidate_refined_float,0),0)
            or c.candidate_applied != case when c.quality_eligible then c.candidate_shift else 0 end
            or c.candidate_pred != round_even(greatest(c.candidate_bounded,0),0)
            or c.candidate_pred is distinct from s.TAXITIME_SEC_mvt then 1 else 0 end),
            max(abs(c.candidate_tree-(cast(p.main as double)+cast(p.airport_new as double)+cast(k.pred as double))/3.0))
            from c join p using(mvt_id) join k using(mvt_id) join s on c.mvt_id=s.MVT_ID_mvt''').fetchone()
        assert n==u==344841 and errors==0 and tree_error<1e-8,(n,u,errors,tree_error)
        nc,uc,wrong = con.execute('''select count(*),count(distinct v.mvt_id),sum(case when
            v.month is distinct from r.month or v.day is distinct from r.day or v.y is distinct from r.y
            or v.control is distinct from r.control or v.pred is distinct from r.candidate then 1 else 0 end)
            from v join r using(mvt_id)''').fetchone()
        assert nc==uc==344419 and wrong==0
        scores = con.execute('select month,sqrt(avg(pow(pred-y,2))) from v group by month').fetchall()
        for m,value in scores:np.testing.assert_allclose(value,result['cv_months'][m],atol=1e-8,rtol=0)
        np.testing.assert_allclose(np.mean([v for m,v in scores]),result['jan_jul_cv'],atol=1e-8,rtol=0)
    f = pd.read_parquet(OUT/'predictions.parquet');ids=f.mvt_id.to_numpy()
    inputs = f[['mvt_id','airport','local_day','runway','eligible','month']]
    raw = pd.read_parquet(parent.daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(parent.constraints.exclusion.OUT/'ranking_matches.parquet')
    for arm in ['control','candidate']:
        _,_,shift = parent.daily.apply(inputs,f[arm+'_refined'].to_numpy(),parent.daily.public(),read(parent.OUT/'parameters.json'))
        np.testing.assert_array_equal(shift,f[arm+'_shift'])
        bounded = parent.aggregate.constrain(ids,raw,f[arm+'_refined'].to_numpy()+f[arm+'_applied'].to_numpy(),matches)
        np.testing.assert_array_equal(bounded,f[arm+'_bounded'])
    template = pq.read_table(ROOT/'data/submitting.parquet');table=pq.read_table(OUT/'candidate.parquet')
    assert table.schema==template.schema
    for name in template.column_names:
        if name!='TAXITIME_SEC_mvt':assert table[name].equals(template[name])
    value = dict(status='pass',identity=art(MANIFEST),result=art(OUT/'result.json'),source=art(__file__),
        ranking_rows=n,cv_rows=nc,all_sql_parent_stages_tree_means_gates_rounding_scores=True,
        tree_mean_max_error=tree_error,all_daily_shifts_final_bounds_replayed=True,
        template_schema_and_unmodified_columns_exact=True,no_version=True,no_upload=True)
    write_json(OUT/'audit.json',value)
    print(value,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['guard','prepare','audit'])
    a=p.parse_args()
    with threadpool_limits(1):globals()[a.stage]()
