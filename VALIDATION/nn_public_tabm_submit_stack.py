"""Replay the complete selected stack with one fixed full-year TabM member.

Recompute the unchanged daily correction after the neural replacement, then add
exactly the retained v78 joint increment. Export its integer delta onto v79.
No robust-runway addition, new calibration, version assignment or upload.
"""
import argparse
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import nn_public_tabm_submit as member
import nn_public_tabm_stack as validation
import daily_main_refinements_submit as parent
import joint_public_moments_submit as joint
from records import write_json
from submit import fill_template

ROOT=member.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_submit_stack'
MANIFEST=OUT/'identity.json'
base=parent.base
art,read,verify=member.art,member.read,member.verify
DESIGN=dict(parent='exact integer v78 with fixed coefficient-one transport onto integer v79',
    change='replace only public/physical member seed42 with selected full-input K32 TabM',
    retained_members=[1042,2042],neural='unchanged half calls15 plus half equal three-member physical pool',
    ordinary='unchanged .71 neural plus .29 three-tree mean; original Rome/rules',
    daily='recompute original quality-gated daily correction using unchanged full2025 calibration',
    joint='unchanged saved v78 increment at .5 after original v77 integer stage',
    transport='rint(max(0,v79+(tabm_v78-v78))); zero integer delta retains exact v79',
    rows=344841,ranking_view='old',cv='frozen nn_public_tabm_stack plus original coefficient-one v79 transport',
    no_new_fit=True,no_new_calibration=True,no_new_weight=True,no_new_gate=True,
    no_robust_runway_addition=True,no_version=True,no_upload=True)


def freeze():
    assert not MANIFEST.exists(),'Preserve the frozen stack identity'
    identity=read(member.MANIFEST)
    assert identity['source']==art(member.__file__) and identity['design']==member.DESIGN
    for record in identity['files']:verify(record)
    completion=member.completed(identity);assert completion is not None
    exported=read(member.OUT/'export.json')
    assert exported['identity']==art(member.MANIFEST) and exported['member']==art(member.OUT/'networks/k0.json')
    assert exported['rows']==DESIGN['rows'] and exported['view']=='old'
    verify(exported['pool'])
    assert exported['retained_members']=={k:identity['parent_members'][k] for k in ['1','2']}
    parent.guard();joint.guard()
    for runner in [parent,joint]:
        result=read(runner.OUT/'result.json');audit=read(runner.OUT/'audit.json')
        assert audit['status']=='pass' and audit['result']==art(runner.OUT/'result.json')
        assert result['identity']==art(runner.MANIFEST)
        for record in result['artifacts']:verify(record)
    cv_result=read(validation.OUT/'result.json');cv_audit=read(validation.OUT/'audit.json')
    assert cv_result['exact_v78_replayed'] and cv_audit['result']==art(validation.OUT/'result.json')
    assert cv_audit['status']=='pass' and cv_result['paired']['replace_tabm_vs_selected']['decision']=='improvement'
    assert cv_result['paired']['replace_tabm_vs_replace_control']['decision']=='improvement'
    verify(cv_result['scored'])
    independent=ROOT/'VALIDATION/reports/public_tabm_20261012/score_audit.json'
    reviewed=read(independent);assert reviewed['status']=='pass'
    for record in reviewed['inputs']:verify(record)
    transport=ROOT/'VALIDATION/runs/v3_20261012_delta_transport/tabm'
    assert read(transport/'result.json')['paired']['replace_tabm']['decision']=='improvement'
    paths={Path(__file__),member.MANIFEST,member.OUT/'export.json',verify(exported['pool']),
        member.OUT/'networks/k0.json',member.OUT/'trained/k0.json',verify(completion['prediction']),
        verify(member.trained(identity)['model']),verify(member.trained(identity)['preprocessing']),
        parent.MANIFEST,parent.OUT/'result.json',parent.OUT/'audit.json',parent.OUT/'predictions.parquet',
        joint.MANIFEST,joint.OUT/'result.json',joint.OUT/'audit.json',joint.OUT/'predictions.parquet',
        validation.OUT/'result.json',validation.OUT/'audit.json',verify(cv_result['scored']),
        transport/'result.json',independent,
        base.OUT/'parameters.json',base.aggregate.OUT/'predictions.parquet',
        base.aggregate.V76/'predictions.parquet',base.daily.SOURCE/'daily_runway.parquet',
        base.daily.base.OUT/'ranking_inputs.parquet',base.constraints.exclusion.OUT/'ranking_matches.parquet',
        ROOT/'data/submitting.parquet',*(ROOT/f'submissions/unique-rose_v{v}.parquet' for v in [77,78,79])}
    for k in range(3):paths.add(verify(identity['parent_members'][str(k)]['prediction']))
    # Retain inherited immutable receipt contents rather than mutable upload state.
    for runner in [parent,joint]:
        paths.update(verify(record) for record in read(runner.MANIFEST)['sources'])
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'.venv'):paths.add(path)
    value=dict(design=DESIGN,source=art(__file__),member_identity=art(member.MANIFEST),
        selected_v78=art(validation.OUT/'result.json'),selected_v79=art(transport/'result.json'),
        sources=[art(path) for path in sorted(paths)])
    OUT.mkdir(parents=True,exist_ok=True);write_json(MANIFEST,value)
    print(dict(frozen=True,identity=art(MANIFEST),sources=len(paths)),flush=True)


def guard():
    identity=read(MANIFEST)
    assert identity['source']==art(__file__) and identity['design']==DESIGN
    assert identity['member_identity']==art(member.MANIFEST)
    assert identity['selected_v78']==art(validation.OUT/'result.json')
    for record in identity['sources']:verify(record)
    return identity


def constrain(ids,raw,values,matches):
    bounded=base.constraints.matched_bound(raw,values)
    bounded,_=base.constraints.exclusion.apply(ids,np.maximum(bounded,0),matches)
    return np.rint(np.maximum(bounded,0))


def build():
    guard();assert not (OUT/'result.json').exists(),'Preserve completed stack'
    data=base.selected.rome.physical.parent.ranking_data('old');ids=np.asarray(data['rank_ids'])
    assert len(ids)==DESIGN['rows'] and pd.Index(ids).is_unique
    previous=pd.read_parquet(parent.OUT/'predictions.parquet')
    original=pd.read_parquet(base.aggregate.V76/'predictions.parquet')
    aggregate=pd.read_parquet(base.aggregate.OUT/'predictions.parquet')
    joint_values=pd.read_parquet(joint.OUT/'predictions.parquet')
    pool=pd.read_parquet(member.OUT/'replacement_pool.parquet')
    for frame in [previous,original,aggregate,joint_values,pool]:np.testing.assert_array_equal(frame.mvt_id,ids)
    template=pq.read_table(ROOT/'data/submitting.parquet')
    assert template.num_rows==len(ids);np.testing.assert_array_equal(template['MVT_ID_mvt'].to_numpy(),ids)
    actual={}
    for version in [77,78,79]:
        table=pq.read_table(ROOT/f'submissions/unique-rose_v{version}.parquet')
        assert table.schema==template.schema
        for column in template.column_names:
            if column!='TAXITIME_SEC_mvt':assert table[column].equals(template[column])
        actual[version]=table['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    np.testing.assert_array_equal(previous.candidate_pred,actual[77])
    np.testing.assert_array_equal(joint_values.parent,actual[77]);np.testing.assert_array_equal(joint_values.pred,actual[78])
    identity=read(member.MANIFEST)
    old_members=[pd.read_parquet(verify(identity['parent_members'][str(k)]['prediction'])) for k in range(3)]
    new_member=pd.read_parquet(verify(member.completed(identity)['prediction']))
    for frame in [*old_members,new_member]:np.testing.assert_array_equal(frame.mvt_id,ids)
    old_pool=np.mean([f.base.to_numpy(float) for f in old_members],axis=0)
    new_pool=np.mean([new_member.base.to_numpy(float),*(f.base.to_numpy(float) for f in old_members[1:])],axis=0)
    np.testing.assert_array_equal(old_pool,pool.parent_pool);np.testing.assert_array_equal(new_pool,pool.replacement_pool)
    np.testing.assert_array_equal(old_pool,original.physical_new)
    calls=original.calls_nn.to_numpy(float);tree=previous.candidate_tree.to_numpy(float)
    probability=original.p.to_numpy(float)
    np.testing.assert_array_equal(.5*(calls+old_pool),original.neural_new)
    raw=pd.read_parquet(base.daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(base.constraints.exclusion.OUT/'ranking_matches.parquet')
    inputs=previous[['mvt_id','airport','local_day','runway','eligible','month']]
    parameters=read(base.OUT/'parameters.json');source=base.daily.public()
    outputs={};cells=[]
    for arm,physical in [('control',old_pool),('tabm',new_pool)]:
        neural=.5*(calls+physical)
        direct,_=base.selected.rome.physical.parent.final_frame(data,neural,tree,probability)
        refined_float=base.selected.main_candidate.constrained(ids,raw,aggregate.new74_serialized.to_numpy(),
            direct.pred.to_numpy(),original.control.to_numpy(),matches)
        refined=np.rint(np.maximum(refined_float,0))
        _,cell,shift=base.daily.apply(inputs,refined,source,parameters)
        applied=np.where(previous.quality_eligible,shift,0.)
        bounded=base.aggregate.constrain(ids,raw,refined+applied,matches)
        v77=np.rint(np.maximum(bounded,0))
        v78=constrain(ids,raw,v77+.5*joint_values.increment.to_numpy(),matches)
        stages=dict(direct=direct.pred.to_numpy(),refined_float=refined_float,refined=refined,
            shift=shift,applied=applied,bounded=bounded,pred=v77)
        if arm=='control':
            for name,values in stages.items():np.testing.assert_array_equal(values,previous['candidate_'+name])
            np.testing.assert_array_equal(v77,actual[77]);np.testing.assert_array_equal(v78,actual[78])
        outputs[arm]=dict(**stages,v78=v78);cells.append(cell.assign(arm=arm))
    f=inputs.copy();f['quality_eligible']=previous.quality_eligible.to_numpy()
    for arm,stages in outputs.items():
        for name,values in stages.items():f[arm+'_'+name]=values
    f['v78']=actual[78];f['v79']=actual[79]
    f['serialized_delta']=f.tabm_v78.to_numpy()-actual[78]
    f['tabm_v79']=np.rint(np.maximum(actual[79]+f.serialized_delta.to_numpy(),0))
    zero=f.serialized_delta.eq(0);np.testing.assert_array_equal(f.loc[zero,'tabm_v79'],f.loc[zero,'v79'])
    assert np.isfinite(f[['tabm_v78','tabm_v79']]).all().all() and f[['tabm_v78','tabm_v79']].ge(0).all().all()
    f.to_parquet(OUT/'predictions.parquet',index=False);pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    for version in [78,79]:
        table=fill_template(template,ids,f['tabm_v'+str(version)].to_numpy())
        assert table.schema==template.schema
        for column in template.column_names:
            if column!='TAXITIME_SEC_mvt':assert table[column].equals(template[column])
        np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),f['tabm_v'+str(version)])
        path=OUT/f'candidate_v{version}.parquet';pq.write_table(table,path);assert pq.read_table(path).equals(table)
    primary=read(validation.OUT/'result.json');secondary=read(ROOT/'VALIDATION/runs/v3_20261012_delta_transport/tabm/result.json')
    result=dict(design=DESIGN,identity=art(MANIFEST),rows=len(ids),
        original_pool_and_all_v77_stages_exact=True,integer_v78_exact=True,zero_delta_v79_exact=True,
        unchanged_calibration=art(base.OUT/'parameters.json'),unchanged_joint_increment=art(joint.OUT/'predictions.parquet'),
        cv_v78=primary['cv']['replace_tabm'],paired_v78=primary['paired']['replace_tabm_vs_selected'],
        cv_v79=secondary['cv']['replace_tabm'],paired_v79=secondary['paired']['replace_tabm'],
        changed_v78=int(f.serialized_delta.ne(0).sum()),changed_v79=int(f.tabm_v79.ne(f.v79).sum()),
        artifacts=[art(OUT/p) for p in ['predictions.parquet','cells.parquet','candidate_v78.parquet','candidate_v79.parquet']],
        no_version=True,no_upload=True)
    write_json(OUT/'result.json',result);print(dict(rows=len(ids),changed=result['changed_v79']),flush=True)


def audit():
    guard();assert not (OUT/'audit.json').exists()
    result=read(OUT/'result.json');assert result['identity']==art(MANIFEST)
    for record in result['artifacts']:verify(record)
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        paths={'f':OUT/'predictions.parquet','p':parent.OUT/'predictions.parquet',
            'j':joint.OUT/'predictions.parquet','s78':OUT/'candidate_v78.parquet','s79':OUT/'candidate_v79.parquet',
            'v78':ROOT/'submissions/unique-rose_v78.parquet','v79':ROOT/'submissions/unique-rose_v79.parquet'}
        for name,path in paths.items():con.read_parquet(str(path)).create_view(name)
        same=' OR '.join(f'f.control_{name} IS DISTINCT FROM p.candidate_{name}' for name in
            ['direct','refined_float','refined','shift','applied','bounded','pred'])
        n,u,bad=con.execute(f'''SELECT count(*),count(distinct f.mvt_id),sum(CASE WHEN {same} OR
            f.control_v78 IS DISTINCT FROM j.pred OR f.v78 IS DISTINCT FROM v78.TAXITIME_SEC_mvt OR
            f.v79 IS DISTINCT FROM v79.TAXITIME_SEC_mvt OR f.serialized_delta IS DISTINCT FROM f.tabm_v78-f.v78 OR
            f.tabm_v79 IS DISTINCT FROM round_even(greatest(0,f.v79+f.serialized_delta),0) OR
            f.tabm_v78 IS DISTINCT FROM s78.TAXITIME_SEC_mvt OR f.tabm_v79 IS DISTINCT FROM s79.TAXITIME_SEC_mvt OR
            f.tabm_refined IS DISTINCT FROM round_even(greatest(0,f.tabm_refined_float),0) OR
            f.tabm_applied IS DISTINCT FROM CASE WHEN f.quality_eligible THEN f.tabm_shift ELSE 0 END OR
            f.tabm_pred IS DISTINCT FROM round_even(greatest(0,f.tabm_bounded),0) OR
            (f.serialized_delta=0 AND f.tabm_v79 IS DISTINCT FROM f.v79)
            THEN 1 ELSE 0 END) FROM f JOIN p USING(mvt_id) JOIN j USING(mvt_id)
            JOIN v78 ON f.mvt_id=v78.MVT_ID_mvt JOIN v79 ON f.mvt_id=v79.MVT_ID_mvt
            JOIN s78 ON f.mvt_id=s78.MVT_ID_mvt JOIN s79 ON f.mvt_id=s79.MVT_ID_mvt''').fetchone()
        assert n==u==DESIGN['rows'] and bad==0,(n,u,bad)
    f=pd.read_parquet(OUT/'predictions.parquet');ids=f.mvt_id.to_numpy()
    raw=pd.read_parquet(base.daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(base.constraints.exclusion.OUT/'ranking_matches.parquet')
    inputs=f[['mvt_id','airport','local_day','runway','eligible','month']]
    source,params=base.daily.public(),read(base.OUT/'parameters.json')
    increment=pd.read_parquet(joint.OUT/'predictions.parquet').set_index('mvt_id').loc[ids,'increment'].to_numpy()
    for arm in ['control','tabm']:
        _,_,shift=base.daily.apply(inputs,f[arm+'_refined'].to_numpy(),source,params)
        np.testing.assert_array_equal(shift,f[arm+'_shift'])
        bounded=base.aggregate.constrain(ids,raw,f[arm+'_refined'].to_numpy()+f[arm+'_applied'].to_numpy(),matches)
        np.testing.assert_array_equal(bounded,f[arm+'_bounded'])
        np.testing.assert_array_equal(constrain(ids,raw,f[arm+'_pred'].to_numpy()+.5*increment,matches),f[arm+'_v78'])
    template=pq.read_table(ROOT/'data/submitting.parquet')
    for version in [78,79]:
        table=pq.read_table(OUT/f'candidate_v{version}.parquet');assert table.schema==template.schema
        for column in template.column_names:
            if column!='TAXITIME_SEC_mvt':assert table[column].equals(template[column])
    write_json(OUT/'audit.json',dict(status='pass',identity=art(MANIFEST),result=art(OUT/'result.json'),source=art(__file__),
        rows=n,all_sql_stage_serialization_transport_checks=True,all_daily_shifts_and_constraints_replayed=True,
        protected_columns_exact=True,scope='Built-in all-row replay/audit; separate independent source review retained',
        no_version=True,no_upload=True))
    print('All-row TabM stack and coefficient-one v79 transport audit passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','build','audit'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
