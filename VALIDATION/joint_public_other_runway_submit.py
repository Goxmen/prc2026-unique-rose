"""Unversioned robust-runway component; cached calibration, no fits or upload."""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import joint_public_other_runway_cv as cv
import joint_public_moments_submit as parent
import target205_delta_transport as transport

ROOT=cv.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_joint_public_other_runway_submit'
V78=ROOT/'VALIDATION/runs/v3_20261011_joint_public_candidate/submission_v78'
V79=ROOT/'VALIDATION/runs/v3_20261012_target205_candidate/submission_v79'
SECONDARY=ROOT/'VALIDATION/runs/v3_20261012_other_runway_delta_transport/other_runway'
BLOCK_AUDIT=ROOT/'VALIDATION/reports/all_discrete_peer_20261012/other_runway_block_audit.json'
art,read,write,verify=cv.art,cv.read,cv.write,cv.verify
DESIGN=dict(settings=cv.model.SETTINGS,rows=344841,months=['2026-01','2026-07'],
    calibration='reuse frozen full2025 parameters and old2026 v78 inference packet; no calibration refit',
    original_clock='original ranking namespace, once-assigned v70 dates; no final-ranking jitter mixing',
    posterior_reference='saved original old_joint; recomputed control atol1e-8 seconds rtol0, log max',
    v78='unchanged matched/recovered bounds and rounding applied to exactv78+0.5*(new_joint-saved_old_joint)',
    v79='unchanged coefficient-one transport: rint(max(0,v79+robust_v78-exactv78))',
    retained='same public/quality/calendar memberships, all supports and noise; same source and helper bytes as CV',
    combination='none; any TabM combination is a separate fixed additive-delta design',
    no_fit=True,no_version=True,no_upload=True,no_weight_search=True,no_new_bounds=True)


def prepare():
    assert not (OUT/'identity.json').exists(),'Preserve prepared component'
    cv.guard();parent_identity=parent.guard()
    ca=read(cv.OUT/'audit.json');assert ca['status']=='pass' and ca['result']==art(cv.OUT/'result.json')
    cv_result=read(cv.OUT/'result.json')
    assert cv_result['paired']['decision']=='improvement' and cv_result['identity']==art(cv.OUT/'identity.json')
    independent=read(BLOCK_AUDIT);assert independent['status']=='pass'
    verify(independent['source'])
    for a in independent['inputs']:verify(a)
    pa=read(parent.OUT/'audit.json')
    assert pa['status']=='pass' and pa['result']==art(parent.OUT/'result.json') and pa['identity']==art(parent.MANIFEST)
    pi=read(parent.OUT/'inference.json');assert pi['identity']==art(parent.MANIFEST)
    for a in pi['artifacts']:verify(a)
    for a in read(parent.OUT/'result.json')['artifacts']:verify(a)
    v78,v79=[read(p/'manifest.json') for p in [V78,V79]]
    for version,manifest in [(78,v78),(79,v79)]:
        assert manifest['version']==version and manifest['state']=='scored'
        assert parent.sha(ROOT/manifest['file'])==manifest['sha256']
    assert read(SECONDARY/'result.json')['all_row_sql_audit']=='pass'
    verify(read(SECONDARY/'result.json')['scored'])
    template=pq.read_table(ROOT/'data/submitting.parquet');ids=template['MVT_ID_mvt'].to_numpy()
    assert len(ids)==len(np.unique(ids))==344841
    for file in [ROOT/v78['file'],ROOT/v79['file'],parent.OUT/'candidate.parquet']:
        table=pq.read_table(file);assert table.schema==template.schema
        for col in template.column_names:
            if col!='TAXITIME_SEC_mvt':assert table[col].equals(template[col])
    assert pq.read_table(ROOT/v78['file']).equals(pq.read_table(parent.OUT/'candidate.parquet'))
    inputs=pd.read_parquet(parent.OUT/'inputs.parquet',columns=cv.model.pilot.INPUTS)
    saved=pd.read_parquet(parent.OUT/'inference.parquet',columns=['mvt_id','joint','touched'])
    assert len(inputs)==len(saved)==26899 and inputs.mvt_id.is_unique and saved.mvt_id.is_unique
    np.testing.assert_array_equal(inputs.mvt_id,saved.mvt_id)
    assert set(inputs.month)==set(DESIGN['months']) and set(inputs.mvt_id).issubset(ids)
    calibration=read(parent.OUT/'parameters.json')
    assert calibration['allowed']==[f'2025-{m:02d}' for m in range(1,13)]
    files=[Path(__file__),Path(cv.__file__),Path(cv.model.__file__),Path(parent.__file__),Path(transport.__file__),
        cv.OUT/'identity.json',cv.OUT/'inference.json',cv.OUT/'result.json',cv.OUT/'audit.json',
        BLOCK_AUDIT,parent.MANIFEST,parent.OUT/'inference.json',parent.OUT/'result.json',parent.OUT/'audit.json',
        V78/'manifest.json',V79/'manifest.json',ROOT/v78['file'],ROOT/v79['file'],
        SECONDARY/'result.json',SECONDARY/'scored.parquet',
        ROOT/'VALIDATION/runs/v3_20261012_other_runway_delta_transport/design.json',
        ROOT/'data/submitting.parquet',parent.OUT/'candidate.parquet',parent.OUT/'inference.parquet',
        *[verify(a) for a in parent_identity['sources']],*[verify(a) for a in read(cv.OUT/'identity.json')['sources']]]
    write(OUT/'identity.json',dict(design=DESIGN,source=art(__file__),sources=[art(p) for p in sorted(set(files))],
        v78_file=art(ROOT/v78['file']),v79_file=art(ROOT/v79['file']),rome_rows=len(inputs),
        training_calibration_reused=True,old_posterior_ids_exact=True,actual_v78_file_exact=True))
    print(dict(prepared=True,rome_rows=len(inputs),full_rows=len(ids)),flush=True)


def guard():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN and identity['source']==art(__file__)
    for a in identity['sources']:verify(a)
    return identity


def infer():
    guard();assert not (OUT/'inference.json').exists() and not (OUT/'inference.parquet').exists()
    start=time.monotonic();rows=pd.read_parquet(parent.OUT/'inputs.parquet',columns=cv.model.pilot.INPUTS)
    saved=pd.read_parquet(parent.OUT/'inference.parquet',columns=['mvt_id','joint','touched']).set_index('mvt_id')
    st=pd.read_parquet(parent.OUT/'taxi_source.parquet');sd=pd.read_parquet(parent.OUT/'delay_source.parquet')
    p=read(parent.OUT/'parameters.json');frames=[];cells=[];checks=[]
    for month,frame in rows.groupby('month',sort=True):
        value,cell=cv.model.apply(frame,st,sd,p['taxi'],p['delay'],p['support'])
        old=saved.loc[value.mvt_id]
        np.testing.assert_array_equal(value.touched,old.touched)
        err=np.abs(value.old_joint.to_numpy()-old.joint.to_numpy())
        np.testing.assert_allclose(value.old_joint,old.joint,atol=1e-8,rtol=0)
        value['recomputed_old_joint']=value.old_joint
        value['old_joint']=old.joint.to_numpy()
        value['new_joint']=np.where(value.extra_cells.gt(0),value.new_joint,value.old_joint)
        value['delta']=value.new_joint-value.old_joint
        assert not value.loc[value.extra_cells.eq(0),'delta'].any()
        frames.append(value.assign(month=month));cells.append(cell.assign(month=month))
        checks.append(dict(month=month,rows=len(value),extra_rows=int(value.extra_cells.gt(0).sum()),
            replay_max_absolute_sec=float(err.max()),delta_against_saved_old=True))
        print(checks[-1],flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'inference.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    write(OUT/'inference.json',dict(identity=art(OUT/'identity.json'),seconds=time.monotonic()-start,
        checks=checks,no_label_values_loaded_or_used=True,artifacts=[art(OUT/p) for p in ['inference.parquet','cells.parquet']]))


def build():
    identity=guard();receipt=read(OUT/'inference.json')
    assert receipt['identity']==art(OUT/'identity.json')
    for a in receipt['artifacts']:verify(a)
    assert not (OUT/'result.json').exists(),'Preserve component'
    template=pq.read_table(ROOT/'data/submitting.parquet');ids=template['MVT_ID_mvt'].to_numpy()
    p78=pq.read_table(verify(identity['v78_file']))['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    p79=pq.read_table(verify(identity['v79_file']))['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    inferred=pd.read_parquet(OUT/'inference.parquet').set_index('mvt_id').reindex(ids)
    delta=inferred.delta.fillna(0).to_numpy();active=inferred.extra_cells.fillna(0).gt(0).to_numpy()
    raw=pd.read_parquet(parent.taxi.daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(parent.taxi.daily.constraints.exclusion.OUT/'ranking_matches.parquet')
    def constrain(p):
        out=parent.taxi.daily.constraints.matched_bound(raw,p)
        out,_=parent.taxi.daily.constraints.exclusion.apply(ids,np.maximum(out,0),matches)
        return np.rint(np.maximum(out,0))
    np.testing.assert_array_equal(constrain(p78),p78)
    new78=constrain(p78+.5*delta)
    np.testing.assert_array_equal(new78[~active],p78[~active])
    new79=transport.transport(p79,p78,new78)
    np.testing.assert_array_equal(transport.transport(p79,p78,p78),p79)
    for name,values in [('v78',new78),('v79',new79)]:
        table=parent.fill_template(template,ids,values);assert table.schema==template.schema
        for col in template.column_names:
            if col!='TAXITIME_SEC_mvt':assert table[col].equals(template[col])
        np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),values)
        pq.write_table(table,OUT/f'candidate_{name}.parquet')
        assert pq.read_table(OUT/f'candidate_{name}.parquet').equals(table)
    pd.DataFrame(dict(mvt_id=ids,v78=p78,v79=p79,posterior_delta=delta,active=active,
        robust_v78=new78,serialized_delta=new78-p78,robust_v79=new79)).to_parquet(OUT/'predictions.parquet',index=False)
    primary,secondary=read(cv.OUT/'result.json'),read(SECONDARY/'result.json')
    write(OUT/'result.json',dict(design=DESIGN,identity=art(OUT/'identity.json'),inference=art(OUT/'inference.json'),
        rows=len(ids),active=int(active.sum()),changed_v78=int(np.sum(new78!=p78)),changed_v79=int(np.sum(new79!=p79)),
        cv_v78=primary['cv'],cv_v79=secondary['cv'],paired_v78=primary['paired'],paired_v79=secondary['paired'],
        parent_zero_delta_and_template_exact=True,artifacts=[art(OUT/p) for p in
            ['candidate_v78.parquet','candidate_v79.parquet','predictions.parquet']],no_version=True,no_upload=True))
    print(dict(rows=len(ids),changed_v78=int(np.sum(new78!=p78)),changed_v79=int(np.sum(new79!=p79))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','infer','build'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
