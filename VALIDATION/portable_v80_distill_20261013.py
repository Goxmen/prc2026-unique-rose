"""Fixed blocked teacher distillation into the audited portable prediction span.

No base model fits. Four label-free CV solves and predictions precede a separate
input audit and score stage. Production is one separately allocated solve.
Every execution stage is exclusive and single-attempt; partial work is retained.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import portable_public_source_20261013 as public
import final_shared_clock_transport_20261013 as uploaded
import tuning_seasonal as seasonal
from records import clean
from submit import fill_template

ROOT = public.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261013_v80_teacher_distillation'
MANIFEST = OUT/'identity.json'
PLAN = ROOT/'VALIDATION/reports/cohort_learning_20261013/v80_distillation_fixed_design.json'
REVIEW = OUT/'source_review.json'
FIXTURE = OUT/'synthetic_preflight.json'
METADATA = OUT/'metadata_preflight.json'
AUDITOR = ROOT/'VALIDATION/tools/audit_portable_v80_distill_20261013.py'
INPUT_AUDITOR = ROOT/'VALIDATION/tools/audit_portable_v80_distill_inputs_20261013.py'
AFFINE = ROOT/'VALIDATION/runs/v3_20261013_portable_affine'
REMOTE = ROOT/'VALIDATION/runs/v3_20261013_portable_feedback_candidates/remote_stats/unique-rose_v80.parquet.json'
VERSIONS = [1,4,8,12,13,14,15,16,17,18,19,20,21,30,33,34,37,38,45,46,54,56,57,58,60,61,64,70]
MONTHS = ['2025-01','2025-07']
META = ['mvt_id','month','day','airport']
PRED_COLUMNS = META+['v82','v80','block','raw','submitted']
art, read, verify = public.art, public.read, public.verify


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(clean(value),stream,indent=2,allow_nan=False);stream.write('\n')


def ids_hash(ids):
    return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()


def projected(record,columns):
    path=verify(record);schema=pq.ParquetFile(path).schema_arrow
    assert set(columns)<=set(schema.names)
    return dict(artifact=record,columns=columns,rows=pq.ParquetFile(path).metadata.num_rows,
                schema_sha256=hashlib.sha256(str(schema).encode()).hexdigest())


def load(spec,truth=False):
    assert truth or not ({'y','TAXITIME_SEC_mvt'}&set(spec['columns'])),spec['columns']
    # TAXITIME_SEC_mvt is permitted only for authenticated submitted predictions.
    path=verify(spec['artifact']);schema=pq.ParquetFile(path).schema_arrow
    assert hashlib.sha256(str(schema).encode()).hexdigest()==spec['schema_sha256']
    frame=pd.read_parquet(path,columns=spec['columns'])
    assert len(frame)==spec['rows'] and list(frame)==spec['columns']
    return frame


def load_prediction(spec):
    # Explicit prediction packets may use the official target-shaped column;
    # source provenance binds them to submissions, never raw training labels.
    assert 'y' not in spec['columns']
    return load(spec,truth=True)


def view_spec(view,a,p,r,old_teacher):
    spec=dict(anchor=projected(r['prediction'],META+['submitted']+(['v80'] if view=='cv' else [])),
        raw=projected(p['specs'][view]['raw'],['mvt_id','month','has_id','nm_gap']),
        matches=projected(p['specs'][view]['matches'],['mvt_id','accepted','recovered_gap','donor_mvt_id','donor_month']))
    if view in ['cv','old']:
        spec['basis']=projected(a['cv_basis' if view=='cv' else 'old_basis'],
            ['mvt_id']+([*META[1:]] if view=='cv' else [])+[f'v{v}' for v in VERSIONS])
    if view=='cv':
        spec['teacher']=projected(a['control_cv'],['mvt_id','three_tabm_runway'])
        spec['truth']=projected(a['control_cv'],['mvt_id','month','day','y'])
    else:
        spec['template']=projected(p['specs'][view]['template'],['MVT_ID_mvt','TAXITIME_SEC_mvt'])
        if view=='old':spec['teacher']=projected(art(old_teacher),['MVT_ID_mvt','TAXITIME_SEC_mvt'])
    return spec


def freeze():
    assert not MANIFEST.exists() and not (OUT/'cv/started.json').exists()
    review=read(REVIEW)
    for key,path in [('source',Path(__file__)),('design',PLAN),('auditor',AUDITOR),('input_auditor',INPUT_AUDITOR),('fixture',FIXTURE),('metadata',METADATA)]:
        assert review[key]==art(path),key
    fixture=read(FIXTURE)
    assert review['status']=='pass' and fixture['status']=='pass'
    assert fixture['source']==art(__file__) and fixture['design']==art(PLAN)
    plan=read(PLAN);assert plan['basis']['versions']==VERSIONS
    a=read(AFFINE/'identity.json');ar=read(AFFINE/'result.json');aa=read(AFFINE/'independent_audit.json')
    assert a['versions']==ar['versions']==VERSIONS and aa['status']=='pass'
    assert ar['identity']==aa['identity']==art(AFFINE/'identity.json')
    assert aa['result']==art(AFFINE/'result.json') and aa['predictions']==ar['predictions']
    p=read(public.MANIFEST)
    metadata_report=read(METADATA)
    assert metadata_report['status']=='pass' and metadata_report['source']==art(__file__) and metadata_report['design']==art(PLAN)
    paths={Path(__file__),PLAN,REVIEW,FIXTURE,METADATA,AUDITOR,INPUT_AUDITOR,public.MANIFEST,REMOTE,
           AFFINE/'identity.json',AFFINE/'result.json',AFFINE/'independent_audit.json',
           ROOT/'data/ranking.parquet',ROOT/'data/final_ranking.parquet'}
    for record in plan['metadata_refs']:
        paths.add(verify(record))
    old_teacher,teacher_paths=uploaded.teacher_contract();paths.update(teacher_paths)
    remote=read(REMOTE)
    assert remote['status']=='success' and remote['name']=='unique-rose_v80.parquet'
    assert remote['metadata']['X-Amz-Meta-Sha256']==art(old_teacher)['sha256'] and remote['size']==old_teacher.stat().st_size
    specs={}
    for view in ['cv','old','final']:
        folder=public.OUT/view;r=read(folder/'result.json');audit=read(folder/'audit.json')
        assert audit['status']=='pass' and audit['identity']==r['identity']==art(public.MANIFEST)
        assert audit['result']==art(folder/'result.json') and audit['prediction']==r['prediction']
        paths.update([folder/'result.json',folder/'audit.json'])
        specs[view]=view_spec(view,a,p,r,old_teacher)
    components={}
    for v in VERSIONS:
        rec=a['components'][str(v)];checked=read(verify(rec['audit']))
        result=read(verify(rec['result'])) if 'result' in rec else None
        assert checked['status']=='pass'
        assert public.correction.audit_binds(checked,rec['final'],rec.get('result'),result)
        components[str(v)]={view:projected(rec[view],[rec['id_column'],rec['value_column']]) for view in ['old','final']}
        components[str(v)].update(id_column=rec['id_column'],value_column=rec['value_column'])
        paths.add(verify(rec['audit']))
        if 'result' in rec:paths.add(verify(rec['result']))
    assert specs==metadata_report['specs']
    assert {v:{view:r[view] for view in ['old','final']} for v,r in components.items()}==metadata_report['components']
    for spec in specs.values():
        paths.update(verify(value['artifact']) for value in spec.values())
    for rec in components.values():
        paths.update(verify(rec[view]['artifact']) for view in ['old','final'])
    for month in MONTHS:
        raw=list((ROOT/'data').glob(f'training_{month}-01_*.parquet'));assert len(raw)==1
        paths.add(raw[0])
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT/'VALIDATION'):paths.add(path)
    write(MANIFEST,dict(source=art(__file__),design=art(PLAN),review=art(REVIEW),fixture=art(FIXTURE),
        auditor=art(AUDITOR),input_auditor=art(INPUT_AUDITOR),metadata=art(METADATA),versions=VERSIONS,specs=specs,components=components,
        sources=[art(path) for path in sorted(paths)],historical_cv_caveat=a['historical_cv_caveat'],
        no_prediction_arrays_decoded=True,teacher_upload=art(old_teacher),teacher_remote=art(REMOTE)))
    print(dict(identity=art(MANIFEST),sources=len(paths)),flush=True)


def guard():
    m=read(MANIFEST)
    for key,path in [('source',Path(__file__)),('design',PLAN),('auditor',AUDITOR),('input_auditor',INPUT_AUDITOR),('review',REVIEW),('fixture',FIXTURE),('metadata',METADATA)]:
        assert m[key]==art(path),key
    assert m['versions']==VERSIONS
    for rec in m['sources']:verify(rec)
    return m


def claim(stage):
    folder=OUT/stage;folder.mkdir(parents=True,exist_ok=True)
    allocation=read(folder/'allocation.json')
    assert allocation['authority']=='/root' and allocation['identity']==art(MANIFEST)
    assert allocation['state']==f'released_for_distillation_{stage}'
    # Exclusive creation closes both duplicate launches and implicit retries.
    write(folder/'started.json',dict(identity=art(MANIFEST),allocation=art(folder/'allocation.json'),pid=os.getpid()))
    return art(folder/'allocation.json')


def align(frame,ids,id_column='mvt_id',full=True):
    assert frame[id_column].is_unique and pd.Index(ids).isin(frame[id_column]).all()
    if full:assert len(frame)==len(ids) and set(frame[id_column])==set(ids)
    return frame.set_index(id_column).loc[ids].reset_index()


def inputs(view,m):
    spec=m['specs'][view];anchor=load(spec['anchor']);ids=anchor.mvt_id.to_numpy()
    assert len(ids)==public.EXPECTED_ROWS[view] and anchor.mvt_id.is_unique
    z=anchor.submitted.to_numpy(float);np.testing.assert_array_equal(z,public.R(z))
    if view in ['cv','old']:
        b=align(load(spec['basis']),ids)
        if view=='cv':
            for key in META:np.testing.assert_array_equal(b[key],anchor[key])
        matrix=b[[f'v{v}' for v in VERSIONS]].to_numpy(float)
        if view=='old':
            for j,v in enumerate(VERSIONS):
                rec=m['components'][str(v)];c=align(load_prediction(rec['old']),ids,rec['id_column'])
                np.testing.assert_array_equal(matrix[:,j],c[rec['value_column']])
    else:
        matrix=np.empty((len(ids),len(VERSIONS)))
        for j,v in enumerate(VERSIONS):
            rec=m['components'][str(v)];c=align(load_prediction(rec['final']),ids,rec['id_column'])
            matrix[:,j]=c[rec['value_column']].to_numpy(float)
    assert np.isfinite(matrix).all() and (matrix>=0).all() and np.array_equal(matrix,np.rint(matrix))
    yteacher=None
    if view=='cv':
        t=align(load(spec['teacher']),ids);yteacher=t.three_tabm_runway.to_numpy(float)
        np.testing.assert_array_equal(yteacher,anchor.v80)
    elif view=='old':
        t=align(load_prediction(spec['teacher']),ids,'MVT_ID_mvt');yteacher=t.TAXITIME_SEC_mvt.to_numpy(float)
    raw=align(load(spec['raw']),ids,full=False);matches=load(spec['matches'])
    np.testing.assert_array_equal(raw.month,anchor.month)
    control=public.constrain(view,raw,ids,z,matches)
    np.testing.assert_array_equal(control,z)
    if view!='cv':
        template=load_prediction(spec['template']);np.testing.assert_array_equal(ids,template.MVT_ID_mvt)
    return anchor,matrix-z[:,None],yteacher,raw,matches


def block_masks(day,block):
    number=pd.to_datetime(day).dt.day.to_numpy()
    assert block in ['A','B']
    held=number<=15 if block=='A' else number>=16
    train=number>=17 if block=='A' else number<=14
    assert not (held&train).any() and held.any() and train.any()
    return train,held


def solve(X,target):
    X=np.asarray(X,dtype=np.float64);target=np.asarray(target,dtype=np.float64)
    assert X.ndim==2 and target.shape==(len(X),) and len(X)>0
    assert np.isfinite(X).all() and np.isfinite(target).all()
    gram=X.T@X/len(X)+np.eye(X.shape[1]);rhs=X.T@target/len(X)
    beta=np.linalg.solve(gram,rhs)
    residual=float(np.max(abs(gram@beta-rhs),initial=0))
    np.testing.assert_allclose(gram@beta,rhs,rtol=1e-10,atol=1e-7)
    assert np.isfinite(beta).all()
    return beta,residual


def coefficient(path,X,target,ids,held_ids,outer,block):
    assert not path.exists()
    beta,error=solve(X,target)
    write(path,dict(identity=art(MANIFEST),outer=outer,block=block,columns=VERSIONS,beta=beta.tolist(),
        train_rows=len(ids),held_rows=len(held_ids),train_ids_sha256=ids_hash(ids),held_ids_sha256=ids_hash(held_ids),
        normal_equation_max_abs=error,ridge=1.,intercept=False))
    return beta


def infer():
    start=time.monotonic();m=guard();allocation=claim('cv')
    frame,X,teacher,raw,matches=inputs('cv',m)
    assert set(frame.month)==set(MONTHS)
    prediction=frame[META].copy();prediction['v82']=frame.submitted.to_numpy()
    prediction['v80']=teacher;prediction['block']='';prediction['raw']=np.nan
    coefficients=[]
    for month in MONTHS:
        indices=np.flatnonzero(frame.month.eq(month));local=frame.iloc[indices]
        for block in ['A','B']:
            tr,te=block_masks(local.day,block);train,held=indices[tr],indices[te]
            path=OUT/'cv/models'/f'{month}_{block}.json'
            beta=coefficient(path,X[train],teacher[train]-prediction.v82.to_numpy()[train],
                frame.mvt_id.to_numpy()[train],frame.mvt_id.to_numpy()[held],month,block)
            coefficients.append(art(path))
            prediction.loc[held,'raw']=prediction.v82.to_numpy()[held]+X[held]@beta
            prediction.loc[held,'block']=block
    assert len(coefficients)==4 and np.isfinite(prediction.raw).all() and prediction.block.ne('').all()
    prediction['submitted']=public.constrain('cv',raw,frame.mvt_id.to_numpy(),prediction.raw.to_numpy(),matches)
    prediction=prediction[PRED_COLUMNS];prediction.to_parquet(OUT/'cv/predictions.parquet',index=False)
    fidelity={month:dict(student_RMSE=float(np.sqrt(np.mean((g.raw-g.v80)**2))),
        anchor_RMSE=float(np.sqrt(np.mean((g.v82-g.v80)**2)))) for month,g in prediction.groupby('month')}
    write(OUT/'cv/inference.json',dict(identity=art(MANIFEST),allocation=allocation,coefficients=coefficients,
        fit_receipts=coefficients,prediction=art(OUT/'cv/predictions.parquet'),rows=len(prediction),
        teacher_fidelity=fidelity,all_four_fits_before_truth=True,seconds=time.monotonic()-start,no_taxi_labels_read=True))
    print(dict(inference=art(OUT/'cv/inference.json'),rows=len(prediction)),flush=True)


def inference_guard():
    result=read(OUT/'cv/inference.json');assert result['identity']==art(MANIFEST)
    assert result['all_four_fits_before_truth'] and len(result['coefficients'])==4
    assert result['fit_receipts']==result['coefficients']
    for rec,(month,block) in zip(result['coefficients'],[(month,b) for month in MONTHS for b in ['A','B']]):
        assert rec==art(OUT/'cv/models'/f'{month}_{block}.json')
        value=read(verify(rec));assert value['identity']==art(MANIFEST) and value['columns']==VERSIONS
        assert value['outer']==month and value['block']==block and len(value['beta'])==28
    verify(result['prediction']);verify(result['allocation'])
    return result


def score():
    m=guard();inference=inference_guard();checked=read(OUT/'cv/input_audit.json')
    assert checked['status']=='pass' and checked['identity']==art(MANIFEST)
    assert checked['inference']==art(OUT/'cv/inference.json')
    allocation=claim('score')
    frame=pd.read_parquet(verify(inference['prediction']),columns=PRED_COLUMNS)
    truth=align(load(m['specs']['cv']['truth'],truth=True),frame.mvt_id.to_numpy())
    for key in ['mvt_id','month','day']:np.testing.assert_array_equal(frame[key],truth[key])
    frame['y']=truth.y.to_numpy(float);assert np.isfinite(frame.y).all()
    months={month:{key:float(np.sqrt(np.mean((g[key]-g.y)**2))) for key in ['v82','v80','submitted']} for month,g in frame.groupby('month')}
    cv={key:float(np.mean([value[key] for value in months.values()])) for key in ['v82','v80','submitted']}
    paired={key:seasonal.paired_bootstrap(frame.day,frame.month,(frame[key]-frame.y)**2,(frame.submitted-frame.y)**2) for key in ['v82','v80']}
    faithful=all(value['student_RMSE']<value['anchor_RMSE'] for value in inference['teacher_fidelity'].values())
    gate=faithful and cv['submitted']-cv['v82']<=-.10 and all(g['submitted']<g['v82'] for g in months.values()) and paired['v82']['ci95'][1]<0
    frame.to_parquet(OUT/'cv/scored.parquet',index=False)
    write(OUT/'cv/score.json',dict(identity=art(MANIFEST),inference=art(OUT/'cv/inference.json'),
        input_audit=art(OUT/'cv/input_audit.json'),allocation=allocation,scored=art(OUT/'cv/scored.parquet'),
        months=months,cv=cv,paired=paired,teacher_fidelity=inference['teacher_fidelity'],gate=bool(gate),
        reused_development=True,unseen_month_transfer_unvalidated=True,independent_audit_pending=True))
    print(dict(cv=cv,gate=bool(gate)),flush=True)


def production():
    m=guard();score_result=read(OUT/'cv/score.json');audit=read(OUT/'cv/audit.json')
    assert score_result['identity']==audit['identity']==art(MANIFEST) and score_result['gate'] and audit['gate']
    assert audit['status']=='pass' and audit['score']==art(OUT/'cv/score.json')
    inference_guard();verify(score_result['scored']);verify(score_result['input_audit'])
    allocation=claim('production')
    frame,X,teacher,raw,matches=inputs('old',m);ids=frame.mvt_id.to_numpy()
    beta=coefficient(OUT/'production/model.json',X,teacher-frame.submitted.to_numpy(float),ids,ids,'2026-all','production')
    views={'cv':dict(rows=344419,prediction=score_result['scored'])}
    for view in ['old','final']:
        if view=='final':frame,X,teacher,raw,matches=inputs(view,m)
        ids=frame.mvt_id.to_numpy();values=frame.submitted.to_numpy(float)+X@beta
        submitted=public.constrain(view,raw,ids,values,matches)
        folder=OUT/'production'/view;folder.mkdir()
        output=frame[META].copy();output['v82']=frame.submitted;output['raw']=values;output['submitted']=submitted
        output.to_parquet(folder/'predictions.parquet',index=False)
        template=pq.read_table(verify(m['specs'][view]['template']['artifact']))
        candidate=fill_template(template,ids,submitted)
        assert candidate.schema==template.schema and candidate['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
        np.testing.assert_array_equal(candidate['TAXITIME_SEC_mvt'].to_numpy(),submitted)
        pq.write_table(candidate,folder/'candidate.parquet')
        assert pq.read_table(folder/'candidate.parquet').equals(candidate)
        views[view]=dict(rows=len(ids),prediction=art(folder/'predictions.parquet'),candidate=art(folder/'candidate.parquet'))
    write(OUT/'production/result.json',dict(identity=art(MANIFEST),allocation=allocation,score=art(OUT/'cv/score.json'),
        cv_audit=art(OUT/'cv/audit.json'),model=art(OUT/'production/model.json'),views=views,
        solves=1,no_taxi_labels_read=True,independent_audit_pending=True,no_upload=True,exact_v80_claim=False,
        old_prediction_caveat='In-sample teacher distillation; original-ranking true taxi labels were never used. Official feedback remains unseen.',
        final_transfer_caveat='Blocked same-month CV does not validate new February/June or clock-shifted final inputs.'))
    print(dict(production=art(OUT/'production/result.json'),views={v:r['rows'] for v,r in views.items()}),flush=True)


def preflight():
    assert not MANIFEST.exists()
    rng=np.random.Generator(np.random.PCG64(20261013));X=rng.normal(size=(96,28));y=rng.normal(size=96)
    beta,error=solve(X,y)
    augmented=np.linalg.lstsq(np.vstack([X,np.sqrt(len(X))*np.eye(28)]),np.r_[y,np.zeros(28)],rcond=None)[0]
    np.testing.assert_allclose(beta,augmented,rtol=1e-12,atol=1e-12)
    day=pd.Series(pd.date_range('2025-01-01',periods=31).strftime('%Y-%m-%d'))
    masks={b:block_masks(day,b) for b in ['A','B']}
    assert np.all(masks['A'][1]^masks['B'][1])
    assert np.flatnonzero(masks['A'][0]).min()==16 and np.flatnonzero(masks['B'][0]).max()==13
    raw=pd.DataFrame(dict(mvt_id=[1,2,3,4],month=['2025-01']*4,has_id=[True,False,False,False],nm_gap=[5000.,np.nan,np.nan,np.nan]))
    matches=pd.DataFrame(dict(mvt_id=[2],accepted=[True],recovered_gap=[4000.],donor_mvt_id=[22],donor_month=['2025-03']))
    constrained=public.constrain('cv',raw,raw.mvt_id.to_numpy(),np.array([1000.,4000.,1.5,-5.]),matches)
    np.testing.assert_array_equal(constrained,[1394,394,2,0])
    np.testing.assert_array_equal(public.constrain('cv',raw,raw.mvt_id.to_numpy(),constrained,matches),constrained)
    blocked=False
    try:load({'columns':['mvt_id','y']})
    except AssertionError:blocked=True
    assert blocked
    write(FIXTURE,dict(status='pass',source=art(__file__),design=art(PLAN),synthetic_only=True,
        ridge_lstsq_max_abs=float(abs(beta-augmented).max()),normal_equation_max_abs=error,
        all_days_exactly_one_held_block=True,raw_constraint_order_and_zero_replay=True,
        y_projection_rejected_before_IO=True,no_prediction_arrays_decoded=True))


def metadata():
    assert not MANIFEST.exists()
    a=read(AFFINE/'identity.json');p=read(public.MANIFEST)
    old_teacher,_=uploaded.teacher_contract()
    specs={view:view_spec(view,a,p,read(public.OUT/view/'result.json'),old_teacher) for view in ['cv','old','final']}
    components={str(v):{view:projected(a['components'][str(v)][view],
        [a['components'][str(v)]['id_column'],a['components'][str(v)]['value_column']]) for view in ['old','final']} for v in VERSIONS}
    write(METADATA,dict(status='pass',source=art(__file__),design=art(PLAN),
        specs=specs,components=components,no_prediction_arrays_decoded=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','metadata','freeze','infer','score','production'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
