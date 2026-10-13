"""One explicit portable ensemble with retained public-source corrections.

Freeze after audited affine and direct-CV packets exist, then infer each view
only under a root allocation. No model or calibration fitting, no coefficient
solve, no upload. Daily t5 posterior checkpoints are saved per airport/month
before later arithmetic, so interrupted postprocessing never repeats them.

  python portable_public_source_20261013.py freeze --base-result ... --base-audit ...
  python portable_public_source_20261013.py infer --view cv|old|final
  python portable_public_source_20261013.py score

Every public correction uses the actual incoming new prediction. The retained
joint/robust increments retain their original v70 prior, explicitly transported
as fixed additive terms; they are not new posterior inferences from this base.
"""
import argparse
import gc
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy
from threadpoolctl import threadpool_limits

import final_public_corrections as correction
import final_public_direct_components as production_direct
import final_joint_public_inference as production_joint
import public_daily_taxi_submit as production_quality
import portable_direct_cv_20261013 as cv_direct
import joint_public_other_runway_cv as cv_robust
import tuning_seasonal as seasonal
from records import write_json
from submit import fill_template

ROOT = correction.ROOT
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261013_portable_public_source'
MANIFEST = OUT / 'identity.json'
DESIGN_PATH = ROOT / 'VALIDATION/reports/portable_public_source_design_20261013.json'
AUDITOR = ROOT / 'VALIDATION/tools/audit_portable_public_source_20261013.py'
WEIGHT = 0.4032872515123992
VIEWS = ['cv', 'old', 'final']
EXPECTED_ROWS = dict(cv=344419, old=344841, final=670790)
heavy, daily, monthly = correction.heavy, production_quality.daily, correction.constraints.monthly
art, read, verify, R = correction.art, correction.read, correction.verify, correction.serialized


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__)


def freeze(base_result, base_audit):
    assert not MANIFEST.exists(), 'Preserve the one predeclared source-correction design'
    base_result, base_audit = Path(base_result).resolve(), Path(base_audit).resolve()
    result, audit = read(base_result), read(base_audit)
    assert audit['status'] == 'pass' and audit['result'] == art(base_result)
    assert audit['identity'] == result['identity']
    assert audit['predictions'] == result['predictions']
    assert set(result['predictions']) == set(VIEWS)
    assert read(correction.V73 / 'geometry.json')['weights']['72'] == WEIGHT
    design = read(DESIGN_PATH)
    assert design['name'] == 'portable_public_source_v1'
    paths = {Path(__file__), AUDITOR, DESIGN_PATH, base_result, base_audit, verify(result['identity']),
             correction.V73 / 'geometry.json', ROOT / 'VALIDATION/submit.py',
             ROOT/'data/ranking.parquet', ROOT/'data/final_ranking.parquet'}
    for month in ['2025-01','2025-07']:
        raw_paths=list((ROOT/'data').glob(f'training_{month}-01_*.parquet'))
        assert len(raw_paths)==1,(month,raw_paths)
        paths.add(raw_paths[0])
    paths.update(verify(r) for r in result['predictions'].values())
    specs = {view:dict(base=result['predictions'][view]) for view in VIEWS}
    cv_result, cv_audit = read(cv_direct.OUT/'result.json'), read(cv_direct.OUT/'audit.json')
    assert cv_audit['status'] == 'pass' and cv_audit['result'] == art(cv_direct.OUT/'result.json')
    assert cv_result['identity'] == cv_audit['identity'] == art(cv_direct.MANIFEST)
    assert all(cv_result[k] for k in ['exact_selected_v77_all_stages','exact_selected_v78','exact_three_tabm_v78'])
    paths.update([cv_direct.MANIFEST, cv_direct.OUT/'result.json', cv_direct.OUT/'audit.json', verify(cv_result['prediction'])])
    cv_files = dict(raw=daily.base.OUT/'rows2025.parquet', monthly=monthly.OUT/'rows2025.parquet',
                    daily_inputs=daily.CACHE, matches=daily.constraints.exclusion.OUT/'matches2025.parquet',
                    daily_parameters=heavy.old.OUT/'screen.json', monthly_parameters=correction.V74/'parameters.json',
                    quality_parameters=daily.OUT/'parameters.json')
    paths.update(cv_files.values())
    specs['cv'].update({k:art(p) for k,p in cv_files.items()})
    specs['cv']['direct'] = cv_result['prediction']
    tail_folder=RUNS/'v3_20261011_daily_heavy_tail_stack'
    tail_identity=read(tail_folder/'identity.json')
    for key in ['daily_parameters','monthly_parameters']:
        assert specs['cv'][key] in tail_identity['sources'],key
    for folder in [tail_folder,daily.OUT]:
        report,checked=read(folder/'result.json'),read(folder/'audit.json')
        assert checked['status']=='pass' and checked['result']==art(folder/'result.json')
        assert checked['identity']==report['identity']==art(folder/'identity.json')
        paths.update(folder/name for name in ['identity.json','result.json','audit.json'])
    assert specs['cv']['quality_parameters'] in read(daily.OUT/'result.json')['artifacts']
    for key, folder in [('joint',cv_robust.parent.OUT), ('robust',cv_robust.OUT)]:
        report, checked = read(folder/'result.json'), read(folder/'audit.json')
        assert checked['status'] == 'pass' and checked['result'] == art(folder/'result.json')
        prediction = report['scored']
        paths.update([folder/'result.json', folder/'audit.json', folder/'identity.json', verify(prediction)])
        specs['cv'][key] = prediction
    input_audit = read(correction.inputs.OUT/'audit.json')
    assert input_audit['status'] == 'pass' and input_audit['identity'] == art(correction.inputs.MANIFEST)
    direct_audit = read(production_direct.OUT/'independent_audit.json')
    assert direct_audit['status'] == 'pass' and direct_audit['identity'] == art(production_direct.MANIFEST)
    paths.update([correction.inputs.MANIFEST, correction.inputs.OUT/'audit.json',
                  production_direct.MANIFEST, production_direct.OUT/'independent_audit.json'])
    for view in ['old','final']:
        spec = specs[view]
        prepared = read(correction.inputs.OUT/view/'prepared.json')
        assert prepared['identity'] == art(correction.inputs.MANIFEST)
        paths.add(correction.inputs.OUT/view/'prepared.json')
        for key, name in [('raw','daily'),('monthly','monthly'),('matches','matches')]:
            path = correction.inputs.OUT/view/f'{name}.parquet'
            assert art(path) in prepared['outputs']
            paths.add(path); spec[key] = art(path)
        receipt = read(production_direct.OUT/f'{view}.json')
        assert receipt['identity'] == art(production_direct.MANIFEST)
        assert direct_audit['views'][view]['prediction'] == receipt['prediction']
        paths.update([production_direct.OUT/f'{view}.json', verify(receipt['prediction'])])
        spec['direct'] = receipt['prediction']
        jr, ja = [read(production_joint.OUT/view/name) for name in ['result.json','audit.json']]
        assert ja['status'] == 'pass' and ja['result'] == art(production_joint.OUT/view/'result.json')
        assert ja['identity'] == jr['identity'] == art(production_joint.MANIFEST)
        assert ja['prediction'] == jr['prediction']
        paths.update([production_joint.MANIFEST, production_joint.OUT/view/'result.json',
                      production_joint.OUT/view/'audit.json', verify(jr['prediction'])])
        spec['increments'] = jr['prediction']
        path = production_joint.FEATURES[view]
        manifest, checked = read(path.parent/'manifest.json'), read(path.parent/'audit.json')
        assert checked['status'] == 'pass' and checked['manifest'] == art(path.parent/'manifest.json')
        assert manifest['records']['ranking' if view=='old' else 'final']['inputs'] == art(path)
        paths.update([path, path.parent/'manifest.json', path.parent/'audit.json'])
        spec['daily_inputs'] = art(path)
        spec['daily_parameters'] = art(correction.V72/'parameters.json')
        spec['monthly_parameters'] = art(correction.V74/'parameters.json')
        spec['quality_parameters'] = art(production_quality.OUT/'parameters.json')
        for key in ['daily_parameters','monthly_parameters','quality_parameters']:
            paths.add(verify(spec[key]))
        spec['template'] = art(ROOT/'data'/('submitting.parquet' if view=='old' else 'final_submitting.parquet'))
        paths.add(verify(spec['template']))
    # Original selected calibration manifests authenticate the reused full2025
    # parameter bytes; fold receipts authenticate the original CV parameters.
    for folder in [correction.V72, correction.V74]:
        manifest = read(folder/'manifest.json')
        record = art(folder/'parameters.json')
        assert manifest['sources'][record['path']] == record['sha256']
        paths.add(folder/'manifest.json')
    qp = read(production_quality.OUT/'parameters.json')
    assert qp['allowed'] == [f'2025-{m:02d}' for m in range(1,13)]
    assert art(production_quality.OUT/'parameters.json') in read(production_quality.OUT/'result.json')['artifacts']
    paths.update([production_quality.MANIFEST, production_quality.OUT/'result.json'])
    paths.update([daily.SOURCE/'daily_runway.parquet',
        *[daily.base.SOURCE/f'{stem}_{year}.csv' for year in [2025,2026]
          for stem in ['all_pre_departure_delays','taxi_out_additional_time']]])
    for module in list(sys.modules.values()):
        filename = getattr(module,'__file__',None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT/'VALIDATION'):
                paths.add(path)
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(MANIFEST, dict(design=design, design_source=art(DESIGN_PATH), source=art(__file__),
        base_result=art(base_result), base_audit=art(base_audit), specs=specs,
        sources=[art(p) for p in sorted(paths)], runtime=runtime(), daily_weight=WEIGHT,
        posterior_checkpoint='One file and receipt per airport/month; no completed posterior repeated',
        joint_provenance='Retained v70 prior, fixed additive transport, no new joint inference',
        no_fit=True,no_solve=True,no_upload=True,selected_for_final=False))
    print(dict(frozen=True,identity=art(MANIFEST),sources=len(paths)),flush=True)


def guard():
    identity = read(MANIFEST)
    assert identity['source'] == art(__file__) and identity['runtime'] == runtime()
    assert identity['design_source'] == art(DESIGN_PATH) and identity['design'] == read(DESIGN_PATH)
    assert identity['daily_weight'] == WEIGHT
    for record in identity['sources']:
        verify(record)
    return identity


def aligned(record, ids, full=False):
    frame = pd.read_parquet(verify(record))
    assert frame.mvt_id.is_unique
    if full:
        assert len(frame) == len(ids) and set(frame.mvt_id) == set(ids)
    assert pd.Index(ids).isin(frame.mvt_id).all()
    return frame.set_index('mvt_id').loc[ids].reset_index()


def load(view, identity):
    spec = identity['specs'][view]
    base = pd.read_parquet(verify(spec['base']))
    ids = base.mvt_id.to_numpy()
    assert len(ids) == len(set(ids)) == EXPECTED_ROWS[view]
    np.testing.assert_array_equal(base.submitted, R(base.submitted))
    if view != 'cv':
        target = pd.read_parquet(verify(spec['template']),columns=['MVT_ID_mvt']).MVT_ID_mvt
        np.testing.assert_array_equal(ids,target)
        assert 'y' not in base
    raw = aligned(spec['raw'],ids)
    mrows = aligned(spec['monthly'],ids)
    direct = aligned(spec['direct'],ids,full=True)
    matches = pd.read_parquet(verify(spec['matches']))
    expected_months={'2025-01','2025-07'} if view=='cv' else (
        {'2026-01','2026-07'} if view=='old' else {'2026-01','2026-02','2026-06','2026-07'})
    assert set(raw.month)==expected_months
    if view == 'cv':
        for frame in [raw,mrows,direct]:
            np.testing.assert_array_equal(frame.y,base.y)
        for key in ['month','day']:
            np.testing.assert_array_equal(raw[key],base[key])
            np.testing.assert_array_equal(direct[key],base[key])
        quality_inputs = daily.rows().drop(columns='y').set_index('mvt_id').loc[ids].reset_index()
        joint = aligned(spec['joint'],ids,full=True)
        robust = aligned(spec['robust'],ids,full=True)
        for frame in [joint,robust]:
            for key in ['y','month','day']:
                np.testing.assert_array_equal(frame[key],base[key])
        increments = pd.DataFrame(dict(mvt_id=ids,increment=joint.increment,
            robust_delta=robust.delta,robust_active=robust.active))
    else:
        assert 'y' not in raw and 'y' not in mrows and 'y' not in direct
        quality_inputs = aligned(spec['daily_inputs'],ids,full=True)
        quality_inputs['eligible'] = mrows.eligible.to_numpy()
        quality_inputs['month'] = quality_inputs.takeoff.dt.strftime('%Y-%m')
        quality_inputs = quality_inputs.drop(columns='takeoff')
        increments = aligned(spec['increments'],ids,full=True)
    raw,mrows = [f.drop(columns=['y'],errors='ignore') for f in [raw,mrows]]
    assert 'y' not in quality_inputs
    np.testing.assert_array_equal(quality_inputs.month,raw.month)
    np.testing.assert_array_equal(quality_inputs.eligible,mrows.eligible)
    assert not increments.loc[~increments.robust_active,'robust_delta'].any()
    return base, raw, mrows, direct, quality_inputs, matches, increments


def constrain(view,raw,ids,values,matches):
    if view == 'cv':
        return R(daily.constrain(raw,ids,raw.month,values,matches))
    return R(correction.constrain(ids,raw,values,matches))


def daily_posterior(view,identity,base,raw):
    folder=OUT/view/'posterior';folder.mkdir(parents=True,exist_ok=True)
    spec=identity['specs'][view]
    params=read(verify(spec['daily_parameters']))['parameters']
    public=daily.base.public(2025 if view=='cv' else 2026).set_index(['airport','day'])
    B=base.submitted.to_numpy(float)
    posterior=np.full(len(base),np.nan);parts=[]
    for (month,airport),indices in raw.groupby(['month','airport'],sort=True).indices.items():
        name=f'{month}_{airport}';receipt_path=folder/f'{name}.json'
        rows=raw.iloc[indices].reset_index(drop=True);ids=rows.mvt_id.to_numpy()
        if receipt_path.exists():
            receipt=read(receipt_path)
            assert receipt['identity']==art(MANIFEST) and receipt['view']==view
            assert receipt['base']==spec['base'] and receipt['month']==month and receipt['airport']==airport
            cached=pd.read_parquet(verify(receipt['prediction']))
            verify(receipt['days'])
            np.testing.assert_array_equal(cached.mvt_id,ids)
            np.testing.assert_array_equal(cached.B,B[indices])
            local=cached.raw_posterior.to_numpy(float)
        else:
            prediction_path,days_path=folder/f'{name}.parquet',folder/f'{name}_days.parquet'
            assert not prediction_path.exists() and not days_path.exists(),'Unrecorded posterior; inspect instead of repeating'
            local=np.full(len(rows),np.nan);logs=[];started=time.monotonic()
            chosen=params[month][airport] if view=='cv' else params[airport]
            for day,at in rows.groupby('day',sort=True).indices.items():
                group=rows.iloc[at];p=public.loc[(airport,day)]
                assert np.isfinite([p.total,p.n_public]).all()
                local[at],info=heavy.condition(group.gap.to_numpy(float),B[indices[at]],
                    group.has_id.to_numpy(bool),group.nm_gap.to_numpy(float),airport,chosen,
                    float(p.total),float(p.n_public-len(group)))
                logs.append(dict(month=month,airport=airport,day=day,rows=len(group),
                                 public_total=float(p.total),public_count=float(p.n_public),**info))
            assert np.isfinite(local).all()
            pd.DataFrame(dict(mvt_id=ids,B=B[indices],raw_posterior=local)).to_parquet(prediction_path,index=False)
            pd.DataFrame(logs).to_parquet(days_path,index=False)
            write_json(receipt_path,dict(identity=art(MANIFEST),view=view,base=spec['base'],month=month,
                airport=airport,rows=len(rows),prediction=art(prediction_path),days=art(days_path),
                seconds=time.monotonic()-started,saved_before_downstream=True))
        posterior[indices]=local;parts.append(art(receipt_path))
    assert np.isfinite(posterior).all()
    return posterior,parts


def infer(view):
    identity=guard();spec=identity['specs'][view]
    folder=OUT/view;folder.mkdir(parents=True,exist_ok=True)
    allocation=read(folder/'allocation.json')
    assert allocation['authority']=='/root' and allocation['identity']==art(MANIFEST)
    assert allocation['view']==view and allocation['state']=='released_for_portable_public_source'
    assert not (folder/'result.json').exists(),'Preserve completed view'
    if view!='cv':
        previous=read(OUT/'cv/audit.json')
        assert previous['status']=='pass' and previous['identity']==art(MANIFEST)
        assert previous['result']==art(OUT/'cv/result.json')
    if view=='final':
        previous=read(OUT/'old/audit.json')
        assert previous['status']=='pass' and previous['identity']==art(MANIFEST)
        assert previous['result']==art(OUT/'old/result.json')
    base,raw,mrows,direct,quality_inputs,matches,increments=load(view,identity)
    ids=base.mvt_id.to_numpy();B=base.submitted.to_numpy(float)
    posterior,receipts=daily_posterior(view,identity,base,raw)
    T=R(posterior);D=R(B+WEIGHT*(T.astype(float)-B))
    monthly_parameters=read(verify(spec['monthly_parameters']))
    quality_parameters=read(verify(spec['quality_parameters']))
    monthly_source=monthly.public(2025 if view=='cv' else 2026)
    taxi_source=daily.public()
    half=np.full(len(ids),np.nan);monthly_cells=[]
    for month,at in raw.groupby('month',sort=True).indices.items():
        params=monthly_parameters['folds'][month] if view=='cv' else monthly_parameters['full_year']
        half[at],cells=monthly.apply(mrows.iloc[at].reset_index(drop=True),D[at].astype(float),monthly_source,params)
        monthly_cells.append(cells.assign(fold=month))
    full=np.maximum(0,2*half-D.astype(float))
    M=constrain(view,raw,ids,full,matches)
    N_raw=M.astype(float)+direct.tabm_direct.to_numpy(float)-direct.control.to_numpy(float)
    N=constrain(view,raw,ids,N_raw,matches)
    quality=quality_inputs[['airport','local_day']].assign(local_month=quality_inputs.local_day.str[:7]).merge(
        monthly_source[['airport','local_month','invalid']],on=['airport','local_month'],how='left',validate='many_to_one')
    quality_gate=quality.invalid.lt(.05).to_numpy()
    shift=np.full(len(ids),np.nan);daily_cells=[]
    for month,at in raw.groupby('month',sort=True).indices.items():
        params=quality_parameters[month] if view=='cv' else quality_parameters
        _,cells,shift[at]=daily.apply(quality_inputs.iloc[at].reset_index(drop=True),N[at].astype(float),taxi_source,params)
        daily_cells.append(cells.assign(fold=month))
    applied=np.where(quality_gate,shift,0.)
    Q=constrain(view,raw,ids,N.astype(float)+applied,matches)
    J=constrain(view,raw,ids,Q.astype(float)+.5*increments.increment.to_numpy(float),matches)
    F=constrain(view,raw,ids,J.astype(float)+.5*increments.robust_delta.to_numpy(float),matches)
    np.testing.assert_array_equal(F[~increments.robust_active],J[~increments.robust_active])
    frame=raw[['mvt_id','month','day','airport']].copy()
    for key,value in dict(B=B,raw_posterior=posterior,T=T,D=D,half_monthly=half,full_monthly=full,M=M,
        direct_control=direct.control.to_numpy(float),tabm_direct=direct.tabm_direct.to_numpy(float),
        N_raw=N_raw,N=N,quality_eligible=quality_gate,invalid_share=quality.invalid.to_numpy(),
        daily_shift=shift,applied_shift=applied,Q=Q,joint_increment=increments.increment.to_numpy(float),
        J=J,robust_delta=increments.robust_delta.to_numpy(float),robust_active=increments.robust_active.to_numpy(),
        submitted=F,direct_reserve=R(direct.tabm_direct)).items():
        frame[key]=value
    for key in ['B','T','D','M','N','Q','J','submitted','direct_reserve']:
        np.testing.assert_array_equal(frame[key],R(frame[key]))
    if view=='cv':
        frame['y']=base.y.to_numpy();frame['v80']=base.v80.to_numpy()
    else:
        template=pq.read_table(verify(spec['template']));candidate=fill_template(template,ids,F)
        assert candidate.schema==template.schema and candidate['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
        for name in template.column_names:
            if name!='TAXITIME_SEC_mvt':assert candidate[name].equals(template[name])
        pq.write_table(candidate,folder/'candidate.parquet')
        assert pq.read_table(folder/'candidate.parquet').equals(candidate)
    assert np.isfinite(frame.drop(columns=['invalid_share']).select_dtypes(include='number')).all().all()
    frame.to_parquet(folder/'stages.parquet',index=False)
    pd.concat(monthly_cells,ignore_index=True).to_parquet(folder/'monthly_cells.parquet',index=False)
    pd.concat(daily_cells,ignore_index=True).to_parquet(folder/'daily_cells.parquet',index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(folder/'stages.parquet'),frame,check_exact=True)
    result=dict(identity=art(MANIFEST),view=view,allocation=art(folder/'allocation.json'),rows=len(frame),
        prediction=art(folder/'stages.parquet'),monthly_cells=art(folder/'monthly_cells.parquet'),
        daily_cells=art(folder/'daily_cells.parquet'),posterior_receipts=receipts,
        independent_audit_pending=True,no_fit=True,no_solve=True,no_upload=True,selected_for_final=False)
    if view!='cv':result['candidate']=art(folder/'candidate.parquet')
    write_json(folder/'result.json',result)
    print(dict(view=view,rows=len(frame),posterior_parts=len(receipts),result=art(folder/'result.json')),flush=True)
    gc.collect()


def score():
    guard();folder=OUT/'cv';result=read(folder/'result.json')
    assert result['identity']==art(MANIFEST)
    assert not (OUT/'score.json').exists(),'One complete score; no tuning'
    frame=pd.read_parquet(verify(result['prediction']))
    columns=['B','direct_reserve','v80','submitted']
    months={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m,g in frame.groupby('month')}
    paired={c:seasonal.paired_bootstrap(frame.day,frame.month,(frame[c]-frame.y)**2,
        (frame.submitted-frame.y)**2) for c in columns[:-1]}
    write_json(OUT/'score.json',dict(identity=art(MANIFEST),result=art(folder/'result.json'),
        months=months,cv={c:float(np.mean([v[c] for v in months.values()])) for c in columns},paired=paired,
        reused_development=True,selected_for_final=False,independent_audit_pending=True,no_upload=True))
    print(read(OUT/'score.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','infer','score'])
    parser.add_argument('--view',choices=VIEWS)
    parser.add_argument('--base-result',type=Path);parser.add_argument('--base-audit',type=Path)
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.stage=='freeze':
            assert args.base_result and args.base_audit
            freeze(args.base_result,args.base_audit)
        elif args.stage=='infer':
            assert args.view
            infer(args.view)
        else:score()
