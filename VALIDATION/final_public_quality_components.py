"""Exact public-model transport and daily quality correction on both views.

Consume audited direct model values and the complete public affine correction
port. Saved full-year calibration is reused. Missing upstream components fail
closed; this module neither trains models nor creates a submission.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import final_public_direct_components as direct
import final_correction_inputs as inputs_cache
import public_daily_taxi_submit as original
from records import write_json

ROOT, RUNS = direct.ROOT, direct.RUNS
OUT = RUNS / 'v3_20261012_final_public_quality_components'
art, read, verify, aligned = direct.art, direct.read, direct.verify, direct.aligned
ARMS = ['airport', 'cat', 'main', 'tabm']


def corrections():
    # Imported only when the complete upstream port is available.
    import final_public_corrections
    return final_public_corrections


def freeze(view):
    dest = OUT / view
    assert not (dest / 'identity.json').exists(), 'Preserve frozen view'
    corrected = corrections().load_stages(view)
    direct_audit = read(direct.OUT / 'independent_audit.json')
    assert direct_audit['status'] == 'pass'
    assert direct_audit['identity'] == art(direct.MANIFEST)
    receipt = read(direct.OUT / f'{view}.json')
    assert receipt['identity'] == art(direct.MANIFEST)
    assert direct_audit['views'][view]['prediction'] == receipt['prediction']
    verify(receipt['prediction'])
    input_audit=read(inputs_cache.OUT / 'audit.json')
    assert input_audit['status'] == 'pass' and input_audit['identity'] == art(inputs_cache.MANIFEST)
    recipe = read(original.MANIFEST)
    # Original calibration and public data bytes are unchanged.
    sources = {Path(__file__), Path(direct.__file__), direct.MANIFEST,
        direct.OUT / 'independent_audit.json', direct.OUT / f'{view}.json', verify(receipt['prediction']),
        inputs_cache.MANIFEST, inputs_cache.OUT / 'audit.json',
        original.MANIFEST, original.OUT / 'result.json', original.OUT / 'parameters.json', original.daily.SOURCE / 'daily_runway.parquet',
        original.daily.base.SOURCE / 'taxi_out_additional_time_2026.csv'}
    for p in [original.daily.SOURCE / 'daily_runway.parquet',
              original.daily.base.SOURCE / 'taxi_out_additional_time_2026.csv']:
        assert recipe['sources'][str(p.relative_to(ROOT))] == art(p)['sha256']
    parameters = read(original.OUT / 'result.json')
    assert art(original.OUT / 'parameters.json') in parameters['artifacts']
    assert read(original.OUT / 'parameters.json')['allowed'] == [f'2025-{m:02d}' for m in range(1,13)]
    for name in ['stages.parquet', 'result.json', 'audit.json']:
        sources.add(corrections().OUT / view / name)
    input_receipt=inputs_cache.OUT / view / 'prepared.json'
    assert read(input_receipt)['identity'] == art(inputs_cache.MANIFEST)
    sources.add(input_receipt)
    for name in ['daily', 'monthly', 'clocks', 'matches']:
        path = inputs_cache.OUT / view / f'{name}.parquet'
        assert art(path) in read(inputs_cache.OUT / view / 'prepared.json')['outputs']
        sources.add(path)
    raw_inputs = (RUNS / 'v3_20261011_public_daily_taxi_features/inputsranking.parquet' if view == 'old'
                  else RUNS / 'v3_20261012_final_daily_main_inference/features/inputsfinal.parquet')
    feature_manifest = raw_inputs.parent / 'manifest.json'
    feature_audit = raw_inputs.parent / 'audit.json'
    assert read(feature_audit)['status'] == 'pass'
    assert read(feature_audit)['manifest'] == art(feature_manifest)
    key = 'ranking' if view == 'old' else 'final'
    assert read(feature_manifest)['records'][key]['inputs'] == art(raw_inputs)
    sources.update([raw_inputs,feature_manifest,feature_audit])
    controls = dict(v76=original.aggregate.V76 / 'predictions.parquet',
        airport=original.OUT / 'predictions.parquet',
        cat=RUNS / 'v3_20261011_catboost_refinements_submit/predictions.parquet',
        main=RUNS / 'v3_20261011_daily_main_refinements_submit/predictions.parquet',
        tabm=RUNS / 'v3_20261012_nn_public_tabm_pool3_submit_stack/predictions.parquet')
    for path in controls.values():
        sources.add(path)
    for module in list(sys.modules.values()):
        filename = getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT/'VALIDATION'):
                sources.add(path)
    dest.mkdir(parents=True,exist_ok=True)
    write_json(dest/'identity.json',dict(view=view,source=art(__file__),
        direct=receipt['prediction'],raw_inputs=art(raw_inputs),
        controls={k:art(p) for k,p in controls.items()},
        sources=[art(p) for p in sorted(sources)],rows=len(corrected),
        calibration='Exact saved full2025 parameters; no fitting',
        quality_gate='Monthly invalid share strictly below0.05',
        no_new_choice=True,no_submission=True))
    print(dict(view=view,frozen=True,sources=len(sources)),flush=True)


def build(view):
    dest=OUT/view
    assert not (dest/'result.json').exists(),'Preserve completed view'
    identity=read(dest/'identity.json')
    assert identity['source']==art(__file__) and identity['view']==view
    for record in identity['sources']:
        verify(record)
    if view=='final':
        assert read(OUT/'old/audit.json')['status']=='pass'
    stages=corrections().load_stages(view)
    ids=stages.mvt_id.to_numpy()
    frame=aligned(identity['direct'],ids)
    raw=aligned(art(inputs_cache.OUT/view/'daily.parquet'),ids)
    monthly=aligned(art(inputs_cache.OUT/view/'monthly.parquet'),ids)
    matches=pd.read_parquet(inputs_cache.OUT/view/'matches.parquet')
    inputs=aligned(identity['raw_inputs'],ids)
    inputs['eligible']=monthly.eligible.to_numpy()
    inputs['month']=inputs.takeoff.dt.strftime('%Y-%m')
    inputs=inputs.drop(columns='takeoff')
    assert 'y' not in inputs
    source=original.daily.public()
    parameters=read(original.OUT/'parameters.json')
    q=inputs[['airport','local_day']].assign(local_month=inputs.local_day.str[:7]).merge(
        original.constraints.monthly.public(2026)[['airport','local_month','invalid']],
        on=['airport','local_month'],how='left',validate='many_to_one')
    quality=q.invalid.lt(.05).to_numpy()
    result=inputs.copy()
    result['quality_eligible']=quality
    result['invalid_share']=q.invalid.to_numpy()
    result['v76_float']=original.selected.main_candidate.constrained(ids,raw,stages.v74.to_numpy(),
        frame.v76_direct.to_numpy(),frame.control.to_numpy(),matches)
    result['v76']=np.rint(np.maximum(result.v76_float.to_numpy(),0)).astype('int32')
    cells=[]
    for arm in ARMS:
        value=original.selected.main_candidate.constrained(ids,raw,stages.refined74.to_numpy(),
            frame[arm+'_direct'].to_numpy(),frame.control.to_numpy(),matches)
        refined=np.rint(np.maximum(value,0))
        _,cell,shift=original.daily.apply(inputs,refined,source,parameters)
        applied=np.where(quality,shift,0.)
        bounded=original.aggregate.constrain(ids,raw,refined+applied,matches)
        pred=np.rint(np.maximum(bounded,0)).astype('int32')
        for name,values in dict(refined_float=value,refined=refined,shift=shift,applied=applied,
                                bounded=bounded,pred=pred).items():
            result[arm+'_'+name]=values
        cells.append(cell.assign(arm=arm))
    count=0
    if view=='old':
        controls={k:aligned(r,ids) for k,r in identity['controls'].items()}
        for out,column in [('v76_float','pred'),('v76','submitted')]:
            np.testing.assert_array_equal(result[out],controls['v76'][column],err_msg=out)
            count+=len(ids)
        for arm in ARMS:
            for key in ['refined_float','refined','shift','applied','bounded','pred']:
                if arm=='airport':
                    column={'shift':'unrestricted_shift','applied':'applied_shift','pred':'submitted'}.get(key,key)
                else:
                    column=('tabm_' if arm=='tabm' else 'candidate_')+key
                np.testing.assert_array_equal(result[arm+'_'+key],controls[arm][column],err_msg=arm+'_'+key)
                count+=len(ids)
            np.testing.assert_array_equal(result.quality_eligible,controls[arm].quality_eligible)
        np.testing.assert_array_equal(result.invalid_share,controls['airport'].invalid_share)
    predcols=['v76']+[a+'_pred' for a in ARMS]
    assert np.isfinite(result[predcols]).all().all() and result[predcols].ge(0).all().all()
    result.to_parquet(dest/'stages.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(dest/'cells.parquet',index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(dest/'stages.parquet'),result,check_exact=True)
    write_json(dest/'result.json',dict(view=view,identity=art(dest/'identity.json'),
        prediction=art(dest/'stages.parquet'),cells=art(dest/'cells.parquet'),
        rows=len(ids),months=inputs.month.value_counts().sort_index().to_dict(),
        old_values_checked=count,old_replay_exact=True,independent_audit_pending=True,
        no_fit=True,no_submission=True))
    print(dict(view=view,rows=len(ids),old_values_checked=count),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','build'])
    parser.add_argument('--view',choices=['old','final'],required=True)
    args=parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage](args.view)
