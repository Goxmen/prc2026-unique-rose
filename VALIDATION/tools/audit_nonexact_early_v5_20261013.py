"""Saved-only negative recovery audit; no fit or model inference."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'VALIDATION/runs/v3_20261012_final_missing_early_reconstruct'
OUT=RUN/'v5'


def read(p):return json.loads(p.read_text())


def art(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())


def verify(a):
    p=ROOT/a['path'];assert art(p)==a,p
    return p


def main():
    target=OUT/'nonexact_independent_audit.json';assert not target.exists()
    identity=read(RUN/'identity.json');trained=read(OUT/'trained.json')
    result=read(OUT/'result.json');old=read(OUT/'old.json')
    worker=read(RUN/'completion_caps_20261013/execution.json')
    plan=read(verify(worker['plan']))
    for a in plan['sources']:verify(a)
    for a in identity['sources']:verify(a)
    assert trained['identity']==result['identity']==art(RUN/'identity.json')
    assert trained['runtime']==identity['runtime']==worker['runtime']
    verify(trained['model']);verify(trained['preflight'])
    assert trained['rounds']==800 and trained['saved_before_inference'] and not trained['historical_source_exact']
    assert old==result['views']['old'] and result['state']=='closed_nonexact_reconstruction'
    assert result['version']==5 and result['no_final_inference'] and result['no_sweep'] and result['no_upload']
    predictions=pd.read_parquet(verify(old['prediction']))
    manifest=read(ROOT/'data/derived/scored_submission_cache_20261011/manifest.json')['5']
    reference=ROOT/manifest['file'];assert art(reference)['sha256']==manifest['sha256']
    truth=pd.read_parquet(reference)
    template=pd.read_parquet(ROOT/'data/submitting.parquet')
    assert len(predictions)==predictions.mvt_id.nunique()==344841
    np.testing.assert_array_equal(predictions.mvt_id,template.MVT_ID_mvt)
    np.testing.assert_array_equal(truth.MVT_ID_mvt,template.MVT_ID_mvt)
    assert np.isfinite(predictions[['pred','submitted']]).all().all()
    assert (predictions[['pred','submitted']]>=0).all().all()
    np.testing.assert_array_equal(predictions.submitted,np.rint(predictions.pred))
    diff=predictions.submitted.to_numpy(float)-truth.TAXITIME_SEC_mvt.to_numpy(float)
    actual=dict(changed_rows=int(np.count_nonzero(diff)),rms_difference=float(np.sqrt(np.mean(diff**2))),
                maximum_absolute_difference=float(abs(diff).max()))
    for k,v in actual.items():assert old[k]==v,(k,old[k],v)
    assert actual['changed_rows']==160 and actual['maximum_absolute_difference']==2
    assert not old['exact_original_integer_replay']
    assert predictions.month.value_counts().sort_index().to_dict()==old['months']=={'2026-01':152719,'2026-07':192122}
    assert worker['state']=='closed_nonexact_stop_batch' and worker['failed_version']==5
    assert worker['result']==art(OUT/'result.json') and worker['completed_versions']==[]
    assert [s['stage'] for s in worker['steps']]==['prepare','fit']
    for s in worker['steps']:verify(s['evidence']);assert s['version']==5
    assert all(not (RUN/f'v{v}').exists() for v in [6,7,9])
    assert not (OUT/'final.parquet').exists() and not (OUT/'final.json').exists()
    report=dict(status='pass_closed_nonexact',source=art(Path(__file__)),identity=art(RUN/'identity.json'),
        result=art(OUT/'result.json'),trained=art(OUT/'trained.json'),reference=art(reference),
        prediction=old['prediction'],worker=art(RUN/'completion_caps_20261013/execution.json'),rows=344841,
        **actual,all_source_runtime_and_artifact_chains_verified=True,
        existing_model_reload_receipt_verified=True,new_model_inferences=0,fits=0,
        decision='Retain v5 as failed reconstruction; no final vector, no v6/v7/v9 expansion, no substitution.')
    target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(audit=art(target),**actual)))


if __name__=='__main__':main()
