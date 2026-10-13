"""Additive numerical-domain recovery for an unbracketed FINAL exponential tilt.

The unchanged original tilt runs first. Its exact weights are returned whenever
the unchanged support mean assertion passes. Only a previously failing interior
and unbracketed root is extended, with the same55bisections and mean assertion.
The37completed groups remain immutable; only three unfinished groups are run.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import portable_public_source_20261013 as producer

ROOT=producer.ROOT
OUT=producer.OUT
FOLDER=OUT/'final/tilt_recovery'
MANIFEST=FOLDER/'identity.json'
AUDITOR=ROOT/'VALIDATION/tools/audit_portable_public_tilt_recovery_20261013.py'
DIAGNOSIS=OUT/'final/support_failure_diagnosis/result.json'
old=producer.heavy.old
ORIGINAL_TILT=old.tilt
ORIGINAL_SUPPORT=old.support
ORIGINAL_POSTERIOR=producer.daily_posterior
ORIGINAL_WRITE=producer.write_json
art,read,verify=producer.art,producer.read,producer.verify
PENDING=[('2026-07','LIRF'),('2026-07','LSZH'),('2026-07','LTFM')]


def mean_check(values,weights,target):
    np.testing.assert_allclose(values@weights,target,atol=1e-5,rtol=1e-8)


def recovered_tilt(values,weights,target,log=None):
    original=ORIGINAL_TILT(values,weights,target)
    try:
        mean_check(values,original,target)
    except AssertionError as failure:
        original_failure=str(failure)
    else:
        return original
    assert np.isfinite(values).all() and np.isfinite(weights).all() and np.isfinite(target)
    assert (weights>=0).all() and weights.sum()>0 and values.min()<target<values.max()
    z=(values-target)/max(np.ptp(values),1)
    lw=np.log(np.maximum(weights,1e-300))
    def at(parameter):
        result=np.exp(lw+parameter*z-logsumexp(lw+parameter*z))
        return result,float(result@z)
    lower,upper=-128.,128.
    lo_mean,hi_mean=at(lower)[1],at(upper)[1]
    assert lo_mean>0 or hi_mean<0,'Only original unbracketed-root failures are recoverable'
    bounds=[dict(lower=lower,upper=upper,lower_mean_z=lo_mean,upper_mean_z=hi_mean)]
    for _ in range(32):
        if lo_mean<=0<=hi_mean:break
        if lo_mean>0:lower*=2
        if hi_mean<0:upper*=2
        lo_mean,hi_mean=at(lower)[1],at(upper)[1]
        assert np.isfinite([lower,upper,lo_mean,hi_mean]).all()
        bounds.append(dict(lower=lower,upper=upper,lower_mean_z=lo_mean,upper_mean_z=hi_mean))
    else:raise AssertionError('Could not bracket finite interior tilt root')
    left,right=lower,upper
    for _ in range(55):
        mid=(left+right)/2;result,mean_z=at(mid)
        if mean_z>0:right=mid
        else:left=mid
    mean_check(values,result,target)
    event=dict(values=values.tolist(),base_weights=weights.tolist(),target=float(target),
        original_weights=original.tolist(),original_mean=float(values@original),original_failure=original_failure,
        bracket=bounds,root_parameter=mid,weights=result.tolist(),mean=float(values@result),
        iterations=55,unchanged_atol=1e-5,unchanged_rtol=1e-8)
    if log is not None:log(event)
    return result


def fixtures():
    unchanged=[]
    for values,weights,target in [(np.array([0.,1.,2.]),np.array([.2,.3,.5]),x) for x in [0.,.25,1.,1.75,2.]]:
        expected=ORIGINAL_TILT(values,weights,target);mean_check(values,expected,target)
        events=[];actual=recovered_tilt(values,weights,target,events.append)
        assert not events and actual.tobytes()==expected.tobytes()
        unchanged.append(dict(target=target,weights_identical=True))
    record=read(DIAGNOSIS);support=record['support'];events=[]
    values=np.array(support['values']);weights=np.array(support['base_weights']);target=record['row']['prior']
    fixed=recovered_tilt(values,weights,target,events.append)
    assert len(events)==1
    np.testing.assert_allclose(fixed,record['fixed']['weights'],atol=1e-12,rtol=0)
    return dict(unchanged_success_cases=unchanged,diagnosed_failure_extended=True,
                original_root_outside_bracket=True,mean=float(values@fixed),event=events[0],
                no_posterior_inference=True,no_tolerance_change=True)


def freeze():
    assert not FOLDER.exists() and not (OUT/'final/result.json').exists()
    identity=producer.guard();assert art(producer.MANIFEST)['sha256']=='b186449ec102d81434c3b0e0ef9483c44e18146a855214e42c6a9c5313c7d660'
    receipts=sorted((OUT/'final/posterior').glob('*.json'));assert len(receipts)==37
    completed=[];paths={Path(__file__),AUDITOR,DIAGNOSIS,Path(producer.__file__),Path(old.__file__),producer.MANIFEST,
                       OUT/'final/completion/execution.json',OUT/'final/completion/01.log',OUT/'final/allocation.json'}
    for view in ['cv','old']:
        audit=read(OUT/view/'audit.json');assert audit['status']=='pass' and audit['result']==art(OUT/view/'result.json')
        paths.update([OUT/view/'audit.json',OUT/view/'result.json'])
    for path in receipts:
        receipt=read(path);assert receipt['identity']==art(producer.MANIFEST) and receipt['view']=='final'
        assert (receipt['month'],receipt['airport']) not in PENDING
        files=[path,verify(receipt['prediction']),verify(receipt['days'])]
        paths.update(files);completed.append(dict(month=receipt['month'],airport=receipt['airport'],files=[art(p) for p in files]))
    assert read(OUT/'final/completion/execution.json')['status']=='failed_preserved_stop'
    checks=fixtures();FOLDER.mkdir()
    for source,name in [(Path(producer.__file__),'original_producer.py'),(Path(old.__file__),'original_support.py')]:
        target=FOLDER/name;target.write_bytes(source.read_bytes());paths.add(target)
    ORIGINAL_WRITE(FOLDER/'fixtures.json',checks);paths.add(FOLDER/'fixtures.json')
    value=dict(source=art(__file__),auditor=art(AUDITOR),original_identity=art(producer.MANIFEST),runtime=producer.runtime(),
        sources=[art(p) for p in sorted(paths)],retained_completed_groups=completed,pending_groups=PENDING,
        mechanism='Original tilt unchanged on exact original assertion success; monotone bracketing extension only on previously failing interior root',
        same_iterations=55,same_atol=1e-5,same_rtol=1e-8,no_original_source_mutation=True,no_old_cv_replay=True,
        no_completed_group_repeat=True,no_new_calibration=True,no_upload=True)
    ORIGINAL_WRITE(MANIFEST,value)
    print(json.dumps(dict(status='frozen',identity=art(MANIFEST),preserved_groups=37,pending=PENDING)),flush=True)


def infer():
    identity=read(MANIFEST)
    assert identity['source']==art(__file__) and identity['runtime']==producer.runtime()
    for record in identity['sources']:verify(record)
    allocation=read(FOLDER/'allocation.json')
    assert allocation['authority']=='/root' and allocation['recovery_identity']==art(MANIFEST)
    assert allocation['state']=='released_for_final_tilt_recovery'
    assert not (FOLDER/'execution.json').exists() and not (OUT/'final/result.json').exists()
    queue=deque();current={};events=[];started=time.time()
    execution=dict(status='running',source=art(__file__),identity=art(MANIFEST),allocation=art(FOLDER/'allocation.json'),
                   started_unix=started,pid=os.getpid())
    ORIGINAL_WRITE(FOLDER/'execution.json',execution)
    def contextual_posterior(view,parent_identity,base,raw):
        assert view=='final'
        prior=base.submitted.to_numpy(float)
        for (month,airport),indices in raw.groupby(['month','airport'],sort=True).indices.items():
            if (month,airport) not in PENDING:continue
            rows=raw.iloc[indices].reset_index(drop=True)
            for day,at in rows.groupby('day',sort=True).indices.items():
                for k in at:
                    row=rows.iloc[k]
                    if row.gap>3600:
                        queue.append(dict(mvt_id=float(row.mvt_id),month=month,airport=airport,day=day,
                            gap=float(row.gap),prior=float(prior[indices[k]]),has_id=bool(row.has_id),
                            nm_gap=float(row.nm_gap) if np.isfinite(row.nm_gap) else None))
        result=ORIGINAL_POSTERIOR(view,parent_identity,base,raw)
        assert not queue,'Every remaining long-row support call must be matched'
        return result
    def support(d,p,nm,nm_gap,airport,parameters):
        assert queue;row=queue.popleft()
        assert row['airport']==airport and row['gap']==d and row['prior']==p and row['has_id']==nm
        assert row['nm_gap']==(float(nm_gap) if np.isfinite(nm_gap) else None)
        current.clear();current.update(row)
        return ORIGINAL_SUPPORT(d,p,nm,nm_gap,airport,parameters)
    def event(record):
        record=dict(row=current.copy(),**record);events.append(record)
        with (FOLDER/'events.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
    def write(path,value):
        path=Path(path)
        if path.parent==OUT/'final/posterior' or path==OUT/'final/result.json':
            value=dict(value,numerical_recovery=art(MANIFEST))
        ORIGINAL_WRITE(path,value)
    old.tilt=lambda values,weights,target:recovered_tilt(values,weights,target,event)
    old.support=support;producer.daily_posterior=contextual_posterior;producer.write_json=write
    try:
        producer.infer('final')
        assert events
        for group in identity['retained_completed_groups']:
            for record in group['files']:verify(record)
        receipt=dict(status='complete',identity=art(MANIFEST),result=art(OUT/'final/result.json'),
            events=art(FOLDER/'events.jsonl'),recoveries=len(events),retained_groups_unchanged=37,new_groups=3,
            no_old_cv_replay=True,no_tolerance_change=True,no_upload=True)
        ORIGINAL_WRITE(FOLDER/'result.json',receipt)
        execution.update(status='complete',finished_unix=time.time(),result=art(FOLDER/'result.json'))
        ORIGINAL_WRITE(FOLDER/'execution.json',execution)
        print(json.dumps(receipt),flush=True)
    except BaseException as error:
        execution.update(status='failed_preserved_stop',finished_unix=time.time(),failure=repr(error),recoveries=len(events))
        ORIGINAL_WRITE(FOLDER/'execution.json',execution)
        raise
    finally:
        old.tilt=ORIGINAL_TILT;old.support=ORIGINAL_SUPPORT;producer.daily_posterior=ORIGINAL_POSTERIOR;producer.write_json=ORIGINAL_WRITE


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','infer'])
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
