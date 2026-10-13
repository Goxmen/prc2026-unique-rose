"""Independent audit of FINAL-only exponential-tilt domain extension.

Uses Brent roots and an independent support-atom construction, never production
bisection. Completed37groups are byte-verified. The unchanged final SQL audit
is composed with its existing two-join repair and explicit supplement binding.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import daily_delay_posterior as original
from tools import recover_portable_public_audit_joins_20261013 as sql_repair
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
FOLDER=OUT/'final/tilt_recovery'
MANIFEST=FOLDER/'identity.json'
ORIGINAL_TILT=original.tilt
EXPECTED_IDENTITY='b186449ec102d81434c3b0e0ef9483c44e18146a855214e42c6a9c5313c7d660'
EXPECTED_SUPPORT='6c47d9efabdeaf445291fa066edf8125944f377efdb87c94eb221d2eb19dd131'
EXPECTED_SQL_REPAIR='28f7bc934a86602bdc61d28b6fc8de725edee37b29cbf00331b735c2eb8a76c3'
PENDING={('2026-07','LIRF'),('2026-07','LSZH'),('2026-07','LTFM')}


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda:stream.read(8<<20),b''):digest.update(b)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest())
def verify(record):
    path=ROOT/record['path'];assert art(path)=={k:record[k] for k in ['path','sha256']};return path
def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def mean_pass(values,weights,target):
    return bool(np.isclose(values@weights,target,atol=1e-5,rtol=1e-8,equal_nan=False))


def soft_weights(values,weights,target,parameter):
    z=(values-target)/max(float(np.ptp(values)),1.)
    logits=np.log(np.maximum(weights,1e-300))+parameter*z
    mass=np.exp(logits-logits.max());mass/=mass.sum()
    return mass,float(mass@z)


def brent_weights(values,weights,target,bounds=None):
    assert np.isfinite(values).all() and np.isfinite(weights).all() and np.isfinite(target)
    assert values.min()<target<values.max() and (weights>=0).all() and weights.sum()>0
    lo,hi=(-128.,128.) if bounds is None else bounds
    if bounds is None:
        for _ in range(33):
            low,high=soft_weights(values,weights,target,lo)[1],soft_weights(values,weights,target,hi)[1]
            if low<=0<=high:break
            if low>0:lo*=2
            if high<0:hi*=2
        else:raise AssertionError('No finite root bracket')
    root=brentq(lambda x:soft_weights(values,weights,target,x)[1],lo,hi,xtol=1e-12,rtol=1e-14,maxiter=200)
    result=soft_weights(values,weights,target,root)[0]
    assert mean_pass(values,result,target)
    return result,root


def audit_tilt(values,weights,target):
    previous=ORIGINAL_TILT(values,weights,target)
    if mean_pass(values,previous,target):return previous
    lo=soft_weights(values,weights,target,-128.)[1];hi=soft_weights(values,weights,target,128.)[1]
    assert lo>0 or hi<0
    return brent_weights(values,weights,target)[0]


def atoms(row,prm):
    d,p,nm,gap=float(row['gap']),float(row['prior']),bool(row['has_id']),row['nm_gap']
    pi=0.
    mixture=row['airport']=='LIRF' and 3600<d<50400 and prm['tau']<p<d
    if mixture:pi=min(max((p-prm['tau'])/(d-prm['tau']),0.),.999)
    loc=(p-pi*d)/(1-pi)
    held=np.maximum(0,loc+np.asarray(prm['spread'],dtype=float))
    values=held.copy();mass=np.full(len(held),1/len(held))
    if mixture:
        values=np.concatenate([held,[d],np.linspace(30.,max(d,30.),9)])
        mass=np.concatenate([np.full(len(held),(1-pi)*.95/len(held)),[pi*.95],np.full(9,.05/9)])
    if nm and gap is not None and np.isfinite(gap):
        lower,upper=max(0.,gap-3606.),gap+3606.
        if lower<=p<=upper:
            take=(values>=lower)&(values<=upper)
            if take.any():values,mass=values[take],mass[take]
    return np.r_[values,p],np.r_[mass,1e-8]


def event_check(event,row,parameters):
    assert event['row']==row
    values,mass=atoms(row,parameters)
    np.testing.assert_array_equal(values,np.asarray(event['values']))
    np.testing.assert_array_equal(mass,np.asarray(event['base_weights']))
    target=float(row['prior']);assert event['target']==target
    previous=ORIGINAL_TILT(values,mass,target)
    np.testing.assert_array_equal(previous,np.asarray(event['original_weights']))
    assert not mean_pass(values,previous,target) and event['original_failure']
    assert abs(float(values@previous)-event['original_mean'])<1e-10
    assert event['iterations']==55 and event['unchanged_atol']==1e-5 and event['unchanged_rtol']==1e-8
    bounds=event['bracket'];assert bounds and bounds[0]['lower']==-128. and bounds[0]['upper']==128.
    for k,bound in enumerate(bounds):
        lo,hi=bound['lower'],bound['upper']
        a,b=soft_weights(values,mass,target,lo)[1],soft_weights(values,mass,target,hi)[1]
        np.testing.assert_allclose([a,b],[bound['lower_mean_z'],bound['upper_mean_z']],atol=1e-12,rtol=0)
        if k<len(bounds)-1:
            assert a>0 or b<0
            nxt=bounds[k+1];assert nxt['lower']==(2*lo if a>0 else lo) and nxt['upper']==(2*hi if b<0 else hi)
        else:assert a<=0<=b
    independent,root=brent_weights(values,mass,target,(bounds[-1]['lower'],bounds[-1]['upper']))
    actual=np.asarray(event['weights']);assert np.isfinite(actual).all() and (actual>=0).all() and abs(actual.sum()-1)<1e-12
    np.testing.assert_allclose(actual,independent,atol=1e-12,rtol=0)
    assert abs(root-event['root_parameter'])<1e-7
    assert abs(values@actual-event['mean'])<1e-10 and mean_pass(values,actual,target)
    return dict(mvt_id=row['mvt_id'],airport=row['airport'],day=row['day'],root=root,
                max_weight_error=float(np.max(np.abs(actual-independent))),mean_error=float(values@actual-target),atoms=len(values))


def audit_events():
    started=time.monotonic();identity=read(MANIFEST);recovery=read(FOLDER/'result.json')
    assert identity['auditor']==art(Path(__file__)) and identity['original_identity']==art(OUT/'identity.json')
    assert identity['original_identity']['sha256']==EXPECTED_IDENTITY
    assert art(Path(original.__file__))['sha256']==EXPECTED_SUPPORT
    assert art(Path(sql_repair.__file__))['sha256']==EXPECTED_SQL_REPAIR
    assert identity['same_iterations']==55 and identity['same_atol']==1e-5 and identity['same_rtol']==1e-8
    assert {tuple(v) for v in identity['pending_groups']}==PENDING
    for record in identity['sources']:verify(record)
    assert recovery['status']=='complete' and recovery['identity']==art(MANIFEST)
    assert recovery['retained_groups_unchanged']==37 and recovery['new_groups']==3
    execution=read(FOLDER/'execution.json');assert execution['status']=='complete' and execution['result']==art(FOLDER/'result.json')
    allocation=read(verify(execution['allocation']))
    assert allocation['authority']=='/root' and allocation['recovery_identity']==art(MANIFEST) and allocation['state']=='released_for_final_tilt_recovery'
    final_result=read(verify(recovery['result']))
    assert final_result['identity']==art(OUT/'identity.json') and final_result['numerical_recovery']==art(MANIFEST)
    groups=set()
    for group in identity['retained_completed_groups']:
        key=(group['month'],group['airport']);assert key not in groups and key not in PENDING;groups.add(key)
        assert len(group['files'])==3
        for record in group['files']:verify(record)
        receipt=read(verify(group['files'][0]));assert receipt['prediction']==group['files'][1] and receipt['days']==group['files'][2]
        assert 'numerical_recovery' not in receipt
    assert len(groups)==37 and len(final_result['posterior_receipts'])==40
    seen=set()
    for record in final_result['posterior_receipts']:
        receipt=read(verify(record));key=receipt['month'],receipt['airport'];assert key not in seen;seen.add(key)
        assert receipt['identity']==art(OUT/'identity.json') and receipt['view']=='final'
        verify(receipt['prediction']);verify(receipt['days'])
        if key in PENDING:assert receipt['numerical_recovery']==art(MANIFEST)
        else:assert key in groups and record==next(g['files'][0] for g in identity['retained_completed_groups'] if (g['month'],g['airport'])==key)
    assert seen==groups|PENDING
    parent=read(OUT/'identity.json');spec=parent['specs']['final']
    parameters=read(verify(spec['daily_parameters']))['parameters']
    # Canonical audited input table and base are aligned independently by ID.
    assert spec['raw'] in parent['sources'] and spec['base'] in parent['sources']
    raw=pd.read_parquet(verify(spec['raw']))
    base=pd.read_parquet(verify(spec['base']))
    assert raw.mvt_id.is_unique and base.mvt_id.is_unique
    assert len(raw)==len(base)==670790 and set(raw.mvt_id)==set(base.mvt_id)
    raw=raw.set_index('mvt_id');base=base.set_index('mvt_id')
    events=[json.loads(line) for line in verify(recovery['events']).read_text().splitlines()]
    assert len(events)==recovery['recoveries'] and events
    checks=[];ids=set()
    for event in events:
        event_row=event['row'];mid=event_row['mvt_id'];assert mid not in ids;ids.add(mid)
        source=raw.loc[mid]
        row=dict(mvt_id=float(mid),month=source.month,day=source.day,airport=source.airport,gap=float(source.gap),
                 prior=float(base.loc[mid,'submitted']),has_id=bool(source.has_id),nm_gap=float(source.nm_gap) if np.isfinite(source.nm_gap) else None)
        assert (row['month'],row['airport']) in PENDING and row['gap']>3600
        checks.append(event_check(event,row,parameters[row['airport']]))
    report=dict(status='pass',source=art(Path(__file__)),recovery_identity=art(MANIFEST),recovery_result=art(FOLDER/'result.json'),
        final_result=recovery['result'],events=recovery['events'],event_checks=checks,preserved_groups=37,new_groups=3,
        independently_rebuilt_atoms_and_Brent_weights=True,no_mean_tolerance_change=True,
        original_success_branch_unmodified=True,no_completed_group_repeated=True,
        unsaved_partial_LIRF_group_restarted=True,no_labels_read=True,no_posterior_inference=True,seconds=time.monotonic()-started)
    write(FOLDER/'independent_event_audit.json',report)
    return report


def audit():
    assert not (OUT/'final/audit.json').exists() and not (FOLDER/'independent_event_audit.json').exists()
    audit_events()
    # Frozen original SQL audit survives unchanged; only its known two JOIN
    # clauses are replaced by the independently reviewed existing adapter.
    sql_repair.freeze('final')
    original.tilt=audit_tilt
    try:sql_repair.audit('final')
    finally:original.tilt=ORIGINAL_TILT
    final=OUT/'final/audit.json';saved=FOLDER/'SQL_numerical_audit.json';assert not saved.exists()
    execution_path=OUT/'final/audit_sql_join_recovery/execution.json'
    execution=read(execution_path);assert execution['state']=='complete' and execution['audit']==art(final)
    archived_execution=FOLDER/'qualified_join_execution_original.json';assert not archived_execution.exists()
    archived_execution.write_bytes(execution_path.read_bytes())
    original_audit=art(final);final.replace(saved)
    relocation=FOLDER/'qualified_join_audit_relocation.json'
    write(relocation,dict(status='explicit_additive_relocation',original_execution=art(archived_execution),
        historical_audit=original_audit,preserved_audit=art(saved),bytes_unchanged=original_audit['sha256']==art(saved)['sha256']))
    report=read(saved);assert report['status']=='pass' and report['result']==art(OUT/'final/result.json')
    report.update(source=art(Path(__file__)),qualified_join_audit=art(saved),numerical_supplement=art(MANIFEST),
                  numerical_supplement_result=art(FOLDER/'result.json'),independent_event_audit=art(FOLDER/'independent_event_audit.json'),
                  qualified_join_execution_relocation=art(relocation),
                  support_audit_method='Original successful tilt unchanged; independent Brent fallback only for originally failing unbracketed interior cases',
                  no_original_source_mutation=True,no_completed_posterior_group_repeated=True)
    write(final,report)
    execution.update(audit=art(final),original_execution=art(archived_execution),supplement_relocation=art(relocation))
    execution_path.write_text(json.dumps(execution,indent=2)+'\n')
    print(json.dumps(dict(status='pass',audit=art(final),events=len(read(FOLDER/'independent_event_audit.json')['event_checks']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['audit'])
    args=parser.parse_args();audit()
