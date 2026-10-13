"""Additive JSON-only recovery of the frozen v83 residual auditor.

Preserves original auditor and numerical identity. Executes the same audit body,
replacing only its report writer to convert numpy.bool_ to JSON bool. No SVD,
certificate, optimizer, inference, or candidate is rerun/created.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_v83_residual_novelty'
RECOVERY=OUT/'audit_serialization_recovery'
ORIGINAL=ROOT/'VALIDATION/tools/audit_v83_residual_novelty_20261013.py'
ORIGINAL_SHA='812cb1458de3b2be848494a93581d7c9686a1dbabae1a066b2a3e47e17cc141c'
SPEC=importlib.util.spec_from_file_location('frozen_v83_novelty_audit',ORIGINAL)
FROZEN=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(FROZEN)
art,read,verify=FROZEN.art,FROZEN.read,FROZEN.verify


def normalize_boolean(value):
    if isinstance(value,np.bool_):return bool(value)
    raise TypeError(f'Unsupported report value: {type(value).__name__}')


def atomic_json(path,value):
    path=Path(path)
    body=json.dumps(value,indent=2,allow_nan=False,default=normalize_boolean)+'\n'
    with path.open('x') as stream:stream.write(body)


def freeze():
    assert art(ORIGINAL)['sha256']==ORIGINAL_SHA
    assert not (RECOVERY/'recovery_identity.json').exists()
    review=read(RECOVERY/'source_review.json')
    assert review['status']=='pass' and review['source']==art(__file__)
    archive=read(RECOVERY/'archive.json')
    for pair in archive['files']:
        verify(pair['original']);verify(pair['archive'])
        assert pair['original']['sha256']==pair['archive']['sha256']
    identity=read(OUT/'identity.json');result=read(OUT/'result.json')
    assert identity['auditor']==art(ORIGINAL) and result['identity']==art(OUT/'identity.json')
    sources=[Path(__file__),ORIGINAL,RECOVERY/'source_review.json',RECOVERY/'archive.json',
             OUT/'identity.json',OUT/'result.json',verify(result['decomposition']),OUT/'allocation.json']
    sources.extend(verify(pair['archive']) for pair in archive['files'])
    atomic_json(RECOVERY/'recovery_identity.json',dict(source=art(__file__),original_auditor=art(ORIGINAL),
        original_identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),decomposition=result['decomposition'],
        partial=art(OUT/'independent_audit.json'),archive=art(RECOVERY/'archive.json'),sources=[art(p) for p in sources],
        scope='Only report serialization np.bool_→bool and explicit recovery provenance; original audit body unchanged.'))
    print(json.dumps({'identity':art(RECOVERY/'recovery_identity.json'),'no_numeric_execution':True}))


def audit():
    identity=read(RECOVERY/'recovery_identity.json')
    assert identity['source']==art(__file__) and identity['original_auditor']==art(ORIGINAL)
    assert art(ORIGINAL)['sha256']==ORIGINAL_SHA
    for rec in identity['sources']:verify(rec)
    assert identity['original_identity']==art(OUT/'identity.json')
    assert not (RECOVERY/'started.json').exists()
    assert not (RECOVERY/'partial_at_retry.json').exists()
    # Preserve the failed path's actual bytes before freeing the final audit name.
    partial=verify(identity['partial'])
    partial.rename(RECOVERY/'partial_at_retry.json')
    atomic_json(RECOVERY/'started.json',dict(identity=art(RECOVERY/'recovery_identity.json'),
        preserved_partial=art(RECOVERY/'partial_at_retry.json'),pid=os.getpid()))
    original_write=FROZEN.write
    def recovered_write(path,value):
        assert Path(path)==OUT/'independent_audit.json' and value['status']=='pass'
        assert value['source']==identity['original_auditor'] and value['identity']==identity['original_identity']
        assert isinstance(value['gate']['minimum_half_second'],np.bool_)
        # Same nested object is subsequently emitted by the frozen stdout print.
        value['gate']['minimum_half_second']=bool(value['gate']['minimum_half_second'])
        value=dict(value,serialization_recovery=art(RECOVERY/'recovery_identity.json'),
            executed_wrapper=art(__file__),original_numeric_source=identity['original_auditor'],
            numeric_body_unchanged=True)
        atomic_json(path,value)
    FROZEN.write=recovered_write
    try:
        FROZEN.audit()
    finally:
        FROZEN.write=original_write
    final=read(OUT/'independent_audit.json')
    assert final['status']=='pass' and final['serialization_recovery']==art(RECOVERY/'recovery_identity.json')
    atomic_json(RECOVERY/'recovery_result.json',dict(status='pass',identity=art(RECOVERY/'recovery_identity.json'),
        original_identity=identity['original_identity'],original_result=identity['result'],
        audit=art(OUT/'independent_audit.json'),original_auditor=identity['original_auditor'],
        executed_wrapper=art(__file__),no_certificate_rerun=True,no_numerical_changes=True))
    print(json.dumps({'recovery':art(RECOVERY/'recovery_result.json'),'audit':art(OUT/'independent_audit.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','audit'])
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
