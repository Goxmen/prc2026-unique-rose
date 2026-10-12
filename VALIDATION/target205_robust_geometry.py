"""Reassess fixed robust affine objective after actual v76-v78 feedback.

The old 48-scored-vector result remains closed. This uses all 51 retained,
successfully scored directions; cancelled v75 is absent. Read-only diagnosis.
"""
import argparse
from pathlib import Path
import types

from threadpoolctl import threadpool_limits
import robust_scored_geometry as original

ROOT = original.ROOT
read, verify, art = original.read, original.verify, original.art
OUT = ROOT / 'VALIDATION/runs/v3_20261012_target205_robust_geometry'
PREPARED = ROOT / 'VALIDATION/runs/v3_20261012_target205_geometry/prepared.json'
DESIGN = {**original.DESIGN,
          'parent': 'all51 retained directions with actual successful v76-v78 scores; v75 removed',
          'reason_for_new_evaluation': 'three actual new scores unavailable to the completed48-vector assessment',
          'no_new_candidate': True}


def diagnose(snapshot):
    assert not (OUT/'result.json').exists()
    assert original.read(PREPARED)['removed_unsubmitted_versions'] == [75]
    assert len(original.read(PREPARED)['versions']) == 51
    live = original.read(snapshot)
    known = {r['version'] for r in live['history'] if r['status']=='Succeeded' and r['usedPairs']==344841}
    assert set(original.read(PREPARED)['versions']) <= known
    previous = original.read(original.OUT/'result.json')
    assert len(previous['available_versions']) == 48
    assert set(original.read(PREPARED)['versions'])-set(previous['available_versions']) == {76,77,78}
    source = original.diagnose
    scope = dict(source.__globals__,OUT=OUT,PREPARED=PREPARED,DESIGN=DESIGN,__file__=__file__)
    fn = types.FunctionType(source.__code__,scope,'new_score_diagnosis',source.__defaults__,source.__closure__)
    assert fn.__code__ is source.__code__
    fn(snapshot)
    original.write_json(OUT/'implementation.json',dict(source=original.art(__file__),
        reused_source=original.art(original.__file__),original48_result=original.art(original.OUT/'result.json'),
        previous51_result=original.art(PREPARED.with_name('result.json')),
        diagnosis_bytecode_unchanged=True,no_candidate_or_upload=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    with threadpool_limits(1):
        diagnose(parser.parse_args().snapshot.resolve())
