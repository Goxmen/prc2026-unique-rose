"""Evaluate the retained prediction basis after removing cancelled v75.

All other vectors remain exact. Ancestry involving v75's prepared predictions
is preserved, but its unobserved score is never used or imputed. Evaluation
requires actual successful scores for v76--v78. No fits or uploads occur here.
"""
import argparse
from pathlib import Path
import types

import pandas as pd
from threadpoolctl import threadpool_limits

import joint_models_geometry as previous
import public_models_geometry as base
from records import sha256, write_json

ROOT = base.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_target205_geometry'
ADDITIONS = [70, 72, 74, 76, 77, 78]
DESIGN = dict(base.DESIGN, parent='retained 52-vector basis with cancelled v75 removed',
    additions=ADDITIONS, target=205., removed_versions=[75],
    new_directions='actual scores for retained public, daily/tree and joint-source candidates',
    preparation='preserve all 51 retained ranking/CV columns exactly; no refits or new weight limits')
read, art, verify = base.read, base.art, base.verify
transport = base.transport


def prepare():
    assert not (OUT / 'prepared.json').exists(), 'Preserve prepared basis'
    old = read(previous.OUT / 'prepared.json')
    audit = read(previous.OUT / 'audit.json')
    assert old['design'] == previous.DESIGN
    assert audit['status'] == 'pass' and audit['prepared'] == art(previous.OUT / 'prepared.json')
    assert len(old['versions']) == 52 and 75 in old['versions']
    for path, digest in old['sources'].items():
        assert sha256(ROOT / path) == digest, path
    sources = dict(old['sources'])
    versions = [version for version in old['versions'] if version != 75]
    assert len(versions) == len(set(versions)) == 51
    OUT.mkdir(parents=True, exist_ok=True)
    for field in ['ranking', 'cv']:
        origin = verify(old[field])
        frame = pd.read_parquet(origin)
        assert 'v75' in frame
        frame.drop(columns=['v75']).to_parquet(OUT / f'{field}_basis.parquet', index=False)
        sources[str(origin.relative_to(ROOT))] = sha256(origin)
    for source in [Path(__file__), Path(previous.__file__), Path(base.__file__),
                   previous.OUT / 'prepared.json', previous.OUT / 'audit.json']:
        sources[str(source.relative_to(ROOT))] = sha256(source)
    additions = {key: value for key, value in old['additions'].items() if key != '75'}
    assert set(map(int, additions)) == set(ADDITIONS)
    write_json(OUT / 'prepared.json', dict(design=DESIGN, versions=versions,
        known_scores={key: value for key, value in old['known_scores'].items() if key != '75'},
        additions=additions, sources=sources, ranking=art(OUT / 'ranking_basis.parquet'),
        cv=art(OUT / 'cv_basis.parquet'), checks=base.check_geometry(),
        historical_cv_caveat=old['historical_cv_caveat'],
        reused_basis=art(previous.OUT / 'prepared.json'),
        removed_unsubmitted_versions=[75], preserved_unsubmitted_ancestry=True))
    print(dict(versions=len(versions), removed=75, required_new_scores=[76, 77, 78]), flush=True)


def inherit(function):
    scope = dict(function.__globals__, OUT=OUT, ADDITIONS=ADDITIONS,
                 DESIGN=DESIGN, __file__=__file__)
    result = types.FunctionType(function.__code__, scope, function.__name__,
                                function.__defaults__, function.__closure__)
    assert result.__code__ is function.__code__
    return result


def audit():
    for field in ['ranking', 'cv']:
        old = pd.read_parquet(previous.OUT / f'{field}_basis.parquet')
        new = pd.read_parquet(OUT / f'{field}_basis.parquet')
        assert 'v75' not in new
        pd.testing.assert_frame_equal(new, old.drop(columns=['v75']), check_exact=True)
    inherit(base.audit)()


def evaluate():
    inherit(base.evaluate)()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'audit', 'evaluate'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
