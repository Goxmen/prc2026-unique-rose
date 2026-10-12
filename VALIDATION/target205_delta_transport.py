"""Predeclared, read-only v79 transport of fixed component deltas tested on v78.

The original v78 comparisons remain primary. This additional comparison checks
the unchanged delta on the best submitted recipe, without searching weights or
changing constraints. It cannot create or upload a submission.
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import tuning_seasonal as validation
from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'VALIDATION/runs/v3_20261012_delta_transport'
PARENT = ROOT / 'VALIDATION/runs/v3_20261012_target205_candidate/submission_v79'
OLD = ROOT / 'VALIDATION/runs/v3_20261011_joint_public_moments'
EXPERIMENTS = {
    'phase': dict(directory='v3_20261012_source_clock_phase_lgb', control='control',
                  arms=['candidate']),
    'tabm': dict(directory='v3_20261012_nn_public_tabm_stack', control='selected',
                 arms=['replace_tabm', 'replace_control']),
}
DESIGN = dict(parent='v79', original_control='exact v78',
    formula='round_even(max(0,v79 + fixed_arm_on_v78 - exact_v78))',
    coefficient=1., new_bounds=False, zero_delta_rows='exact v79',
    purpose='secondary transport check; retain original and same-machine comparisons',
    experiments=EXPERIMENTS, no_fit=True, no_search=True, no_submission=True,
    limitation='reused development data and fixed additive transport; not independent confirmation')


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def freeze():
    assert not (OUT / 'design.json').exists()
    parent = read(PARENT / 'manifest.json')
    assert parent['state'] == 'scored' and parent['score'] == 219.613
    assert read(OLD / 'audit.json')['status'] == 'pass'
    for experiment in EXPERIMENTS.values():
        assert not (ROOT / 'VALIDATION/runs' / experiment['directory'] / 'result.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'design.json', dict(design=DESIGN, sources=[artifact(p) for p in
        [Path(__file__), PARENT / 'cv.parquet', PARENT / 'manifest.json',
         OLD / 'scored.parquet', OLD / 'result.json', OLD / 'audit.json']]))
    print('Secondary comparison frozen before either experimental score exists.')


def transport(parent, control, candidate):
    arrays = [np.asarray(x, dtype=float) for x in [parent, control, candidate]]
    assert len({x.shape for x in arrays}) == 1
    assert all(np.isfinite(x).all() for x in arrays)
    parent, control, candidate = arrays
    result = np.rint(np.maximum(0, parent + candidate - control))
    np.testing.assert_array_equal(result[candidate == control], parent[candidate == control])
    return result


def score(experiment):
    identity = read(OUT / 'design.json')
    assert identity['design'] == DESIGN
    for record in identity['sources']:
        assert sha256(ROOT / record['path']) == record['sha256']
    config = EXPERIMENTS[experiment]
    source = ROOT / 'VALIDATION/runs' / config['directory']
    audit = read(source / 'audit.json')
    assert audit['status'] == 'pass'
    assert audit['result'] == artifact(source / 'result.json')
    destination = OUT / experiment
    assert not destination.exists(), 'Preserve completed comparison'
    parent = pd.read_parquet(PARENT / 'cv.parquet')
    old = pd.read_parquet(OLD / 'scored.parquet').set_index('mvt_id')
    candidate = pd.read_parquet(source / 'scored.parquet').set_index('mvt_id')
    assert len(parent) == len(old) == len(candidate) == 344419
    assert parent.mvt_id.is_unique and old.index.is_unique and candidate.index.is_unique
    ids = parent.mvt_id
    old, candidate = old.loc[ids], candidate.loc[ids]
    for column in ['y', 'month', 'day']:
        np.testing.assert_array_equal(candidate[column], parent[column])
        np.testing.assert_array_equal(old[column], parent[column])
    np.testing.assert_array_equal(candidate[config['control']], old.candidate)
    frame = parent[['mvt_id', 'y', 'month', 'day']].copy()
    frame['control'] = parent.pred
    for arm in config['arms']:
        frame[arm] = transport(parent.pred, candidate[config['control']], candidate[arm])
    names = ['control', *config['arms']]
    scores = {month: {name: float(np.sqrt(np.mean((q[name] - q.y)**2)))
        for name in names} for month, q in frame.groupby('month')}
    paired = {arm: validation.paired_bootstrap(frame.day, frame.month,
        (frame.control - frame.y)**2, (frame[arm] - frame.y)**2) for arm in config['arms']}
    destination.mkdir()
    frame.to_parquet(destination / 'scored.parquet', index=False)
    # Separate SQL checks every scalar against source frames and parent truth.
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('f', destination / 'scored.parquet'), ('p', PARENT / 'cv.parquet'),
                           ('c', source / 'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        bad_arms = ' or '.join(f'f.{arm} is distinct from round_even(greatest(0,p.pred+'
            f'c.{arm}-c.{config["control"]}),0)' for arm in config['arms'])
        n, unique, bad = con.execute(f'''select count(*),count(distinct f.mvt_id),
            sum(case when f.y is distinct from p.y or f.month is distinct from p.month
                or f.day is distinct from p.day or f.control is distinct from p.pred
                or {bad_arms} then 1 else 0 end)
            from f join p using(mvt_id) join c using(mvt_id)''').fetchone()
        assert n == unique == 344419 and bad == 0
        for name in names:
            for month, value in con.execute(
                f'select month,sqrt(avg(pow({name}-y,2))) from f group by month').fetchall():
                assert abs(value - scores[month][name]) < 1e-8
    result = dict(design=DESIGN, experiment=experiment, rows=n, scores=scores,
        cv={name: float(np.mean([s[name] for s in scores.values()])) for name in names},
        paired=paired, all_row_sql_audit='pass', exact_v78_control=True,
        zero_delta_rows_exact_v79=True, identity=artifact(OUT / 'design.json'),
        sources=[artifact(source / p) for p in ['audit.json', 'result.json', 'scored.parquet']],
        scored=artifact(destination / 'scored.parquet'))
    write_json(destination / 'result.json', result)
    print(json.dumps(dict(cv=result['cv'], paired=paired), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'score'])
    parser.add_argument('--experiment', choices=list(EXPERIMENTS))
    args = parser.parse_args()
    with threadpool_limits(1):
        freeze() if args.stage == 'freeze' else score(args.experiment)
