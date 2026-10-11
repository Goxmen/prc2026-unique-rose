"""One fixed broad-support follow-up to the inconclusive count/sum screen."""
import argparse
from pathlib import Path
import types

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import taxi_count_sum_jan_jul as parent
import taxi_count_sum_support as posterior
from records import write_json

ROOT = parent.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_taxi_count_sum_support'
art, read, verify = parent.art, parent.read, parent.verify
DESIGN = dict(parent.DESIGN, change='retained broad/quantile supports replace narrow Gaussian/copy modes in count/sum inference',
    settings=posterior.SETTINGS, calibration='reuse every fitted count/sum coefficient and covariance exactly',
    declared_after='inconclusive first screen and its false-copy diagnosis; reused development selection',
    no_new_fit=True)


def guard():
    assert read(parent.OUT/'audit.json')['status'] == 'pass'
    paths = [Path(__file__), Path(posterior.__file__), Path(posterior.original.__file__),
        Path(posterior.retained.__file__), Path(parent.__file__), parent.OUT/'identity.json',
        parent.OUT/'scored.parquet', parent.OUT/'result.json', parent.OUT/'audit.json',
        parent.OUT/'parameters.json', parent.OUT/'calibration.parquet', parent.TAU,
        parent.daily.CACHE, parent.daily.SOURCE/'daily_runway.parquet',
        parent.quality.OUT/'scored.parquet', parent.daily.base.OUT/'rows2025.parquet',
        parent.daily.constraints.exclusion.OUT/'matches2025.parquet']
    value = dict(design=DESIGN, checks=posterior.checks(), sources=[art(p) for p in paths])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'identity.json'
    if path.exists():
        assert read(path) == value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(path, value)
    return value


def calibration(rows, source, allowed):
    month = next(m for m, a in parent.daily.base.FOLDS.items() if a == allowed)
    assert set(rows.month).issubset(allowed)
    params = read(parent.OUT/'parameters.json')[month]
    params['held_parameters'] = read(parent.TAU)['parameters'][month]['LIRF']
    cells = pd.read_parquet(parent.OUT/'calibration.parquet')
    return params, cells[cells.fold.eq(month)].drop(columns='fold').reset_index(drop=True)


def screen():
    namespace = {**parent.screen.__globals__, 'OUT': OUT, 'guard': guard, 'calibrate': calibration,
                 'posterior': posterior, 'DESIGN': DESIGN}
    fn = types.FunctionType(parent.screen.__code__, namespace, 'broad_support_score',
                            parent.screen.__defaults__, parent.screen.__closure__)
    fn()


def audit():
    guard()
    assert not (OUT/'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT/'result.json')
    for artifact in result['artifacts']:
        verify(artifact)
    parameters, old_parameters = read(OUT/'parameters.json'), read(parent.OUT/'parameters.json')
    for month, p in parameters.items():
        assert {k: v for k, v in p.items() if k != 'held_parameters'} == old_parameters[month]
        assert p['held_parameters'] == read(parent.TAU)['parameters'][month]['LIRF']
    pd.testing.assert_frame_equal(pd.read_parquet(OUT/'calibration.parquet'),
                                  pd.read_parquet(parent.OUT/'calibration.parquet'), check_exact=True)
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('c', OUT/'scored.parquet'), ('p', parent.OUT/'scored.parquet'),
                           ('r', parent.daily.CACHE)]:
            con.read_parquet(str(path)).create_view(name)
        n, u, errors = con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.refined is distinct from p.refined
            or c.refined_quality is distinct from p.refined_quality or c.month is distinct from p.month
            or c.day is distinct from p.day or (not c.touched and c.candidate!=p.refined_quality)
            or (c.touched and (c.airport!='LIRF' or not c.eligible or not c.quality_eligible))
            then 1 else 0 end) from c join p using(mvt_id) join r using(mvt_id)''').fetchone()
        assert n == u == 344419 and errors == 0
        for col in result['cv']:
            values = con.execute(f'select month,sqrt(avg(pow({col}-y,2))) from c group by month').fetchall()
            for m, v in values:
                np.testing.assert_allclose(v, result['scores'][m][col], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in values]), result['cv'][col], atol=1e-8, rtol=0)
    frame = pd.read_parquet(OUT/'scored.parquet')
    ids = frame.mvt_id.to_numpy()
    raw = pd.read_parquet(parent.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    detail = pd.read_parquet(OUT/'probabilities.parquet').set_index('mvt_id')
    assert detail.index.is_unique and len(detail) == int(frame.touched.sum())
    selected = frame[frame.touched].set_index('mvt_id').loc[detail.index]
    np.testing.assert_allclose(detail.prior_mean, selected.refined, atol=1e-5, rtol=1e-8)
    np.testing.assert_array_equal(detail.posterior_mean, selected.candidate_unconstrained)
    matches = pd.read_parquet(parent.daily.constraints.exclusion.OUT/'matches2025.parquet')
    np.testing.assert_array_equal(parent.daily.constrain(raw, ids, frame.month, frame.candidate_unconstrained.to_numpy(), matches), frame.candidate)
    report = dict(status='pass', rows=n, identity=art(OUT/'identity.json'), result=art(OUT/'result.json'),
        no_refit_parameters_calibration_and_training_supports_exact=True,
        independent_sql_parents_truth_dates_scores=True, unchanged_outside_scope=True,
        prior_means_and_final_constraints_replayed=True, source=art(__file__))
    write_json(OUT/'audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'screen', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
