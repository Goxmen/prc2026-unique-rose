"""Redistribute the selected daily correction by fixed ensemble disagreement.

Preserve each source cell's total correction and its original independent/common
variance budget. Only the within-cell allocation changes, without label fitting.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_daily_taxi_quality as quality
import public_stack_cv as joint
from records import write_json

daily = quality.original
rounded = daily.parent
ROOT = daily.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_daily_taxi_uncertainty'
art, read, verify = daily.art, daily.read, daily.verify
NAMES = ['callsign', 'physical0', 'physical1', 'physical2', 'main', 'airport', 'catboost']
WEIGHTS = np.array([.71*.5]+[.71*.5/3]*3+[.29/3]*3)
DESIGN = dict(parent='selected production-rounded daily-quality refinement',
    change='allocate each unchanged daily taxi mean shift using relative ensemble disagreement',
    members=NAMES, member_weights=WEIGHTS.tolist(),
    disagreement='weighted variance of each stand-alone ordinary member through the fixed classifier/corrections/serialized bounds',
    independent_variance='rescale disagreements within eligible runway-day to mean200^2; zero variance falls back to200^2',
    shared_variance='unchanged15^2 common component',
    allocation='uniform shift*(individual_variance+n*15^2)/(200^2+n*15^2)',
    invariant='same group total before final constraints and rounding; same source/gain/cap/quality gate',
    parameters='all retained; no fitted error model, no outcome gate, no new scale or blend search',
    limitation='disagreement is a proxy for relative uncertainty; reused Jan/Jul development',
    no_new_fit=True, no_production=True, no_upload=True)


def allocate(variance):
    variance = np.asarray(variance, float)
    assert len(variance) and np.isfinite(variance).all() and (variance >= 0).all()
    n = len(variance)
    average = float(variance.mean())
    independent = 40000*variance/average if average > 0 else np.full(n, 40000.)
    weight = (independent+225*n)/(40000+225*n)
    np.testing.assert_allclose(weight.mean(), 1., atol=1e-12, rtol=0)
    assert (weight > 0).all()
    return weight


def checks():
    np.testing.assert_array_equal(allocate(np.zeros(6)), np.ones(6))
    np.testing.assert_array_equal(allocate(np.ones(6)*9), np.ones(6))
    x = np.array([0., 1., 4., 9., 50.])
    w = allocate(x)
    np.testing.assert_allclose(w, allocate(x*7), atol=1e-12, rtol=0)
    order = np.array([4, 1, 0, 3, 2])
    np.testing.assert_array_equal(allocate(x[order]), w[order])
    np.testing.assert_allclose(w.sum()*37., len(x)*37., atol=1e-10)
    assert np.all(np.diff(w) > 0)
    # Direct Gaussian conditioning with heterogeneous independent variances.
    diagonal = x/x.mean()*40000
    covariance = np.diag(diagonal)+np.ones((5, 5))*225
    independent = covariance.sum(axis=1)/(covariance.sum()/5)
    np.testing.assert_allclose(w, independent, atol=1e-12, rtol=0)
    return dict(equal_and_zero_variance_uniform=True, scale_invariant=True, permutation=True,
        group_sum_preserved=True, independent_covariance_derivation=True)


def dependencies():
    neural = joint.neural
    identity = read(neural.single.MANIFEST)
    replay = neural.cat.r20.physical.replay
    hpo = joint.main.parent.hpo
    paths = [Path(__file__), Path(quality.__file__), Path(daily.__file__), Path(rounded.__file__),
        Path(joint.__file__), Path(hpo.__file__), Path(hpo.ts.__file__),
        quality.OUT/'scored.parquet', quality.OUT/'result.json', quality.OUT/'audit.json',
        daily.OUT/'parameters.json', daily.OUT/'cells.parquet',
        rounded.OUT/'scored.parquet', rounded.original.airport.OUT/'scored.parquet',
        neural.cat.OUT/'scored.parquet', daily.base.OUT/'rows2025.parquet',
        daily.constraints.exclusion.OUT/'matches2025.parquet', replay.combo.sequence.CALLS_PRED,
        hpo.pred_path('catboost', 'K07', 42), hpo.record_path('catboost'),
        hpo.pred_path('rome', hpo.winner('rome')['trial'], 42)]
    for month in daily.base.FOLDS:
        split = hpo.ts.split_dir(hpo.SPLITS_FROM, month)
        paths.extend([split/'heldout.pkl', split/'meta.json',
            joint.main.OUT/'predictions'/f'{month}_s42.parquet',
            rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet'])
        for k in range(3):
            record = neural.completed(month, k, identity)
            assert record is not None
            paths.append(verify(record['prediction']))
    return paths


def guard():
    assert read(quality.OUT/'audit.json')['status'] == 'pass'
    assert read(rounded.original.airport.OUT/'audit.json')['status'] == 'pass'
    value = dict(design=DESIGN, checks=checks(), sources=[art(p) for p in dependencies()])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/'identity.json'
    if p.exists():
        assert read(p) == value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(p, value)
    return value


def members():
    guard()
    assert not (OUT/'members.parquet').exists(), 'Preserve prepared member predictions'
    neural, hpo = joint.neural, joint.main.parent.hpo
    identity = read(neural.single.MANIFEST)
    replay = neural.cat.r20.physical.replay
    prior = pd.read_parquet(neural.cat.OUT/'scored.parquet').set_index('mvt_id')
    rounded_saved = pd.read_parquet(rounded.OUT/'scored.parquet').set_index('mvt_id')
    direct_saved = pd.read_parquet(rounded.original.airport.OUT/'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    rome = pd.read_parquet(hpo.pred_path('rome', hpo.winner('rome')['trial'], 42)).set_index('mvt_id')
    oldcat = pd.read_parquet(hpo.pred_path('catboost', 'K07', 42)).set_index('mvt_id')
    parts = []
    for month in daily.base.FOLDS:
        split = hpo.ts.split_dir(hpo.SPLITS_FROM, month)
        held, meta = pd.read_pickle(split/'heldout.pkl'), read(split/'meta.json')
        ids = held['raw'].mvt_id.to_numpy()
        main = replay.values(joint.main.OUT/'predictions'/f'{month}_s42.parquet', ids, 'pred')
        airport = replay.values(rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet', ids, 'pred')
        catboost = oldcat.loc[ids, hpo.winner('catboost')['checkpoint']].to_numpy(float)
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        physical = [replay.values(verify(neural.completed(month, k, identity)['prediction']), ids, 'base') for k in range(3)]
        probability = .5*prior.loc[ids, 'newcat'].to_numpy()+.5*rome.loc[ids, hpo.winner('rome')['checkpoint']].to_numpy(float)
        boundary = rounded_saved.loc[ids, 'candidate_boundary'].to_numpy()
        baseline = prior.loc[ids, 'control'].to_numpy()
        permitted = matches[matches.donor_month.isin(daily.base.FOLDS[month]+[month])]

        def final(ordinary):
            direct = hpo.ts.final_prediction(ordinary, probability, held['V'], held['raw'], meta['parameters'], meta['corrections'])
            pred = daily.constraints.matched_bound(raw.loc[ids].reset_index(), boundary+direct-baseline)
            pred, _ = daily.constraints.exclusion.apply(ids, np.maximum(pred, 0), permitted)
            return direct, np.rint(np.maximum(pred, 0))

        # Reproduce the selected member arithmetic and both correction boundaries.
        ordinary = .71*.5*(calls+np.mean(physical, axis=0))+.29*np.mean([main, airport, catboost], axis=0)
        direct, control = final(ordinary)
        np.testing.assert_array_equal(direct, direct_saved.loc[ids, 'candidate'])
        np.testing.assert_array_equal(control, rounded_saved.loc[ids, 'candidate'])
        values = [final(v)[1] for v in [calls, *physical, main, airport, catboost]]
        matrix = np.column_stack(values)
        mean = matrix@WEIGHTS
        variance = ((matrix-mean[:, None])**2)@WEIGHTS
        frame = pd.DataFrame(dict(mvt_id=ids, month=month, mean=mean, variance=variance,
                                  **dict(zip(NAMES, values))))
        parts.append(frame)
    pd.concat(parts, ignore_index=True).to_parquet(OUT/'members.parquet', index=False)
    write_json(OUT/'members.json', dict(identity=art(OUT/'identity.json'), prediction=art(OUT/'members.parquet'),
        actual_selected_direct_and_serialized_controls_exact=True))


def screen():
    guard()
    assert not (OUT/'result.json').exists(), 'Preserve completed score'
    record = read(OUT/'members.json')
    assert record['identity'] == art(OUT/'identity.json')
    verify(record['prediction'])
    f = pd.read_parquet(quality.OUT/'scored.parquet')
    ids = f.mvt_id.to_numpy()
    member = pd.read_parquet(OUT/'members.parquet').set_index('mvt_id').loc[ids]
    raw = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    weights = np.ones(len(f))
    cell_info = []
    selected = f.eligible & f.refined.between(0, 7200)
    for key, ix in f[selected].groupby(['month', *daily.KEY]).groups.items():
        ix = np.asarray(ix, int)
        variance = member.variance.iloc[ix].to_numpy()
        weights[ix] = allocate(variance)
        shift = f.refined_shift.iloc[ix].to_numpy()
        np.testing.assert_allclose(shift, shift[0], atol=0, rtol=0)
        np.testing.assert_allclose((weights[ix]*shift).sum(), shift.sum(), atol=1e-7, rtol=1e-12)
        cell_info.append(dict(fold=key[0], airport=key[1], local_day=key[2], runway=key[3], n=len(ix),
            mean_variance=float(variance.mean()), uniform_sum=float(shift.sum()),
            weighted_sum=float((weights[ix]*shift).sum()), max_weight=float(weights[ix].max())))
    original_shift = np.where(f.quality_eligible, f.refined_shift, 0.)
    np.testing.assert_array_equal(daily.constrain(raw, ids, f.month, f.refined.to_numpy()+original_shift, matches), f.refined_quality)
    adjusted = f.refined.to_numpy()+weights*original_shift
    candidate = daily.constrain(raw, ids, f.month, adjusted, matches)
    f['variance'], f['allocation'], f['candidate_unconstrained'], f['candidate'] = member.variance.to_numpy(), weights, adjusted, candidate
    names = ['refined', 'refined_quality', 'candidate']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in names} for m, g in f.groupby('month')}
    paired = daily.parent.original.airport.parent.ts.paired_bootstrap(f.day, f.month,
        (f.refined_quality-f.y)**2, (f.candidate-f.y)**2)
    f.to_parquet(OUT/'scored.parquet', index=False)
    pd.DataFrame(cell_info).to_parquet(OUT/'cells.parquet', index=False)
    result = dict(design=DESIGN, identity=art(OUT/'identity.json'), scores=scores,
        cv={c: float(np.mean([v[c] for v in scores.values()])) for c in names}, paired=paired,
        artifacts=[art(OUT/p) for p in ['scored.parquet', 'cells.parquet', 'members.parquet']],
        uniform_control_exact=True, every_preconstraint_cell_sum_preserved=True,
        changed=int(f.candidate.ne(f.refined_quality).sum()))
    write_json(OUT/'result.json', result)
    print(result, flush=True)


def audit():
    guard()
    assert not (OUT/'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT/'result.json')
    for a in result['artifacts']:
        verify(a)
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('c', OUT/'scored.parquet'), ('p', quality.OUT/'scored.parquet'),
                           ('m', OUT/'members.parquet'), ('r', daily.base.OUT/'rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        mean = '+'.join(f'{float(w)!r}*m.{n}' for n, w in zip(NAMES, WEIGHTS))
        variance = '+'.join(f'{float(w)!r}*pow(m.{n}-({mean}),2)' for n, w in zip(NAMES, WEIGHTS))
        n, u, errors, delta = con.execute(f'''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.month is distinct from p.month or c.day is distinct from p.day
            or c.refined is distinct from p.refined or c.refined_quality is distinct from p.refined_quality
            or c.variance is distinct from m.variance
            or c.candidate_unconstrained != c.refined+c.allocation*(case when c.quality_eligible then c.refined_shift else 0 end)
            then 1 else 0 end),max(abs(m.variance-({variance})))
            from c join p using(mvt_id) join r using(mvt_id) join m using(mvt_id)''').fetchone()
        assert n == u == 344419 and errors == 0
        assert delta < 1e-5, delta
        for col in result['cv']:
            values = con.execute(f'select month,sqrt(avg(pow({col}-y,2))) from c group by month').fetchall()
            for month, v in values:
                np.testing.assert_allclose(v, result['scores'][month][col], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in values]), result['cv'][col], atol=1e-8, rtol=0)
    f = pd.read_parquet(OUT/'scored.parquet')
    cells = pd.read_parquet(OUT/'cells.parquet')
    np.testing.assert_allclose(cells.uniform_sum, cells.weighted_sum, atol=1e-7, rtol=1e-12)
    eligible = f.eligible & f.refined.between(0, 7200)
    for _, g in f[eligible].groupby(['month', *daily.KEY]):
        variance = g.variance.to_numpy()
        individual = variance/variance.mean()*40000 if variance.mean() > 0 else np.full(len(g), 40000.)
        expected = (individual+225*len(g))/(40000+225*len(g))
        np.testing.assert_allclose(expected, g.allocation, atol=1e-12, rtol=0)
    ids = f.mvt_id.to_numpy()
    raw = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    np.testing.assert_array_equal(daily.constrain(raw, ids, f.month, f.candidate_unconstrained.to_numpy(), matches), f.candidate)
    report = dict(status='pass', rows=n, identity=art(OUT/'identity.json'), result=art(OUT/'result.json'),
        independent_sql_variance_truth_controls_arithmetic_scores=True, variance_sql_max_error=delta,
        all_cell_allocations_sums_and_final_constraints_replayed=True, source=art(__file__))
    write_json(OUT/'audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'members', 'screen', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
