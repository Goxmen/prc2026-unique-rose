"""Measure the daily-taxi main tree inside the selected CatBoost/quality recipe.

Reuse every fit and calibration. Evaluate the four fixed old/new main and
CatBoost combinations, reconstructing both serialized boundaries and the
unchanged daily source correction. No search, refits, or upload.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import catboost_refinements_cv as parent
import public_daily_taxi_lgb as main
from records import write_json

ROOT, cv, daily, joint = parent.ROOT, parent.cv, parent.daily, parent.joint
quality, rounded = parent.quality, parent.rounded
OUT = ROOT / 'VALIDATION/runs/v3_20261011_daily_main_refinements_cv'
art, read, verify = parent.art, parent.read, parent.verify
DESIGN = dict(parent='selected full174 CatBoost plus airport/t5/full-monthly/daily-quality recipe',
    change='replace only main LightGBM with retained184-input daily-taxi model',
    primary='combined versus full_cat after unchanged daily-quality correction',
    secondary='four fixed main/CatBoost combinations, before and after quality correction',
    no_fits=True, no_weight_or_parameter_search=True, no_upload=True,
    limitation='reused Jan/Jul development; historical CatBoost control used another GPU')
ARMS = {'refined': (False, False), 'daily_main': (True, False),
        'full_cat': (False, True), 'combined': (True, True)}


def guard():
    assert read(parent.OUT / 'audit.json')['status'] == 'pass'
    assert read(main.OUT / 'audit.json')['status'] == 'pass'
    assert read(main.OUT / 'result.json')['paired']['refined']['decision'] == 'improvement'
    paths = [Path(__file__), Path(parent.__file__), Path(main.__file__),
        Path(daily.__file__), Path(quality.__file__), Path(joint.__file__),
        Path(daily.constraints.__file__), Path(daily.constraints.exclusion.__file__),
        parent.OUT / 'identity.json', parent.OUT / 'result.json', parent.OUT / 'audit.json',
        parent.OUT / 'scored.parquet', main.OUT / 'identity.json', main.OUT / 'result.json',
        main.OUT / 'audit.json', main.OUT / 'scored.parquet',
        quality.OUT / 'scored.parquet', quality.OUT / 'audit.json',
        daily.OUT / 'parameters.json', daily.CACHE, daily.SOURCE / 'daily_runway.parquet',
        daily.base.OUT / 'rows2025.parquet', daily.constraints.exclusion.OUT / 'matches2025.parquet',
        rounded.OUT / 'scored.parquet', joint.neural.cat.OUT / 'scored.parquet']
    for month in daily.base.FOLDS:
        paths.extend([verify(main.completed(month)['prediction']),
            verify(cv.completed(month)['prediction']),
            joint.main.OUT / 'predictions' / f'{month}_s42.parquet',
            rounded.original.airport.OUT / 'predictions' / f'{month}_s42.parquet'])
    value = dict(design=DESIGN, sources=[art(p) for p in paths])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'identity.json'
    if path.exists():
        assert read(path) == value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(path, value)


def score():
    guard()
    assert not (OUT / 'result.json').exists(), 'Preserve completed score'
    neural = joint.neural
    replay = neural.cat.r20.physical.replay
    sc, rome, _, trees = replay.baseline.baseline()
    identity = read(neural.single.MANIFEST)
    prior = pd.read_parquet(neural.cat.OUT / 'scored.parquet').set_index('mvt_id')
    serialized = pd.read_parquet(rounded.OUT / 'scored.parquet').set_index('mvt_id')
    parent_saved = pd.read_parquet(parent.OUT / 'scored.parquet').set_index('mvt_id')
    main_saved = pd.read_parquet(main.OUT / 'scored.parquet').set_index('mvt_id')
    selected = pd.read_parquet(quality.OUT / 'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(daily.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    inputs = daily.rows().drop(columns='y').set_index('mvt_id')
    source, parameters = daily.public(), read(daily.OUT / 'parameters.json')
    matches = pd.read_parquet(daily.constraints.exclusion.OUT / 'matches2025.parquet')
    frames, cells = [], []
    for month in daily.base.FOLDS:
        d = sc.d[month]; ids = d['ids']
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        physical = np.mean([replay.values(verify(neural.completed(month, k, identity)['prediction']),
            ids, 'base') for k in range(3)], axis=0)
        old_main = replay.values(joint.main.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        new_main = replay.values(verify(main.completed(month)['prediction']), ids, 'pred')
        airport = replay.values(rounded.original.airport.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        new_cat = replay.values(verify(cv.completed(month)['prediction']), ids, 'pred')
        probability = .5 * prior.loc[ids, 'newcat'].to_numpy() + .5 * rome[month]
        permitted = matches[matches.donor_month.isin(daily.base.FOLDS[month] + [month])]
        outputs = {}
        for arm, (use_main, use_cat) in ARMS.items():
            ordinary = .71 * .5 * (calls + physical) + .29 * np.mean([
                new_main if use_main else old_main, airport,
                new_cat if use_cat else trees['catboost'][42][month]], axis=0)
            direct = sc.final(month, ordinary, probability)
            start = serialized.loc[ids, 'candidate_boundary'].to_numpy() + direct - prior.loc[ids, 'control'].to_numpy()
            refined = daily.constraints.matched_bound(raw.loc[ids].reset_index(), start)
            refined, _ = daily.constraints.exclusion.apply(ids, np.maximum(refined, 0), permitted)
            refined = np.rint(np.maximum(refined, 0))
            updated, cell, shift = daily.apply(inputs.loc[ids].reset_index(), refined, source, parameters[month])
            gated = np.where(selected.loc[ids, 'quality_eligible'], updated, refined)
            final = daily.constrain(raw.loc[ids].reset_index(), ids,
                pd.Series(month, index=range(len(ids))), gated, matches)
            if arm in ['refined', 'full_cat']:
                old_arm = 'candidate' if use_cat else 'control'
                for suffix, value in [('_refined', refined), ('_shift', shift), ('', final)]:
                    np.testing.assert_array_equal(value, parent_saved.loc[ids, old_arm + suffix])
            elif arm == 'daily_main':
                np.testing.assert_array_equal(refined, main_saved.loc[ids, 'refined_daily'])
            outputs[arm + '_before'], outputs[arm + '_shift'], outputs[arm] = refined, shift, final
            cells.append(cell.assign(fold=month, arm=arm))
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'], **outputs)))
    f = pd.concat(frames, ignore_index=True)
    assert len(f) == 344419 and f.mvt_id.is_unique
    names = [a + suffix for a in ARMS for suffix in ['_before', '']]
    scores = {m: {c: float(np.sqrt(np.mean((g[c] - g.y)**2))) for c in names} for m, g in f.groupby('month')}
    paired = {key: cv.parent.ts.paired_bootstrap(f.day, f.month,
        (f[a] - f.y)**2, (f[b] - f.y)**2) for key, a, b in [
            ('primary', 'full_cat', 'combined'), ('daily_main_alone', 'refined', 'daily_main'),
            ('before_quality', 'full_cat_before', 'combined_before')]}
    f.to_parquet(OUT / 'scored.parquet', index=False)
    pd.concat(cells, ignore_index=True).to_parquet(OUT / 'cells.parquet', index=False)
    result = dict(design=DESIGN, identity=art(OUT / 'identity.json'), scores=scores,
        cv={c: float(np.mean([v[c] for v in scores.values()])) for c in names}, paired=paired,
        all_parent_controls_shifts_and_main_refined_exact=True,
        artifacts=[art(OUT / 'scored.parquet'), art(OUT / 'cells.parquet')])
    write_json(OUT / 'result.json', result)
    print(result, flush=True)


def audit():
    guard()
    assert not (OUT / 'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT / 'result.json')
    for a in result['artifacts']:
        verify(a)
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('c', OUT / 'scored.parquet'), ('p', parent.OUT / 'scored.parquet'),
            ('m', main.OUT / 'scored.parquet'), ('r', daily.base.OUT / 'rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, u, errors = con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.month is distinct from r.month or c.day is distinct from r.day
            or c.refined is distinct from p.control or c.full_cat is distinct from p.candidate
            or c.refined_before is distinct from p.control_refined or c.full_cat_before is distinct from p.candidate_refined
            or c.refined_shift is distinct from p.control_shift or c.full_cat_shift is distinct from p.candidate_shift
            or c.daily_main_before is distinct from m.refined_daily then 1 else 0 end)
            from c join p using(mvt_id) join m using(mvt_id) join r using(mvt_id)''').fetchone()
        assert n == u == 344419 and errors == 0
        for col in result['cv']:
            values = con.execute(f'select month,sqrt(avg(pow({col}-y,2))) from c group by month').fetchall()
            for m, value in values:
                np.testing.assert_allclose(value, result['scores'][m][col], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in values]), result['cv'][col], atol=1e-8, rtol=0)
    f = pd.read_parquet(OUT / 'scored.parquet'); ids = f.mvt_id.to_numpy()
    raw = pd.read_parquet(daily.base.OUT / 'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    selected = pd.read_parquet(quality.OUT / 'scored.parquet').set_index('mvt_id').loc[ids]
    matches = pd.read_parquet(daily.constraints.exclusion.OUT / 'matches2025.parquet')
    source, parameters = daily.public(), read(daily.OUT / 'parameters.json')
    inputs = daily.rows().drop(columns='y').set_index('mvt_id')
    for arm in ARMS:
        for month in daily.base.FOLDS:
            ix = np.flatnonzero(f.month.eq(month))
            _, _, shift = daily.apply(inputs.loc[ids[ix]].reset_index(), f[arm + '_before'].to_numpy()[ix], source, parameters[month])
            np.testing.assert_array_equal(shift, f[arm + '_shift'].to_numpy()[ix])
        pred = f[arm + '_before'].to_numpy() + np.where(selected.quality_eligible, f[arm + '_shift'], 0.)
        np.testing.assert_array_equal(daily.constrain(raw, ids, f.month, pred, matches), f[arm])
    write_json(OUT / 'audit.json', dict(status='pass', rows=n,
        identity=art(OUT / 'identity.json'), result=art(OUT / 'result.json'),
        independent_sql_truth_calendar_controls_scores=True,
        all_daily_shifts_gates_and_constraints_replayed=True, source=art(__file__)))
    print('All344419 rows: controls, shifts, constraints and independent SQL scores pass', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'score', 'audit'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
