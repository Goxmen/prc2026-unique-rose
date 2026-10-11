"""Check the passing full-input CatBoost inside selected daily refinements."""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import catboost_full_inputs as cv
import public_daily_taxi_quality as quality
from records import write_json

daily, joint = quality.original, cv.joint
rounded = daily.parent
ROOT = cv.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_catboost_refinements_cv'
art, read, verify = cv.art, cv.read, cv.verify
DESIGN = dict(parent='selected airport/t5/full-monthly/daily-quality refinement',
    change='replace only ordinary CatBoost with retained full174-input K07/10000 seed42',
    method='rebuild ordinary blend and both serialized boundaries; recompute unchanged daily source correction',
    comparison='old/new CatBoost before and after fixed daily-quality correction',
    no_fits=True, no_weight_or_parameter_search=True, no_upload=True,
    limitation='reused Jan/Jul development; old CatBoost fitted on laptop, new on desktop')


def guard():
    assert read(cv.OUT/'audit.json')['status'] == 'pass'
    assert read(cv.OUT/'result.json')['paired']['candidate_transfer_vs_control_transfer']['decision'] == 'improvement'
    assert read(quality.OUT/'audit.json')['status'] == 'pass'
    paths = [Path(__file__), Path(cv.__file__), Path(daily.__file__), Path(quality.__file__),
        Path(joint.__file__), Path(daily.constraints.__file__), Path(daily.constraints.exclusion.__file__),
        cv.OUT/'identity.json', cv.OUT/'result.json', cv.OUT/'audit.json',
        quality.OUT/'scored.parquet', quality.OUT/'result.json', quality.OUT/'audit.json',
        daily.OUT/'parameters.json', daily.CACHE, daily.SOURCE/'daily_runway.parquet',
        daily.base.OUT/'rows2025.parquet', daily.constraints.exclusion.OUT/'matches2025.parquet',
        rounded.OUT/'scored.parquet', rounded.original.airport.OUT/'scored.parquet',
        joint.neural.cat.OUT/'scored.parquet']
    for month in daily.base.FOLDS:
        paths.extend([cv.paths(month)[1], cv.paths(month)[2],
            joint.main.OUT/'predictions'/f'{month}_s42.parquet',
            rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet'])
    value = dict(design=DESIGN, sources=[art(p) for p in paths])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'identity.json'
    if path.exists():
        assert read(path) == value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(path, value)
    return value


def score():
    guard()
    assert not (OUT/'result.json').exists(), 'Preserve completed score'
    neural = joint.neural
    replay = neural.cat.r20.physical.replay
    sc, rome, _, trees = replay.baseline.baseline()
    identity = read(neural.single.MANIFEST)
    prior = pd.read_parquet(neural.cat.OUT/'scored.parquet').set_index('mvt_id')
    serialized = pd.read_parquet(rounded.OUT/'scored.parquet').set_index('mvt_id')
    direct_saved = pd.read_parquet(rounded.original.airport.OUT/'scored.parquet').set_index('mvt_id')
    selected = pd.read_parquet(quality.OUT/'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    inputs = daily.rows().drop(columns='y').set_index('mvt_id')
    source, parameters = daily.public(), read(daily.OUT/'parameters.json')
    matches = pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    frames, cells = [], []
    for month in daily.base.FOLDS:
        d = sc.d[month]; ids = d['ids']
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        physical = np.mean([replay.values(verify(neural.completed(month, k, identity)['prediction']), ids, 'base') for k in range(3)], axis=0)
        main = replay.values(joint.main.OUT/'predictions'/f'{month}_s42.parquet', ids, 'pred')
        airport = replay.values(rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet', ids, 'pred')
        new_cat = replay.values(verify(cv.completed(month)['prediction']), ids, 'pred')
        probability = .5*prior.loc[ids, 'newcat'].to_numpy()+.5*rome[month]
        permitted = matches[matches.donor_month.isin(daily.base.FOLDS[month]+[month])]
        outputs = {}
        for arm, catboost in [('control', trees['catboost'][42][month]), ('candidate', new_cat)]:
            ordinary = .71*.5*(calls+physical)+.29*np.mean([main, airport, catboost], axis=0)
            direct = sc.final(month, ordinary, probability)
            if arm == 'control':
                np.testing.assert_array_equal(direct, direct_saved.loc[ids, 'candidate'])
            start = serialized.loc[ids, 'candidate_boundary'].to_numpy()+direct-prior.loc[ids, 'control'].to_numpy()
            refined = daily.constraints.matched_bound(raw.loc[ids].reset_index(), start)
            refined, _ = daily.constraints.exclusion.apply(ids, np.maximum(refined, 0), permitted)
            refined = np.rint(np.maximum(refined, 0))
            if arm == 'control':
                np.testing.assert_array_equal(refined, serialized.loc[ids, 'candidate'])
            updated, c, shift = daily.apply(inputs.loc[ids].reset_index(), refined, source, parameters[month])
            gated = np.where(selected.loc[ids, 'quality_eligible'], updated, refined)
            final = daily.constrain(raw.loc[ids].reset_index(), ids, pd.Series(month, index=range(len(ids))), gated, matches)
            if arm == 'control':
                np.testing.assert_array_equal(final, selected.loc[ids, 'refined_quality'])
            outputs[arm+'_refined'], outputs[arm+'_shift'], outputs[arm] = refined, shift, final
            cells.append(c.assign(fold=month, arm=arm))
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'], **outputs)))
    f = pd.concat(frames, ignore_index=True)
    names = ['control_refined', 'candidate_refined', 'control', 'candidate']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in names} for m, g in f.groupby('month')}
    paired = {suffix or 'quality': cv.parent.ts.paired_bootstrap(f.day, f.month,
        (f['control'+suffix]-f.y)**2, (f['candidate'+suffix]-f.y)**2) for suffix in ['_refined', '']}
    f.to_parquet(OUT/'scored.parquet', index=False)
    pd.concat(cells, ignore_index=True).to_parquet(OUT/'cells.parquet', index=False)
    result = dict(design=DESIGN, identity=art(OUT/'identity.json'), scores=scores,
        cv={c: float(np.mean([v[c] for v in scores.values()])) for c in names}, paired=paired,
        all_direct_serialized_quality_controls_exact=True,
        artifacts=[art(OUT/'scored.parquet'), art(OUT/'cells.parquet')])
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
                           ('r', daily.base.OUT/'rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, u, errors = con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.month is distinct from r.month or c.day is distinct from r.day
            or c.control is distinct from p.refined_quality or c.control_refined is distinct from p.refined
            or c.control_shift is distinct from p.refined_shift
            then 1 else 0 end) from c join p using(mvt_id) join r using(mvt_id)''').fetchone()
        assert n == u == 344419 and errors == 0
        for col in result['cv']:
            values = con.execute(f'select month,sqrt(avg(pow({col}-y,2))) from c group by month').fetchall()
            for m, v in values:
                np.testing.assert_allclose(v, result['scores'][m][col], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in values]), result['cv'][col], atol=1e-8, rtol=0)
    f = pd.read_parquet(OUT/'scored.parquet'); ids = f.mvt_id.to_numpy()
    raw = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    selected = pd.read_parquet(quality.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    matches = pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    source, parameters = daily.public(), read(daily.OUT/'parameters.json')
    inputs = daily.rows().drop(columns='y').set_index('mvt_id')
    for arm in ['control', 'candidate']:
        for month in daily.base.FOLDS:
            ix = np.flatnonzero(f.month.eq(month))
            _, _, shift = daily.apply(inputs.loc[ids[ix]].reset_index(), f[arm+'_refined'].to_numpy()[ix], source, parameters[month])
            np.testing.assert_array_equal(shift, f[arm+'_shift'].to_numpy()[ix])
        pred = f[arm+'_refined'].to_numpy()+np.where(selected.quality_eligible, f[arm+'_shift'], 0.)
        np.testing.assert_array_equal(daily.constrain(raw, ids, f.month, pred, matches), f[arm])
    report = dict(status='pass', rows=n, identity=art(OUT/'identity.json'), result=art(OUT/'result.json'),
        independent_sql_truth_calendar_controls_scores=True, all_daily_shifts_gates_and_constraints_replayed=True,
        source=art(__file__))
    write_json(OUT/'audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'score', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
