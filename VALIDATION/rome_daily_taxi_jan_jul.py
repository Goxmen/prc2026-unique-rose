"""Fixed ten-input daily taxi-out screen on the selected Rome CatBoost pool.

The three seeds, fold-specific rounds, rows, targets and half R20 mixture are
unchanged. Compare serialized v76 and the selected daily-quality refinement.
"""
import argparse
import gc
from pathlib import Path
import platform
import time

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_daily_taxi_features as features
import public_daily_taxi_quality as quality
import public_stack_cv as joint
from records import write_json

cat = joint.neural.cat
r20 = cat.r20
daily = quality.original
rounded = daily.parent
ROOT = cat.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_rome_daily_taxi'
MANIFEST = OUT / 'identity.json'
art, read, verify = joint.art, joint.read, joint.verify
DESIGN = dict(parent='production-rounded v76 public Cat3/original R20',
    change='ten public daily runway/airport taxi inputs in the CatBoost classifier',
    added=features.FEATURES, old_columns=80, columns=90,
    seeds=cat.parent.SEEDS, rounds=cat.parent.ROUNDS,
    preserved='original classifier rows, targets, parameters, rounds and equal Cat3/R20 mixture',
    primary='production-rounded v76 substitution',
    secondary='selected airport/t5/full-monthly/daily-quality recipe; recompute fixed daily shifts',
    no_weight_or_parameter_search=True, no_new_stopping=True,
    limitation='reused Jan/Jul development; public taxi source is a different cohort',
    no_production=True, no_upload=True)


def guard():
    for path in [features.OUT/'audit.json', cat.OUT/'audit.json',
                 rounded.OUT/'audit.json', quality.OUT/'audit.json']:
        assert read(path)['status'] == 'pass'
    paths = [Path(__file__), Path(features.__file__), Path(cat.__file__), Path(cat.parent.__file__),
        Path(r20.__file__), Path(joint.__file__), Path(daily.__file__), Path(quality.__file__),
        Path(rounded.__file__), Path(daily.constraints.__file__), Path(daily.constraints.exclusion.__file__),
        features.OUT/'manifest.json', features.OUT/'audit.json', features.OUT/'features2025.parquet',
        cat.OUT/'scored.parquet', cat.OUT/'result.json', cat.OUT/'audit.json',
        rounded.OUT/'scored.parquet', rounded.OUT/'audit.json', joint.OUT/'scored.parquet',
        rounded.original.airport.OUT/'scored.parquet', quality.OUT/'scored.parquet',
        quality.OUT/'result.json', quality.OUT/'audit.json', daily.OUT/'parameters.json',
        daily.CACHE, daily.SOURCE/'daily_runway.parquet', daily.base.OUT/'rows2025.parquet',
        daily.constraints.exclusion.OUT/'matches2025.parquet']
    for month in cat.parent.ROUNDS:
        paths.append(r20.OUT/'inputs'/f'{month}.pkl')
        for seed in cat.parent.SEEDS:
            paths.extend([cat.OUT/'predictions'/f'{month}_s{seed}.json',
                          cat.OUT/'predictions'/f'{month}_s{seed}.parquet',
                          cat.OUT/'models'/f'{month}_s{seed}.cbm'])
    value = dict(design=DESIGN, sources=[art(p) for p in paths])
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('**/*.cbm')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST, value)
    return value


def prepare():
    guard()
    extra = pd.read_parquet(features.OUT/'features2025.parquet').set_index('mvt_id')
    truth = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    (OUT/'inputs').mkdir(exist_ok=True)
    checks = []
    for month in cat.parent.ROUNDS:
        path = OUT/'inputs'/f'{month}.pkl'
        if path.exists():
            record = read(path.with_suffix('.json'))
            assert record['identity'] == art(MANIFEST) and record['prepared'] == art(path)
            checks.append(record)
            continue
        d = pd.read_pickle(r20.OUT/'inputs'/f'{month}.pkl')
        original, old = cat.parent.validated_parent(month)
        for field in ['target', 'train_ids', 'ids', 'use']:
            np.testing.assert_array_equal(d[field], old[field])
        assert set(truth.loc[d['train_ids'], 'month']) == set(daily.base.FOLDS[month])
        assert set(truth.loc[d['ids'], 'month']) == {month}
        assert not np.intersect1d(d['train_ids'], d['ids']).size
        assert truth.loc[d['train_ids'], 'airport'].eq('LIRF').all()
        assert truth.loc[d['ids'][d['use']], 'airport'].eq('LIRF').all()
        for part, ids in [('A', d['train_ids']), ('B', d['ids'][d['use']])]:
            before = d[part]
            assert len(before.columns) == 80
            pd.testing.assert_frame_equal(before[original['columns']], old[part][original['columns']])
            d[part] = pd.concat([before, extra.loc[ids, features.FEATURES].set_axis(before.index)], axis=1)
            pd.testing.assert_frame_equal(d[part][before.columns], before, check_exact=True)
            assert len(d[part].columns) == 90
        pd.to_pickle(d, path)
        record = dict(identity=art(MANIFEST), prepared=art(path), rows=len(d['A']),
            held_rome_rows=len(d['B']), held_rows=len(d['ids']),
            months=sorted(set(truth.loc[d['train_ids'], 'month'])),
            original80_exact=True, original_ids_targets_use_exact=True,
            columns=list(d['A']), categoricals=original['categoricals'])
        write_json(path.with_suffix('.json'), record)
        checks.append(record)
        del d, old
        gc.collect()
    write_json(OUT/'preflight.json', dict(status='pass', identity=art(MANIFEST), checks=checks))
    print([(x['rows'], x['held_rome_rows']) for x in checks], flush=True)


def completed(month, seed):
    path = OUT/'predictions'/f'{month}_s{seed}.parquet'
    record_path = path.with_suffix('.json')
    if not record_path.exists():
        return None
    record = read(record_path)
    assert record['identity'] == art(MANIFEST)
    verify(record['prediction']); verify(record['model'])
    assert record['input'] == art(OUT/'inputs'/f'{month}.pkl')
    return record


def fit():
    from catboost import CatBoostClassifier, Pool
    guard()
    assert read(OUT/'preflight.json')['identity'] == art(MANIFEST)
    for directory in ['models', 'predictions']:
        (OUT/directory).mkdir(exist_ok=True)
    for month in cat.parent.ROUNDS:
        d = pd.read_pickle(OUT/'inputs'/f'{month}.pkl')
        control = read(cat.OUT/'predictions'/f'{month}_s42.json')
        cats = control['categoricals']
        X, V = [cat.parent.parent.cbj.frame(d[p], cats) for p in ['A', 'B']]
        assert list(X) == control['columns']+features.FEATURES
        for seed in cat.parent.SEEDS:
            if completed(month, seed):
                continue
            # Every retained control replays on all Rome held rows before the new fit.
            old = CatBoostClassifier()
            old.load_model(str(cat.OUT/'models'/f'{month}_s{seed}.cbm'))
            expected = pd.read_parquet(cat.OUT/'predictions'/f'{month}_s{seed}.parquet')
            np.testing.assert_array_equal(expected.mvt_id, d['ids'])
            np.testing.assert_array_equal(old.predict_proba(Pool(V[control['columns']], cat_features=cats),
                thread_count=2)[:, 1], expected.p.to_numpy()[d['use']])
            assert old.feature_names_ == control['columns']
            del old, expected
            params = dict(control['params'], random_seed=seed)
            mp = OUT/'models'/f'{month}_s{seed}.cbm'
            pp = OUT/'predictions'/f'{month}_s{seed}.parquet'
            intent = dict(identity=art(MANIFEST), input=art(OUT/'inputs'/f'{month}.pkl'),
                month=month, seed=seed, params=params, rounds=cat.parent.ROUNDS[month])
            ip = mp.with_suffix('.intent.json')
            if ip.exists():
                assert read(ip) == intent
            else:
                assert not mp.exists() and not pp.exists()
                write_json(ip, intent)
            started = time.monotonic()
            restored = CatBoostClassifier()
            if mp.exists():
                restored.load_model(str(mp))
            else:
                assert not pp.exists()
                model = CatBoostClassifier(iterations=cat.parent.ROUNDS[month], **params)
                model.fit(Pool(X, d['target'], cat_features=cats))
                model.save_model(str(mp))  # Preserve the completed fit before any diagnostics.
                restored.load_model(str(mp))
                np.testing.assert_array_equal(model.predict_proba(Pool(V, cat_features=cats))[:, 1],
                    restored.predict_proba(Pool(V, cat_features=cats), thread_count=2)[:, 1])
                del model
            assert restored.tree_count_ == cat.parent.ROUNDS[month] and restored.feature_names_ == list(X)
            prediction = np.zeros(len(d['ids']))
            prediction[d['use']] = restored.predict_proba(Pool(V, cat_features=cats), thread_count=2)[:, 1]
            assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
            f = pd.DataFrame(dict(mvt_id=d['ids'], p=prediction))
            if pp.exists():
                pd.testing.assert_frame_equal(pd.read_parquet(pp), f, check_exact=True)
            else:
                f.to_parquet(pp, index=False)
            write_json(pp.with_suffix('.json'), dict(**intent, model=art(mp), prediction=art(pp),
                original80_control_exact=True, all_held_reload_exact=True,
                columns=list(X), categoricals=cats, machine=platform.node(),
                minutes=(time.monotonic()-started)/60))
            del restored
            gc.collect()
        del X, V, d
        gc.collect()


def score():
    guard()
    assert not (OUT/'result.json').exists(), 'Preserve completed score'
    neural = joint.neural
    replay = r20.physical.replay
    sc, rome, _, trees = replay.baseline.baseline()
    identity = neural.guard()
    prior = pd.read_parquet(cat.OUT/'scored.parquet').set_index('mvt_id')
    selected = pd.read_parquet(joint.OUT/'scored.parquet').set_index('mvt_id')
    refined = pd.read_parquet(rounded.original.airport.OUT/'scored.parquet').set_index('mvt_id')
    serialized = pd.read_parquet(rounded.OUT/'scored.parquet').set_index('mvt_id')
    quality_control = pd.read_parquet(quality.OUT/'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id')
    inputs = daily.rows().drop(columns='y').set_index('mvt_id')
    source, parameters = daily.public(), read(daily.OUT/'parameters.json')
    constraints = daily.constraints
    matches = pd.read_parquet(constraints.exclusion.OUT/'matches2025.parquet')
    frames, cells = [], []
    for month in cat.parent.ROUNDS:
        d = sc.d[month]; ids = d['ids']
        probabilities = []
        for seed in cat.parent.SEEDS:
            record = completed(month, seed)
            assert record is not None
            probabilities.append(replay.values(verify(record['prediction']), ids, 'p'))
        oldcat = prior.loc[ids, 'newcat'].to_numpy()
        newcat = np.mean(probabilities, axis=0)
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        physical = np.mean([replay.values(verify(neural.completed(month, k, identity)['prediction']), ids, 'base')
                            for k in range(3)], axis=0)
        main = replay.values(joint.main.OUT/'predictions'/f'{month}_s42.parquet', ids, 'pred')
        new_airport = replay.values(rounded.original.airport.OUT/'predictions'/f'{month}_s42.parquet', ids, 'pred')
        permitted = matches[matches.donor_month.isin(daily.base.FOLDS[month]+[month])]
        outputs = {}
        for arm, airport, boundary, expected in [
            ('v76', trees['airport'][42][month], 'control', selected.loc[ids, 'combined']),
            ('refined', new_airport, 'candidate', refined.loc[ids, 'candidate'])]:
            ordinary = .71*.5*(calls+physical)+.29*np.mean([main, airport, trees['catboost'][42][month]], axis=0)
            for suffix, cp in [('', oldcat), ('_dailycat', newcat)]:
                direct = sc.final(month, ordinary, .5*cp+.5*rome[month])
                if not suffix:
                    np.testing.assert_array_equal(direct, expected)
                start = serialized.loc[ids, boundary+'_boundary'].to_numpy()+direct-prior.loc[ids, 'control'].to_numpy()
                pred = constraints.matched_bound(raw.loc[ids].reset_index(), start)
                pred, _ = constraints.exclusion.apply(ids, np.maximum(pred, 0), permitted)
                pred = np.rint(np.maximum(pred, 0))
                if not suffix:
                    np.testing.assert_array_equal(pred, serialized.loc[ids, boundary])
                outputs[arm+suffix] = pred
                if arm == 'refined':
                    q, c, shift = daily.apply(inputs.loc[ids].reset_index(), pred, source, parameters[month])
                    gated = np.where(quality_control.loc[ids, 'quality_eligible'], q, pred)
                    bounded = daily.constrain(raw.loc[ids].reset_index(), ids, pd.Series(month, index=range(len(ids))), gated, matches)
                    outputs['quality'+suffix] = bounded
                    outputs['quality_shift'+suffix] = shift
                    cells.append(c.assign(fold=month, arm='quality'+suffix))
                    if not suffix:
                        np.testing.assert_array_equal(bounded, quality_control.loc[ids, 'refined_quality'])
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'],
            airport=raw.loc[ids, 'airport'].to_numpy(), oldcat=oldcat, newcat=newcat, **outputs)))
    f = pd.concat(frames, ignore_index=True)
    assert len(f) == 344419 and f.mvt_id.is_unique
    names = [n+s for n in ['v76', 'refined', 'quality'] for s in ['', '_dailycat']]
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in names} for m, g in f.groupby('month')}
    paired = {n: cat.parent.shared.ts.paired_bootstrap(f.day, f.month, (f[n]-f.y)**2, (f[n+'_dailycat']-f.y)**2)
              for n in ['v76', 'refined', 'quality']}
    f.to_parquet(OUT/'scored.parquet', index=False)
    pd.concat(cells, ignore_index=True).to_parquet(OUT/'cells.parquet', index=False)
    result = dict(design=DESIGN, identity=art(MANIFEST), prediction=art(OUT/'scored.parquet'),
        cells=art(OUT/'cells.parquet'), scores=scores, paired=paired,
        cv={n: float(np.mean([v[n] for v in scores.values()])) for n in names},
        serialized_unserialized_and_quality_controls_exact=True)
    write_json(OUT/'result.json', result)
    print(result, flush=True)


def audit():
    from catboost import CatBoostClassifier, Pool
    guard()
    assert not (OUT/'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT/'result.json')
    verify(result['prediction']); verify(result['cells'])
    reports = []
    extra = pd.read_parquet(features.OUT/'features2025.parquet').set_index('mvt_id')
    for month in cat.parent.ROUNDS:
        d = pd.read_pickle(OUT/'inputs'/f'{month}.pkl')
        old = pd.read_pickle(r20.OUT/'inputs'/f'{month}.pkl')
        for key in ['target', 'train_ids', 'ids', 'use']:
            np.testing.assert_array_equal(d[key], old[key])
        for part, ids in [('A', d['train_ids']), ('B', d['ids'][d['use']])]:
            pd.testing.assert_frame_equal(d[part][old[part].columns], old[part], check_exact=True)
            pd.testing.assert_frame_equal(d[part][features.FEATURES], extra.loc[ids, features.FEATURES].set_axis(d[part].index), check_exact=True)
        for seed in cat.parent.SEEDS:
            record = completed(month, seed)
            model = CatBoostClassifier(); model.load_model(str(verify(record['model'])))
            V = cat.parent.parent.cbj.frame(d['B'], record['categoricals'])
            p = model.predict_proba(Pool(V, cat_features=record['categoricals']), thread_count=2)[:, 1]
            saved = pd.read_parquet(verify(record['prediction']))
            np.testing.assert_array_equal(saved.mvt_id, d['ids'])
            np.testing.assert_array_equal(saved.p.to_numpy()[d['use']], p)
            assert saved.p.to_numpy()[~d['use']].sum() == 0
            assert model.feature_names_ == record['columns'] == list(d['B'])
            reports.append(dict(month=month, seed=seed, held_rome_rows=len(p), all_reload_exact=True))
        del d, old, V, model
        gc.collect()
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('c', OUT/'scored.parquet'), ('r', daily.base.OUT/'rows2025.parquet'),
                           ('p', rounded.OUT/'scored.parquet'), ('q', quality.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, u, errors = con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.day is distinct from r.day or c.month is distinct from r.month
            or c.airport is distinct from r.airport or c.v76 is distinct from p.control
            or c.refined is distinct from p.candidate or c.quality is distinct from q.refined_quality
            or (c.airport!='LIRF' and (c.v76!=c.v76_dailycat or c.refined!=c.refined_dailycat
                                      or c.quality!=c.quality_dailycat))
            then 1 else 0 end) from c join r using(mvt_id) join p using(mvt_id) join q using(mvt_id)''').fetchone()
        assert n == u == 344419 and errors == 0
        for column in result['cv']:
            values = con.execute(f'select month,sqrt(avg(pow({column}-y,2))) from c group by month').fetchall()
            for month, score_value in values:
                np.testing.assert_allclose(score_value, result['scores'][month][column], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in values]), result['cv'][column], atol=1e-8, rtol=0)
    f = pd.read_parquet(OUT/'scored.parquet')
    ids = f.mvt_id.to_numpy()
    raw = pd.read_parquet(daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    q = pd.read_parquet(quality.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    matches = pd.read_parquet(daily.constraints.exclusion.OUT/'matches2025.parquet')
    for suffix in ['', '_dailycat']:
        start = f['refined'+suffix].to_numpy()+np.where(q.quality_eligible, f['quality_shift'+suffix], 0)
        np.testing.assert_array_equal(daily.constrain(raw, ids, f.month, start, matches), f['quality'+suffix])
    audit_result = dict(status='pass', identity=art(MANIFEST), result=art(OUT/'result.json'),
        models=reports, rows=n, original80_and_new10_inputs_exact=True,
        independent_sql_truth_calendar_controls_scores_non_rome=True,
        all_quality_gates_and_constraints_replayed=True, source=art(__file__))
    write_json(OUT/'audit.json', audit_result)
    print(audit_result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'prepare', 'fit', 'score', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(2):
        globals()[args.stage]()
