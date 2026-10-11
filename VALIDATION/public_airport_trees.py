"""Fixed public-daily-input screen for the ten retained airport LightGBMs.

Keep P04/5000 rounds, seed42, original156 inputs and ordinary training rows.
Append the already audited twelve public inputs. Score inside the complete v76
stack. Each airport checkpoint is retained and reloaded before resuming a run.
This runner does not fit production models, select weights or upload anything.
"""
import argparse
import gc
from pathlib import Path
import platform
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_stack_cv as joint
from records import sha256, write_json

ROOT = joint.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_public_airport_trees'
MANIFEST = OUT / 'identity.json'
parent = joint.main.parent
features = joint.main.features
art, verify, read = joint.art, joint.verify, joint.read
DESIGN = dict(parent='v76', change='12 public daily inputs in every airport LightGBM',
    original_inputs=156, added_inputs=features.FEATURES, seed=42, trial='P04',
    rounds=5000, threads=4, original_training_rows=True, original_labels=True,
    no_callsign_inputs=True, no_new_stopping=True, no_parameter_search=True,
    all_other_parts='selected public main/physical/Rome and original callsign15/CatBoost',
    comparison='original saved airport models versus public airport models in selected stack',
    caveat='Reused Jan/Jul development; historical and new tree thread counts can differ',
    no_production=True, no_upload=True)


def params():
    winner = parent.hpo.winner('airport')
    assert winner['trial'] == 'P04' and winner['checkpoint'] == 'c5000'
    spec = parent.hpo.trial_spec('airport', 'P04')
    assert spec['rounds'] == 5000 and spec['learning_rate'] == .02
    value = parent.ts.lgb_params(dict(parent.hpo.V44, **spec['overrides']), 42, .02)
    value['num_threads'] = DESIGN['threads']
    return value


def guard():
    assert read(joint.OUT / 'audit.json')['status'] == 'pass'
    assert read(features.OUT / 'audit.json')['status'] == 'pass'
    paths = [Path(__file__), Path(joint.__file__), Path(features.__file__),
        Path(parent.hpo.__file__), Path(parent.ts.__file__), Path(parent.modeling.__file__),
        joint.OUT / 'scored.parquet', joint.OUT / 'audit.json',
        features.OUT / 'features2025.parquet', features.OUT / 'audit.json',
        features.base.OUT / 'rows2025.parquet',
        parent.hpo.pred_path('airport', 'P04', 42), parent.hpo.record_path('airport')]
    paths.extend(parent.input_cache.input_path(month) for month in parent.ts.SPLITS)
    value = dict(design=DESIGN, params=params(), sources=[art(p) for p in paths])
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('**/*.txt')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST, value)
    return value


def member_paths(month, airport):
    stem = f'{month}_{airport}_s42'
    return OUT / 'models' / (stem + '.txt'), OUT / 'members' / (stem + '.parquet'), OUT / 'members' / (stem + '.json')


def completed(month, airport):
    mp, pp, rp = member_paths(month, airport)
    if not rp.exists():
        assert not mp.exists() and not pp.exists(), 'Unrecorded airport fit; inspect before resuming'
        return None
    record = read(rp)
    assert record['identity'] == art(MANIFEST)
    assert record['params'] == params() and record['rounds'] == 5000
    verify(record['model']); verify(record['prediction'])
    return record


def inputs(month, public, truth, held_only=False):
    source = parent.input_cache.input_path(month)
    data = pd.read_pickle(source)
    names = data['baseline']
    assert len(names) == len(set(names)) == 156
    assert not set(names).intersection(features.FEATURES)
    train_ids, held_ids = np.asarray(data['train_ids']), np.asarray(data['ids'])
    assert not np.intersect1d(train_ids, held_ids).size
    assert set(truth.loc[train_ids, 'month']) == set(parent.ts.FOLDS[month])
    assert set(truth.loc[held_ids, 'month']) == {month}
    np.testing.assert_array_equal(data['y'], truth.loc[train_ids, 'y'])
    result = dict(train_ids=train_ids, held_ids=held_ids, y=data['y'])
    for part, ids in [('X', train_ids), ('V', held_ids)]:
        if held_only and part == 'X':
            continue
        original = data[part].loc[:, names]
        added = public.loc[ids, features.FEATURES].set_axis(original.index)
        result[part] = pd.concat([original, added], axis=1)
        assert list(result[part]) == list(names) + features.FEATURES
        pd.testing.assert_frame_equal(result[part].loc[:, names], original, check_exact=True)
    return result


def fit():
    identity = guard()
    public = pd.read_parquet(features.OUT / 'features2025.parquet').set_index('mvt_id')
    truth = pd.read_parquet(features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    for directory in ['models', 'members', 'predictions']:
        (OUT / directory).mkdir(exist_ok=True)
    for month in parent.ts.SPLITS:
        d = inputs(month, public, truth)
        X, V = d['X'], d['V']
        airports = truth.loc[d['train_ids'], 'airport'].to_numpy()
        v_airports = truth.loc[d['held_ids'], 'airport'].to_numpy()
        assert len(set(airports)) == 10 and set(v_airports) <= set(airports)
        cats = parent.ideas.signals._categorical(X.columns) + [c for c in parent.ideas.ef.CATEGORICAL if c in X]
        assembled = np.full(len(V), np.nan, dtype=np.float32)
        members = []
        for airport in sorted(set(airports)):
            train, held = airports == airport, v_airports == airport
            mp, pp, rp = member_paths(month, airport)
            record = completed(month, airport)
            if record is None:
                started = time.monotonic()
                parent.log(f'{month} {airport}: {train.sum():,} rows,168 inputs,P04/5000')
                dataset = lgb.Dataset(X.loc[train], d['y'][train], categorical_feature=cats)
                model = lgb.train(identity['params'], dataset, num_boost_round=5000)
                prediction = parent.modeling.predict(model, V.loc[held], threads=4).astype(np.float32)
                assert np.isfinite(prediction).all()
                model.save_model(str(mp))
                restored = lgb.Booster(model_file=str(mp))
                assert restored.feature_name() == list(X)
                replay = np.maximum(restored.predict(V.loc[held], num_threads=4), 0).astype(np.float32)
                np.testing.assert_array_equal(prediction, replay)
                pd.DataFrame(dict(mvt_id=d['held_ids'][held], pred=prediction)).to_parquet(pp, index=False)
                write_json(rp, dict(identity=art(MANIFEST), model=art(mp), prediction=art(pp),
                    input=art(parent.input_cache.input_path(month)), params=identity['params'], rounds=5000,
                    airport=airport, month=month, columns=list(X), categoricals=cats,
                    train_rows=int(train.sum()), held_rows=int(held.sum()),
                    all_predictions_reloaded_exact=True, minutes=(time.monotonic()-started)/60,
                    machine=platform.node()))
                record = completed(month, airport)
                del model, restored, dataset
                gc.collect()
            frame = pd.read_parquet(verify(record['prediction']))
            np.testing.assert_array_equal(frame.mvt_id, d['held_ids'][held])
            assembled[held] = frame.pred
            members.append(art(rp))
        assert np.isfinite(assembled).all()
        pp = OUT / 'predictions' / f'{month}_s42.parquet'
        frame = pd.DataFrame(dict(mvt_id=d['held_ids'], pred=assembled))
        if pp.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(pp), frame, check_exact=True)
        else:
            frame.to_parquet(pp, index=False)
        write_json(pp.with_suffix('.json'), dict(identity=art(MANIFEST), prediction=art(pp), members=members))
        parent.log(f'{month}: all ten public airport models retained and reloaded')
        del d, X, V
        gc.collect()


def score():
    guard()
    assert not (OUT / 'result.json').exists(), 'Preserve completed score'
    neural, cat = joint.neural, joint.neural.cat
    replay = cat.r20.physical.replay
    sc, rome, _, trees = replay.baseline.baseline()
    identity = neural.guard()
    prior = pd.read_parquet(cat.OUT / 'scored.parquet').set_index('mvt_id')
    selected = pd.read_parquet(joint.OUT / 'scored.parquet').set_index('mvt_id')
    truth = pd.read_parquet(features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(cat.r20.constraints.exclusion.OUT / 'matches2025.parquet')
    frames = []
    for month in parent.ts.SPLITS:
        d = sc.d[month]; ids = d['ids']
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        physical = np.mean([replay.values(verify(neural.completed(month, k, identity)['prediction']), ids, 'base') for k in range(3)], axis=0)
        main = replay.values(joint.main.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        new = replay.values(OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        probability = .5*prior.loc[ids, 'newcat'].to_numpy()+.5*rome[month]
        permitted = matches[matches.donor_month.isin(parent.ts.FOLDS[month]+[month])]
        outputs = {}
        for name, airport in [('control', trees['airport'][42][month]), ('candidate', new)]:
            ordinary = .71*.5*(calls+physical)+.29*np.mean([main, airport, trees['catboost'][42][month]], axis=0)
            final = sc.final(month, ordinary, probability)
            transferred = cat.r20.constraints.matched_bound(truth.loc[ids].reset_index(),
                prior.loc[ids, 'v74'].to_numpy()+final-prior.loc[ids, 'control'].to_numpy())
            transferred, _ = cat.r20.constraints.exclusion.apply(ids, np.maximum(transferred, 0), permitted)
            outputs[name] = final; outputs[name+'_transfer'] = transferred
        np.testing.assert_array_equal(outputs['control'], selected.loc[ids, 'combined'])
        np.testing.assert_array_equal(outputs['control_transfer'], selected.loc[ids, 'combined_transfer'])
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'], **outputs)))
    frame = pd.concat(frames, ignore_index=True)
    assert len(frame) == 344419 and frame.mvt_id.is_unique and np.isfinite(frame[list(outputs)]).all().all()
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in outputs} for m, g in frame.groupby('month')}
    paired = {f'candidate{suffix}_vs_control{suffix}': parent.ts.paired_bootstrap(frame.day, frame.month,
        (frame['control'+suffix]-frame.y)**2, (frame['candidate'+suffix]-frame.y)**2) for suffix in ['', '_transfer']}
    frame.to_parquet(OUT / 'scored.parquet', index=False)
    result = dict(design=DESIGN, identity=art(MANIFEST), predictions=art(OUT / 'scored.parquet'),
        scores=scores, cv={c: float(np.mean([v[c] for v in scores.values()])) for c in outputs},
        paired=paired, selected_control_exact=True)
    write_json(OUT / 'result.json', result)
    print(dict(cv=result['cv'], paired=paired), flush=True)


def audit():
    guard()
    assert not (OUT / 'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT / 'result.json'); verify(result['predictions'])
    public = pd.read_parquet(features.OUT / 'features2025.parquet').set_index('mvt_id')
    truth = pd.read_parquet(features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    reloaded = []
    for month in parent.ts.SPLITS:
        d = inputs(month, public, truth, held_only=True)
        airport_values = truth.loc[d['held_ids'], 'airport'].to_numpy()
        combined = pd.read_parquet(OUT / 'predictions' / f'{month}_s42.parquet').set_index('mvt_id')
        for airport in sorted(set(airport_values)):
            record = completed(month, airport); assert record is not None
            model = lgb.Booster(model_file=str(verify(record['model'])))
            held = airport_values == airport
            assert model.feature_name() == record['columns'] == list(d['V'])
            prediction = np.maximum(model.predict(d['V'].loc[held], num_threads=2), 0).astype(np.float32)
            member = pd.read_parquet(verify(record['prediction']))
            np.testing.assert_array_equal(member.mvt_id, d['held_ids'][held])
            np.testing.assert_array_equal(prediction, member.pred)
            np.testing.assert_array_equal(prediction, combined.loc[member.mvt_id, 'pred'])
            reloaded.append(dict(month=month, airport=airport, rows=len(member), all_predictions_exact=True))
        del d, model
        gc.collect()
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('candidate', OUT/'scored.parquet'), ('truth', features.base.OUT/'rows2025.parquet'), ('selected', joint.OUT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n, u, errors = con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from t.y or c.month is distinct from t.month or c.day is distinct from t.day
            or c.control is distinct from s.combined or c.control_transfer is distinct from s.combined_transfer
            then 1 else 0 end) from candidate c join truth t using(mvt_id) join selected s using(mvt_id)''').fetchone()
        assert n == u == 344419 and errors == 0
        for name in result['cv']:
            values = con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from candidate group by month').fetchall()
            for month, value in values:
                np.testing.assert_allclose(value, result['scores'][month][name], atol=1e-8, rtol=0)
            np.testing.assert_allclose(np.mean([v for m, v in values]), result['cv'][name], atol=1e-8, rtol=0)
    report = dict(status='pass', identity=art(MANIFEST), result=art(OUT/'result.json'),
        all_twenty_models_reloaded=reloaded, independent_sql_truth_calendar_control_scores=True,
        all_rows=n, source=art(__file__))
    write_json(OUT/'audit.json', report); print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'fit', 'score', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(4):
        globals()[args.stage]()
