"""Fixed XGBoost fourth-tree screen on the selected physical/callsign stack.

Existing 162 inputs, original rows/labels and original fold boundaries. The
new member receives one quarter of the unchanged 0.29 tree share. No stopping,
weight or parameter search on held labels. Run after the allocated desktop job.
"""
import argparse
import gc
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from threadpoolctl import threadpool_limits

import callsign_lgb_jan_jul as parent
import rome_public_delay_jan_jul as selected
from records import write_json

ROOT = parent.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_xgboost_diversity'
ROUNDS = 4000
PARAMS = dict(objective='reg:squarederror', tree_method='hist', device='cuda',
    eta=.03, max_depth=8, min_child_weight=20, reg_lambda=100,
    subsample=.8, colsample_bytree=.8, max_bin=256, grow_policy='depthwise',
    max_cat_to_onehot=4, max_cat_threshold=16, seed=42, nthread=4)
DESIGN = dict(parent='v70 selected physical3/calls15 stack',
    change='add XGBoost as equally weighted fourth tree member',
    original_columns=162, rounds=ROUNDS, params=PARAMS,
    primary='quarter of existing tree share; all other parts unchanged',
    secondary='fixed final-prediction delta transferred to v74, restore bounds',
    development='reused Jan/Jul; no hyperparameter, stopping or weight search')


def artifact(path):
    return selected.features.base.artifact(path)


def convert(frame, cats, category_maps=None):
    """Encode categories against the training frame's retained vocabulary.

    The original LightGBM packet contains categorical strings, not numerical
    category IDs. Held unknowns map to NaN; numeric columns stay numerical.
    """
    names = list(frame)
    a = np.empty(frame.shape, dtype=np.float32)
    types = ['c' if name in cats else 'q' for name in names]
    maps = {} if category_maps is None else dict(category_maps)
    for j, (name, kind) in enumerate(zip(names, types)):
        if kind == 'c':
            if category_maps is None:
                maps[name] = frame[name].astype('category').cat.categories.tolist()
            values = pd.Categorical(frame[name], categories=maps[name]).codes.astype(np.float32)
            a[:, j] = values
            a[a[:, j] < 0, j] = np.nan
            v = a[np.isfinite(a[:, j]), j]
            np.testing.assert_array_equal(v, np.floor(v))
        else:
            a[:, j] = frame[name].to_numpy(dtype=np.float32, na_value=np.nan)
    a[~np.isfinite(a)] = np.nan
    return a, types, maps


def check(device='cpu'):
    frame = pd.DataFrame({'num': [0., 1., np.inf, -2., 3., 4., 5., 6.],
                          'cat': ['A', 'B', None, 'A', 'C', 'B', 'C', 'A']})
    a, ft, maps = convert(frame, ['cat'])
    assert np.isnan(a[2]).all() and ft == ['q', 'c']
    held = frame.copy()
    held.loc[0, 'cat'] = 'UNSEEN'
    held['cat'] = pd.Categorical(held['cat'], categories=['C', 'B', 'A', 'UNSEEN'])
    va, vft, _ = convert(held, ['cat'], maps)
    assert np.isnan(va[0, 1]) and vft == ft
    np.testing.assert_array_equal(va[1:], a[1:])
    y = np.array([1, 3, 2, 1, 5, 6, 7, 8], dtype=np.float32)
    d = xgb.QuantileDMatrix(a, y, feature_names=list(frame), feature_types=ft,
                          enable_categorical=True, max_bin=16, nthread=1)
    p = dict(PARAMS, device=device, min_child_weight=1, max_depth=2, max_bin=16)
    model = xgb.train(p, d, num_boost_round=5)
    first = model.predict(d)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'model.ubj'
        model.save_model(path)
        reload = xgb.Booster(model_file=path)
        reload.set_param({'device': device, 'nthread': 1})
        np.testing.assert_array_equal(first, reload.predict(d))
    assert np.isfinite(first).all() and model.feature_types == ft
    return dict(device=device, missing_codes_preserved=True,
                string_categories_aligned=True, unknown_categories_missing=True,
                held_category_order_invariant=True, categorical_types_exact=True,
                serialization_exact=True)


def guard():
    OUT.mkdir(parents=True, exist_ok=True)
    parent.guard()
    current = dict(design=DESIGN, source=artifact(__file__),
        helpers=[artifact(m.__file__) for m in [parent, selected]],
        xgboost=xgb.__version__, build=xgb.build_info(),
        inputs={m: artifact(parent.input_cache.input_path(m)) for m in parent.ts.SPLITS})
    path = OUT / 'identity.json'
    if path.exists():
        assert parent.ts.read(path) == current, 'Frozen XGBoost identity changed'
    else:
        write_json(path, current)
    return current


def fit():
    checks = [check('cpu'), check('cuda')]
    write_json(OUT / 'preflight.json', dict(checks=checks))
    (OUT / 'models').mkdir(exist_ok=True)
    (OUT / 'predictions').mkdir(exist_ok=True)
    for month in parent.ts.SPLITS:
        path = OUT / 'predictions' / f'{month}_s42.parquet'
        mp = OUT / 'models' / f'{month}_s42.ubj'
        if path.exists():
            rec = parent.ts.read(path.with_suffix('.json'))
            assert rec['identity'] == artifact(OUT / 'identity.json')
            assert rec['model'] == artifact(mp) and rec['prediction'] == artifact(path)
            continue
        assert not mp.exists(), 'Unrecorded model; inspect before refitting'
        source = parent.input_cache.input_path(month)
        d = pd.read_pickle(source)
        X, V = parent.append_pair(d, month)
        assert len(X.columns) == 162
        cats = parent.ideas.signals._categorical(X.columns) + [
            c for c in parent.ideas.ef.CATEGORICAL if c in X]
        names = list(X)
        xa, ft, category_maps = convert(X, cats)
        va, vft, _ = convert(V, cats, category_maps)
        assert ft == vft and names == list(V)
        y, ids = d['y'].copy(), d['ids'].copy()
        train_ids = d['train_ids'].copy()
        assert not np.intersect1d(ids, train_ids).size
        del d, X, V
        gc.collect()
        started = time.monotonic()
        parent.log(f'{month}: {len(y):,} rows, 162 inputs, XGBoost {ROUNDS} fixed rounds')
        train = xgb.QuantileDMatrix(xa, y, feature_names=names, feature_types=ft,
            enable_categorical=True, max_bin=PARAMS['max_bin'], nthread=4)
        held = xgb.QuantileDMatrix(va, ref=train, feature_names=names, feature_types=ft,
            enable_categorical=True, max_bin=PARAMS['max_bin'], nthread=4)
        del xa, va
        gc.collect()
        model = xgb.train(PARAMS, train, num_boost_round=ROUNDS,
                          evals=[(train, 'training')], verbose_eval=500)
        prediction = model.predict(held)
        assert np.isfinite(prediction).all()
        model.save_model(mp)
        reload = xgb.Booster(model_file=mp)
        reload.set_param({'device': PARAMS['device'], 'nthread': 4})
        np.testing.assert_array_equal(prediction, reload.predict(held))
        pd.DataFrame(dict(mvt_id=ids, pred=prediction)).to_parquet(path, index=False)
        write_json(path.with_suffix('.json'), dict(identity=artifact(OUT / 'identity.json'),
            input=artifact(source), model=artifact(mp), prediction=artifact(path),
            params=PARAMS, rounds=ROUNDS, columns=names, feature_types=ft,
            category_maps=category_maps,
            train_rows=len(y), held_rows=len(ids), all_held_reload_exact=True,
            minutes=(time.monotonic()-started)/60, host=platform.node()))
        parent.log(f'{month}: completed in {(time.monotonic()-started)/60:.1f} min')
        del model, reload, train, held, y, prediction
        gc.collect()


def score():
    ph = selected.physical
    replay = ph.replay
    sc, rome, _, trees = replay.baseline.baseline()
    original = pd.read_parquet(ph.OUT / 'pool3/cv/predictions.parquet').set_index('mvt_id')
    best = pd.read_parquet(selected.constraints.OUT / 'submission_v74/cv.parquet').set_index('mvt_id')
    raw = pd.read_parquet(selected.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(selected.constraints.exclusion.OUT / 'matches2025.parquet')
    pool_id = parent.ts.read(ph.production.cv.MANIFEST)
    frames = []
    for month in parent.ts.SPLITS:
        d = sc.d[month]
        ids = d['ids']
        path = OUT / 'predictions' / f'{month}_s42.parquet'
        rec = parent.ts.read(path.with_suffix('.json'))
        assert rec['identity'] == artifact(OUT / 'identity.json')
        assert rec['prediction'] == artifact(path)
        new = pd.read_parquet(path)
        np.testing.assert_array_equal(new.mvt_id, ids)
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        physical = np.mean([replay.values(ph.verify(ph.pool.completed(month, k, pool_id)['prediction']), ids)
                            for k in [0, 1, 2]], axis=0)
        nn = .5 * (calls + physical)
        main = replay.values(replay.combo.parent.control.candidate.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        t = [main, trees['airport'][42][month], trees['catboost'][42][month]]
        p = .5 * replay.combo.parent.cat.probabilities(month, ids) + .5 * rome[month]
        control = sc.final(month, .71 * nn + .29 * np.mean(t, axis=0), p)
        np.testing.assert_allclose(control, original.loc[ids, 'pred'], rtol=0, atol=1e-9)
        pred = sc.final(month, .71 * nn + .29 * np.mean(t + [new.pred.to_numpy(float)], axis=0), p)
        v74 = best.loc[ids, 'pred'].to_numpy(float)
        q = selected.constraints.matched_bound(raw.loc[ids].reset_index(), v74 + pred - control)
        permitted = matches[matches.donor_month.isin(selected.features.base.FOLDS[month] + [month])]
        q, _ = selected.constraints.exclusion.apply(ids, np.maximum(0, q), permitted)
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'],
            airport=raw.loc[ids, 'airport'].to_numpy(), control=control, pred=pred, v74=v74, transferred=q)))
    f = pd.concat(frames, ignore_index=True)
    f.to_parquet(OUT / 'scored.parquet', index=False)
    cols = ['control', 'pred', 'v74', 'transferred']
    scores = {m: {c: float(np.sqrt(np.mean((q[c]-q.y)**2))) for c in cols} for m, q in f.groupby('month')}
    paired = {f'{b}_vs_{a}': parent.ts.paired_bootstrap(f.day, f.month, (f[a]-f.y)**2, (f[b]-f.y)**2)
              for a, b in [('control', 'pred'), ('v74', 'transferred')]}
    result = dict(design=DESIGN, scores=scores,
        cv={c: float(np.mean([v[c] for v in scores.values()])) for c in cols}, paired=paired)
    write_json(OUT / 'result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'fit', 'score'])
    args = parser.parse_args()
    with threadpool_limits(4):
        if args.stage == 'check':
            print(check())
        else:
            guard()
            globals()[args.stage]()
