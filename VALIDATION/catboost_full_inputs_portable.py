"""Fit the frozen full174 CatBoost on the laptop's verified training cache.

Desktop prepare checks the original complete source graph. The worker verifies
the portable files and imported local helpers, reuses the original matrix and
labels, and saves the model before diagnostics. Original desktop fit/audit then
reuse that model for ranking predictions; no second training fit is allowed.
"""
import argparse
import gc
from pathlib import Path
import platform
import sys
import time

import catboost
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import catboost_full_inputs_submit as parent
from records import write_json

ROOT, OUT = parent.ROOT, parent.OUT
WORK = OUT / 'portable_worker'
MANIFEST = WORK / 'manifest.json'
art, read, verify = parent.art, parent.read, parent.verify


def prepare():
    recipe = parent.guard()
    original = parent.parent.validate_prepared()
    paths = {Path(__file__), parent.MANIFEST, parent.parent.OUT / 'prepared.json'}
    # Bind every imported repository helper; third-party versions are explicit.
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = Path(file).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix == '.py' and '.venv' not in path.parts:
                paths.add(path)
    inputs = {key: recipe[key] for key in ['original_inputs', 'training_ids', 'training_features', 'truth']}
    inputs['callsign_features'] = original['train_features']['artifact']
    paths.update(verify(entry) for entry in inputs.values())
    value = dict(source=art(__file__), identity=art(parent.MANIFEST),
        original_prepared=art(parent.parent.OUT / 'prepared.json'), inputs=inputs,
        files=[art(p) for p in sorted(paths)], train_rows=recipe['train_rows'],
        columns=recipe['columns'], categoricals=recipe['categoricals'], params=recipe['params'],
        catboost_version=catboost.__version__, full_original_desktop_guard_pass=True,
        training='unchanged full-year ordinary population and uncapped labels',
        scope='portable cached inputs; complete raw-source guard repeated on desktop before predictions',
        worker_host='GLAPTOP', no_version=True, no_upload=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not (OUT / 'model.json').exists() and not (OUT / 'models/catboost.cbm').exists()
        write_json(MANIFEST, value)
    validate()
    print('Portable manifest ready; preserve and reuse cached original 2.55 GB input file', flush=True)


def validate():
    manifest = read(MANIFEST)
    assert manifest['source'] == art(__file__)
    assert manifest['identity'] == art(parent.MANIFEST)
    assert manifest['catboost_version'] == catboost.__version__
    for entry in manifest['files']:
        verify(entry)
    recipe = read(parent.MANIFEST)
    original = read(verify(manifest['original_prepared']))
    assert manifest['full_original_desktop_guard_pass'] is True
    for key in ['params', 'columns', 'categoricals', 'train_rows']:
        assert manifest[key] == recipe[key]
    assert original['original_inputs'] == recipe['original_inputs'] == manifest['inputs']['original_inputs']
    assert original['training_ids'] == recipe['training_ids'] == manifest['inputs']['training_ids']
    assert original['train_features']['artifact'] == manifest['inputs']['callsign_features']
    assert original['training_rows'] == recipe['train_rows'] == 2051327
    assert len(recipe['columns']) == 174 and recipe['rounds'] == 10000
    return manifest, recipe, original


def fit():
    manifest, recipe, original = validate()
    assert platform.node() == manifest['worker_host'], 'Wrong allocated host'
    assert get_gpu_device_count() == 1
    mp, rp = OUT / 'models/catboost.cbm', OUT / 'model.json'
    if rp.exists():
        record = read(rp)
        assert record['identity'] == art(parent.MANIFEST)
        assert record['portable_worker'] == art(MANIFEST)
        verify(record['model'])
        print('Completed portable fit verified; no repeat', flush=True)
        return
    assert not mp.exists() and not mp.with_suffix('.cbm.tmp').exists(), 'Inspect unrecorded checkpoint before retry'
    data = pd.read_pickle(verify(recipe['original_inputs']))
    membership = pd.read_parquet(verify(recipe['training_ids']))
    np.testing.assert_array_equal(membership.original_index, data['Xn'].index)
    ids = membership.mvt_id.to_numpy(np.int64)
    assert len(ids) == len(data['yn']) == recipe['train_rows'] and pd.Index(ids).is_unique
    truth = pd.read_parquet(verify(recipe['truth'])).set_index('mvt_id').loc[ids]
    assert set(truth.month) == {f'2025-{m:02}' for m in range(1, 13)}
    y = np.asarray(data['yn'], float)
    np.testing.assert_array_equal(y, truth.y)
    base = data['Xn'][original['original_columns']]
    clocks = parent.cv.parent.clocks.load_features(verify(manifest['inputs']['callsign_features']))
    frame = parent.cv.parent.clocks.add_features(base, ids, clocks)
    public = pd.read_parquet(verify(recipe['training_features'])).set_index('mvt_id')
    assert public.index.is_unique
    frame = pd.concat([frame, public.loc[ids, parent.cv.features.FEATURES].set_axis(frame.index)], axis=1)
    cats = parent.cv.parent.ideas.signals._categorical(frame.columns) + [
        c for c in parent.cv.parent.ideas.ef.CATEGORICAL if c in frame]
    assert list(frame) == recipe['columns'] and cats == recipe['categoricals']
    pd.testing.assert_frame_equal(frame[original['original_columns']], base, check_exact=True)
    X = parent.cv.cb.frame(frame, cats)
    old = parent.cv.cb.frame(base[recipe['original_columns']],
        parent.cv.parent.ideas.signals._categorical(recipe['original_columns']))
    pd.testing.assert_frame_equal(X[recipe['original_columns']], old, check_exact=True)
    write_json(WORK / 'input_audit.json', dict(status='pass', identity=art(parent.MANIFEST),
        manifest=art(MANIFEST), rows=len(ids), columns=list(X), categoricals=cats,
        all_training_labels_and_original134_exact=True, original156_exact=True,
        all12_months=True, callsign_and_public_join_unique=True, host=platform.node()))
    del data, membership, truth, base, clocks, public, frame, old
    gc.collect()
    started = time.monotonic()
    print(f'Laptop full-year: {len(X):,} rows, 174 inputs, K07/10000, seed 42', flush=True)
    pool = Pool(X, y, cat_features=cats)
    model = CatBoostRegressor(**recipe['params'])
    model.fit(pool)
    assert model.tree_count_ == 10000 and model.feature_names_ == recipe['columns']
    mp.parent.mkdir(exist_ok=True)
    temporary = mp.with_suffix('.cbm.tmp')
    model.save_model(str(temporary)); temporary.replace(mp)
    record = dict(identity=art(parent.MANIFEST), model=art(mp), params=recipe['params'],
        actual_params=model.get_all_params(), columns=recipe['columns'], categoricals=cats,
        train_rows=len(ids), training_ids=recipe['training_ids'], original_inputs=recipe['original_inputs'],
        all_training_labels_and_original134_exact=True, rounds=10000, seed=42,
        minutes=(time.monotonic()-started)/60, host=platform.node(),
        portable_worker=art(MANIFEST), portable_input_audit=art(WORK / 'input_audit.json'))
    write_json(rp, record)
    print(f'Saved portable model {record["model"]["sha256"]}; desktop owns ranking audit', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validate', 'fit'])
    with threadpool_limits(4):
        globals()[parser.parse_args().stage]()
