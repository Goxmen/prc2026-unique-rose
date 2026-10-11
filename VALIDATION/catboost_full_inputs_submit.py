"""Fit one full174-input CatBoost component for the selected combined recipe.

Retain the model and audited predictions, without assigning a version or upload.
"""
import argparse
import gc
from pathlib import Path
import platform
import time

from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import catboost_full_inputs as cv
import catboost_refinements_cv as combination
import public_delay_lgb_submit as main
from records import write_json

ROOT = cv.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261011_catboost_full_inputs_submit'
MANIFEST = OUT/'identity.json'
parent, old = main.parent, main.old
art, read, verify = cv.art, cv.read, cv.verify


def guard():
    assert read(cv.OUT/'audit.json')['status'] == 'pass'
    assert read(combination.OUT/'audit.json')['status'] == 'pass'
    result = read(combination.OUT/'result.json')
    assert result['paired']['quality']['decision'] == 'improvement'
    assert all(v['candidate'] < v['control'] for v in result['scores'].values())
    records = [cv.completed(month) for month in cv.parent.ts.SPLITS]
    first = records[0]
    assert all(r is not None for r in records)
    for r in records:
        for key in ['params', 'columns', 'categoricals', 'original_columns']:
            assert r[key] == first[key]
    original = parent.validate_prepared()
    main_recipe = read(main.OUT/'identity.json')
    assert main_recipe['columns'] == first['columns'] and len(first['columns']) == 174
    assert len(first['original_columns']) == 134
    paths = [Path(__file__), Path(cv.__file__), Path(cv.cb.__file__), Path(combination.__file__),
        Path(main.__file__), Path(old.__file__), Path(parent.__file__),
        cv.OUT/'identity.json', cv.OUT/'result.json', cv.OUT/'audit.json',
        combination.OUT/'result.json', combination.OUT/'audit.json',
        parent.OUT/'prepared.json', main.OUT/'identity.json', parent.hs.INPUTS,
        verify(original['training_ids']), cv.features.OUT/'features2025.parquet',
        cv.features.OUT/'audit.json', main.public_rome.OUT/'ranking_features.parquet',
        main.public_rome.OUT/'ranking_feature_audit.json', cv.features.base.OUT/'rows2025.parquet']
    value = dict(source=art(__file__), sources=[art(p) for p in paths],
        columns=first['columns'], original_columns=first['original_columns'],
        categoricals=first['categoricals'], params=cv.params(), seed=42, rounds=10000,
        train_rows=2051327, training_ids=original['training_ids'], original_inputs=art(parent.hs.INPUTS),
        training_features=art(cv.features.OUT/'features2025.parquet'),
        ranking_features=art(main.public_rome.OUT/'ranking_features.parquet'),
        truth=art(cv.features.base.OUT/'rows2025.parquet'), view='old',
        selection='small incremental component in the already selected combined recipe; no separate candidate claim',
        caveat='reused CV; cross-machine historical control; combined gain small and January nearly flat',
        no_version=True, no_upload=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('**/*.cbm')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST, value)
    return value


def ranking(recipe, original):
    packet, provenance = old.ranking_inputs('old', original)
    ids = np.asarray(packet['rank_ids'])
    assert len(ids) == 344841 and pd.Index(ids).is_unique and packet['view'] == 'old'
    base = packet['Vn'][original['original_columns']]
    clocks, _ = parent.features.load_features('old')
    frame = cv.parent.clocks.add_features(base, ids, clocks)
    public = pd.read_parquet(verify(recipe['ranking_features'])).set_index('mvt_id')
    frame = pd.concat([frame, public.loc[ids, cv.features.FEATURES].set_axis(frame.index)], axis=1)
    assert list(frame) == recipe['columns']
    cats = cv.parent.ideas.signals._categorical(frame.columns)+[c for c in cv.parent.ideas.ef.CATEGORICAL if c in frame]
    assert cats == recipe['categoricals']
    pd.testing.assert_frame_equal(frame[original['original_columns']], base, check_exact=True)
    converted = cv.cb.frame(frame, cats)
    old_converted = cv.cb.frame(base[recipe['original_columns']], cv.parent.ideas.signals._categorical(recipe['original_columns']))
    pd.testing.assert_frame_equal(converted[recipe['original_columns']], old_converted, check_exact=True)
    return converted, ids, provenance


def fit():
    recipe = guard()
    original = parent.validate_prepared()
    mp, rp = OUT/'models/catboost.cbm', OUT/'model.json'
    if rp.exists():
        record = read(rp)
        assert record['identity'] == art(MANIFEST)
        verify(record['model'])
    else:
        assert not mp.exists(), 'Unrecorded model: inspect saved checkpoint before any retry'
        data = pd.read_pickle(verify(recipe['original_inputs']))
        membership = pd.read_parquet(verify(recipe['training_ids']))
        np.testing.assert_array_equal(membership.original_index, data['Xn'].index)
        ids = membership.mvt_id.to_numpy(np.int64)
        assert len(ids) == len(data['yn']) == original['training_rows'] == recipe['train_rows']
        truth = pd.read_parquet(verify(recipe['truth'])).set_index('mvt_id').loc[ids]
        assert set(truth.month) == {f'2025-{m:02}' for m in range(1, 13)}
        y = np.asarray(data['yn'], float)
        np.testing.assert_array_equal(y, truth.y)
        base = data['Xn'][original['original_columns']]
        clocks, _ = parent.features.load_features('train')
        frame = cv.parent.clocks.add_features(base, ids, clocks)
        public = pd.read_parquet(verify(recipe['training_features'])).set_index('mvt_id')
        frame = pd.concat([frame, public.loc[ids, cv.features.FEATURES].set_axis(frame.index)], axis=1)
        cats = cv.parent.ideas.signals._categorical(frame.columns)+[c for c in cv.parent.ideas.ef.CATEGORICAL if c in frame]
        assert list(frame) == recipe['columns'] and cats == recipe['categoricals']
        pd.testing.assert_frame_equal(frame[original['original_columns']], base, check_exact=True)
        X = cv.cb.frame(frame, cats)
        old_converted = cv.cb.frame(base[recipe['original_columns']], cv.parent.ideas.signals._categorical(recipe['original_columns']))
        pd.testing.assert_frame_equal(X[recipe['original_columns']], old_converted, check_exact=True)
        del data, membership, truth, base, clocks, public, frame, old_converted
        gc.collect()
        started = time.monotonic()
        print(f'Full year: {len(X):,} ordinary rows,174 inputs,K07/10000,seed42', flush=True)
        pool = Pool(X, y, cat_features=cats)
        model = CatBoostRegressor(**recipe['params'])
        model.fit(pool)
        assert model.tree_count_ == 10000 and model.feature_names_ == recipe['columns']
        mp.parent.mkdir(exist_ok=True)
        temporary = mp.with_suffix('.cbm.tmp')
        model.save_model(str(temporary)); temporary.replace(mp)
        # Record the saved fit before any prediction/reload diagnostics.
        record = dict(identity=art(MANIFEST), model=art(mp), params=recipe['params'],
            actual_params=model.get_all_params(), columns=recipe['columns'], categoricals=cats,
            train_rows=len(ids), training_ids=recipe['training_ids'], original_inputs=recipe['original_inputs'],
            all_training_labels_and_original134_exact=True, rounds=10000, seed=42,
            minutes=(time.monotonic()-started)/60, host=platform.node())
        write_json(rp, record)
        del model, X, y, pool
        gc.collect()
    pp = OUT/'predictions/old.parquet'
    if pp.with_suffix('.json').exists():
        saved = read(pp.with_suffix('.json'))
        assert saved['identity'] == art(MANIFEST) and saved['model'] == record['model']
        verify(saved['prediction'])
        return
    assert not pp.exists(), 'Unrecorded predictions; inspect before repeating'
    V, ids, provenance = ranking(recipe, original)
    model = CatBoostRegressor(); model.load_model(str(verify(record['model'])))
    assert model.feature_names_ == recipe['columns'] and model.tree_count_ == 10000
    pool = Pool(V, cat_features=recipe['categoricals'])
    values = np.maximum(model.predict(pool, thread_count=4), 0).astype(np.float32)
    assert np.isfinite(values).all() and (values >= 0).all()
    restored = CatBoostRegressor(); restored.load_model(str(verify(record['model'])))
    np.testing.assert_array_equal(values, np.maximum(restored.predict(pool, thread_count=2), 0).astype(np.float32))
    pp.parent.mkdir(exist_ok=True)
    pd.DataFrame(dict(mvt_id=ids, pred=values)).to_parquet(pp, index=False)
    write_json(pp.with_suffix('.json'), dict(identity=art(MANIFEST), model=record['model'],
        prediction=art(pp), rows=len(ids), ranking_packet=provenance,
        original134_categories_and_values_exact=True, all_ranking_reload_exact=True))
    print('Full-year CatBoost and all344841 ranking predictions saved/reloaded', flush=True)


def audit():
    recipe = guard()
    assert not (OUT/'audit.json').exists(), 'Preserve completed audit'
    original = parent.validate_prepared()
    V, ids, provenance = ranking(recipe, original)
    record, prediction = read(OUT/'model.json'), read(OUT/'predictions/old.json')
    assert record['identity'] == prediction['identity'] == art(MANIFEST)
    assert prediction['model'] == record['model']
    saved = pd.read_parquet(verify(prediction['prediction']))
    np.testing.assert_array_equal(saved.mvt_id, ids)
    model = CatBoostRegressor(); model.load_model(str(verify(record['model'])))
    assert model.feature_names_ == recipe['columns'] and model.tree_count_ == 10000
    values = np.maximum(model.predict(Pool(V, cat_features=recipe['categoricals']), thread_count=2), 0).astype(np.float32)
    np.testing.assert_array_equal(values, saved.pred)
    report = dict(status='pass', rows=len(ids), all344841_predictions_reload_exact=True,
        original134_values_and_categories_exact=True, ranking_packet=provenance,
        identity=art(MANIFEST), model=art(OUT/'model.json'), prediction=art(OUT/'predictions/old.json'),
        source=art(__file__), no_version=True, no_upload=True)
    write_json(OUT/'audit.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'fit', 'audit'])
    args = parser.parse_args()
    with threadpool_limits(4):
        globals()[args.stage]()
