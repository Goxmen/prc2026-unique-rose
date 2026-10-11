"""Portable full-year fit of the selected184-input daily-taxi main LightGBM.

Prepare validates complete desktop sources and cached ranking equivalence.
The laptop reuses those exact cached matrices, one fit only. Its checkpoint is
saved before prediction diagnostics. Full ranking reconstruction is audited
on the desktop after return. This component has no version or upload stage.
"""
import argparse
import gc
from pathlib import Path
import platform
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_main_refinements_cv as combination
import catboost_full_inputs_portable as cache
import public_delay_lgb_submit as original_main
from records import write_json

ROOT, cv = combination.ROOT, combination.main
OUT = ROOT / 'VALIDATION/runs/v3_20261011_daily_main_submit'
MANIFEST = OUT / 'identity.json'
art, read, verify = combination.art, combination.read, combination.verify


def prepare():
    assert read(combination.OUT / 'audit.json')['status'] == 'pass'
    result = read(combination.OUT / 'result.json')
    assert result['paired']['primary']['decision'] == 'improvement'
    assert all(x['combined'] < x['full_cat'] for x in result['scores'].values())
    original = original_main.parent.validate_prepared()
    portable, _, _ = cache.validate()
    records = [cv.completed(m) for m in cv.parent.ts.SPLITS]
    first = records[0]
    for record in records:
        for key in ['columns', 'params', 'categoricals', 'rounds']:
            assert record[key] == first[key]
    assert first['rounds'] == 10000 and len(first['columns']) == 184
    assert first['params']['num_threads'] == 4
    # The already cached old ranking view must be identical to the original
    # ranking builder before the remote worker may use it.
    packet, provenance = original_main.old.ranking_inputs('old', original)
    data = pd.read_pickle(verify(portable['inputs']['original_inputs']))
    pd.testing.assert_frame_equal(packet['Vn'][original['original_columns']],
        data['Vn'][original['original_columns']], check_exact=True)
    np.testing.assert_array_equal(packet['rank_ids'], data['rank_ids'])
    assert len(data['rank_ids']) == 344841
    del data, packet
    gc.collect()
    inputs = dict(portable['inputs'])
    inputs.update(callsign_ranking=original['views']['old']['features']['artifact'],
        public_ranking=art(original_main.public_rome.OUT / 'ranking_features.parquet'),
        daily_train=art(cv.features.OUT / 'features2025.parquet'),
        daily_ranking=art(cv.features.OUT / 'featuresranking.parquet'))
    paths = {Path(__file__), cache.MANIFEST, original_main.parent.OUT / 'prepared.json',
        combination.OUT / 'identity.json', combination.OUT / 'result.json', combination.OUT / 'audit.json',
        cv.OUT / 'audit.json', cv.features.OUT / 'manifest.json', cv.features.OUT / 'audit.json'}
    paths.update(verify(x) for x in inputs.values())
    paths.update(verify(x) for x in portable['files'])
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            p = Path(file).resolve()
            if p.is_file() and p.is_relative_to(ROOT) and p.suffix == '.py' and '.venv' not in p.parts:
                paths.add(p)
    value = dict(source=art(__file__), files=[art(p) for p in sorted(paths)], inputs=inputs,
        portable_parent=art(cache.MANIFEST), original_prepared=art(original_main.parent.OUT / 'prepared.json'),
        columns=first['columns'], categoricals=first['categoricals'], params=first['params'],
        original_columns=original['original_columns'], rounds=10000, seed=42, train_rows=2051327,
        ranking_rows=344841, ranking_packet=provenance, cached_ranking_all_values_and_ids_exact=True,
        selection=art(combination.OUT / 'result.json'), lightgbm=lgb.__version__, worker_host='GLAPTOP',
        caveat='small selected component; reused Jan/Jul; production fit on laptop', no_version=True, no_upload=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('**/*.txt')) and not list(OUT.glob('**/*.parquet'))
        write_json(MANIFEST, value)
    validate()
    print('Full-year daily main portable inputs and exact cached ranking checked', flush=True)


def validate():
    recipe = read(MANIFEST)
    assert recipe['source'] == art(__file__) and recipe['lightgbm'] == lgb.__version__
    for entry in recipe['files']:
        verify(entry)
    assert recipe['portable_parent'] == art(cache.MANIFEST)
    assert recipe['cached_ranking_all_values_and_ids_exact'] is True
    assert recipe['params']['num_threads'] == 4 and recipe['rounds'] == 10000
    assert len(recipe['columns']) == 184 and recipe['train_rows'] == 2051327
    assert read(combination.OUT / 'audit.json')['status'] == 'pass'
    return recipe


def assemble(base, ids, recipe, view):
    inputs = recipe['inputs']
    clocks = cv.parent.clocks.load_features(verify(inputs[
        'callsign_features' if view == 'train' else 'callsign_ranking']))
    frame = cv.parent.clocks.add_features(base, ids, clocks)
    for key, names in [('training_features' if view == 'train' else 'public_ranking', cv.main.features.FEATURES),
                       ('daily_train' if view == 'train' else 'daily_ranking', cv.features.FEATURES)]:
        table = pd.read_parquet(verify(inputs[key])).set_index('mvt_id')
        assert table.index.is_unique
        frame = pd.concat([frame, table.loc[ids, names].set_axis(frame.index)], axis=1)
    pd.testing.assert_frame_equal(frame[recipe['original_columns']], base, check_exact=True)
    assert list(frame) == recipe['columns']
    cats = cv.parent.ideas.signals._categorical(frame.columns) + [
        c for c in cv.parent.ideas.ef.CATEGORICAL if c in frame]
    assert cats == recipe['categoricals']
    return frame


def fit():
    recipe = validate()
    assert platform.node() == recipe['worker_host'], 'Wrong allocated production host'
    mp, rp = OUT / 'models/lgb.txt', OUT / 'model.json'
    if rp.exists():
        record = read(rp)
        assert record['identity'] == art(MANIFEST)
        verify(record['model'])
    else:
        assert not mp.exists() and not mp.with_suffix('.txt.tmp').exists(), 'Inspect unrecorded saved model before retry'
        data = pd.read_pickle(verify(recipe['inputs']['original_inputs']))
        membership = pd.read_parquet(verify(recipe['inputs']['training_ids']))
        np.testing.assert_array_equal(membership.original_index, data['Xn'].index)
        ids = membership.mvt_id.to_numpy(np.int64)
        assert len(ids) == len(data['yn']) == recipe['train_rows'] and pd.Index(ids).is_unique
        truth = pd.read_parquet(verify(recipe['inputs']['truth'])).set_index('mvt_id').loc[ids]
        assert set(truth.month) == {f'2025-{m:02}' for m in range(1, 13)}
        y = np.asarray(data['yn'], float)
        np.testing.assert_array_equal(y, truth.y)
        X = assemble(data['Xn'][recipe['original_columns']], ids, recipe, 'train')
        write_json(OUT / 'input_audit.json', dict(status='pass', identity=art(MANIFEST), rows=len(ids),
            all_original156_exact=True, all_labels_and_membership_exact=True, all12_months=True,
            appended_tables_unique=True, columns=list(X), categoricals=recipe['categoricals'], host=platform.node()))
        del data, membership, truth
        gc.collect()
        started = time.monotonic()
        print(f'Laptop full-year main: {len(X):,} rows, 184 inputs, LB1/10000, four threads', flush=True)
        dataset = lgb.Dataset(X, y, categorical_feature=recipe['categoricals'])
        model = lgb.train(recipe['params'], dataset, num_boost_round=10000)
        assert model.current_iteration() == 10000 and model.feature_name() == recipe['columns']
        mp.parent.mkdir(exist_ok=True)
        temporary = mp.with_suffix('.txt.tmp'); model.save_model(str(temporary)); temporary.replace(mp)
        record = dict(identity=art(MANIFEST), model=art(mp), params=recipe['params'],
            columns=recipe['columns'], categoricals=recipe['categoricals'], train_rows=len(ids),
            rounds=10000, seed=42, input_audit=art(OUT / 'input_audit.json'),
            minutes=(time.monotonic()-started)/60, host=platform.node(), lightgbm=lgb.__version__)
        write_json(rp, record)
        del X, y, dataset, model
        gc.collect()
    pp = OUT / 'predictions/old.parquet'
    if pp.with_suffix('.json').exists():
        saved = read(pp.with_suffix('.json'))
        assert saved['identity'] == art(MANIFEST) and saved['model'] == record['model']
        verify(saved['prediction'])
        return
    assert not pp.exists(), 'Inspect unrecorded predictions before retry'
    data = pd.read_pickle(verify(recipe['inputs']['original_inputs']))
    ids = np.asarray(data['rank_ids'])
    assert len(ids) == 344841 and pd.Index(ids).is_unique
    V = assemble(data['Vn'][recipe['original_columns']], ids, recipe, 'old')
    del data
    gc.collect()
    model = lgb.Booster(model_file=str(verify(record['model'])))
    values = np.maximum(model.predict(V, num_threads=4), 0).astype(np.float32)
    assert np.isfinite(values).all()
    restored = lgb.Booster(model_file=str(verify(record['model'])))
    np.testing.assert_array_equal(values, np.maximum(restored.predict(V, num_threads=2), 0).astype(np.float32))
    pp.parent.mkdir(exist_ok=True)
    pd.DataFrame(dict(mvt_id=ids, pred=values)).to_parquet(pp, index=False)
    write_json(pp.with_suffix('.json'), dict(identity=art(MANIFEST), model=record['model'],
        prediction=art(pp), rows=len(ids), all_ranking_reload_exact=True, host=platform.node()))
    print('Daily main model and all344841 predictions saved/reloaded', flush=True)


def audit():
    recipe = validate()
    assert not (OUT / 'audit.json').exists(), 'Preserve completed audit'
    original = original_main.parent.validate_prepared()
    packet, provenance = original_main.old.ranking_inputs('old', original)
    ids = np.asarray(packet['rank_ids'])
    V = assemble(packet['Vn'][recipe['original_columns']], ids, recipe, 'old')
    record, saved = read(OUT / 'model.json'), read(OUT / 'predictions/old.json')
    assert record['identity'] == saved['identity'] == art(MANIFEST)
    assert saved['model'] == record['model']
    model = lgb.Booster(model_file=str(verify(record['model'])))
    assert model.feature_name() == recipe['columns'] and model.current_iteration() == 10000
    values = np.maximum(model.predict(V, num_threads=2), 0).astype(np.float32)
    prediction = pd.read_parquet(verify(saved['prediction']))
    np.testing.assert_array_equal(prediction.mvt_id, ids)
    np.testing.assert_array_equal(prediction.pred, values)
    assert len(ids) == 344841
    write_json(OUT / 'audit.json', dict(status='pass', rows=len(ids), identity=art(MANIFEST),
        source=art(__file__), all_ranking_original_inputs_and_reload_exact=True,
        ranking_packet=provenance, model=art(OUT / 'model.json'), prediction=art(OUT / 'predictions/old.json'),
        no_version=True, no_upload=True))
    print('Original ranking source reconstruction and all344841 predictions pass', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'validate', 'fit', 'audit'])
    with threadpool_limits(4):
        globals()[parser.parse_args().stage]()
