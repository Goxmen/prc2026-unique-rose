"""Fit the passing public-input main LightGBM once on all ordinary 2025 rows.

No upload or submission assembly. Retain a reusable audited tree prediction
for a later combination, after the selected-Cat incremental CV audit passes.
"""
import argparse
import gc
from pathlib import Path
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_delay_lgb_jan_jul as cv
import callsign_lgb_submit as old
import rome_public_delay_submit as public_rome
from records import write_json, sha256

ROOT = cv.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_public_delay_lgb_submit'
parent = old.parent
artifact, verify, read = old.artifact, old.verify, old.read


def guard():
    result = read(cv.OUT / 'result.json')
    audit = read(cv.OUT / 'audit.json')
    assert result['paired']['transferred_vs_v74']['decision'] == 'improvement'
    assert audit['status'] == 'pass'
    assert audit['combined']['paired']['transferred_vs_v75']['decision'] == 'improvement'
    records = [read(cv.OUT / 'predictions' / f'{m}_s42.json') for m in cv.parent.ts.SPLITS]
    first = records[0]
    for r in records:
        assert r['identity'] == artifact(cv.OUT / 'identity.json')
        for key in ['params', 'columns', 'categoricals']:
            assert r[key] == first[key]
        verify(r['model'])
        verify(r['prediction'])
    assert len(first['columns']) == 174 and first['params']['num_threads'] == 4
    expected = cv.parent.ts.lgb_params(dict(cv.parent.hpo.V44,
        **cv.parent.hpo.trial_spec('lgb', 'LB1')['overrides']), 42, .02)
    expected['num_threads'] = 4
    assert first['params'] == expected
    value = dict(source=artifact(__file__), helpers=[artifact(m.__file__) for m in [cv, old, parent, public_rome]],
        params=first['params'], columns=first['columns'], categoricals=first['categoricals'],
        rounds=10000, seed=42, train_rows=2051327,
        cv=artifact(cv.OUT / 'result.json'), combination_audit=artifact(cv.OUT / 'audit.json'),
        parent_prepared=artifact(parent.OUT / 'prepared.json'),
        training_features=artifact(cv.features.OUT / 'features2025.parquet'),
        ranking_features=artifact(public_rome.OUT / 'ranking_features.parquet'),
        ranking_feature_audit=artifact(public_rome.OUT / 'ranking_feature_audit.json'),
        view='old', no_new_selection=True, no_upload=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'identity.json'
    if path.exists():
        assert read(path) == value
    else:
        assert not list(OUT.glob('**/*.txt')) and not list(OUT.glob('**/*.parquet'))
        write_json(path, value)
    return value


def fit():
    recipe = guard()
    original = parent.validate_prepared()
    recpath, modelpath = OUT / 'model.json', OUT / 'models/lgb.txt'
    if recpath.exists():
        rec = read(recpath)
        assert rec['identity'] == artifact(OUT / 'identity.json')
        model = lgb.Booster(model_file=str(verify(rec['model'])))
    else:
        assert not modelpath.exists(), 'Unrecorded model: inspect before refitting'
        data = pd.read_pickle(parent.hs.INPUTS)
        membership = pd.read_parquet(verify(original['training_ids']))
        np.testing.assert_array_equal(membership.original_index, data['Xn'].index)
        ids = membership.mvt_id.to_numpy(np.int64)
        assert len(ids) == original['training_rows'] == recipe['train_rows'] == len(data['yn'])
        base = data['Xn'][original['original_columns']]
        clocks, _ = parent.features.load_features('train')
        X = cv.parent.clocks.add_features(base, ids, clocks)
        public = pd.read_parquet(verify(recipe['training_features'])).set_index('mvt_id')
        X = pd.concat([X, public.loc[ids, cv.features.FEATURES].set_axis(X.index)], axis=1)
        pd.testing.assert_frame_equal(X[original['original_columns']], base, check_exact=True)
        cats = cv.parent.ideas.signals._categorical(X.columns) + [c for c in cv.parent.ideas.ef.CATEGORICAL if c in X]
        assert list(X) == recipe['columns'] and cats == recipe['categoricals']
        y = np.asarray(data['yn'], float)
        assert np.isfinite(y).all()
        del base, public, clocks, membership, data
        gc.collect()
        start = time.monotonic()
        print(f'Full year: {len(X)} ordinary rows,174 inputs,10000 rounds,four threads', flush=True)
        dataset = lgb.Dataset(X, y, categorical_feature=cats)
        model = lgb.train(recipe['params'], dataset, num_boost_round=10000)
        assert model.current_iteration() == 10000
        modelpath.parent.mkdir(exist_ok=True)
        temp = modelpath.with_suffix('.txt.tmp')
        model.save_model(str(temp))
        temp.replace(modelpath)
        rec = dict(identity=artifact(OUT / 'identity.json'), model=artifact(modelpath),
            params=recipe['params'], columns=recipe['columns'], categoricals=cats,
            train_rows=len(ids), training_ids=original['training_ids'],
            original_inputs=original['original_inputs'], rounds=10000, seed=42,
            minutes=(time.monotonic()-start)/60, host=platform.node(), lightgbm=lgb.__version__)
        write_json(recpath, rec)
        del X, y, dataset
        gc.collect()
    assert model.feature_name() == recipe['columns']
    out = OUT / 'predictions/old.parquet'
    if out.with_suffix('.json').exists():
        saved = read(out.with_suffix('.json'))
        assert saved['identity'] == artifact(OUT / 'identity.json') and saved['model'] == rec['model']
        verify(saved['prediction'])
        assert saved['all_ranking_reload_exact']
        return
    assert not out.exists(), 'Unrecorded prediction'
    packet, provenance = old.ranking_inputs('old', original)
    ids = np.asarray(packet['rank_ids'])
    assert len(ids) == 344841 and pd.Index(ids).is_unique and packet['view'] == 'old'
    clocks, _ = parent.features.load_features('old')
    V = cv.parent.clocks.add_features(packet['Vn'][original['original_columns']], ids, clocks)
    public = pd.read_parquet(verify(recipe['ranking_features'])).set_index('mvt_id')
    V = pd.concat([V, public.loc[ids, cv.features.FEATURES].set_axis(V.index)], axis=1)
    assert list(V) == recipe['columns']
    values = cv.parent.modeling.predict(model, V, threads=4).astype(np.float32)
    assert np.isfinite(values).all() and (values >= 0).all()
    restored = lgb.Booster(model_file=str(verify(rec['model'])))
    np.testing.assert_array_equal(values, np.maximum(restored.predict(V, num_threads=4), 0).astype(np.float32))
    out.parent.mkdir(exist_ok=True)
    pd.DataFrame(dict(mvt_id=ids, pred=values)).to_parquet(out, index=False)
    write_json(out.with_suffix('.json'), dict(identity=artifact(OUT / 'identity.json'),
        model=rec['model'], prediction=artifact(out), ranking_packet=provenance,
        feature_table=recipe['ranking_features'], rows=len(ids), all_ranking_reload_exact=True))
    print('Production main tree and all344841 audited predictions saved', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit'])
    args = parser.parse_args()
    with threadpool_limits(4):
        globals()[args.stage]()
