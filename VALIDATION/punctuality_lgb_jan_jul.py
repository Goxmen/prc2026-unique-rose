"""Test public punctuality in the ordinary regressor, independently of Rome.

Add nine already-audited inputs to the passing174-column public main tree.
Retain LB1/10000/seed42 and every original row. Compare against the actual
public-main/public-Cat combination, not against the older162-column model.
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

import public_delay_lgb_jan_jul as parent
import public_punctuality_features as punctuality
import rome_public_delay_cat as cat
from records import write_json

ROOT = parent.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_punctuality_lgb'
artifact = parent.features.base.artifact
DESIGN = dict(parent='passing public main and public Rome Cat3 combination',
    change='nine public punctuality inputs in ordinary main LightGBM only',
    original_columns=174, added=punctuality.FEATURES, columns=183,
    rounds=10000, seed=42, trial='LB1', threads=4,
    rows='unchanged original ordinary population', machine='desktop; same-machine174-column controls',
    no_parameter_or_weight_search=True, no_stopping_selection=True, production_or_upload=False)


def guard():
    OUT.mkdir(parents=True, exist_ok=True)
    assert parent.parent.ts.read(parent.OUT / 'audit.json')['status'] == 'pass'
    assert parent.parent.ts.read(punctuality.OUT / 'audit.json')['status'] == 'pass'
    value = dict(design=DESIGN, source=artifact(__file__),
        helpers=[artifact(m.__file__) for m in [parent, parent.parent, punctuality, cat]],
        main_result=artifact(parent.OUT / 'result.json'), main_audit=artifact(parent.OUT / 'audit.json'),
        main_predictions=artifact(parent.OUT / 'combined_public_cat.parquet'),
        public_features=artifact(parent.features.OUT / 'features2025.parquet'),
        punctuality_features=artifact(punctuality.OUT / 'features2025.parquet'),
        punctuality_audit=artifact(punctuality.OUT / 'audit.json'),
        folds={m: dict(matrix=artifact(parent.parent.input_cache.input_path(m)),
            control=artifact(parent.OUT / 'predictions' / f'{m}_s42.json')) for m in parent.parent.ts.SPLITS})
    path = OUT / 'identity.json'
    if path.exists():
        assert parent.parent.ts.read(path) == value
    else:
        write_json(path, value)


def fit():
    public = pd.read_parquet(parent.features.OUT / 'features2025.parquet').set_index('mvt_id')
    new_features = pd.read_parquet(punctuality.OUT / 'features2025.parquet').set_index('mvt_id')
    for sub in ['models', 'predictions']:
        (OUT / sub).mkdir(exist_ok=True)
    for month in parent.parent.ts.SPLITS:
        dest, mp = OUT / 'predictions' / f'{month}_s42.parquet', OUT / 'models' / f'{month}_s42.txt'
        if dest.with_suffix('.json').exists():
            rec = parent.parent.ts.read(dest.with_suffix('.json'))
            assert rec['identity'] == artifact(OUT / 'identity.json')
            assert rec['prediction'] == artifact(dest) and rec['model'] == artifact(mp)
            continue
        assert not dest.exists() and not mp.exists(), 'Unrecorded fit artifacts'
        control = parent.parent.ts.read(parent.OUT / 'predictions' / f'{month}_s42.json')
        d = pd.read_pickle(parent.parent.input_cache.input_path(month))
        A, B = parent.parent.append_pair(d, month)
        built = []
        for frame, ids in [(A, d['train_ids']), (B, d['ids'])]:
            old = pd.concat([frame, public.loc[ids, parent.features.FEATURES].set_axis(frame.index)], axis=1)
            assert list(old) == control['columns'] and old.shape[1] == 174
            value = pd.concat([old, new_features.loc[ids, punctuality.FEATURES].set_axis(old.index)], axis=1)
            pd.testing.assert_frame_equal(value[old.columns], old)
            assert value.shape[1] == 183
            built.append(value)
        X, V = built
        del A, B, built, value, old, frame
        gc.collect()
        assert control['params']['num_threads'] == 4
        start = time.monotonic()
        print(month, len(X), 'rows,183 inputs,10000 rounds', flush=True)
        dataset = lgb.Dataset(X, d['y'], categorical_feature=control['categoricals'])
        model = lgb.train(control['params'], dataset, num_boost_round=10000)
        assert model.current_iteration() == 10000
        prediction = parent.parent.modeling.predict(model, V, threads=4).astype('float32')
        assert np.isfinite(prediction).all() and (prediction >= 0).all()
        model.save_model(str(mp))
        restored = lgb.Booster(model_file=str(mp))
        np.testing.assert_array_equal(prediction, np.maximum(restored.predict(V, num_threads=4), 0).astype('float32'))
        pd.DataFrame(dict(mvt_id=d['ids'], pred=prediction)).to_parquet(dest, index=False)
        write_json(dest.with_suffix('.json'), dict(identity=artifact(OUT / 'identity.json'),
            model=artifact(mp), prediction=artifact(dest), params=control['params'],
            columns=list(X), categoricals=control['categoricals'],
            rows=len(X), held_rows=len(V), rounds=10000, seed=42, all_held_reload_exact=True,
            minutes=(time.monotonic()-start)/60, host=platform.node()))
        del d, X, V, model, restored, dataset
        gc.collect()


def score():
    ph = parent.selected.physical
    replay = ph.replay
    sc, rome, _, trees = replay.baseline.baseline()
    prior = pd.read_parquet(parent.OUT / 'combined_public_cat.parquet').set_index('mvt_id')
    old = pd.read_parquet(cat.OUT / 'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(parent.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(parent.selected.constraints.exclusion.OUT / 'matches2025.parquet')
    identity = parent.parent.ts.read(ph.production.cv.MANIFEST)
    frames = []
    for month in parent.parent.ts.SPLITS:
        d = sc.d[month]
        ids = d['ids']
        path = OUT / 'predictions' / f'{month}_s42.parquet'
        rec = parent.parent.ts.read(path.with_suffix('.json'))
        assert rec['identity'] == artifact(OUT / 'identity.json') and rec['prediction'] == artifact(path)
        new = pd.read_parquet(path)
        np.testing.assert_array_equal(new.mvt_id, ids)
        old_main = replay.values(parent.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        neural = np.mean([replay.values(ph.verify(ph.pool.completed(month, k, identity)['prediction']), ids)
            for k in [0, 1, 2]], axis=0)
        p = .5*old.loc[ids, 'newcat'].to_numpy()+.5*rome[month]
        def final(main):
            ordinary = .71*.5*(calls+neural)+.29*np.mean([main, trees['airport'][42][month], trees['catboost'][42][month]], axis=0)
            return sc.final(month, ordinary, p)
        np.testing.assert_array_equal(final(old_main), prior.loc[ids, 'candidate'])
        pred = final(new.pred.to_numpy(float))
        q = parent.selected.constraints.matched_bound(raw.loc[ids].reset_index(),
            old.loc[ids, 'v74'].to_numpy()+pred-old.loc[ids, 'control'].to_numpy())
        permitted = matches[matches.donor_month.isin(parent.features.base.FOLDS[month]+[month])]
        q, _ = parent.selected.constraints.exclusion.apply(ids, np.maximum(q, 0), permitted)
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'],
            control=prior.loc[ids, 'candidate'].to_numpy(), pred=pred,
            transferred_control=prior.loc[ids, 'transferred'].to_numpy(), transferred=q)))
    f = pd.concat(frames, ignore_index=True)
    f.to_parquet(OUT / 'scored.parquet', index=False)
    cols = ['control', 'pred', 'transferred_control', 'transferred']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in cols} for m, g in f.groupby('month')}
    paired = {f'{b}_vs_{a}': parent.parent.ts.paired_bootstrap(f.day, f.month, (f[a]-f.y)**2, (f[b]-f.y)**2)
        for a, b in [('control', 'pred'), ('transferred_control', 'transferred')]}
    result = dict(design=DESIGN, scores=scores, cv={c: float(np.mean([s[c] for s in scores.values()])) for c in cols}, paired=paired)
    write_json(OUT / 'result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'score'])
    args = parser.parse_args()
    with threadpool_limits(4):
        guard()
        globals()[args.stage]()
