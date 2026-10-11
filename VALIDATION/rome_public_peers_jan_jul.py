"""Cross-fitted peer budgets added to the selected public Rome CatBoost pool.

For each outer fold, three interleaved month groups produce training-row
probabilities without their own months. Five departure-label history columns
are removed from these inner classifiers. A fourth fit on all allowed months
predicts the outer holdout. Public measurements never enter these probability
models. Final CatBoost rows, parameters, seeds and rounds match v75's recipe.
"""
import argparse
import gc
from pathlib import Path
import time
import types

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import rome_public_delay_cat as cat
import rome_public_peer_features as peer
from records import write_json, sha256

r20 = cat.r20
ROOT = cat.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_rome_public_peers'
DESIGN = dict(parent='selected public Cat3/R20 recipe, prepared v75',
    features=peer.FEATURES, inner='three interleaved groups of allowed months',
    inner_drop=peer.DROP_HISTORY, inner_columns=63, inner_rounds=1700,
    inner_trial='R20', seed=42, final_columns=90,
    seeds=cat.parent.SEEDS, final_rounds=cat.parent.ROUNDS,
    missing_peer='no-ID gap>=4h uses zero-delay copy mode; other missing probability .5 with count flag',
    public_in_inner_classifier=False, departure_label_history_in_inner_classifier=False,
    target_usage='only permitted inner training labels; held labels for final score only',
    selection='fixed new features; no stopping/weight/parameter search; reused development months')


def artifact(path):
    return r20.features.base.artifact(path)


def guard():
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(design=DESIGN, source=artifact(__file__),
        helpers=[artifact(m.__file__) for m in [cat, r20, peer, cat.parent, r20.shared]],
        inputs={m: artifact(r20.OUT / 'inputs' / f'{m}.pkl') for m in cat.parent.ROUNDS},
        parent_result=artifact(cat.OUT / 'result.json'),
        parent_predictions=artifact(cat.OUT / 'scored.parquet'),
        raw_metadata=artifact(r20.features.base.OUT / 'rows2025.parquet'),
        raw_rules=artifact(r20.features.OUT / 'inputs2025.parquet'),
        public_totals=artifact(r20.features.base.SOURCE / 'all_pre_departure_delays_2025.csv'),
        public_features=artifact(r20.features.OUT / 'manifest.json'))
    path = OUT / 'identity.json'
    if path.exists():
        assert cat.parent.shared.ts.read(path) == value
    else:
        write_json(path, value)


def prepare():
    checks = peer.checks()
    (OUT / 'inputs').mkdir(exist_ok=True)
    (OUT / 'inner/models').mkdir(parents=True, exist_ok=True)
    (OUT / 'inner/predictions').mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(r20.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    rules = pd.read_parquet(r20.features.OUT / 'inputs2025.parquet', columns=['mvt_id', 'fltrule']).set_index('mvt_id')
    raw = raw[raw.airport.eq('LIRF')].drop(columns='y').join(rules)
    source = r20.features.base.public(2025)
    spec = r20.shared.hpo.trial_spec('rome', 'R20')
    params = dict(r20.shared.rome_hybrid.CLASSIFIER_PARAMS, **spec['overrides'],
                  seed=42, bagging_seed=43, feature_fraction_seed=44, num_threads=2)
    for month, allowed in r20.features.base.FOLDS.items():
        dest = OUT / 'inputs' / f'{month}.pkl'
        if dest.exists():
            saved = cat.parent.shared.ts.read(dest.with_suffix('.json'))
            assert saved['identity'] == artifact(OUT / 'identity.json')
            assert saved['prepared'] == artifact(dest)
            continue
        d = pd.read_pickle(r20.OUT / 'inputs' / f'{month}.pkl')
        names = [c for c in d['A'] if c not in r20.features.FEATURES+peer.DROP_HISTORY]
        assert len(names) == 63
        assert not set(names).intersection(peer.DROP_HISTORY+r20.features.FEATURES)
        tm = raw.loc[d['train_ids'], 'month'].to_numpy()
        assert set(tm) == set(allowed) and month not in allowed
        groups = [sorted(allowed)[k::3] for k in range(3)]
        probabilities = []
        audits = []
        for k in range(4):
            excluded = groups[k] if k < 3 else [month]
            tr = ~np.isin(tm, excluded)
            if k < 3:
                hold = ~tr
                X, V = d['A'].loc[tr, names], d['A'].loc[hold, names]
                held_ids = d['train_ids'][hold]
            else:
                X, V = d['A'][names], d['B'][names]
                held_ids = d['ids'][d['use']]
            fit_ids = d['train_ids'][tr]
            assert not np.intersect1d(fit_ids, held_ids).size
            fit_months = sorted(set(raw.loc[fit_ids, 'month']))
            held_months = sorted(set(raw.loc[held_ids, 'month']))
            assert not set(fit_months).intersection(held_months)
            tag = f'{month}_g{k}'
            mp = OUT / 'inner/models' / f'{tag}.txt'
            pp = OUT / 'inner/predictions' / f'{tag}.parquet'
            rp = pp.with_suffix('.json')
            contract = dict(identity=artifact(OUT / 'identity.json'),
                fitting_months=fit_months, predicted_months=held_months,
                fit_ids_sha256=sha256_bytes(fit_ids), held_ids_sha256=sha256_bytes(held_ids),
                fit_rows=len(fit_ids), held_rows=len(held_ids), columns=names,
                params=params, rounds=1700)
            if rp.exists():
                rec = cat.parent.shared.ts.read(rp)
                assert all(rec[key] == value for key, value in contract.items())
                assert rec['prediction'] == artifact(pp) and rec['model'] == artifact(mp)
                f = pd.read_parquet(pp)
                np.testing.assert_array_equal(f.mvt_id, held_ids)
            else:
                assert not mp.exists() and not pp.exists(), 'Unrecorded inner artifacts'
                started = time.monotonic()
                dataset = lgb.Dataset(X, d['target'][tr], categorical_feature=r20.shared.signals._categorical(names))
                model = lgb.train(params, dataset, num_boost_round=1700)
                values = model.predict(V, num_threads=2)
                assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
                model.save_model(str(mp))
                restored = lgb.Booster(model_file=str(mp))
                np.testing.assert_array_equal(values, restored.predict(V, num_threads=2))
                f = pd.DataFrame(dict(mvt_id=held_ids, p=values))
                f.to_parquet(pp, index=False)
                rec = dict(contract, prediction=artifact(pp), model=artifact(mp),
                    all_held_reload_exact=True, minutes=(time.monotonic()-started)/60)
                write_json(rp, rec)
                del dataset, model, restored
                print(tag, fit_months, '->', held_months, rec['minutes'], flush=True)
            probabilities.append(f.set_index('mvt_id').p)
            audits.append(rec)
        p = pd.concat(probabilities)
        assert p.index.is_unique
        assert set(p.index) == set(d['train_ids']) | set(d['ids'][d['use']])
        rows = raw[raw.month.isin(allowed+[month])].reset_index()
        extra = peer.build(rows, p, source).set_index('mvt_id')
        for part, ids in [('A', d['train_ids']), ('B', d['ids'][d['use']])]:
            old = d[part]
            d[part] = pd.concat([old, extra.loc[ids, peer.FEATURES].set_axis(old.index)], axis=1)
            pd.testing.assert_frame_equal(d[part][old.columns], old)
        assert d['A'].shape[1] == d['B'].shape[1] == 90
        pd.to_pickle(d, dest)
        extra.reset_index().to_parquet(dest.with_name(f'{month}_features.parquet'), index=False)
        write_json(dest.with_suffix('.json'), dict(identity=artifact(OUT / 'identity.json'),
            prepared=artifact(dest), feature_table=artifact(dest.with_name(f'{month}_features.parquet')),
            checks=checks, inner_fits=audits, public=artifact(r20.features.base.SOURCE / 'all_pre_departure_delays_2025.csv')))
        del d, extra
        gc.collect()


def sha256_bytes(values):
    import hashlib
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def fit():
    prepare()
    # Reuse the unchanged selected CatBoost fit code with this run's inputs.
    proxy = types.SimpleNamespace(**{**vars(r20), 'OUT': OUT})
    namespace = {**cat.fit.__globals__, 'OUT': OUT, 'r20': proxy}
    fn = types.FunctionType(cat.fit.__code__, namespace, 'peer_cat_fit',
                            cat.fit.__defaults__, cat.fit.__closure__)
    assert fn.__code__ is cat.fit.__code__
    fn()


def score():
    ph = r20.physical
    replay = ph.replay
    sc, rome, _, trees = replay.baseline.baseline()
    old = pd.read_parquet(cat.OUT / 'scored.parquet').set_index('mvt_id')
    raw = pd.read_parquet(r20.features.base.OUT / 'rows2025.parquet').set_index('mvt_id')
    matches = pd.read_parquet(r20.constraints.exclusion.OUT / 'matches2025.parquet')
    pool_id = cat.parent.shared.ts.read(ph.production.cv.MANIFEST)
    frames = []
    for month in cat.parent.ROUNDS:
        d = sc.d[month]
        ids = d['ids']
        probabilities = []
        for seed in cat.parent.SEEDS:
            path = OUT / 'predictions' / f'{month}_s{seed}.parquet'
            rec = cat.parent.shared.ts.read(path.with_suffix('.json'))
            assert rec['identity'] == artifact(OUT / 'identity.json') and rec['prediction'] == artifact(path)
            f = pd.read_parquet(path)
            np.testing.assert_array_equal(f.mvt_id, ids)
            probabilities.append(f.p.to_numpy(float))
        calls = replay.values(replay.combo.sequence.CALLS_PRED, ids, 'candidate_nn')
        physical = np.mean([replay.values(ph.verify(ph.pool.completed(month, k, pool_id)['prediction']), ids)
                            for k in [0, 1, 2]], axis=0)
        main = replay.values(replay.combo.parent.control.candidate.OUT / 'predictions' / f'{month}_s42.parquet', ids, 'pred')
        ordinary = .71*.5*(calls+physical)+.29*np.mean([main, trees['airport'][42][month], trees['catboost'][42][month]], axis=0)
        prior = old.loc[ids]
        expected = sc.final(month, ordinary, .5*prior.newcat.to_numpy()+.5*rome[month])
        np.testing.assert_array_equal(expected, prior.cat_only)
        pred = sc.final(month, ordinary, .5*np.mean(probabilities, axis=0)+.5*rome[month])
        q = r20.constraints.matched_bound(raw.loc[ids].reset_index(), prior.v74.to_numpy()+pred-prior.control.to_numpy())
        permitted = matches[matches.donor_month.isin(r20.features.base.FOLDS[month]+[month])]
        q, _ = r20.constraints.exclusion.apply(ids, np.maximum(q, 0), permitted)
        frames.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'], y=d['y'],
            airport=raw.loc[ids, 'airport'].to_numpy(), control=prior.cat_only.to_numpy(), pred=pred,
            v75=prior.cat_only_transfer.to_numpy(), transferred=q)))
    f = pd.concat(frames, ignore_index=True)
    f.to_parquet(OUT / 'scored.parquet', index=False)
    cols = ['control', 'pred', 'v75', 'transferred']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in cols} for m, g in f.groupby('month')}
    paired = {f'{b}_vs_{a}': cat.parent.shared.ts.paired_bootstrap(f.day, f.month, (f[a]-f.y)**2, (f[b]-f.y)**2)
              for a, b in [('control', 'pred'), ('v75', 'transferred')]}
    result = dict(design=DESIGN, scores=scores, cv={c: float(np.mean([v[c] for v in scores.values()])) for c in cols}, paired=paired)
    write_json(OUT / 'result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'score'])
    args = parser.parse_args()
    with threadpool_limits(2):
        guard()
        globals()[args.stage]()
