"""Fixed three-seed extension of the original Rome CatBoost screen.

Reuse seed 42 exactly. Refit seeds 142 and 242 with the original parameters and
its training-selected rounds (1182 January, 1401 July), without another stopping
search. Score the mean probability and its fixed 50/50 mean with R20 through
unchanged v58 parts. Reused development months; no submission is produced.
"""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import rome_catboost_jan_jul as parent

shared = parent.shared
ROOT = shared.ROOT
OUT = shared.ts.RUNS / 'v3_20261010_rome_catboost_bagging_jan_jul'
SEEDS = [42, 142, 242]
ROUNDS = {'2025-01': 1182, '2025-07': 1401}
DESIGN = dict(parent='v58', part='Rome recording classifier',
    parent_experiment=str(parent.OUT.relative_to(ROOT)), seeds=SEEDS,
    reused_seed=42, rounds=ROUNDS, stopping='none; frozen parent training-selected rounds',
    parameter_changes=['random_seed only'], train_membership='original R20 after_drop',
    new_features=[], network_share=.71,
    arms=['catboost_three_seed', 'half_catboost_three_seed_half_R20'],
    heldout_months=shared.ts.SPLITS, exploratory_reused_development_months=True)


def parent_paths(month):
    prediction = parent.OUT / 'predictions' / f'{month}_s42.parquet'
    return prediction, prediction.with_suffix('.json')


def validated_parent(month, load_input=True):
    prediction, record_path = parent_paths(month)
    record = shared.ts.read(record_path)
    assert record['design'] == parent.DESIGN
    assert record['source_sha256'] == shared.sha256(Path(parent.__file__))
    for relative, digest in record['helper_sha256'].items():
        assert shared.sha256(ROOT / relative) == digest, relative
    assert record['rounds'] == ROUNDS[month]
    assert record['params']['random_seed'] == 42
    assert record['params']['task_type'] == 'GPU'
    assert record['prediction_sha256'] == shared.sha256(prediction)
    source = shared.OUT / 'inputs' / f'{month}.pkl'
    assert record['input_sha256'] == shared.sha256(source)
    d = pd.read_pickle(source) if load_input else None
    if d is not None:
        names = [c for c in d['A'] if c not in shared.NEW]
        assert record['columns'] == names
        assert record['categoricals'] == shared.signals._categorical(names)
        assert len(d['A']) == record['training_rows'] == len(d['target']) == len(d['train_ids'])
        assert set(np.unique(d['target'])).issubset({0, 1})
        assert int(np.sum(d['target'])) == record['positives']
        assert len(np.unique(d['train_ids'])) == len(d['train_ids'])
        assert len(np.unique(d['ids'])) == len(d['ids'])
        assert not np.intersect1d(d['train_ids'], d['ids']).size
        assert len(d['B']) == int(np.sum(d['use']))
        f = pd.read_parquet(prediction)
        np.testing.assert_array_equal(f.mvt_id, d['ids'])
        assert np.isfinite(f.p).all() and f.p.between(0, 1).all()
    return record, d


def identity(month, seed, record):
    prediction, record_path = parent_paths(month)
    return dict(design=DESIGN, source_sha256=shared.sha256(Path(__file__)),
        parent_record_sha256=shared.sha256(record_path),
        parent_prediction_sha256=record['prediction_sha256'],
        input_sha256=record['input_sha256'], helper_sha256=record['helper_sha256'],
        parent_source_sha256=record['source_sha256'],
        rounds=ROUNDS[month], params={**record['params'], 'random_seed': seed},
        columns=record['columns'], categoricals=record['categoricals'],
        training_rows=record['training_rows'], positives=record['positives'])


def check():
    reports = {}
    for month in shared.ts.SPLITS:
        record, d = validated_parent(month)
        # Same conversion function and exact original ordering for every new seed.
        X = parent.cbj.frame(d['A'][record['columns']], record['categoricals'])
        V = parent.cbj.frame(d['B'][record['columns']], record['categoricals'])
        assert list(X) == list(V) == record['columns']
        assert all(X[c].map(type).eq(str).all() and V[c].map(type).eq(str).all()
                   for c in record['categoricals'])
        reports[month] = dict(training_rows=len(X), held_rome_rows=len(V),
            held_all_rows=len(d['ids']), columns=len(X.columns),
            categoricals=len(record['categoricals']), rounds=ROUNDS[month],
            parent_prediction_sha256=record['prediction_sha256'])
        del X, V, d
        gc.collect()
    shared.write_json(OUT / 'input_check.json', reports)
    shared.log(json.dumps(reports, indent=2))


def fit():
    from catboost import CatBoostClassifier, Pool
    for month in shared.ts.SPLITS:
        record, d = validated_parent(month)
        names, cats = record['columns'], record['categoricals']
        X, V = parent.cbj.frame(d['A'][names], cats), parent.cbj.frame(d['B'][names], cats)
        for seed in SEEDS[1:]:
            dest = OUT / 'predictions' / f'{month}_s{seed}.parquet'
            expected = identity(month, seed, record)
            if dest.exists():
                old = shared.ts.read(dest.with_suffix('.json'))
                assert all(old.get(k) == v for k, v in expected.items()), 'Saved seed provenance changed'
                assert old['prediction_sha256'] == shared.sha256(dest)
                continue
            started = time.monotonic()
            shared.log(f'{month} seed {seed}: fixed {ROUNDS[month]} rounds on {len(X):,} original rows')
            model = CatBoostClassifier(iterations=ROUNDS[month], **expected['params'])
            model.fit(Pool(X, d['target'], cat_features=cats))
            p = np.zeros(len(d['ids']))
            p[d['use']] = model.predict_proba(Pool(V, cat_features=cats))[:, 1]
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            model_path = OUT / 'models' / f'{month}_s{seed}.cbm'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(model_path))
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_suffix('.tmp')
            pd.DataFrame(dict(mvt_id=d['ids'], p=p)).to_parquet(temp, index=False)
            temp.replace(dest)
            shared.write_json(dest.with_suffix('.json'), dict(**expected,
                model_sha256=shared.sha256(model_path), prediction_sha256=shared.sha256(dest),
                minutes=(time.monotonic()-started)/60))
            del model
            gc.collect()
        del X, V, d
        gc.collect()


def probabilities(month, expected_ids):
    record, _ = validated_parent(month, load_input=False)
    values = []
    for seed in SEEDS:
        if seed == 42:
            path = parent_paths(month)[0]
        else:
            path = OUT / 'predictions' / f'{month}_s{seed}.parquet'
            old = shared.ts.read(path.with_suffix('.json'))
            assert all(old.get(k) == v for k, v in identity(month, seed, record).items())
            assert old['prediction_sha256'] == shared.sha256(path)
        f = pd.read_parquet(path)
        np.testing.assert_array_equal(f.mvt_id, expected_ids)
        assert f.p.between(0, 1).all()
        values.append(f.p.to_numpy())
    return np.mean(values, axis=0)


def score():
    sc, original, nn, trees = shared.baseline.baseline()
    errors = {name: {} for name in ['control'] + DESIGN['arms']}
    rows = []
    for month in shared.ts.SPLITS:
        p = probabilities(month, sc.d[month]['ids'])
        arms = dict(control=original[month], catboost_three_seed=p,
                    half_catboost_three_seed_half_R20=(p+original[month])/2)
        for name, prob in arms.items():
            errors[name][month] = np.mean([sc.se(month, seed, nn=nn[month], p=prob,
                **{shared.hpo.MEMBER_OF[part]: trees[part][seed][month] for part in trees})
                for seed in shared.hpo.SEEDS], axis=0)
        rows.append(pd.DataFrame(dict(mvt_id=sc.d[month]['ids'], month=month,
            day=sc.d[month]['day'], **{name: errors[name][month] for name in errors})))
    frame = pd.concat(rows, ignore_index=True)
    scores = {name: sc.cv(value) for name, value in errors.items()}
    assert abs(scores['control']['cv']-295.7414415741871) < 1e-7
    comparisons = {name: shared.ts.paired_bootstrap(frame.day.to_numpy(), frame.month.to_numpy(),
        frame.control.to_numpy(), frame[name].to_numpy()) for name in DESIGN['arms']}
    result = dict(design=DESIGN, scores=scores, comparisons=comparisons,
                  parent_seed42_result=shared.ts.read(parent.OUT / 'result.json'))
    frame.to_parquet(OUT / 'scored_errors.parquet', index=False)
    shared.write_json(OUT / 'result.json', result)
    shared.log(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'fit', 'score'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'design.json'
    if path.exists():
        assert shared.ts.read(path) == DESIGN, 'Frozen bagging design changed'
    shared.write_json(path, DESIGN)
    with threadpool_limits(2):
        {'check': check, 'fit': fit, 'score': score}[args.stage]()


if __name__ == '__main__':
    main()
