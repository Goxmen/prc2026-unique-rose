"""Fixed daily punctuality screen on v75's Rome CatBoost pool.

Nine new inputs, identical rows/targets, three seeds, retained rounds.
The separate queued peer-budget experiment does not enter this comparison.
"""
import argparse
import gc
import types

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_punctuality_features as features
import rome_public_peers_jan_jul as peers
from records import write_json

cat, r20 = peers.cat, peers.r20
ROOT = cat.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_rome_punctuality'
artifact = peers.artifact
DESIGN = dict(parent='prepared v75 public Cat3/R20', change='nine public airport-day punctuality inputs',
    features=features.FEATURES, seeds=cat.parent.SEEDS, rounds=cat.parent.ROUNDS,
    columns=89, no_parameter_or_weight_search=True, no_new_early_stopping=True,
    source_population='airline schedules; not an exact challenge-population identity',
    held_labels='final reused development scoring only', production_or_upload=False)


def guard():
    OUT.mkdir(parents=True, exist_ok=True)
    features.prepare()
    value = dict(design=DESIGN, source=artifact(__file__),
        helpers=[artifact(m.__file__) for m in [cat, r20, peers, features, cat.parent]],
        feature_manifest=artifact(features.OUT / 'manifest.json'),
        feature_audit=artifact(features.OUT / 'audit.json'),
        inputs={m: artifact(r20.OUT / 'inputs' / f'{m}.pkl') for m in cat.parent.ROUNDS},
        parent_result=artifact(cat.OUT / 'result.json'), parent_predictions=artifact(cat.OUT / 'scored.parquet'))
    path = OUT / 'identity.json'
    if path.exists():
        assert cat.parent.shared.ts.read(path) == value
    else:
        write_json(path, value)


def fit():
    (OUT / 'inputs').mkdir(exist_ok=True)
    extra = pd.read_parquet(features.OUT / 'features2025.parquet').set_index('mvt_id')
    for month in cat.parent.ROUNDS:
        dest = OUT / 'inputs' / f'{month}.pkl'
        if dest.exists():
            rec = cat.parent.shared.ts.read(dest.with_suffix('.json'))
            assert rec['identity'] == artifact(OUT / 'identity.json') and rec['prepared'] == artifact(dest)
            continue
        d = pd.read_pickle(r20.OUT / 'inputs' / f'{month}.pkl')
        for part, ids in [('A', d['train_ids']), ('B', d['ids'][d['use']])]:
            old = d[part]
            assert len(old.columns) == 80
            d[part] = pd.concat([old, extra.loc[ids, features.FEATURES].set_axis(old.index)], axis=1)
            pd.testing.assert_frame_equal(d[part][old.columns], old)
            assert len(d[part].columns) == 89
        pd.to_pickle(d, dest)
        write_json(dest.with_suffix('.json'), dict(identity=artifact(OUT / 'identity.json'), prepared=artifact(dest)))
        del d
        gc.collect()
    proxy = types.SimpleNamespace(**{**vars(r20), 'OUT': OUT})
    namespace = {**cat.fit.__globals__, 'OUT': OUT, 'r20': proxy}
    fn = types.FunctionType(cat.fit.__code__, namespace, 'punctuality_cat_fit', cat.fit.__defaults__, cat.fit.__closure__)
    fn()


def score():
    # Same already-reviewed score arithmetic as the independent peer experiment.
    fn = types.FunctionType(peers.score.__code__, {**peers.score.__globals__, 'OUT': OUT, 'DESIGN': DESIGN},
        'punctuality_score', peers.score.__defaults__, peers.score.__closure__)
    fn()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'score'])
    args = parser.parse_args()
    with threadpool_limits(2):
        guard()
        globals()[args.stage]()
