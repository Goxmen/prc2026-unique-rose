"""Fixed v58 main-LightGBM screen of six retained callsign/squawk clocks.

The original 156 features, rows, LB1 settings, 10,000 rounds and seed 42 are
unchanged. Only six frozen numeric observation clocks/flags are appended. Daily source
aggregates are not asserted to be physical off-blocks or contiguous episodes.
Cached callsign tables are fold scoped and never rebuilt by this runner. Every other v58 ensemble component stays fixed.

The laptop reuses arrival_plan_jan_jul.input_path; only four small feature
parquets, their manifests and code need transfer. No full-year fit or upload.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import callsign_clocks as clocks
import arrival_plan_jan_jul as input_cache
import hpo_jan_jul as hpo
import ideas_jan_jul as ideas
import modeling
import open_meteo_jan_jul as matched
import tuning_seasonal as ts
from records import sha256, write_json

OUT = ts.RUNS / 'v3_20261010_callsign_lgb_jan_jul'
FEATURES = clocks.FEATURE_NAMES
DESIGN = dict(parent='v58', changed_part='main LightGBM', features=FEATURES,
              trial='LB1', rounds=10000, seed=42,
              network_share=.71, feature_design=clocks.SETTINGS,
              holdout_usage='fixed Jan/Jul screen; no feature or parameter selection')


def log(message):
    print(time.strftime('[%H:%M:%S] ') + message, flush=True)


def feature_path(month, part):
    return clocks.DEFAULT / f'{month}_{part}.parquet'


def checked_manifest():
    """Check transfer integrity without needing the large shared identity cache."""
    manifest = ts.read(clocks.DEFAULT / 'manifest.json')
    if manifest['settings'] != clocks.SETTINGS:
        raise ValueError('Callsign feature design changed')
    if manifest['source_sha256'] != sha256(Path(clocks.__file__)):
        raise ValueError('Callsign builder code differs from audited feature source')
    for month in ts.SPLITS:
        for part in ['train', 'held']:
            entry = manifest['folds'][month][part]
            expected = ts.FOLDS[month] + ([month] if part == 'held' else [])
            if entry['reference_months'] != expected:
                raise ValueError('Fold reference months changed')
            if sha256(feature_path(month, part)) != entry['sha256']:
                raise ValueError('Transferred feature table hash differs')
    return manifest


def guard():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'design.json'
    if path.exists() and ts.read(path) != DESIGN:
        raise ValueError('Run design changed; use a distinct directory')
    write_json(path, DESIGN)
    manifest = checked_manifest()
    identity = dict(runner_sha256=sha256(Path(__file__)),
        builder_sha256=sha256(Path(clocks.__file__)),
        fold_manifest_sha256=sha256(clocks.DEFAULT / 'manifest.json'),
        source_identity_sha256=manifest['identity_sha256'])
    path = OUT / 'code_identity.json'
    if path.exists() and ts.read(path) != identity:
        raise ValueError('Run code or feature identity changed')
    write_json(path, identity)
    return identity


def append_pair(data, month):
    names = data['baseline']
    if len(names) != 156 or len(set(names)) != 156 or set(names) & set(FEATURES):
        raise ValueError('Original v58 feature list changed')
    if np.intersect1d(data['train_ids'], data['ids']).size:
        raise ValueError('Training/held row overlap')
    X0, V0 = data['X'].loc[:, names], data['V'].loc[:, names]
    X = clocks.add_features(X0, data['train_ids'],
                              clocks.load_features(feature_path(month, 'train')))
    V = clocks.add_features(V0, data['ids'],
                              clocks.load_features(feature_path(month, 'held')))
    pd.testing.assert_frame_equal(X.loc[:, names], X0, check_exact=True)
    pd.testing.assert_frame_equal(V.loc[:, names], V0, check_exact=True)
    if list(X.columns) != names + FEATURES or list(V.columns) != list(X.columns):
        raise ValueError('Appended feature order changed')
    return X, V


def fit(threads, months):
    implementation = guard()
    spec = hpo.trial_spec('lgb', 'LB1')
    winner = hpo.winner('lgb')
    if winner['trial'] != 'LB1' or winner['checkpoint'] != 'c10000':
        raise ValueError('Saved v58 main-LightGBM recipe changed')
    params = ts.lgb_params(dict(hpo.V44, **spec['overrides']), 42, spec['learning_rate'])
    params['num_threads'] = threads
    for month in months:
        source = input_cache.input_path(month)
        if not source.exists():
            raise FileNotFoundError(f'Reuse an existing original matrix: {source}')
        identity = dict(**implementation, inputs_sha256=sha256(source),
            feature_hashes={p: sha256(feature_path(month, p)) for p in ['train', 'held']},
            params=params, rounds=10000, seed=42)
        path = OUT / 'predictions' / f'{month}_s42.parquet'
        if path.exists():
            old = ts.read(path.with_suffix('.json'))
            if any(old.get(k) != v for k, v in identity.items()):
                raise ValueError('Existing prediction provenance changed')
            if sha256(path) != old['prediction_sha256']:
                raise ValueError('Existing prediction bytes changed')
            log(f'{month}: audited prediction exists, skip')
            continue
        data = pd.read_pickle(source)
        X, V = append_pair(data, month)
        cats = ideas.signals._categorical(X.columns) + [
            c for c in ideas.ef.CATEGORICAL if c in X.columns]
        if any(not pd.api.types.is_numeric_dtype(X[c]) for c in FEATURES):
            raise ValueError('All six callsign features must remain numeric')
        started = time.monotonic()
        log(f'{month}: {len(X):,} rows, 156+6 inputs, LB1/10000 seed42, {threads} threads')
        dataset = lgb.Dataset(X, data['y'], categorical_feature=cats)
        booster = lgb.train(params, dataset, num_boost_round=10000)
        pred = modeling.predict(booster, V, threads=threads)
        if not np.isfinite(pred).all():
            raise ValueError('Nonfinite predictions')
        model_path = OUT / 'models' / f'{month}_s42.txt'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(model_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        pd.DataFrame(dict(mvt_id=data['ids'], pred=pred.astype('float32'))).to_parquet(temporary, index=False)
        temporary.replace(path)
        write_json(path.with_suffix('.json'), dict(**identity, machine=platform.node(),
            input_source=str(source), columns=list(X.columns), categorical=cats,
            rows=len(X), held_rows=len(V),
            prediction_sha256=sha256(path), model_sha256=sha256(model_path),
            minutes=(time.monotonic()-started)/60,
            held_missing={c: int(V[c].isna().sum()) for c in FEATURES},
            new_feature_gain={c: float(v) for c,v in zip(booster.feature_name(),
                booster.feature_importance('gain')) if c in FEATURES}))
        log(f'{month}: saved in {(time.monotonic()-started)/60:.1f} minutes')
        del data, X, V, dataset, booster
        gc.collect()


def score():
    guard()
    sc, rome, pool, reps = matched.baseline()
    old, new, rows = {}, {}, []
    for month in ts.SPLITS:
        path = OUT / 'predictions' / f'{month}_s42.parquet'
        metadata = ts.read(path.with_suffix('.json'))
        if sha256(path) != metadata['prediction_sha256']:
            raise ValueError('Prediction hash changed')
        frame = pd.read_parquet(path)
        np.testing.assert_array_equal(frame.mvt_id.to_numpy(), sc.d[month]['ids'])
        members = {hpo.MEMBER_OF[p]: reps[p][42][month] for p in reps}
        old[month] = sc.se(month, 42, nn=pool[month], p=rome[month], **members)
        members[hpo.MEMBER_OF['lgb']] = frame.pred.to_numpy(float)
        new[month] = sc.se(month, 42, nn=pool[month], p=rome[month], **members)
        rows.append(pd.DataFrame(dict(mvt_id=frame.mvt_id, month=month,
            day=sc.d[month]['day'], airport=sc.d[month]['held']['raw'].ADEP_mvt.to_numpy(),
            se_control=old[month], se_candidate=new[month])))
    audit = pd.concat(rows, ignore_index=True)
    interval = ts.paired_bootstrap(audit.day.to_numpy(), audit.month.to_numpy(),
                                   audit.se_control.to_numpy(), audit.se_candidate.to_numpy())
    control, candidate = sc.cv(old), sc.cv(new)
    if abs(control['cv'] - 295.7301261) > .0001:
        raise ValueError(f'v58 control differs: {control}')
    result = dict(design=DESIGN, control=control, candidate=candidate, change=interval,
        airport_month=[dict(airport=ap,month=m,rows=len(g),
            control=float(np.sqrt(g.se_control.mean())),candidate=float(np.sqrt(g.se_candidate.mean())))
            for (ap,m),g in audit.groupby(['airport','month'])],
        caveats=['Reused Jan/Jul development labels; bootstrap is not independent confirmation.',
                 'Original matrix values, indices, labels, IDs, dtypes and category orders match across machines; '
                 'candidate uses laptop and saved control uses desktop, so machine numerical variation remains unresolved.'])
    audit.to_parquet(OUT / 'scored_rows.parquet', index=False)
    write_json(OUT / 'result.json', result)
    log(json.dumps({k:v for k,v in result.items() if k != 'airport_month'}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=['check', 'fit', 'score'])
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--months', nargs='+', choices=ts.SPLITS, default=ts.SPLITS)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        if args.stage == 'check':
            guard()
            log('Frozen design and four feature-transfer hashes pass')
        elif args.stage == 'fit':
            fit(args.threads, args.months)
        else:
            score()


if __name__ == '__main__':
    main()
