"""Untested arrival-plan duration inputs, on v58's fixed Jan/Jul CV recipe.

Only the main LightGBM changes. Four raw timing differences and four deviations
from training-only route/type or airport medians are added. No departure label
or block timestamp is read by the feature builder. Training reference medians
leave the query's entire calendar month out; heldout references use only the
permitted training months. This is a single predefined feature family, not HPO.

The cached ERA5 matrices are restricted to their original 156 baseline columns.
Fit settings/rounds and every other ensemble member stay at v58's values. No
full-year fit, submission file, remote upload, or ledger entry is produced.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import hpo_jan_jul as hpo
import open_meteo_jan_jul as matched
import tuning_seasonal as ts
from records import sha256, write_json

OUT = ts.RUNS / 'v3_20261010_arrival_plan_jan_jul'
INPUTS = matched.OUT / 'inputs'
RAW = ['fp_block1_sec', 'fp_block3_sec', 'fp_block_revision_sec', 'fp_air3_sec']
FEATURES = RAW + [c.replace('_sec', '_excess_sec') for c in RAW]
GROUP = ['ADEP_mvt', 'ADES_mvt', 'AIRCRAFT_TYPE_mvt']
MIN_GROUP = 20


def log(message):
    print(time.strftime('[%H:%M:%S] ') + message, flush=True)


def input_path(month):
    cached = INPUTS / f'{month}.pkl'
    return cached if cached.exists() else OUT / 'inputs' / f'{month}.pkl'


def ensure_inputs():
    """On the laptop reuse its earlier split/feature caches, avoiding a large transfer."""
    missing = [m for m in ts.SPLITS if not input_path(m).exists()]
    if not missing:
        return
    import adsblol
    import extra_features as ef
    import ideas_jan_jul as ideas
    import nn_improve_jan_jul as nni
    import overnight_run
    import seasonal
    dep, arr = overnight_run.load_inputs(1.)
    ab, extra = adsblol.load_all(), ef.load()
    for month in missing:
        item = nni.load_month(month, dep, arr, ab)
        td, _, _ = seasonal.split_frames(dep, arr, dict(train=ts.FOLDS[month], val=[month]))
        ids = td.loc[item['X'].index, 'mvt_id'].to_numpy()
        val_ids = item['raw'].mvt_id.to_numpy()
        X = ef.add(item['X'], ids, extra, ideas.COMBO)
        V = ef.add(item['Vn'], val_ids, extra, ideas.COMBO)
        assert len(X.columns) == 156 and list(X.columns) == list(V.columns)
        np.testing.assert_array_equal(item['y'], td.loc[item['X'].index, 'taxi_sec'].to_numpy())
        path = input_path(month)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(dict(X=X, V=V, y=item['y'], train_ids=ids, ids=val_ids,
                          baseline=list(X.columns)), path)
        write_json(path.with_suffix('.json'), dict(input_sha256=sha256(path),
            source_sha256=sha256(Path(__file__)), rows=len(X), held_rows=len(V),
            existing_splits=str(nni.SPLITS_FROM), columns=list(X.columns)))
        log(f'{month}: assembled original 156 inputs from local verified caches')
        del item, td, X, V
        gc.collect()


def raw_inputs():
    import duckdb
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET TimeZone='UTC'")
    frame = con.execute('''SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt, ADES_mvt,
        AIRCRAFT_TYPE_mvt, MVT_TIME_UTC_mvt, EOBT_1_flt, AOBT_3_flt,
        ARVT_1_flt, ARVT_3_flt FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''',
        [str(ROOT / 'data/training_2025-*.parquet')]).df()
    con.close()
    if frame.mvt_id.duplicated().any():
        raise ValueError('Movement IDs are not unique')
    frame['month'] = pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    for c in GROUP:
        frame[c] = frame[c].fillna('<MISSING>')
    return frame.set_index('mvt_id')


def durations(raw):
    """Pure input arithmetic, in seconds. Invalid endpoints remain missing."""
    stamp = lambda c: pd.to_datetime(raw[c], utc=True)
    f = pd.DataFrame(index=raw.index)
    f[RAW[0]] = (stamp('ARVT_1_flt') - stamp('EOBT_1_flt')).dt.total_seconds()
    f[RAW[1]] = (stamp('ARVT_3_flt') - stamp('AOBT_3_flt')).dt.total_seconds()
    f[RAW[2]] = f[RAW[1]] - f[RAW[0]]
    f[RAW[3]] = (stamp('ARVT_3_flt') - stamp('MVT_TIME_UTC_mvt')).dt.total_seconds()
    return f


def deviations(reference, query, reference_values, query_values):
    """Hierarchical input-only reference; do not fabricate values for unknown airports."""
    r = pd.concat([reference[GROUP], reference_values], axis=1)
    key = pd.MultiIndex.from_frame(query[GROUP])
    airport = query.ADEP_mvt
    by_route = r.groupby(GROUP, observed=True)[RAW].agg(['median', 'count'])
    by_airport = r.groupby('ADEP_mvt', observed=True)[RAW].median()
    result = query_values.copy()
    for c in RAW:
        full = by_route[c].reindex(key)
        med = full['median'].to_numpy(float)
        fallback = airport.map(by_airport[c]).to_numpy(float)
        med = np.where(full['count'].to_numpy(float) >= MIN_GROUP, med, fallback)
        result[c.replace('_sec', '_excess_sec')] = query_values[c].to_numpy(float) - med
    return result[FEATURES]


def feature_pair(train_raw, held_raw):
    a, b = durations(train_raw), durations(held_raw)
    train = pd.DataFrame(np.nan, index=train_raw.index, columns=FEATURES)
    for month in sorted(train_raw.month.unique()):
        own = train_raw.month.eq(month)
        if own.all():
            raise ValueError('At least two training months are required')
        part = deviations(train_raw.loc[~own], train_raw.loc[own], a.loc[~own], a.loc[own])
        train.loc[own, FEATURES] = part
    held = deviations(train_raw, held_raw, a, b)
    return train, held


def prepare():
    ensure_inputs()
    raw = raw_inputs()
    for month in ts.SPLITS:
        path = OUT / 'features' / f'{month}.pkl'
        source = input_path(month)
        identity = dict(source_sha256=sha256(Path(__file__)), inputs_sha256=sha256(source),
                        features=FEATURES, min_group=MIN_GROUP)
        if path.exists():
            saved = ts.read(path.with_suffix('.json'))
            if any(saved.get(k) != v for k, v in identity.items()):
                raise ValueError(f'{path}: changed input/source; use a new run directory')
            continue
        data = pd.read_pickle(source)
        tr, va = raw.loc[data['train_ids']], raw.loc[data['ids']]
        assert set(tr.month) == set(ts.FOLDS[month]) and set(va.month) == {month}
        assert not np.intersect1d(tr.index, va.index).size
        a, b = feature_pair(tr, va)
        a.index, b.index = data['X'].index, data['V'].index
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(dict(X=a, V=b, train_ids=data['train_ids'], ids=data['ids']), path)
        write_json(path.with_suffix('.json'), dict(**identity, rows=len(a), held_rows=len(b),
            held_missing=b.isna().mean().to_dict(), feature_sha256=sha256(path)))
        log(f'{month}: prepared {len(a):,}/{len(b):,} rows, {len(FEATURES)} input-only features')
        del data, tr, va, a, b
        gc.collect()


def fit(seeds, threads):
    import modeling
    import ideas_jan_jul as ideas
    spec = hpo.trial_spec('lgb', 'LB1')
    rounds = int(hpo.winner('lgb')['checkpoint'][1:])
    assert rounds == 10000
    for month in ts.SPLITS:
        source = input_path(month)
        feature_path = OUT / 'features' / f'{month}.pkl'
        data, extra = pd.read_pickle(source), pd.read_pickle(feature_path)
        assert np.array_equal(extra['train_ids'], data['train_ids'])
        assert np.array_equal(extra['ids'], data['ids'])
        assert data['X'].index.equals(extra['X'].index)
        assert data['V'].index.equals(extra['V'].index)
        X = pd.concat([data['X'][data['baseline']], extra['X']], axis=1)
        V = pd.concat([data['V'][data['baseline']], extra['V']], axis=1)
        identity = dict(source_sha256=sha256(Path(__file__)), inputs_sha256=sha256(source),
                        features_sha256=sha256(feature_path), columns=list(X.columns), rounds=rounds)
        for seed in seeds:
            path = OUT / 'predictions' / f'{month}_s{seed}.parquet'
            if path.exists():
                previous = ts.read(path.with_suffix('.json'))
                if any(previous.get(k) != v for k, v in identity.items()):
                    raise ValueError(f'{path}: stale prediction')
                continue
            started = time.monotonic()
            params = ts.lgb_params(dict(hpo.V44, **spec['overrides']), seed, spec['learning_rate'])
            params['num_threads'] = threads
            log(f'{month} seed {seed}: fit {rounds} rounds with {threads} threads')
            booster = ideas.train(X, data['y'], params, rounds)
            pred = modeling.predict(booster, V, threads=threads)
            assert np.isfinite(pred).all()
            model_path = OUT / 'models' / f'{month}_s{seed}.txt'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(model_path))
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            pd.DataFrame(dict(mvt_id=data['ids'], pred=pred.astype('float32'))).to_parquet(temp, index=False)
            temp.replace(path)
            write_json(path.with_suffix('.json'), dict(**identity, seed=seed, params=params,
                prediction_sha256=sha256(path), model_sha256=sha256(model_path),
                minutes=(time.monotonic()-started)/60,
                new_feature_gain={c: float(v) for c, v in zip(booster.feature_name(), booster.feature_importance('gain'))
                                  if c in FEATURES}))
            log(f'{month} seed {seed}: saved in {(time.monotonic()-started)/60:.1f} min')
            del booster
        del data, extra, X, V
        gc.collect()


def score(seeds):
    sc, rome, pool, reps = matched.baseline()
    records, old, new, days, months = {}, [], [], [], []
    for seed in seeds:
        a, b = {}, {}
        for month in ts.SPLITS:
            f = pd.read_parquet(OUT / 'predictions' / f'{month}_s{seed}.parquet')
            assert np.array_equal(f.mvt_id.to_numpy(), sc.d[month]['ids'])
            members = {hpo.MEMBER_OF[p]: reps[p][seed][month] for p in reps}
            a[month] = sc.se(month, seed, nn=pool[month], p=rome[month], **members)
            members[hpo.MEMBER_OF['lgb']] = f.pred.to_numpy(float)
            b[month] = sc.se(month, seed, nn=pool[month], p=rome[month], **members)
            old.append(a[month]); new.append(b[month])
            days.append(sc.d[month]['day']); months.append(np.full(len(f), month))
        records[seed] = dict(control=sc.cv(a), candidate=sc.cv(b))
    interval = ts.paired_bootstrap(np.concatenate(days), np.concatenate(months),
                                   np.concatenate(old), np.concatenate(new))
    result = dict(features=FEATURES, seeds=records, change=interval,
        caution='Exploratory reuse of Jan/Jul development labels; bootstrap is not independent confirmation.')
    write_json(OUT / ('result_s' + '_'.join(map(str, seeds)) + '.json'), result)
    log(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('stage', choices=['prepare', 'fit', 'score', 'all'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        if args.stage in ('prepare', 'all'): prepare()
        if args.stage in ('fit', 'all'): fit(args.seeds, args.threads)
        if args.stage in ('score', 'all'): score(args.seeds)


if __name__ == '__main__':
    main()
