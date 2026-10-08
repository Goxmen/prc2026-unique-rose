"""v58 + 11 Open-Meteo ERA5 inputs in the main LightGBM only.

Fixed LB1 settings and 10,000 rounds; unchanged training rows and final rules.
The network share remains 0.71. Other HPO parts are reused. No event inputs.
Run prepare; fit --seeds 42 142 242; score --seeds 42 142 242.
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
import external_conditions as wx
import hpo_jan_jul as hpo
import ideas_jan_jul as ideas
import modeling
import rome_hybrid
import seasonal
import tuning_seasonal as ts
from records import sha256, write_json

OUT = ts.RUNS / 'v3_20261008_open_meteo_jan_jul'


def log(msg):
    print(time.strftime('[%H:%M:%S] ') + msg, flush=True)


def prepare():
    import adsblol
    import extra_features as ef
    import overnight_run
    out = OUT / 'inputs'
    out.mkdir(parents=True, exist_ok=True)
    table, ab, extra = wx.weather_table(), adsblol.load_all(), ef.load()
    dep, arr = overnight_run.load_inputs(1.)
    for month in ts.SPLITS:
        path = out / f'{month}.pkl'
        if path.exists():
            continue
        d = ts.split_dir(hpo.SPLITS_FROM, month)
        meta = ts.read(d / 'meta.json')
        td, _, _ = seasonal.split_frames(dep, arr, dict(train=ts.FOLDS[month], val=[month]))
        train, held = pd.read_pickle(d / 'after_drop.pkl'), pd.read_pickle(d / 'heldout.pkl')
        names = meta['regressor_features']
        s = rome_hybrid.substituted(train['X'], train['y'], ts.minimal_raw(train['airport'], train['X'].index))
        X = train['X'].loc[~s, names]
        raw = td.loc[X.index]
        ids = raw.mvt_id.to_numpy()
        X = adsblol.add_adsblol(X, raw[['mvt_id']], ab, ideas.V46)
        X = ef.add(X, ids, extra, ideas.COMBO)
        vr = held['raw']
        V = adsblol.add_adsblol(held['V'][names], vr[['mvt_id']].set_index(held['V'].index), ab, ideas.V46)
        V = ef.add(V, vr.mvt_id.to_numpy(), extra, ideas.COMBO)
        baseline = names + ideas.V46 + ideas.COMBO
        X, V = X[baseline], V[baseline]
        a, b = wx.weather_join(raw, table), wx.weather_join(vr, table)
        assert X.index.equals(a.index) and V.index.equals(b.index)
        assert not a.isna().any().any() and not b.isna().any().any()
        assert not np.intersect1d(ids, vr.mvt_id.to_numpy()).size
        assert np.allclose(train['y'][~s], raw.taxi_sec.to_numpy())
        pd.to_pickle(dict(X=pd.concat([X, a], axis=1), V=pd.concat([V, b], axis=1),
                          y=np.asarray(train['y'], float)[~s], ids=vr.mvt_id.to_numpy(),
                          train_ids=ids, baseline=baseline), path)
        write_json(out / f'{month}.json', dict(rows=len(X), held_rows=len(V), columns=baseline + wx.WEATHER,
            source=sha256(Path(__file__)), weather_code=sha256(Path(wx.__file__)),
            weather_manifest=sha256(wx.CACHE / 'weather_manifest.json'), input_sha256=sha256(path),
            weather_train_ranges={c: [float(a[c].min()), float(a[c].max())] for c in wx.WEATHER},
            weather_held_ranges={c: [float(b[c].min()), float(b[c].max())] for c in wx.WEATHER}))
        log(f'prepared {month}: {len(X):,} training / {len(V):,} held-out, complete weather coverage')
        del train, held, X, V, a, b, td, raw
        gc.collect()


def fit(seeds):
    spec = hpo.trial_spec('lgb', 'LB1')
    rounds = int(hpo.winner('lgb')['checkpoint'][1:])
    assert rounds == 10000
    for month in ts.SPLITS:
        data = pd.read_pickle(OUT / 'inputs' / f'{month}.pkl')
        for seed in seeds:
            dest = OUT / 'predictions' / f'{month}_s{seed}.parquet'
            if dest.exists():
                continue
            params = ts.lgb_params(dict(hpo.V44, **spec['overrides']), seed, spec['learning_rate'])
            started = time.monotonic()
            log(f'fit {month} seed {seed}: 10,000 rounds, {params["num_threads"]} threads')
            booster = ideas.train(data['X'], data['y'], params, rounds)
            pred = modeling.predict(booster, data['V'], threads=ts.THREADS)
            assert np.isfinite(pred).all()
            dest.parent.mkdir(parents=True, exist_ok=True)
            model = OUT / 'models' / f'{month}_s{seed}.txt'
            model.parent.mkdir(parents=True, exist_ok=True)
            booster.save_model(str(model))
            pd.DataFrame(dict(mvt_id=data['ids'], pred=pred.astype('float32'))).to_parquet(dest, index=False)
            write_json(dest.with_suffix('.json'), dict(seed=seed, month=month, rounds=rounds,
                params=params, minutes=(time.monotonic()-started)/60,
                model_sha256=sha256(model), prediction_sha256=sha256(dest),
                inputs_sha256=sha256(OUT / 'inputs' / f'{month}.pkl'),
                importance=dict(zip(booster.feature_name(), booster.feature_importance('gain').tolist()))))
            log(f'finished {month} seed {seed} in {(time.monotonic()-started)/60:.1f} minutes')
            del booster
            gc.collect()
        del data
        gc.collect()


def baseline():
    hpo.W_NN = .71
    sc = hpo.Scorer()
    wins = {p: hpo.winner(p) for p in ['rome', 'nn', 'catboost', 'lgb', 'airport']}
    f = sc.frame('rome', wins['rome']['trial'], 42)
    rome = {m: f[m][wins['rome']['checkpoint']].to_numpy(float) for m in ts.SPLITS}
    t = wins['nn']['trial']
    frames = [sc.frame('nn', t, 42), sc.frame('nn', t + '_k34', 42)] + [sc.frame('nn', t, s) for s in hpo.SEEDS[1:]]
    pool = {m: np.concatenate([fr[m].to_numpy(float) for fr in frames], axis=1).mean(1) for m in ts.SPLITS}
    assert sum(fr[ts.SPLITS[0]].shape[1] for fr in frames) == 15
    reps = {p: {s: hpo.member_preds(sc, p, wins[p], s) for s in hpo.SEEDS} for p in ['catboost', 'lgb', 'airport']}
    return sc, rome, pool, reps


def score(seeds):
    sc, rome, pool, reps = baseline()
    result, per_month, saved = {}, {}, []
    for s in seeds:
        levels = {}
        for m in ts.SPLITS:
            f = pd.read_parquet(OUT / 'predictions' / f'{m}_s{s}.parquet')
            assert np.array_equal(f.mvt_id.to_numpy(), sc.d[m]['ids'])
            kw = {hpo.MEMBER_OF[p]: reps[p][s][m] for p in reps}
            e0 = sc.se(m, s, nn=pool[m], p=rome[m], **kw)
            kw['combo'] = f.pred.to_numpy(float)
            e1 = sc.se(m, s, nn=pool[m], p=rome[m], **kw)
            per_month.setdefault(m, []).append((e0, e1))
            levels[m] = dict(control=float(np.sqrt(e0.mean())), candidate=float(np.sqrt(e1.mean())))
            raw = sc.d[m]['held']['raw']
            saved.append(pd.DataFrame(dict(mvt_id=f.mvt_id, seed=s, month=m, day=sc.d[m]['day'],
                airport=raw.ADEP_mvt.to_numpy(), se_control=e0, se_candidate=e1)))
        result[s] = dict(months=levels, control=float(np.mean([v['control'] for v in levels.values()])),
                        candidate=float(np.mean([v['candidate'] for v in levels.values()])))
    a = {m: np.mean([p[0] for p in v], axis=0) for m, v in per_month.items()}
    b = {m: np.mean([p[1] for p in v], axis=0) for m, v in per_month.items()}
    days = np.concatenate([sc.d[m]['day'] for m in ts.SPLITS])
    months = np.concatenate([np.full(len(sc.d[m]['ids']), m) for m in ts.SPLITS])
    change = ts.paired_bootstrap(days, months, np.concatenate(list(a.values())), np.concatenate(list(b.values())))
    audit = pd.concat(saved, ignore_index=True)
    audit.to_parquet(OUT / 'scored_rows.parquet', index=False)
    by_airport = {}
    for airport, g in audit.groupby('airport'):
        by_airport[airport] = {m: dict(rows=int(len(f)/len(seeds)), control=float(np.sqrt(f.se_control.mean())),
                candidate=float(np.sqrt(f.se_candidate.mean()))) for m, f in g.groupby('month')}
    report = dict(parent='v58', change_description='11 ERA5 inputs in main LightGBM', network_share=.71,
        seeds=result, control=sc.cv(a), candidate=sc.cv(b), change=change, airports=by_airport,
        weather_columns=wx.WEATHER, reused_development_months=True)
    if len(seeds) == 3:
        assert abs(report['control']['cv'] - 295.7414) < .0001, report['control']
    write_json(OUT / 'result.json', report)
    log(json.dumps({k:v for k,v in report.items() if k != 'airports'}, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'fit', 'score'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 142, 242])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1, user_api='blas'):
        if args.stage == 'prepare': prepare()
        elif args.stage == 'fit': fit(args.seeds)
        else: score(args.seeds)
