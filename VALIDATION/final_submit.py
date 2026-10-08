"""v58's recipe on the final test set (`final_ranking.parquet` / `final_submitting.parquet`).

The organisers published a final test set on 2026-10-08: the January/July 2026 departures plus
February and June 2026 (670,790 departures), with every movement time jittered by up to 12 s.
Training data is unchanged, so v58 is not refitted where its models were kept:

  lgb, airport, catboost  v54's saved full-year models, applied to the final inputs
  rome                    v53's classifier (trial R20, 1,700 rounds) refitted on the same 2025 LIRF
                          rows (v61's prepared matrix without its weather columns); not saved before
  nn                      v54's 15 `ple` networks refitted on v51's full-year rows with v54's
                          settings and recorded stop epochs (not saved before; GPU fits differ slightly)
Blend 0.71 x networks + 0.29 x mean(trees), then v46's Rome mix, corrections and four-hour rule.

Inputs are rebuilt from `final_ranking.parquet` with v51's code; the three external tables keyed
by flight (OPDI, ADSB.lol, extra features) are rebuilt for every final departure from the same
sources (`tools/fetch_opdi_final.py`; ADSB.lol days fetched by `tools/download_adsblol_heatmap.py`).
Controls on the old ranking file: rebuilt tables against the original tables, assembled inputs
against v51's saved inputs, saved trees against v54's saved predictions, the Rome refit against
v53's probabilities.

Stages (one at a time on the desktop):
  tables --set old|final   build the OPDI / ADSB.lol / extra tables; old is compared to the originals
  inputs --set old|final   ranking inputs; old must equal v51's saved inputs
  trees                    saved tree models on the final inputs (reproducing v54 on the old ones)
  rome                     refit and apply the Rome classifier (reproducing v53 on the old inputs)
  nn                       refit the 15 networks; predictions for final and old rows
  build --version N        blend, rules, submission file and manifest
  upload --version N
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')    # as nn_improve_jan_jul.py, before torch loads
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import external_conditions as wx
import extra_features as ef
import hpo_submit as hs
import modeling
import opdi
import overnight_run
import overnight_submit
from records import runtime_identity, sha256, write_json
import rome_hpo_submit as rs
from rome_hybrid_submit import FREEZE

ids, nis, ts = hs.ids, hs.nis, hs.ts
cv, features, signals, weather, sf = nis.cv, nis.features, nis.signals, nis.weather, nis.sf
RANKING = dict(old=ROOT / 'data/ranking.parquet', final=ROOT / 'data/final_ranking.parquet')
TEMPLATE = ROOT / 'data/final_submitting.parquet'
MONTHS = dict(old=['2026-01', '2026-07'], final=['2026-01', '2026-02', '2026-06', '2026-07'])
OUT = hs.RUNS / 'v3_20261008_final_submit'
WORK = OUT / 'work'
TABLES = ROOT / 'data/derived/final_tables'
OPDI_FINAL = ROOT / 'data/external/opdi_final'
ORIGINAL = dict(opdi=opdi.FEATURES_PATH, adsb=adsblol.FEATURES_PATH, adsb_stand=adsblol.STAND_PATH, extra=ef.PATH)
V54_PARTS = hs.WORK / 'parts'
V53 = hs.V53
V58 = hs.RUNS / 'v3_20261006_blend_weight_submit' / 'submission_v58'
ROME_MATRIX = hs.RUNS / 'v3_20261008_open_meteo_submit' / 'work' / 'rome.pkl'
W_NN = 0.71
FINAL_ROWS = 670_790
THREADS = 8
FIRST_DAY = '2025-12-31'    # Dec 31 departures are the 60-min neighbours of Jan 1 ones


def log(message):
    print(time.strftime('[%H:%M:%S] ') + message, flush=True)


def table_path(s, name):
    return TABLES / s / f'{name}.parquet'


# ---------------------------------------------------------------- tables
def build_opdi(s):
    opdi.TRAIN_GLOB = ROOT / 'data/training_2025-12-01_2026-01-01.parquet'
    opdi.RANKING = RANKING[s]
    opdi.LIST_GLOB = OPDI_FINAL / 'flight_list' / 'flight_list_*.parquet'
    opdi.EVENTS_GLOB = OPDI_FINAL / 'events' / 'flight_events_*.parquet'
    import duckdb
    con = duckdb.connect()
    con.execute(f'SET threads={THREADS}')
    return opdi.build_features(con, path=table_path(s, 'opdi'))


def build_adsb(s):
    adsblol.RANKING = RANKING[s]
    deps = adsblol.departures()
    deps = deps[deps.day >= FIRST_DAY]
    days = sorted(deps.day.unique())
    missing = [d for d in days if not (adsblol.HEATMAP / 'positions' / f'{d}.parquet').exists()]
    if missing:
        raise ValueError(f'ADSB.lol positions missing for {missing}')
    groups = [g for _, g in deps.groupby('day')]
    tables = sf.load()
    with ProcessPoolExecutor(THREADS) as pool:
        own = [p for p in pool.map(adsblol.build_day, days, groups, chunksize=2) if p is not None]
        stand = [p for p in pool.map(adsblol.stand_day, days, groups, [tables] * len(days), chunksize=2)
                 if p is not None]
    own = adsblol.neighbours(pd.concat(own, ignore_index=True))
    own = deps[['mvt_id', 'airport', 'to_ms']].merge(own.drop(columns=['airport', 'to_ms']), on='mvt_id', how='left')
    own = own[['mvt_id', 'airport', 'to_ms'] + adsblol.ADSB_FEATURES].sort_values('mvt_id').reset_index(drop=True)
    stand = adsblol.prior_mean(pd.concat(stand, ignore_index=True), 'ab_stand_exit_sec',
                               lambda t: t.ab_exit_gap_sec <= adsblol.OWN_EXIT_GAP_SEC, 'ab_nb_exit')
    stand = deps[['mvt_id']].merge(stand.drop(columns=['airport', 'to_ms']), on='mvt_id', how='left')
    stand = stand[['mvt_id'] + adsblol.STAND_FEATURES].sort_values('mvt_id').reset_index(drop=True)
    for name, frame in (('adsb', own), ('adsb_stand', stand)):
        overnight_run.save_frame(table_path(s, name), frame)
    return len(own), len(days)


def build_extra(s):
    overnight_run.RANKING = RANKING[s]
    ef.build(path=table_path(s, 'extra'))


def compare_table(name, s):
    """Rebuilt table against the original one on the ranking departures of the old file."""
    rank = pd.read_parquet(RANKING[s], columns=['MVT_ID_mvt', 'PHASE_mvt'])
    want = rank.MVT_ID_mvt[rank.PHASE_mvt == 'DEP'].to_numpy()
    new = pd.read_parquet(table_path(s, name)).set_index('mvt_id').reindex(want)
    old = pd.read_parquet(ORIGINAL[name]).set_index('mvt_id').reindex(want)
    out = {}
    for c in new.columns:
        a, b = new[c], old[c]
        if a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu':
            a, b = a.to_numpy(float), b.to_numpy(float)
            diff = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        else:
            diff = ~((a.astype(object) == b.astype(object)) | (a.isna() & b.isna())).to_numpy()
        out[c] = int(diff.sum())
    return dict(rows=len(want), differing_rows=out)


def tables(s):
    (TABLES / s).mkdir(parents=True, exist_ok=True)
    started, record = time.monotonic(), {}
    if not table_path(s, 'opdi').exists():
        record['opdi_rows'] = build_opdi(s)
        log(f'OPDI table built ({record["opdi_rows"]:,} rows)')
    if not table_path(s, 'adsb_stand').exists():
        record['adsb_rows'], record['adsb_days'] = build_adsb(s)
        log(f'ADSB.lol tables built ({record["adsb_rows"]:,} rows, {record["adsb_days"]} days)')
    if not table_path(s, 'extra').exists():
        build_extra(s)
        log('extra table built')
    record['sha256'] = {n: sha256(table_path(s, n)) for n in ORIGINAL}
    if s == 'old':
        record['against_original'] = {n: compare_table(n, s) for n in ORIGINAL}
        log(json.dumps(record['against_original']))
    record['minutes'] = (time.monotonic() - started) / 60
    write_json(OUT / f'tables_{s}.json', record)


# ---------------------------------------------------------------- inputs
def adsb_table(s):
    paths = (ORIGINAL['adsb'], ORIGINAL['adsb_stand']) if s == 'old' else \
        (table_path(s, 'adsb'), table_path(s, 'adsb_stand'))
    return adsblol.load_features(paths[0]).join(adsblol.load_features(paths[1]), how='left')


def widen(parts):
    """Month-specific category sets (e.g. airline): widen to the union, values unchanged (as v51)."""
    for c in parts[0].columns:
        if isinstance(parts[0][c].dtype, pd.CategoricalDtype):
            values = [q[c].cat.categories.to_series() if isinstance(q[c].dtype, pd.CategoricalDtype)
                      else pd.Series(q[c].dropna().unique()) for q in parts]
            union = pd.CategoricalDtype(pd.Index(pd.concat(values, ignore_index=True).unique()))
            for q in parts:
                q[c] = q[c].astype(object).astype(union)
    return parts


def same_frame(a, b):
    """Equal columns and values (numbers exactly, everything else by value; missing equals missing)."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        log(f'  columns or lengths differ: {len(a)} vs {len(b)} rows')
        return False
    for c in a.columns:
        x, y = a[c], b[c]
        numeric = all(pd.api.types.is_numeric_dtype(z) and not isinstance(z.dtype, pd.CategoricalDtype) for z in (x, y))
        if numeric:
            x, y = x.to_numpy(float), y.to_numpy(float)
            ok = (x == y) | (np.isnan(x) & np.isnan(y))
        else:
            text = lambda z: z.astype(object).where(z.notna(), '<NA>').astype(str).to_numpy()
            ok = text(x) == text(y)
        if not ok.all():
            log(f'  column {c}: {int((~ok).sum())} rows differ')
            return False
    return True


def inputs(s):
    started = time.monotonic()
    data = pd.read_pickle(hs.INPUTS)
    names, parameters, fitted = data['names'], data['parameters'], data['fitted']
    saved = None
    if s == 'old':
        saved = dict(months={m: dict(V=data['months'][m]['V'], raw=data['months'][m]['raw']) for m in MONTHS[s]},
                     Vn=data['Vn'], rank_ids=data['rank_ids'])
    del data
    v46 = ts.read(nis.V46 / 'submission_v46.json')
    config = v46['identity']['config']
    overnight_run.RANKING = RANKING[s]
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1., ranking=True)
    if s == 'final':
        template = pq.read_table(TEMPLATE).column('MVT_ID_mvt').to_numpy()
        if len(rank) != len(template) or set(rank.mvt_id) != set(template):
            raise ValueError('Final ranking departures differ from the final template')
    cats = overnight_run.read_categories(ids.WORK / 'categories.json')
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    tables_ = sf.load()
    ab = adsb_table(s)
    extra = ef.load(ORIGINAL['extra'] if s == 'old' else table_path(s, 'extra'))
    cv.recipe._TABLE = opdi.load_features(ORIGINAL['opdi'] if s == 'old' else table_path(s, 'opdi'))
    months, parts = {}, []
    for month in MONTHS[s]:
        raw_m, ra = modeling.month_rows(rank, month), modeling.month_rows(rank_arr, month)
        V = modeling.validation_matrix(dep, arr, raw_m, ra, FREEZE, cats, config)
        arrivals = pd.concat([arr, ra], ignore_index=True)
        V = cv.recipe.add_inputs(V, raw_m, reports, pd.concat([dep, raw_m], ignore_index=True), arrivals)
        V = features.add_features(V, raw_m, arrivals, grid)
        V = adsblol.add_adsblol(sf.add_features(V, raw_m, tables_), raw_m, ab, nis.AB)
        parts.append(ef.add(V[names].copy(), raw_m.mvt_id.to_numpy(), extra, ids.COMBO))
        months[month] = dict(V=V, raw=raw_m)
        log(f'{month}: {len(raw_m):,} departures')
    Vn = pd.concat(widen(parts), ignore_index=True)
    rank_ids = np.concatenate([months[m]['raw'].mvt_id.to_numpy() for m in MONTHS[s]])
    if len(rank_ids) != len(rank) or len(set(rank_ids)) != len(rank_ids):
        raise ValueError('Ranking months do not cover every departure exactly once')
    record = dict(rows=len(rank_ids), months={m: len(months[m]['raw']) for m in MONTHS[s]},
                  missing_share={c: round(float(Vn[c].isna().mean()), 4) for c in Vn.columns if Vn[c].isna().any()})
    if s == 'old':
        ok = np.array_equal(rank_ids, saved['rank_ids']) and same_frame(Vn, saved['Vn']) and all(
            same_frame(months[m]['V'], saved['months'][m]['V']) and same_frame(months[m]['raw'], saved['months'][m]['raw'])
            for m in MONTHS[s])
        record['reproduces_v51_inputs'] = bool(ok)
        if not ok:
            raise ValueError('Old-file inputs do not reproduce v51 saved inputs')
        log('old-file inputs reproduce v51 saved inputs exactly')
    WORK.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(dict(months=months, Vn=Vn, rank_ids=rank_ids, names=names, parameters=parameters, fitted=fitted),
                 WORK / f'inputs_{s}.pkl')
    record.update(minutes=(time.monotonic() - started) / 60, sha256=sha256(WORK / f'inputs_{s}.pkl'))
    write_json(OUT / f'inputs_{s}.json', record)


# ---------------------------------------------------------------- parts
def tree_predictions(Vn, names):
    import lightgbm as lgb
    from catboost import CatBoostRegressor, Pool
    import catboost_jan_jul as cbj
    models = hs.WORK / 'models'
    cols = names + ids.COMBO
    out = {}
    out['lgb'] = modeling.predict(lgb.Booster(model_file=str(models / 'lgb.txt')), Vn[cols], threads=THREADS)
    pred, vap = np.full(len(Vn), np.nan), Vn.airport.astype(str).to_numpy()
    for ap in sorted(set(vap)):
        booster = lgb.Booster(model_file=str(models / f'airport_{ap}.txt'))
        pred[vap == ap] = modeling.predict(booster, Vn[cols][vap == ap], threads=THREADS)
    out['airport'] = pred
    cats = signals._categorical(names)
    model = CatBoostRegressor()
    model.load_model(str(models / 'catboost.cbm'))
    out['catboost'] = np.maximum(model.predict(Pool(cbj.frame(Vn[names], cats), cat_features=cats),
                                               thread_count=THREADS), 0)
    for p, v in out.items():
        if not np.isfinite(v).all():
            raise ValueError(f'{p}: non-finite predictions')
    return {p: v.astype(np.float32) for p, v in out.items()}


def trees():
    started = time.monotonic()
    record = {}
    old = pd.read_pickle(hs.INPUTS)
    pred = tree_predictions(old['Vn'], old['names'])
    for p, v in pred.items():
        saved = pd.read_parquet(V54_PARTS / f'{p}.parquet')
        if not np.array_equal(saved.mvt_id.to_numpy(), old['rank_ids']):
            raise ValueError(f'{p}: v54 rows differ')
        record[f'{p}_v54_max_abs_diff'] = float(np.max(np.abs(v - saved.pred.to_numpy(np.float32))))
    log(f'saved trees on old inputs vs v54: {record}')
    if max(record.values()) > 1e-3:
        raise ValueError('Saved tree models do not reproduce v54')
    del old
    final = pd.read_pickle(WORK / 'inputs_final.pkl')
    pred = tree_predictions(final['Vn'], final['names'])
    frame = pd.DataFrame(dict(mvt_id=final['rank_ids'], **pred))
    overnight_run.save_frame(WORK / 'parts' / 'trees.parquet', frame)
    record.update(minutes=(time.monotonic() - started) / 60, sha256=sha256(WORK / 'parts' / 'trees.parquet'))
    write_json(WORK / 'parts' / 'trees.json', record)


def rome():
    started = time.monotonic()
    spec = hs.searched()['rome']
    d = pd.read_pickle(ROME_MATRIX)
    X = d['X'].drop(columns=wx.WEATHER)
    final = pd.read_pickle(WORK / 'inputs_final.pkl')
    V, raw = final['months']['2026-01']['V'], final['months']['2026-01']['raw']
    lirf = raw.ADEP_mvt.to_numpy() == 'LIRF'
    if list(X.columns) != list(signals.classifier_matrix(V.loc[lirf][:100], raw.loc[lirf][:100], rs.KIND,
                                                         cv.CLASSIFIER).columns):
        raise ValueError('Rome training columns differ from the classifier inputs')
    booster = rs.classifier(X, d['y'], spec['overrides'], rs.ROUNDS)
    (WORK / 'models').mkdir(parents=True, exist_ok=True)
    booster.save_model(str(WORK / 'models' / 'rome.txt'))
    old = pd.read_pickle(hs.INPUTS)
    gap = 0.
    for m in MONTHS['old']:
        p = signals.substitution_probability(booster, old['months'][m]['V'], old['months'][m]['raw'], rs.KIND,
                                             THREADS, cv.CLASSIFIER)
        saved = pd.read_parquet(V53 / f'ranking_{m}.parquet')
        if not np.array_equal(saved.mvt_id.to_numpy(), old['months'][m]['raw'].mvt_id.to_numpy()):
            raise ValueError(f'{m}: v53 rows differ')
        gap = max(gap, float(np.max(np.abs(p - saved.p.to_numpy(float)))))
    del old
    log(f'Rome refit vs v53 probabilities: max abs diff {gap}')
    if gap > 1e-9:
        raise ValueError('Rome classifier refit does not reproduce v53')
    p = np.concatenate([signals.substitution_probability(booster, final['months'][m]['V'], final['months'][m]['raw'],
                                                         rs.KIND, THREADS, cv.CLASSIFIER) for m in MONTHS['final']])
    overnight_run.save_frame(WORK / 'parts' / 'rome.parquet', pd.DataFrame(dict(mvt_id=final['rank_ids'], p=p)))
    write_json(WORK / 'parts' / 'rome.json', dict(trial=spec['trial'], rounds=rs.ROUNDS, v53_max_abs_diff=gap,
                                                  training_rows=len(X), minutes=(time.monotonic() - started) / 60,
                                                  model_sha256=sha256(WORK / 'models' / 'rome.txt')))


def nn():
    import torch
    import nn_improve_jan_jul as nni
    import nn_jan_jul
    torch.set_num_threads(THREADS)
    started = time.monotonic()
    spec = hs.searched()['nn']
    cfg = hs.hpo.nn_config(spec['overrides'])
    hs.hpo.apply_nn(cfg, nn_jan_jul, nni)
    rec = ts.read(V54_PARTS / 'nn.json')
    if rec['config'] != cfg:
        raise ValueError('Network settings differ from v54')
    data = pd.read_pickle(hs.INPUTS)
    final = pd.read_pickle(WORK / 'inputs_final.pkl')
    cols = data['names'] + ids.COMBO
    n_final = len(final['Vn'])
    Vn = pd.concat(widen([final['Vn'][cols].copy(), data['Vn'][cols].copy()]), ignore_index=True)
    ts.FOLDS.setdefault(ids.FULL, [f'2025-{m:02d}' for m in range(1, 13)])
    item = dict(X=data['Xn'][cols], y=data['yn'], Vn=Vn, tmonth=data['tmonth'], meta=dict(month=ids.FULL),
                raw=pd.DataFrame({'taxi_sec': np.full(len(Vn), np.nan)}))
    d = nni.Data(item, 'cuda')
    old_ids = data['rank_ids']
    del data, item
    (WORK / 'nn').mkdir(parents=True, exist_ok=True)
    for g in ids.GROUPS:
        stop = rec['groups'][str(g)]['stop_epoch']
        for k in range(hs.hpo.NN_FINAL_NETS):
            path = WORK / 'nn' / f'n{g}_{k}.parquet'
            if path.exists():
                continue
            _, pred, _ = nni.train('ple', d, np.arange(len(d.y)), g + 1000 * k, stop, (d.held, d.yv))
            pred = np.asarray(pred, np.float32)
            overnight_run.save_frame(path, pd.DataFrame(dict(
                mvt_id=np.concatenate([final['rank_ids'], old_ids]), pred=pred,
                set=np.r_[np.full(n_final, 'final'), np.full(len(old_ids), 'old')])))
            torch.cuda.empty_cache()
            log(f'network {g}/{k} (stop epoch {stop}) saved')
    frames = [pd.read_parquet(WORK / 'nn' / f'n{g}_{k}.parquet') for g in ids.GROUPS
              for k in range(hs.hpo.NN_FINAL_NETS)]
    mean = np.mean([f.pred.to_numpy(float) for f in frames], axis=0)
    v54 = pd.read_parquet(V54_PARTS / 'nn.parquet')
    if not np.array_equal(v54.mvt_id.to_numpy(), old_ids):
        raise ValueError('v54 network rows differ')
    a, b = mean[n_final:], v54.drop(columns='mvt_id').to_numpy(float).mean(axis=1)
    check = dict(corr=float(np.corrcoef(a, b)[0, 1]), rms_diff=float(np.sqrt(np.mean((a - b) ** 2))),
                 mean_diff=float(np.mean(a - b)))
    log(f'refitted networks vs v54 networks on the old inputs: {check}')
    overnight_run.save_frame(WORK / 'parts' / 'nn.parquet',
                             pd.DataFrame(dict(mvt_id=final['rank_ids'], pred=mean[:n_final].astype(np.float32))))
    write_json(WORK / 'parts' / 'nn.json', dict(config=cfg, stop_epochs={g: rec['groups'][str(g)]['stop_epoch']
                                                                          for g in ids.GROUPS},
                                                networks=len(frames), against_v54_old_inputs=check,
                                                device=torch.cuda.get_device_name(),
                                                minutes=(time.monotonic() - started) / 60))


# ---------------------------------------------------------------- submission
def submission(version):
    return ROOT / 'submissions' / f'unique-rose_v{version}.parquet'


def build(version):
    directory = OUT / f'submission_v{version}'
    if (directory / f'submission_v{version}.json').exists() or submission(version).exists():
        raise ValueError(f'v{version} is already prepared; inspect it before continuing')
    final = pd.read_pickle(WORK / 'inputs_final.pkl')
    rank_ids = final['rank_ids']
    parts = {}
    for name in ('nn', 'trees', 'rome'):
        f = pd.read_parquet(WORK / 'parts' / f'{name}.parquet')
        if not np.array_equal(f.mvt_id.to_numpy(), rank_ids):
            raise ValueError(f'{name}: rows differ from the final inputs')
        parts[name] = f
    nn_all = parts['nn'].pred.to_numpy(float)
    gbm_all = parts['trees'][hs.GBM].to_numpy(float).mean(axis=1)
    p_all = parts['rome'].p.to_numpy(float)
    directory.mkdir(parents=True, exist_ok=True)
    pieces, summary, offset = [], {}, 0
    for m in MONTHS['final']:
        d = final['months'][m]
        sl = slice(offset, offset + len(d['raw']))
        offset += len(d['raw'])
        base = W_NN * nn_all[sl] + (1 - W_NN) * gbm_all[sl]
        frame, masks = nis.final(base, p_all[sl], d['V'], d['raw'], final['parameters'], final['fitted'])
        frame['nn_base'], frame['gbm_base'], frame['blend_base'], frame['p'], frame['month'] = \
            nn_all[sl], gbm_all[sl], base, p_all[sl], m
        overnight_run.save_frame(directory / f'ranking_{m}.parquet', frame)
        summary[m] = dict(rows=len(frame), mean_pred=float(frame.pred.mean()), rule_triggered=int(frame.rule_triggered.sum()),
                          correction_masks=masks)
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True)
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or not frame.mvt_id.is_unique:
        raise ValueError('Invalid predictions')
    template = pq.read_table(TEMPLATE)
    if template.num_rows != FINAL_ROWS:
        raise ValueError('Final template row count changed')
    table = hs.fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    overnight_submit.EXPECTED_ROWS = FINAL_ROWS    # the shared check's count is the old file's 344,841
    nis.transfer.validate_submission(table, template)
    if not table['MVT_ID_mvt'].equals(template['MVT_ID_mvt']):
        raise ValueError('Movement IDs or their order differ from the final template')
    # Jan/Jul against v58 (same models; jittered inputs and refitted networks/Rome classifier)
    mine = table.to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    v58 = pq.read_table(hs.submission(58)).to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    a, b = mine.reindex(v58.index).to_numpy(float), v58.to_numpy(float)
    against = dict(rows=len(b), mean_change=float((a - b).mean()), mean_abs_change=float(np.abs(a - b).mean()),
                   rms_change=float(np.sqrt(np.mean((a - b) ** 2))), max_abs_change=float(np.abs(a - b).max()),
                   corr=float(np.corrcoef(a, b)[0, 1]))
    path = submission(version)
    pq.write_table(table, path)
    manifest = dict(version=version, state='prepared', file=str(path.relative_to(ROOT)), sha256=sha256(path),
                    test_set='final_ranking.parquet (Jan/Feb/Jun/Jul 2026)', rows=table.num_rows,
                    parent='v58', change="v58's recipe on the final test set", network_share=W_NN,
                    summary=summary, jan_jul_against_v58=against,
                    parts={n: ts.read(WORK / 'parts' / f'{n}.json') for n in ('nn', 'trees', 'rome')},
                    template_sha256=sha256(TEMPLATE), ranking_sha256=sha256(RANKING['final']),
                    source_sha256=sha256(Path(__file__)), runtime=runtime_identity(), prepared_at=hs.now())
    write_json(directory / f'submission_v{version}.json', manifest)
    log(json.dumps(dict(summary=summary, jan_jul_against_v58=against), indent=1))
    log(f'prepared {path}; not uploaded')


def upload(version):
    import hashlib
    path = OUT / f'submission_v{version}' / f'submission_v{version}.json'
    record = ts.read(path)
    local = submission(version)
    if sha256(local) != record['sha256']:
        raise ValueError('Local file differs from the prepared one')
    mc = shutil.which('mc') or str(Path.home() / '.local/bin/mc')
    remote = 'opensky/prc-2026-unique-rose/' + local.name
    if subprocess.run([mc, 'stat', remote], capture_output=True).returncode == 0:
        raise ValueError(f'{remote} already exists; inspect it before uploading')
    subprocess.run([mc, 'cp', str(local), remote], check=True)
    proc = subprocess.Popen([mc, 'cat', remote], stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    for block in iter(lambda: proc.stdout.read(1 << 20), b''):
        digest.update(block)
    if proc.wait() != 0 or digest.hexdigest() != record['sha256']:
        raise ValueError('Remote bytes differ from the local file')
    record.update(state='uploaded', uploaded_at=hs.now(), remote=remote, remote_sha256=digest.hexdigest())
    write_json(path, record)
    log(f'v{version} uploaded; remote bytes verified')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('stage', choices=['tables', 'inputs', 'trees', 'rome', 'nn', 'build', 'upload'])
    ap.add_argument('--set', choices=['old', 'final'], default='final')
    ap.add_argument('--version', type=int)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (WORK / 'parts').mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1, user_api='blas'):
        if args.stage == 'tables':
            tables(args.set)
        elif args.stage == 'inputs':
            inputs(args.set)
        elif args.stage == 'build':
            build(args.version)
        elif args.stage == 'upload':
            upload(args.version)
        else:
            globals()[args.stage]()
