"""Submit v54: v51 (C4) with every part's searched settings, refitted on all of 2025.

Settings from the Jan/Jul search (`VALIDATION/reports/HPO_JAN_JUL.md`, run
`v3_20261005_hpo_jan_jul`); Jan/Jul CV of all five together (C5) 295.3603 against C4's 298.2138.

  nn        15 `ple` networks on v46's + the 22 combo inputs, batch 2,048 (trial N09); seed groups
            42/142/242 x 5, each group's epoch chosen on December 2025 as for v51
  lgb       combo LightGBM: v44's settings with max_cat_threshold 8, learning rate 0.02,
            10,000 rounds (trial LB1)
  airport   per-airport combo LightGBMs: 63 leaves, >= 50 rows per leaf, lambda_l2 10,
            learning rate 0.02, 5,000 rounds (trial P04)
  catboost  v46's inputs, depth 8, learning rate 0.1, l2_leaf_reg 3, Bernoulli row sampling 0.7,
            10,000 iterations (trial K07; iterations chosen with the settings on Jan/Jul)
  rome      LIRF classifier R20, refitted on all 2025 LIRF departures by `rome_hpo_submit.py`;
            its ranking probabilities are read from v53's saved frames
Blend: 0.6 x networks + 0.4 x mean(lgb, airport, catboost), then v46's Rome mix, corrections and
four-hour rule (`nn_improve_submit.final`). No flight is held at an earlier prediction: the user
asked for v54 without v53's hold on ITA flight ITY108.

Inputs are v51's full-year rows and ranking inputs (`v3_20261003_ideas_submit/work/inputs.pkl`,
read only). Every part runs in its own process, one at a time.

Run: .venv/bin/python VALIDATION/hpo_submit.py fit --part {nn,catboost,lgb,airport}
     .venv/bin/python VALIDATION/hpo_submit.py prepare --version 54
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')    # as nn_improve_jan_jul.py, before torch loads

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import hpo_jan_jul as hpo
import ideas_submit as ids
import modeling
import nn_improve_submit as nis
import overnight_run
from records import RUNS, runtime_identity, sha256, source_snapshot, write_json
from rome_hybrid_submit import MONTHS, SEED, now, submission
from submit import fill_template
import tuning_seasonal as ts

INPUTS = ids.WORK / 'inputs.pkl'
V53 = RUNS / 'v3_20261005_rome_hpo_submit' / 'submission_v53'
OUT = RUNS / 'v3_20261006_hpo_submit'
WORK = OUT / 'work'
PARTS = ['nn', 'catboost', 'lgb', 'airport']
GBM = ['lgb', 'airport', 'catboost']
THREADS = 14
TRIALS = dict(nn='N09', lgb='LB1', airport='P04', catboost='K07', rome='R20')


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def searched():
    """Each part's winning trial and round count, as recorded by the search."""
    wins = ts.read(hpo.OUT / 'report.json')['winners']     # winners.json predates the CatBoost pull
    out = {}
    for part, trial in TRIALS.items():
        rec = ts.read(hpo.OUT / f'{part}.json')
        if part != 'rome' and wins[part]['trial'] != trial:
            raise ValueError(f'{part}: the search winner is {wins[part]["trial"]}, not {trial}')
        out[part] = dict(trial=trial, overrides=rec['R20'] if part == 'rome' else rec['trials'][trial]['overrides'],
                         checkpoint=None if part == 'nn' else rec['scores'][trial]['best_checkpoint'],
                         learning_rate=rec['trials'][trial].get('learning_rate') if part in ('lgb', 'airport') else None)
    return out


def network_part(data, spec):
    import nn_improve_jan_jul as nni
    import nn_jan_jul
    hpo.apply_nn(hpo.nn_config(spec['overrides']), nn_jan_jul, nni)
    cols = data['names'] + ids.COMBO
    ts.FOLDS.setdefault(ids.FULL, [f'2025-{m:02d}' for m in range(1, 13)])
    item = dict(X=data['Xn'][cols], y=data['yn'], Vn=data['Vn'][cols], tmonth=data['tmonth'], meta=dict(month=ids.FULL),
                raw=pd.DataFrame({'taxi_sec': np.full(len(data['Vn']), np.nan)}))
    d = nni.Data(item, 'cuda')
    if d.info['inner_month'] != '2025-12':
        raise ValueError('Epoch choice is not on December 2025')
    out, record = pd.DataFrame({'mvt_id': data['rank_ids']}), dict(config=hpo.nn_config(spec['overrides']), groups={})
    inner = (d.subset(d.inner_val), d.y[d.inner_val])
    for g in ids.GROUPS:
        _, _, curve = nni.train('ple', d, d.inner_train, g, nni.NEW['epochs'], inner)
        stop = min(curve, key=lambda c: c['rmse'])['epoch']
        record['groups'][g] = dict(stop_epoch=stop, december_rmse=curve[stop - 1]['rmse'])
        for k in range(hpo.NN_FINAL_NETS):
            _, base, _ = nni.train('ple', d, np.arange(len(d.y)), g + 1000 * k, stop, (d.held, d.yv))
            out[f'n{g}_{k}'] = base.astype(np.float32)
        log(f'nn group {g}: stop {stop}, five networks fitted')
    return out, record


def gbm_part(data, part, spec):
    import ideas_jan_jul as ideas
    rounds = int(spec['checkpoint'][1:])
    settings = dict(hpo.V44, **spec['overrides'])
    params = dict(ts.lgb_params(settings, SEED, spec['learning_rate']), num_threads=THREADS)
    cols = data['names'] + ids.COMBO
    X, Vn = data['Xn'][cols], data['Vn'][cols]
    pred = np.full(len(Vn), np.nan)
    (WORK / 'models').mkdir(parents=True, exist_ok=True)
    if part == 'lgb':
        booster = ideas.train(X, data['yn'], params, rounds)
        booster.save_model(str(WORK / 'models' / 'lgb.txt'))
        pred = modeling.predict(booster, Vn, threads=THREADS)
    else:
        tap, vap = X.airport.astype(str).to_numpy(), Vn.airport.astype(str).to_numpy()
        for ap in sorted(set(tap)):
            booster = ideas.train(X[tap == ap], data['yn'][tap == ap], params, rounds)
            booster.save_model(str(WORK / 'models' / f'airport_{ap}.txt'))
            if (vap == ap).any():
                pred[vap == ap] = modeling.predict(booster, Vn[vap == ap], threads=THREADS)
            log(f'airport {ap}: {int((tap == ap).sum()):,} rows')
    if np.isnan(pred).any():
        raise ValueError(f'{part}: ranking rows without a prediction')
    return (pd.DataFrame({'mvt_id': data['rank_ids'], 'pred': pred.astype(np.float32)}),
            dict(settings=settings, learning_rate=spec['learning_rate'], rounds=rounds, seed=SEED))


def catboost_part(data, spec):
    from catboost import CatBoostRegressor, Pool
    import catboost_jan_jul as cbj
    import rome_signals as signals
    iterations = int(spec['checkpoint'][1:])
    params = dict(hpo.CB_BASE, **spec['overrides'])
    cols = data['names']
    cats = signals._categorical(cols)
    X, V = cbj.frame(data['Xn'][cols], cats), cbj.frame(data['Vn'][cols], cats)
    model = CatBoostRegressor(loss_function='RMSE', iterations=iterations, random_seed=SEED, task_type='GPU',
                              devices='0', verbose=1000, allow_writing_files=False, **params)
    model.fit(Pool(X, data['yn'], cat_features=cats))
    (WORK / 'models').mkdir(parents=True, exist_ok=True)
    model.save_model(str(WORK / 'models' / 'catboost.cbm'))
    pred = np.maximum(model.predict(Pool(V, cat_features=cats)), 0)
    return (pd.DataFrame({'mvt_id': data['rank_ids'], 'pred': pred.astype(np.float32)}),
            dict(params=params, iterations=iterations, seed=SEED))


def fit(part):
    started = time.monotonic()
    spec = searched()[part]
    data = pd.read_pickle(INPUTS)
    (WORK / 'parts').mkdir(parents=True, exist_ok=True)
    if part == 'nn':
        frame, record = network_part(data, spec)
    elif part == 'catboost':
        frame, record = catboost_part(data, spec)
    else:
        with threadpool_limits(limits=1, user_api='blas'):
            frame, record = gbm_part(data, part, spec)
    path = WORK / 'parts' / f'{part}.parquet'
    overnight_run.save_frame(path, frame)
    import torch
    record.update(trial=spec['trial'], minutes=(time.monotonic() - started) / 60, finished_at=now(),
                  device=torch.cuda.get_device_name() if part in ('nn', 'catboost') else 'cpu',
                  artifacts=overnight_run.artifact_hashes(WORK, path))
    write_json(WORK / 'parts' / f'{part}.json', record)
    log(f'{part} done in {record["minutes"]:.1f} min')


def part_values(part, rank_ids):
    f = pd.read_parquet(WORK / 'parts' / f'{part}.parquet')
    if not np.array_equal(f.mvt_id.to_numpy(), rank_ids):
        raise ValueError(f'{part}: rows differ from the ranking rows')
    cols = [c for c in f.columns if c != 'mvt_id']
    if part == 'nn' and len(cols) != len(ids.GROUPS) * hpo.NN_FINAL_NETS:
        raise ValueError('nn: expected 15 networks')
    return f[cols].to_numpy(float).mean(axis=1)


def prepare(version):
    started = time.monotonic()
    directory = OUT / f'submission_v{version}'
    if directory.exists() or submission(version).exists():
        raise ValueError(f'v{version} is already prepared; inspect it before continuing')
    spec = searched()
    v53 = ts.read(V53 / 'submission_v53.json')
    if v53['state'] != 'scored' or v53['sha256'] != sha256(submission(53)):
        raise ValueError('v53 is not the scored upload this script reads its Rome probabilities from')
    data = pd.read_pickle(INPUTS)
    rank_ids = data['rank_ids']
    parts = {p: part_values(p, rank_ids) for p in PARTS}
    v51_parts = {p: ids.part_values(n, data) for p, n in
                 dict(nn='nn_combo', lgb='lgb_combo', airport='lgb_combo_airport', catboost='catboost').items()}
    nn_all, gbm_all = parts['nn'], np.mean([parts[p] for p in GBM], axis=0)
    directory.mkdir(parents=True)
    write_json(directory / 'refit.json', dict(searched=spec, started_at=now(), runtime=runtime_identity(),
                                              parts={p: ts.read(WORK / 'parts' / f'{p}.json') for p in PARTS},
                                              source=source_snapshot(directory)))
    pieces, summary, offset = [], {}, 0
    for month in MONTHS:
        d = data['months'][month]
        n = len(d['raw'])
        sl = slice(offset, offset + n)
        offset += n
        saved = pd.read_parquet(V53 / f'ranking_{month}.parquet')
        if not np.array_equal(saved.mvt_id.to_numpy(), d['raw'].mvt_id.to_numpy()):
            raise ValueError(f'{month}: v53 rows differ')
        p = saved.p.to_numpy(float)
        base = ids.W_NN * nn_all[sl] + (1 - ids.W_NN) * gbm_all[sl]
        frame, masks = nis.final(base, p, d['V'], d['raw'], data['parameters'], data['fitted'])
        frame['nn_base'], frame['gbm_base'], frame['blend_base'], frame['month'] = nn_all[sl], gbm_all[sl], base, month
        for q in PARTS:
            frame[f'{q}_part'] = parts[q][sl]
        overnight_run.save_frame(directory / f'ranking_{month}.parquet', frame)
        summary[month] = dict(rows=n, correction_masks=masks,
                              part_vs_v51={q: dict(corr=float(np.corrcoef(parts[q][sl], v51_parts[q][sl])[0, 1]),
                                                   mean_change=float((parts[q][sl] - v51_parts[q][sl]).mean()))
                                           for q in PARTS})
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(rank_ids).reset_index()
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or frame.rule_triggered.sum() != 43:
        raise ValueError('Invalid predictions or changed rule eligibility')
    template = pq.read_table(nis.TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    nis.transfer.validate_submission(table, template)
    path = submission(version)
    pq.write_table(table, path)
    mine = table.to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    against = {}
    for label in (51, 53):
        other = pq.read_table(submission(label)).to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(mine.index)
        a, b = mine.to_numpy(float), other.to_numpy(float)
        against[f'v{label}'] = dict(mean_change=float((a - b).mean()), mean_abs_change=float(np.abs(a - b).mean()),
                                    max_abs_change=float(np.abs(a - b).max()), corr=float(np.corrcoef(a, b)[0, 1]))
    manifest = dict(version=version, state='prepared', file=str(path.relative_to(ROOT)), sha256=sha256(path),
                    parent='v51 (C4)', change='every part with its searched settings (C5)', held_at_v51=[],
                    searched=spec, jan_jul_cv=dict(C4=298.2138, C5=295.3603), summary=summary, against=against,
                    minutes=(time.monotonic() - started) / 60, prepared_at=now())
    write_json(directory / f'submission_v{version}.json', manifest)
    log(json.dumps(dict(summary=summary, against=against), indent=1))
    log(f'prepared {path}; not uploaded')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('action', choices=['fit', 'prepare'])
    ap.add_argument('--part', choices=PARTS)
    ap.add_argument('--version', type=int)
    args = ap.parse_args()
    if args.action == 'fit':
        fit(args.part)
    else:
        prepare(args.version)


if __name__ == '__main__':
    main()
