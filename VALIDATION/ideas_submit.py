"""Submit candidates C1 and C4 of `VALIDATION/reports/IDEAS_JAN_JUL.md` (Jan/Jul CV, run
v3_20261003_ideas_jan_jul) on v46's full-year rows.

  C1  0.6 x mean of 15 `ple` networks (seed groups 42/142/242 x 5) on v46's inputs
      + 0.4 x v46's full-year LightGBM (reused unchanged)
  C4  0.6 x mean of 15 `ple` networks on v46's inputs + the 22 inputs of arrtaxi/peers/rwycfg
      + 0.4 x equal mean of three full-year GBMs: LightGBM with those inputs (v44 settings, seed
      42, 5,000 rounds), one such LightGBM per airport (2,500 rounds), CatBoost on v46's inputs
      (iterations chosen on December, seed 42)
Both then take v46's Rome classifier mix, corrections and four-hour rule, as v48 did.

Stages (each part runs in its own process so CPU and GPU parts can overlap):
  inputs                  rebuild v46's full-year training rows and ranking inputs with the checks of
                          `nn_improve_submit.py`, reproduce v46's ranking predictions, attach the new
                          inputs by movement ID; saved to WORK/inputs.pkl
  fit --part P            one part on all of 2025; its ranking predictions go to WORK/parts/P.parquet
                          (nn_v46, nn_combo, lgb_combo, lgb_combo_airport, catboost)
  prepare --candidate C --version N    blend, final prediction, checks, submission file and manifest
  upload / result --version N
Network epochs and CatBoost iterations are chosen as in validation: train on January-November,
keep the epoch / iteration with the lowest December RMSE, refit on all of 2025 and stop there.
GPU training is not bit-for-bit repeatable, so every part's ranking predictions are kept.
"""
import argparse
import gc
import glob
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')    # as nn_improve_jan_jul.py, before torch loads

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import catboost_jan_jul as cbj
import extra_features as ef
import ideas_jan_jul as ideas
import modeling
import nn_improve_jan_jul as nni
import nn_improve_submit as nis
import nn_jan_jul
import overnight_run
from records import RUNS, data_identity, runtime_identity, sha256, source_snapshot, write_json
import rome_hybrid
from rome_hybrid_submit import FREEZE, KIND, MONTHS, SEED, file_name, now, submission
import stand_day_submit as transfer
from submit import fill_template
import tuning_seasonal as ts

cv, features, signals, weather, run, sf, v43s = nis.cv, nis.features, nis.signals, nis.weather, nis.run, nis.sf, nis.v43s
V46, V47, AB, ORDINARY, MODEL_KEYS = nis.V46, nis.V47, nis.AB, nis.ORDINARY, nis.MODEL_KEYS
IDEAS = ideas.OUT
OUT = RUNS / 'v3_20261003_ideas_submit'
WORK = OUT / 'work'
V48 = RUNS / 'v3_20261003_nn_improve_submit' / 'submission_v48' / 'submission_v48.json'
COMBO = ideas.COMBO
GROUPS = [42, 142, 242]
W_NN = 0.6
PARTS = ['nn_v46', 'nn_combo', 'lgb_combo', 'lgb_combo_airport', 'catboost']
CANDIDATES = {'C1': dict(score_key='C1_ple15', network='nn_v46', gbm=['v46']),
              'C4': dict(score_key='C4_all_nn_features', network='nn_combo',
                         gbm=['lgb_combo', 'lgb_combo_airport', 'catboost'])}
FULL = '2025'


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def evidence(candidate):
    c = ts.read(IDEAS / 'candidates.json')[CANDIDATES[candidate]['score_key']]
    base = ts.read(IDEAS / 'candidates.json')['v48']
    if c['change']['decision'] != 'improvement' or any(c['by_seed'][s] >= base['by_seed'][s] for s in c['by_seed']):
        raise ValueError(f'{candidate} does not improve Jan/Jul CV at every seed')
    fits = ts.read(IDEAS / 'fits.json')['_source']
    for arm in ('combo', 'combo_airport'):
        src = fits[arm]
        if (src['runner'] != sha256(ROOT / 'VALIDATION/ideas_jan_jul.py') or src['features'] != sha256(ef.PATH)
                or src['extra_features_code'] != sha256(ROOT / 'VALIDATION/extra_features.py') or src['columns'] != COMBO):
            raise ValueError(f'{arm}: code, inputs or columns changed since validation')
    if ts.read(IDEAS / 'catboost.json')['recipe'] != cbj.RECIPE:
        raise ValueError('CatBoost recipe changed since validation')
    nn_combo = ts.read(IDEAS / 'nn_combo.json')
    if nn_combo['columns'] != COMBO or nn_combo['recipe']['new'] != nni.NEW:
        raise ValueError('Network recipe or inputs changed since validation')
    if ts.read(nni.OUT / 'result.json')['source_sha256']['runner'] != sha256(ROOT / 'VALIDATION/nn_improve_jan_jul.py'):
        raise ValueError('nn_improve_jan_jul.py changed since validation')
    return dict(candidate=candidate, jan_jul=c, v48=base['level'])


def inputs():
    """v46's full-year rows and ranking inputs (nn_improve_submit.prepare's checks) + the new inputs."""
    started = time.monotonic()
    for c in CANDIDATES:
        evidence(c)
    v46 = ts.read(V46 / 'submission_v46.json')
    if v46['state'] != 'scored' or sha256(submission(46)) != v46['sha256']:
        raise ValueError('v46 is not scored, or its local file is not the one uploaded')
    v48 = ts.read(V48)
    if v48['state'] != 'scored' or sha256(submission(48)) != v48['sha256']:
        raise ValueError('v48 is not scored, or its local file is not the one uploaded')
    for key, files in (('data_identity', [*glob.glob(nis.TRAIN_GLOB), nis.RANKING, weather.METAR]),
                       ('opdi_identity', [cv.opdi.FEATURES_PATH]), ('stand_identity', [sf.COORDS, sf.ENDS])):
        if data_identity(files) != v46['identity'][key]:
            raise ValueError(f'{key} differs from v46')
    if dict(features=sha256(adsblol.FEATURES_PATH), stand_features=sha256(adsblol.STAND_PATH)) != \
            v46['identity']['adsblol_tables_sha256']:
        raise ValueError('ADS-B tables differ from v46')
    v46_progress = ts.read(V46 / 'progress.json')
    overnight_run.check_artifacts(V46, v46_progress)
    WORK.mkdir(parents=True, exist_ok=True)
    config = v46['identity']['config']
    previous = ts.read(v43s.PARENT_MANIFEST)
    expected = previous['model']['parameters']
    tables, table, extra = sf.load(), adsblol.load_all(), ef.load()
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1., ranking=True)
    fitted, _ = v43s.v42.fit_corrections(dep, arr, config, expected['late_mean'])
    v43s.v42.check_parameters(fitted, previous['model']['corrections'])
    X, y, cats, raw, parameters, membership = cv.training_inputs(dep, arr, config, reports, grid, expected)
    if len(membership) != 2084659 or membership.excluded_by_rule.sum() != 127 or len(raw) != 2084532:
        raise ValueError('Full-year training population differs from v46')
    if parameters != v46['training']['parameters'] or fitted != v46['training']['corrections']:
        raise ValueError('Training parameters or corrections differ from v46')
    write_json(WORK / 'categories.json', overnight_run.category_values(cats))
    if sha256(WORK / 'categories.json') != sha256(V46 / 'categories.json'):
        raise ValueError('Category values differ from v46')
    X = adsblol.add_adsblol(sf.add_features(X, raw, tables), raw, table, AB)
    s = rome_hybrid.substituted(X, y, raw)
    names = rome_hybrid.regressor_features(cv.recipe.KIND, ORDINARY)
    boosters = {part: run.load_model(V46, v46_progress, key) for part, key in MODEL_KEYS.items()}
    if boosters['ordinary'].feature_name() != list(names):
        raise ValueError('v46 ordinary regressor inputs differ')
    ids = raw.mvt_id.to_numpy()[~s]
    Xn = ef.add(X.loc[~s, names], ids, extra, COMBO)
    yn = np.asarray(y, float)[~s]
    tmonth = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m').to_numpy()[~s]
    if len(Xn) != v46['training']['ordinary_rows'] or sorted(set(tmonth)) != [f'2025-{m:02d}' for m in range(1, 13)]:
        raise ValueError('Training rows differ from v46 ordinary regressor rows')
    del X, y, raw, membership
    gc.collect()
    cats = overnight_run.read_categories(WORK / 'categories.json')
    months = {}
    for month in MONTHS:
        log(f'ranking inputs {month}')
        raw_m, ra = modeling.month_rows(rank, month), modeling.month_rows(rank_arr, month)
        V = modeling.validation_matrix(dep, arr, raw_m, ra, FREEZE, cats, config)
        arrivals = pd.concat([arr, ra], ignore_index=True)
        V = cv.recipe.add_inputs(V, raw_m, reports, pd.concat([dep, raw_m], ignore_index=True), arrivals)
        V = features.add_features(V, raw_m, arrivals, grid)
        V = adsblol.add_adsblol(sf.add_features(V, raw_m, tables), raw_m, table, AB)
        lgb_base = signals.ordinary_prediction(boosters['ordinary'], V, KIND, run.THREADS, ORDINARY)
        p = signals.substitution_probability(boosters['classifier'], V, raw_m, KIND, run.THREADS, cv.CLASSIFIER)
        saved = pd.read_parquet(V46 / f'ranking_{month}.parquet').set_index('mvt_id').reindex(raw_m.mvt_id)
        frame0, _ = nis.final(lgb_base, p, V, raw_m, parameters, fitted)
        gap = max(float(np.max(np.abs(lgb_base - saved.ordinary.to_numpy(float)))),
                  float(np.max(np.abs(frame0.pred.to_numpy(float) - saved.pred.to_numpy(float)))))
        if gap > 1e-6:
            raise ValueError(f'{month}: v46 ranking predictions not reproduced ({gap})')
        months[month] = dict(V=V, raw=raw_m, lgb=lgb_base, p=p, reproduction=gap,
                             Vn=ef.add(V[names].copy(), raw_m.mvt_id.to_numpy(), extra, COMBO))
        del arrivals, ra
        gc.collect()
    parts = [months[m]['Vn'] for m in MONTHS]
    for c in Xn.columns:   # month-specific category sets (e.g. airline): widen to the union, values unchanged
        if isinstance(Xn[c].dtype, pd.CategoricalDtype):
            values = [q[c].cat.categories.to_series() if isinstance(q[c].dtype, pd.CategoricalDtype)
                      else pd.Series(q[c].dropna().unique()) for q in parts]
            union = pd.CategoricalDtype(pd.Index(pd.concat(values, ignore_index=True).unique()))
            for q in parts:
                q[c] = q[c].astype(object).astype(union)
    Vn = pd.concat(parts, ignore_index=True)
    for m in MONTHS:
        del months[m]['Vn']
    rank_ids = np.concatenate([months[m]['raw'].mvt_id.to_numpy() for m in MONTHS])
    pd.to_pickle(dict(Xn=Xn, yn=yn, tmonth=tmonth, Vn=Vn, rank_ids=rank_ids, names=list(names), months=months,
                      parameters=parameters, fitted=fitted, v46_identity=v46['identity'],
                      v46_models={k: v46_progress['models'][k]['artifacts'] for k in MODEL_KEYS.values()},
                      ordinary_rows=len(Xn), substituted_rows=int(s.sum())), WORK / 'inputs.pkl')
    log(f'inputs saved: training {Xn.shape}, ranking {Vn.shape}, missing new inputs in ranking '
        f'{ {c: round(float(Vn[c].isna().mean()), 4) for c in COMBO} } ({(time.monotonic() - started) / 60:.1f} min)')


def network_part(data, part):
    cols = data['names'] + (COMBO if part == 'nn_combo' else [])
    ts.FOLDS.setdefault(FULL, [f'2025-{m:02d}' for m in range(1, 13)])
    item = dict(X=data['Xn'][cols], y=data['yn'], Vn=data['Vn'][cols], tmonth=data['tmonth'], meta=dict(month=FULL),
                raw=pd.DataFrame({'taxi_sec': np.full(len(data['Vn']), np.nan)}))
    d = nni.Data(item, 'cuda')
    if d.info['inner_month'] != '2025-12':
        raise ValueError('Epoch choice is not on December 2025')
    out, record = pd.DataFrame({'mvt_id': data['rank_ids']}), dict(inputs=d.info, groups={})
    inner = (d.subset(d.inner_val), d.y[d.inner_val])
    for g in GROUPS:
        _, _, curve = nni.train('ple', d, d.inner_train, g, nni.NEW['epochs'], inner)
        stop = min(curve, key=lambda c: c['rmse'])['epoch']
        record['groups'][g] = dict(stop_epoch=stop, december_rmse=curve[stop - 1]['rmse'])
        for k in range(nni.NETWORKS['ple']):
            _, base, _ = nni.train('ple', d, np.arange(len(d.y)), g + 1000 * k, stop, (d.held, d.yv))
            out[f'n{g}_{k}'] = base.astype(np.float32)
        log(f'{part} group {g}: stop {stop}, five networks fitted')
    return out, record


def gbm_part(data, part):
    chosen = ts.read(ideas.CHOSEN)
    params = dict(ts.lgb_params(chosen['settings'], SEED, chosen['learning_rate']), num_threads=14)
    cols = data['names'] + COMBO
    X, Vn = data['Xn'][cols], data['Vn'][cols]
    pred, record = np.full(len(Vn), np.nan), {}
    (WORK / 'models').mkdir(exist_ok=True)
    if part == 'lgb_combo':
        booster = ideas.train(X, data['yn'], params, chosen['rounds'])
        booster.save_model(str(WORK / 'models' / 'lgb_combo.txt'))
        pred = modeling.predict(booster, Vn, threads=14)
        record = dict(rounds=chosen['rounds'], settings=chosen['settings'], seed=SEED)
    else:
        tap, vap = X.airport.astype(str).to_numpy(), Vn.airport.astype(str).to_numpy()
        for ap in sorted(set(tap)):
            booster = ideas.train(X[tap == ap], data['yn'][tap == ap], params, ideas.ROUNDS['combo_airport'])
            booster.save_model(str(WORK / 'models' / f'lgb_combo_airport_{ap}.txt'))
            if (vap == ap).any():
                pred[vap == ap] = modeling.predict(booster, Vn[vap == ap], threads=14)
            log(f'{part} {ap}: {int((tap == ap).sum()):,} rows')
        record = dict(rounds=ideas.ROUNDS['combo_airport'], settings=chosen['settings'], seed=SEED)
    if np.isnan(pred).any():
        raise ValueError(f'{part}: ranking rows without a prediction')
    return pd.DataFrame({'mvt_id': data['rank_ids'], 'pred': pred.astype(np.float32)}), record


def catboost_part(data):
    cbj.GPU_RAM_PART = 0.6
    cols = data['names']
    cats = signals._categorical(cols)
    X, V = cbj.frame(data['Xn'][cols], cats), cbj.frame(data['Vn'][cols], cats)
    last = data['tmonth'] == '2025-12'
    inner = cbj.model(SEED, cbj.RECIPE['iterations'], od_type='Iter', od_wait=cbj.RECIPE['od_patience'],
                      use_best_model=True)
    inner.fit(cbj.Pool(X[~last], data['yn'][~last], cat_features=cats),
              eval_set=cbj.Pool(X[last], data['yn'][last], cat_features=cats))
    best = int(inner.get_best_iteration()) + 1
    record = dict(iterations=best, december_rmse=float(inner.get_best_score()['validation']['RMSE']), seed=SEED)
    del inner
    final = cbj.model(SEED, best)
    final.fit(cbj.Pool(X, data['yn'], cat_features=cats))
    (WORK / 'models').mkdir(exist_ok=True)
    final.save_model(str(WORK / 'models' / 'catboost.cbm'))
    pred = np.maximum(final.predict(cbj.Pool(V, cat_features=cats)), 0)
    return pd.DataFrame({'mvt_id': data['rank_ids'], 'pred': pred.astype(np.float32)}), record


def fit(part):
    started = time.monotonic()
    data = pd.read_pickle(WORK / 'inputs.pkl')
    (WORK / 'parts').mkdir(exist_ok=True)
    if part.startswith('nn_'):
        frame, record = network_part(data, part)
    elif part == 'catboost':
        frame, record = catboost_part(data)
    else:
        with threadpool_limits(limits=1, user_api='blas'):
            frame, record = gbm_part(data, part)
    path = WORK / 'parts' / f'{part}.parquet'
    overnight_run.save_frame(path, frame)
    record.update(minutes=(time.monotonic() - started) / 60, finished_at=now(),
                  device=torch.cuda.get_device_name() if part.startswith('nn_') or part == 'catboost' else 'cpu',
                  artifacts=overnight_run.artifact_hashes(WORK, path))
    write_json(WORK / 'parts' / f'{part}.json', record)
    log(f'{part} done in {record["minutes"]:.1f} min')


def part_values(name, data):
    if name == 'v46':
        return np.concatenate([data['months'][m]['lgb'] for m in MONTHS])
    f = pd.read_parquet(WORK / 'parts' / f'{name}.parquet')
    if not np.array_equal(f.mvt_id.to_numpy(), data['rank_ids']):
        raise ValueError(f'{name}: rows differ from the ranking rows')
    cols = [c for c in f.columns if c != 'mvt_id']
    if name.startswith('nn_') and len(cols) != len(GROUPS) * nni.NETWORKS['ple']:
        raise ValueError(f'{name}: expected 15 networks')
    return f[cols].to_numpy(float).mean(axis=1)


def prepare(candidate, version):
    started = time.monotonic()
    jan_jul = evidence(candidate)
    spec = CANDIDATES[candidate]
    directory = OUT / f'submission_v{version}'
    if directory.exists() or submission(version).exists():
        raise ValueError(f'v{version} is already prepared; inspect it before continuing')
    data = pd.read_pickle(WORK / 'inputs.pkl')
    v46 = ts.read(V46 / 'submission_v46.json')
    nn_all = part_values(spec['network'], data)
    gbm_all = np.mean([part_values(g, data) for g in spec['gbm']], axis=0)
    directory.mkdir(parents=True)
    identity = dict(version=version, candidate=candidate, spec=spec, weight=W_NN,
                    signature={n: sha256(ROOT / n) for n in (
                        'VALIDATION/ideas_submit.py', 'VALIDATION/ideas_jan_jul.py', 'VALIDATION/extra_features.py',
                        'VALIDATION/catboost_jan_jul.py', 'VALIDATION/nn_improve_jan_jul.py',
                        'VALIDATION/nn_features_jan_jul.py', 'VALIDATION/tools/score_candidates_jan_jul.py')},
                    extra_features=sha256(ef.PATH), v46_identity=data['v46_identity'], v46_models=data['v46_models'],
                    parts={p: ts.read(WORK / 'parts' / f'{p}.json') for p in [spec['network']] + spec['gbm'] if p != 'v46'},
                    torch=torch.__version__)
    write_json(directory / 'refit.json', dict(identity=identity, runtime=runtime_identity(), started_at=now(),
                                              source=source_snapshot(directory)))
    pieces, masks_all, offset, summary = [], {}, 0, {}
    for month in MONTHS:
        d = data['months'][month]
        n = len(d['raw'])
        nn_base, gbm_base = nn_all[offset:offset + n], gbm_all[offset:offset + n]
        offset += n
        base = W_NN * nn_base + (1 - W_NN) * gbm_base
        frame, masks = nis.final(base, d['p'], d['V'], d['raw'], data['parameters'], data['fitted'])
        frame['lgb_v46_base'], frame['nn_base'], frame['gbm_base'], frame['blend_base'] = d['lgb'], nn_base, gbm_base, base
        frame['month'] = month
        path = directory / f'ranking_{month}.parquet'
        overnight_run.save_frame(path, frame)
        if masks != v46['correction_masks'][month]:
            raise ValueError(f'{month}: correction rows differ from v46')
        masks_all[month] = masks
        summary[month] = dict(rows=n, corr_nn_gbm=float(np.corrcoef(nn_base, gbm_base)[0, 1]),
                              nn_minus_gbm_mean=float((nn_base - gbm_base).mean()),
                              artifacts=overnight_run.artifact_hashes(directory, path))
        pieces.append(frame)
    rank_ids = data['rank_ids']
    frame = pd.concat(pieces, ignore_index=True)
    if frame.mvt_id.duplicated().any() or set(frame.mvt_id) != set(rank_ids):
        raise ValueError('Invalid or incomplete ranking predictions')
    frame = frame.set_index('mvt_id').reindex(rank_ids).reset_index()
    template = pq.read_table(nis.TEMPLATE)
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or frame.rule_triggered.sum() != 43:
        raise ValueError('Invalid predictions or changed rule eligibility')
    table_out = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    transfer.validate_submission(table_out, template)
    mine = table_out.to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    month_of = frame.set_index('mvt_id').month.reindex(mine.index)
    against = {}
    for label, other_path in (('v48', submission(48)), ('v46', submission(46))):
        other = pq.read_table(other_path).to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(mine.index)
        for month in MONTHS:
            k = (month_of == month).to_numpy()
            a, b = mine.to_numpy(float)[k], other.to_numpy(float)[k]
            against.setdefault(label, {})[month] = dict(rows=int(k.sum()), mean_change=float((a - b).mean()),
                mean_abs_change=float(np.abs(a - b).mean()), corr=float(np.corrcoef(a, b)[0, 1]))
    path = submission(version)
    temp = path.with_suffix('.parquet.tmp')
    pq.write_table(table_out, temp)
    temp.replace(path)
    manifest = dict(version=version, file=file_name(version), recipe=f'v48-ideas-{candidate.lower()}', parent='v48',
                    state='prepared', prepared_at=now(), sha256=sha256(path), rows=table_out.num_rows, weight=W_NN,
                    jan_jul_evidence=jan_jul, parts=spec, ranking=summary, correction_masks=masks_all,
                    rule_triggered_rows=int(frame.rule_triggered.sum()), against=against,
                    prediction_summary=frame.pred.describe().to_dict(), identity=identity,
                    external_data=ts.read(V48)['external_data'], minutes=(time.monotonic() - started) / 60)
    write_json(directory / f'submission_v{version}.json', manifest)
    log(f'Prepared {file_name(version)} ({candidate}): SHA256 {manifest["sha256"]}')
    for label, changes in against.items():
        for month, c in changes.items():
            log(f'  {month} against {label}: mean {c["mean_change"]:+.2f} s, mean |change| {c["mean_abs_change"]:.2f} s, '
                f'corr {c["corr"]:.4f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('action', choices=['inputs', 'fit', 'prepare', 'upload', 'result'])
    parser.add_argument('--part', choices=PARTS)
    parser.add_argument('--candidate', choices=sorted(CANDIDATES))
    parser.add_argument('--version', type=int)
    args = parser.parse_args(argv)
    if args.action == 'inputs':
        with threadpool_limits(limits=1, user_api='blas'):
            inputs()
    elif args.action == 'fit':
        fit(args.part)
    elif args.action == 'prepare':
        prepare(args.candidate, args.version)
    else:
        manifest = ts.read(OUT / f'submission_v{args.version}' / f'submission_v{args.version}.json')
        transfer.VERSION, transfer.NAME = args.version, manifest['recipe']
        transfer.RUN = OUT / f'submission_v{args.version}'
        transfer.MANIFEST = transfer.RUN / f'submission_v{args.version}.json'
        (transfer.upload if args.action == 'upload' else transfer.result)()


if __name__ == '__main__':
    main()
