"""Submit v46 blended with the Jan/Jul-validated neural network (v47 by default).

prepare rebuilds v46's full-year training rows and ranking inputs, reuses v46's saved
full-year boosters (ordinary regressor and Rome classifier, seed 42) unchanged, fits one
network on all of 2025 with `nn_jan_jul.fit_nn` (the validated recipe, seed 42), and
blends the two ordinary predictions 0.7 LightGBM / 0.3 network before v46's Rome
classifier mix, corrections and four-hour rule. upload performs one conditional PUT;
result reads the official score.

Evidence: `VALIDATION/tools/score_nn_jan_jul.py` (run v3_20261002_nn_v46_jan_jul): Jan/Jul CV
303.7536 -> 302.3701 at three seeds, -1.38 [-1.94, -0.95], every seed improves.

Checks: v46 scored and its local file is the one uploaded; the network runner is the one
validated; data/OPDI/stand/ADS-B hashes, training population, correction parameters and
category values equal v46's; with weight 0 the pipeline reproduces v46's saved ranking
predictions; the same correction rows and 43 rule rows. GPU training is not bit-for-bit
repeatable, so the network's ranking predictions are saved with the submission.
"""
import argparse
import gc
import glob
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import adsblol_submit as v46s
from common import RANKING, TEMPLATE, TRAIN_GLOB
import modeling
import nn_jan_jul
import overnight_run
from records import RUNS, data_identity, runtime_identity, sha256, source_snapshot, write_json
import rome_hybrid
from rome_hybrid_submit import FREEZE, KIND, MONTHS, SEED, file_name, now, submission
import stand_day_submit as transfer
from submit import fill_template
import tuning_seasonal as ts

v43s = v46s.v43s
cv, features, signals, weather, run, sf = v43s.cv, v43s.features, v43s.signals, v43s.weather, v43s.run, v43s.sf
V46 = v46s.OUT / 'submission_v46'
NN_RUN = RUNS / 'v3_20261002_nn_v46_jan_jul'
OUT = RUNS / 'v3_20261002_nn_submit'
NAME = 'v46-nn-blend'
WEIGHT = 0.3
AB = adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES
ORDINARY = run.CANDIDATE + AB
MODEL_KEYS = {'ordinary': 'refit2025__ordinary_adsblol_stand__42', 'classifier': 'refit2025__classifier__42'}


def evidence():
    score, result = ts.read(NN_RUN / 'score_3seed.json'), ts.read(NN_RUN / 'result_seeds42_142_242.json')
    if score['weight'] != WEIGHT or score['seeds'] != [42, 142, 242] or result['features'] != AB:
        raise ValueError('Network evidence is not for this weight, these seeds or v46 inputs')
    if result['source_sha256']['runner'] != sha256(ROOT / 'VALIDATION/nn_jan_jul.py'):
        raise ValueError('nn_jan_jul.py changed since it was validated')
    if result['recipe'] != nn_jan_jul.RECIPE:
        raise ValueError('Network recipe differs from the validated one')
    change = score['change']['blend']
    if change['decision'] != 'improvement' or any(v['blend'] >= v['lgb'] for v in score['by_seed'].values()):
        raise ValueError('The blend does not improve Jan/Jul CV at every seed')
    return dict(level=score['level'], change=change, by_seed=score['by_seed'],
                runner_sha256=result['source_sha256']['runner'])


def prepare(version):
    started = time.monotonic()
    jan_jul = evidence()
    v46 = ts.read(V46 / 'submission_v46.json')
    if v46['state'] != 'scored' or v46['version'] != 46 or sha256(submission(46)) != v46['sha256']:
        raise ValueError('v46 is not the scored parent, or its local file is not the one uploaded')
    for key, files in (('data_identity', [*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]),
                       ('opdi_identity', [cv.opdi.FEATURES_PATH]), ('stand_identity', [sf.COORDS, sf.ENDS])):
        if data_identity(files) != v46['identity'][key]:
            raise ValueError(f'{key} differs from v46')
    tables_sha = dict(features=sha256(adsblol.FEATURES_PATH), stand_features=sha256(adsblol.STAND_PATH))
    if tables_sha != v46['identity']['adsblol_tables_sha256']:
        raise ValueError('ADS-B tables differ from v46')
    v46_progress = ts.read(V46 / 'progress.json')
    overnight_run.check_artifacts(V46, v46_progress)
    directory = OUT / f'submission_v{version}'
    if directory.exists() or submission(version).exists():
        raise ValueError(f'v{version} is already prepared; inspect it before continuing')
    directory.mkdir(parents=True)
    (directory / 'models').mkdir()
    identity = dict(version=version, parent='v46', weight=WEIGHT, recipe=nn_jan_jul.RECIPE, seed=SEED,
        signature={n: sha256(ROOT / n) for n in ('VALIDATION/nn_submit.py', 'VALIDATION/nn_jan_jul.py',
                   'VALIDATION/tools/score_nn_jan_jul.py', 'VALIDATION/adsblol_submit.py', 'VALIDATION/adsblol.py')},
        v46_identity=v46['identity'], parent_models={k: v46_progress['models'][k]['artifacts'] for k in MODEL_KEYS.values()},
        anchors={str(p.relative_to(ROOT)): sha256(p) for p in (V46 / 'submission_v46.json', submission(46),
                 NN_RUN / 'score_3seed.json', NN_RUN / 'result_seeds42_142_242.json')},
        torch=torch.__version__, device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu')
    write_json(directory / 'refit.json', dict(identity=identity, runtime=runtime_identity(), started_at=now(),
                                              source=source_snapshot(directory)))
    config = v46['identity']['config']
    previous = ts.read(v43s.PARENT_MANIFEST)
    expected = previous['model']['parameters']
    tables, table = sf.load(), adsblol.load_all()
    print(f'v{version}: verified network evidence, v46 parent, source and data hashes; loading inputs', flush=True)
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1., ranking=True)
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    if months != MONTHS or len(rank) != transfer.EXPECTED_ROWS or rank.mvt_id.duplicated().any():
        raise ValueError('Ranking population changed')

    # Full-year training rows, exactly as v46 built them.
    fitted, _ = v43s.v42.fit_corrections(dep, arr, config, expected['late_mean'])
    v43s.v42.check_parameters(fitted, previous['model']['corrections'])
    X, y, cats, raw, parameters, membership = cv.training_inputs(dep, arr, config, reports, grid, expected)
    if len(membership) != 2084659 or membership.excluded_by_rule.sum() != 127 or len(raw) != 2084532:
        raise ValueError('Full-year training population differs from v46')
    if parameters != v46['training']['parameters'] or fitted != v46['training']['corrections']:
        raise ValueError('Training parameters or corrections differ from v46')
    write_json(directory / 'categories.json', overnight_run.category_values(cats))
    if sha256(directory / 'categories.json') != sha256(V46 / 'categories.json'):
        raise ValueError('Category values differ from v46')
    X = adsblol.add_adsblol(sf.add_features(X, raw, tables), raw, table, AB)
    s = rome_hybrid.substituted(X, y, raw)
    names = rome_hybrid.regressor_features(cv.recipe.KIND, ORDINARY)
    boosters = {part: run.load_model(V46, v46_progress, key) for part, key in MODEL_KEYS.items()}
    run.check_classifier(boosters['classifier'])
    if boosters['ordinary'].feature_name() != list(names):
        raise ValueError('v46 ordinary regressor inputs differ from the network inputs')
    Xn, yn = X.loc[~s, names], np.asarray(y, float)[~s]
    if len(Xn) != v46['training']['ordinary_rows']:
        raise ValueError('Network training rows differ from v46 ordinary regressor rows')
    training = dict(rows=len(Xn), substituted_rows=int(s.sum()), ordinary_rows_v46=v46['training']['ordinary_rows'])
    del X, y, raw, membership
    gc.collect()

    # Ranking inputs and v46's own predictions, reproduced before blending.
    cats = overnight_run.read_categories(directory / 'categories.json')
    months_data = {}
    for month in MONTHS:
        print(f'v{version}: ranking inputs {month}', flush=True)
        raw_m, ra = modeling.month_rows(rank, month), modeling.month_rows(rank_arr, month)
        V = modeling.validation_matrix(dep, arr, raw_m, ra, FREEZE, cats, config)
        arrivals = pd.concat([arr, ra], ignore_index=True)
        V = cv.recipe.add_inputs(V, raw_m, reports, pd.concat([dep, raw_m], ignore_index=True), arrivals)
        V = features.add_features(V, raw_m, arrivals, grid)
        V = adsblol.add_adsblol(sf.add_features(V, raw_m, tables), raw_m, table, AB)
        lgb_base = signals.ordinary_prediction(boosters['ordinary'], V, KIND, run.THREADS, ORDINARY)
        p = signals.substitution_probability(boosters['classifier'], V, raw_m, KIND, run.THREADS, cv.CLASSIFIER)
        saved = pd.read_parquet(V46 / f'ranking_{month}.parquet').set_index('mvt_id').reindex(raw_m.mvt_id)
        frame0, _ = final(lgb_base, p, V, raw_m, parameters, fitted)
        gaps = dict(ordinary=float(np.max(np.abs(lgb_base - saved.ordinary.to_numpy(float)))),
                    final=float(np.max(np.abs(frame0.pred.to_numpy(float) - saved.pred.to_numpy(float)))))
        if max(gaps.values()) > 1e-6:
            raise ValueError(f'{month}: v46 ranking predictions not reproduced ({gaps})')
        print(f'  reproduces v46 to {max(gaps.values()):.2e} s', flush=True)
        months_data[month] = dict(V=V, raw=raw_m, lgb=lgb_base, p=p, reproduction=gaps)
        del arrivals, ra
        gc.collect()
    del dep, arr, rank_arr
    gc.collect()

    # One network on all of 2025; predicts both ranking months.
    parts = [months_data[m]['V'][names].copy() for m in MONTHS]
    for c in names:   # month-specific category sets (e.g. airline): widen to the union, values unchanged
        if isinstance(Xn[c].dtype, pd.CategoricalDtype):
            values = [q[c].cat.categories.to_series() if isinstance(q[c].dtype, pd.CategoricalDtype)
                      else pd.Series(q[c].dropna().unique()) for q in parts]
            union = pd.CategoricalDtype(pd.Index(pd.concat(values, ignore_index=True).unique()))
            for q in parts:
                q[c] = q[c].astype(object).astype(union)
    Vn = pd.concat(parts, ignore_index=True)
    del parts
    bad = [c for c in Xn.columns if isinstance(Xn[c].dtype, pd.CategoricalDtype)
           and not isinstance(Vn[c].dtype, pd.CategoricalDtype)]
    if bad:
        raise ValueError(f'Ranking categorical inputs lost their categories: {bad}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'v{version}: fitting the network on {len(Xn):,} rows ({device})', flush=True)
    t0 = time.monotonic()
    item = dict(X=Xn, y=yn, Vn=Vn, raw=pd.DataFrame({'taxi_sec': np.full(len(Vn), np.nan)}))
    nn_all, curve, info = nn_jan_jul.fit_nn(item, SEED, device)
    network = dict(seed=SEED, seconds=time.monotonic() - t0, train_curve=[
        dict(epoch=c['epoch'], train_mse_std=c['train_mse_std']) for c in curve], info=info, training=training)
    del item, Xn, yn, Vn
    gc.collect()

    pieces, masks_all, offset = [], {}, 0
    for month in MONTHS:
        d = months_data[month]
        nn_base = nn_all[offset:offset + len(d['raw'])]
        offset += len(d['raw'])
        base = (1 - WEIGHT) * d['lgb'] + WEIGHT * nn_base
        frame, masks = final(base, d['p'], d['V'], d['raw'], parameters, fitted)
        frame['lgb_base'], frame['nn_base'], frame['blend_base'] = d['lgb'], nn_base, base
        frame['month'] = month
        path = directory / f'ranking_{month}.parquet'
        overnight_run.save_frame(path, frame)
        if masks != v46['correction_masks'][month]:
            raise ValueError(f'{month}: correction rows differ from v46')
        masks_all[month] = masks
        network.setdefault('ranking', {})[month] = dict(
            rows=len(frame), corr_with_lgb=float(np.corrcoef(d['lgb'], nn_base)[0, 1]),
            nn_minus_lgb_mean=float((nn_base - d['lgb']).mean()), reproduction=d['reproduction'],
            artifacts=overnight_run.artifact_hashes(directory, path))
        pieces.append(frame)
    if offset != len(nn_all):
        raise ValueError('Network prediction rows do not match the ranking months')
    frame = pd.concat(pieces, ignore_index=True)
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any() or set(frame.mvt_id) != set(rank.mvt_id):
        raise ValueError('Invalid or incomplete ranking predictions')
    frame = frame.set_index('mvt_id').reindex(rank.mvt_id).reset_index()
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or frame.rule_triggered.sum() != 43:
        raise ValueError('Invalid predictions or changed rule eligibility')
    template = pq.read_table(TEMPLATE)
    table_out = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    transfer.validate_submission(table_out, template)
    mine = table_out.to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    month_of = frame.set_index('mvt_id').month.reindex(mine.index)
    against = {}
    for label, other_path in (('v46', submission(46)), ('v45', submission(45))):
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
    manifest = dict(version=version, file=file_name(version), recipe=NAME, parent='v46', state='prepared',
        prepared_at=now(), sha256=sha256(path), rows=table_out.num_rows, weight=WEIGHT, jan_jul_evidence=jan_jul,
        lightgbm=dict(reused_from=str(V46.relative_to(ROOT)), models=MODEL_KEYS), network=network,
        ordinary_extra=ORDINARY, classifier_extra=cv.CLASSIFIER, correction_masks=masks_all,
        rule_triggered_rows=int(frame.rule_triggered.sum()), against=against,
        prediction_summary=frame.pred.describe().to_dict(), identity=identity,
        external_data=v46['external_data'], minutes=(time.monotonic() - started) / 60)
    write_json(directory / f'submission_v{version}.json', manifest)
    print(f'Prepared {file_name(version)}: {len(frame):,} rows, SHA256 {manifest["sha256"]}', flush=True)
    for label, changes in against.items():
        for month, c in changes.items():
            print(f'  {month} against {label}: mean {c["mean_change"]:+.2f} s, mean |change| '
                  f'{c["mean_abs_change"]:.2f} s, corr {c["corr"]:.4f}', flush=True)


def final(base, p, V, raw, parameters, fitted):
    out = signals.combine(base, p, V, raw, late_mean=parameters['late_mean'], dayshift_weight=None)
    frame, masks = v43s.v42.parent.apply_corrections(out['final'], raw, V, fitted, out['p'])
    frame['ordinary'] = base
    frame['rule_triggered'] = out['rule_triggered']
    return frame, {name: int(mask.sum()) for name, mask in masks.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload', 'result'])
    parser.add_argument('--version', type=int, default=47)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        with threadpool_limits(limits=1, user_api='blas'):
            prepare(args.version)
    else:
        transfer.VERSION, transfer.NAME = args.version, NAME
        transfer.RUN = OUT / f'submission_v{args.version}'
        transfer.MANIFEST = transfer.RUN / f'submission_v{args.version}.json'
        (transfer.upload if args.action == 'upload' else transfer.result)()


if __name__ == '__main__':
    main()
