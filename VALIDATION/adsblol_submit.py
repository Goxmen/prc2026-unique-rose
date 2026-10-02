"""Refit and submit v44 with the ADSB.lol inputs in the ordinary regressor (v45, v46).

prepare fits seed 42 on all of 2025 and writes the ranking file; upload performs one
conditional PUT; result reads the official score. This is v44's submission
(`tuned_submit.py`) step for step: v43's inputs, Rome classifier, corrections and rules,
and the Jan/Jul-tuned ordinary regressor (search trial B00, learning rate 0.02, 5,000
rounds). The only recipe change is `adsblol.ADSB_FEATURES` added to the ordinary
regressor, validated by `adsblol_jan_jul.py`; the user dropped the 16-fold validations on
2026-10-02.

    v45  + adsblol.ADSB_FEATURES                     Jan/Jul CV 314.2129 -> 306.9405 (3 seeds)
    v46  + adsblol.ADSB_FEATURES + STAND_FEATURES    Jan/Jul CV 314.2129 -> 303.7536 (3 seeds)

v43/v44's full-year model files are on the Apple M3, and LightGBM refits differ across
machines, so the byte-for-byte parent checks of tuned_submit.py are replaced by: the same
data, OPDI and stand hashes, training population, correction parameters and category values
as v44, the same 43 rule rows, and a comparison with v44's official file.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
from common import RANKING, TEMPLATE, TRAIN_GLOB
import modeling
import overnight_run
from records import RUNS, data_identity, runtime_identity, sha256, source_snapshot, write_json
import rome_hybrid
from rome_hybrid_run import save_booster
from rome_hybrid_submit import FREEZE, KIND, MONTHS, SEED, file_name, now, submission
import stand_coords_submit as v43s
import stand_day_submit as transfer
from submit import fill_template
import tuned_submit as v44s
import tuning_seasonal as ts

cv, features, signals, weather, run, sf = v43s.cv, v43s.features, v43s.signals, v43s.weather, v43s.run, v43s.sf
V44 = ts.DEFAULT / 'submission_v44'
JAN_JUL = RUNS / 'v3_20261002_adsblol_jan_jul'
V44_OFFICIAL = ROOT / 'tmp/adsblol_features_20261002/v44_official/unique-rose_v44.parquet'
OUT = RUNS / 'v3_20261002_adsblol_submit'
RECIPES = {
    45: dict(name='v44-adsblol', evidence='result.json', arm='adsblol', key='adsblol',
             features=adsblol.ADSB_FEATURES, tables=dict(features=adsblol.FEATURES_PATH)),
    46: dict(name='v44-adsblol-stand', evidence='result_stand_3seed.json', arm='stand', key='adsblol_stand',
             features=adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES,
             tables=dict(features=adsblol.FEATURES_PATH, stand_features=adsblol.STAND_PATH)),
}
EVIDENCE, NAME, ORDINARY, MODEL_KEYS, TABLES = None, None, None, None, None


def configure(version):
    global EVIDENCE, NAME, ORDINARY, MODEL_KEYS, TABLES
    r = RECIPES[version]
    EVIDENCE, NAME, TABLES = JAN_JUL / r['evidence'], r['name'], r['tables']
    ORDINARY = run.CANDIDATE + r['features']
    MODEL_KEYS = {'ordinary': f'refit2025__ordinary_{r["key"]}__42', 'classifier': 'refit2025__classifier__42'}
    return r
EXTERNAL = ('ADSB.lol globe_history heatmaps (GitHub adsblol/globe_history_2025 and _2026; CC0 1.0 / ODbL 1.0 '
            'per archive), fetched 2026-10-02')


def evidence(arm):
    result = ts.read(EVIDENCE)
    if result['chosen']['settings'] != v44s.CHOSEN['settings'] or result['seeds'] != [42, 142, 242] \
            or result.get('arm', 'adsblol') != arm:
        raise ValueError('Jan/Jul evidence is not for this arm at v44 settings and three seeds')
    stale = [k for k, path in TABLES.items() if result['source_sha256'].get(k) != sha256(path)]
    if stale:
        raise ValueError(f'ADS-B tables differ from the ones validated: {stale}')
    change = result['seed_average_change']['all']
    if change['decision'] != 'improvement' or any(v['change_all']['change'] >= 0 for v in result['by_seed'].values()):
        raise ValueError('ADS-B inputs do not improve Jan/Jul CV at every seed')
    return dict(control=result['seed_average']['control']['all'], adsblol=result['seed_average']['adsblol']['all'],
                change=change, by_seed={s: v['change_all'] for s, v in result['by_seed'].items()},
                tables_sha256={k: result['source_sha256'][k] for k in TABLES})


def fit_ordinary(directory, progress, X, y, raw, params, rounds):
    key = MODEL_KEYS['ordinary']
    if key in progress['models']:
        return
    started = time.monotonic()
    s = rome_hybrid.substituted(X, y, raw)
    names = rome_hybrid.regressor_features(cv.recipe.KIND, ORDINARY)
    model = signals._train(params, X.loc[~s, names], np.asarray(y, float)[~s], rounds, f'ordinary+{len(ORDINARY)}/{SEED}')
    if model.feature_name() != list(names):
        raise ValueError(f'Model feature order changed: {key}')
    path = directory / 'models' / f'{key}.txt'
    save_booster(model, path)
    progress['models'][key] = dict(path=str(path.relative_to(directory)), part='ordinary', seed=SEED,
        features=model.feature_name(), params=params, rounds=rounds,
        artifacts=overnight_run.artifact_hashes(directory, path), seconds=time.monotonic() - started)
    write_json(directory / 'progress.json', progress)
    print(f'Fitted {key} in {time.monotonic() - started:.1f}s', flush=True)


def predict(ordinary, classifier, V, raw, parameters, fitted):
    out = signals.predict(ordinary, classifier, V, raw, KIND, threads=run.THREADS, extra=ORDINARY,
                          classifier_extra=cv.CLASSIFIER, late_mean=parameters['late_mean'], dayshift_weight=None)
    frame, masks = v43s.v42.parent.apply_corrections(out['final'], raw, V, fitted, out['p'])
    frame['ordinary'] = out['ordinary']
    frame['rule_triggered'] = out['rule_triggered']
    for column in features.FEATURES + sf.FEATURES + [c for c in ORDINARY if c.startswith('ab_')]:
        frame[column] = V[column].to_numpy()
    return frame, {name: int(mask.sum()) for name, mask in masks.items()}


def prepare(version, resume):
    started = time.monotonic()
    recipe = configure(version)
    record = ts.read(run.DEFAULT / 'main' / 'result.json')
    jan_jul = evidence(recipe['arm'])
    v44 = ts.read(V44 / 'submission_v44.json')
    if v44['state'] != 'scored' or v44['version'] != 44 or sha256(V44_OFFICIAL) != v44['sha256']:
        raise ValueError('v44 is not the scored parent, or its official file is not the one uploaded')
    v44_cv_sources = ['VALIDATION/tuned_submit.py', 'VALIDATION/tuning_seasonal.py', 'VALIDATION/stand_coords_submit.py']
    changed = [n for n in v44_cv_sources if v44['identity']['signature'].get(n) != sha256(ROOT / n)]
    if changed:
        raise ValueError(f'v44 submission sources changed: {changed}')
    for key, files in (('data_identity', [*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]),
                       ('opdi_identity', [cv.opdi.FEATURES_PATH]), ('stand_identity', [sf.COORDS, sf.ENDS])):
        if data_identity(files) != v44['identity'][key]:
            raise ValueError(f'{key} differs from v44')
    config = record['config'].copy()
    config.update(threads=run.THREADS, prediction_threads=run.THREADS, fit_workers=1)
    params = v44s.tuned_params()
    if params != v44['params']:
        raise ValueError('Ordinary-regressor parameters differ from v44')
    directory = OUT / f'submission_v{version}'
    manifest_path = directory / f'submission_v{version}.json'
    if manifest_path.exists():
        raise ValueError('Submission is already prepared; use upload/result')
    if submission(version).exists():
        raise ValueError('Submission file already exists; inspect it before continuing')
    sources = {name: sha256(ROOT / name) for name in (
        'VALIDATION/adsblol_submit.py', 'VALIDATION/adsblol.py', 'VALIDATION/adsblol_jan_jul.py',
        'VALIDATION/tools/download_adsblol_heatmap.py', *v44_cv_sources)}
    identity = dict(version=version, signature=sources, config=config, params=params, rounds=v44s.CHOSEN['rounds'],
        data_identity=v44['identity']['data_identity'], opdi_identity=v44['identity']['opdi_identity'],
        stand_identity=v44['identity']['stand_identity'], adsblol_tables_sha256={k: sha256(v) for k, v in TABLES.items()},
        template_sha256=sha256(TEMPLATE), anchors={str(p.relative_to(ROOT)): sha256(p) for p in (
            V44 / 'submission_v44.json', EVIDENCE, V44_OFFICIAL)})
    if identity['adsblol_tables_sha256'] != jan_jul['tables_sha256']:
        raise ValueError('ADS-B feature tables differ from the ones validated')
    if resume:
        if ts.read(directory / 'refit.json')['identity'] != identity:
            raise ValueError('Cannot resume changed code, recipe or data')
        progress = ts.read(directory / 'progress.json')
        overnight_run.check_artifacts(directory, progress)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'models').mkdir()
        write_json(directory / 'refit.json', dict(identity=identity, runtime=runtime_identity(), started_at=now(),
                                                  source=source_snapshot(directory)))
        progress = dict(models={}, predictions={})
        write_json(directory / 'progress.json', progress)
    previous = ts.read(v43s.PARENT_MANIFEST)
    expected = previous['model']['parameters']
    tables, table = sf.load(), (adsblol.load_all() if 'stand_features' in TABLES else adsblol.load_features())
    print(f'v{version}: verified Jan/Jul evidence, v44 parent, source and data hashes; loading inputs', flush=True)
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1., ranking=True)
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    if months != MONTHS or len(rank) != transfer.EXPECTED_ROWS or rank.mvt_id.duplicated().any():
        raise ValueError('Ranking population changed')
    if set(progress['models']) != set(MODEL_KEYS.values()):
        print('Preparing full-year features; fitting the classifier and the ADS-B regressor', flush=True)
        fitted, correction_audit = v43s.v42.fit_corrections(dep, arr, config, expected['late_mean'])
        v43s.v42.check_parameters(fitted, previous['model']['corrections'])
        X, y, cats, raw, parameters, membership = cv.training_inputs(dep, arr, config, reports, grid, expected)
        if len(membership) != 2084659 or membership.excluded_by_rule.sum() != 127 or len(raw) != 2084532:
            raise ValueError('Full-year training population differs from v40-v44')
        if parameters != v44['training']['parameters'] or fitted != v44['training']['corrections']:
            raise ValueError('Training parameters or corrections differ from v44')
        X = adsblol.add_adsblol(sf.add_features(X, raw, tables), raw, table, [c for c in ORDINARY if c.startswith('ab_')])
        overnight_run.save_frame(directory / 'training_membership.parquet', membership)
        overnight_run.save_frame(directory / 'correction_audit.parquet', correction_audit)
        write_json(directory / 'categories.json', overnight_run.category_values(cats))
        if sha256(directory / 'categories.json') != sha256(V44 / 'categories.json'):
            raise ValueError('Category values differ from v44')
        progress['training'] = dict(parameters=parameters, corrections=fitted,
            rows_before_exclusion=len(membership), excluded=127, rows=len(raw),
            ordinary_rows=int(membership.used_by_ordinary.sum()),
            classifier_rows=int(membership.used_by_classifier.sum()), stand_coverage=sf.coverage(X),
            adsblol_coverage=dict(matched=float(np.nanmean(X.ab_matched)), seg=float(X.ab_seg_sec.notna().mean())),
            artifacts=overnight_run.artifact_hashes(directory, directory / 'training_membership.parquet',
                directory / 'correction_audit.parquet', directory / 'categories.json'))
        write_json(directory / 'progress.json', progress)
        cv.fitted_model(directory, progress, MODEL_KEYS['classifier'], X, y, raw, 'classifier',
                        cv.CLASSIFIER, SEED, config)
        fit_ordinary(directory, progress, X, y, raw, params, v44s.CHOSEN['rounds'])
        del X, y, raw, membership, correction_audit
        gc.collect()
    cats = overnight_run.read_categories(directory / 'categories.json')
    parameters, fitted = (progress['training'][name] for name in ('parameters', 'corrections'))
    boosters = {part: run.load_model(directory, progress, key) for part, key in MODEL_KEYS.items()}
    run.check_classifier(boosters['classifier'])
    pieces, all_masks = [], {}
    for month in MONTHS:
        path = directory / f'ranking_{month}.parquet'
        if month in progress['predictions']:
            cv.prior.checked_file(directory, progress['predictions'][month])
            frame, masks = pd.read_parquet(path), progress['predictions'][month]['masks']
        else:
            print(f'v{version}: predicting ranking month {month}', flush=True)
            raw, ra = modeling.month_rows(rank, month), modeling.month_rows(rank_arr, month)
            V = modeling.validation_matrix(dep, arr, raw, ra, FREEZE, cats, config)
            arrivals = pd.concat([arr, ra], ignore_index=True)
            V = cv.recipe.add_inputs(V, raw, reports, pd.concat([dep, raw], ignore_index=True), arrivals)
            V = features.add_features(V, raw, arrivals, grid)
            V = adsblol.add_adsblol(sf.add_features(V, raw, tables), raw, table, [c for c in ORDINARY if c.startswith('ab_')])
            frame, masks = predict(boosters['ordinary'], boosters['classifier'], V, raw, parameters, fitted)
            frame['month'] = month
            overnight_run.save_frame(path, frame)
            progress['predictions'][month] = dict(path=path.name, rows=len(frame), masks=masks,
                stand_coverage=sf.coverage(V), adsblol_coverage=dict(
                    matched=float(np.nanmean(V.ab_matched)), seg=float(V.ab_seg_sec.notna().mean())),
                artifacts=overnight_run.artifact_hashes(directory, path))
            write_json(directory / 'progress.json', progress)
            del V, arrivals, raw, ra
            gc.collect()
        if masks != v44['correction_masks'][month]:
            raise ValueError(f'{month}: correction rows differ from v44')
        pieces.append(frame)
        all_masks[month] = masks
    frame = pd.concat(pieces, ignore_index=True)
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
        raise ValueError('Invalid prediction movement IDs')
    if len(frame) != len(rank) or set(frame.mvt_id) != set(rank.mvt_id):
        raise ValueError('Incomplete ranking predictions')
    frame = frame.set_index('mvt_id').reindex(rank.mvt_id).reset_index()
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or frame.rule_triggered.sum() != 43:
        raise ValueError('Invalid predictions or changed rule eligibility')
    template = pq.read_table(TEMPLATE)
    table_out = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    transfer.validate_submission(table_out, template)
    mine = table_out.to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    month_of = frame.set_index('mvt_id').month.reindex(mine.index)

    def compare(other_path):
        other = pq.read_table(other_path).to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(mine.index)
        out = {}
        for month in MONTHS:
            k = (month_of == month).to_numpy()
            a, b = mine.to_numpy(float)[k], other.to_numpy(float)[k]
            out[month] = dict(rows=int(k.sum()), mean_change=float((a - b).mean()), mean_abs_change=float(np.abs(a - b).mean()),
                              rmse_change=float(np.sqrt(((a - b) ** 2).mean())), corr=float(np.corrcoef(a, b)[0, 1]))
        return out

    against = compare(V44_OFFICIAL)
    against_previous = {}
    if version > 45:
        v45 = ts.read(OUT / 'submission_v45' / 'submission_v45.json')
        if v45['state'] != 'scored' or sha256(submission(45)) != v45['sha256']:
            raise ValueError('v45 is not scored, or its local file is not the one uploaded')
        against_previous = compare(submission(45))
    path = submission(version)
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix('.parquet.tmp')
    pq.write_table(table_out, temp)
    temp.replace(path)
    manifest = dict(version=version, file=file_name(version), recipe=NAME, parent='v44', state='prepared',
        prepared_at=now(), sha256=sha256(path), rows=table_out.num_rows, seed=SEED, rounds=v44s.CHOSEN['rounds'],
        learning_rate=v44s.CHOSEN['learning_rate'], tuned_settings=v44s.CHOSEN['settings'], params=params,
        classifier_rounds=300, freeze=FREEZE, ranking_months=MONTHS, jan_jul_evidence=jan_jul,
        ordinary_extra=ORDINARY, classifier_extra=cv.CLASSIFIER, adsblol=dict(
            features=[c for c in ORDINARY if c.startswith('ab_')], tables_sha256=identity['adsblol_tables_sha256']),
        disruption=features.SETTINGS, stand=sf.SETTINGS, late=True, dayshift=False, exclude_rule_rows=True,
        model=progress['models'], training=progress['training'], predictions=progress['predictions'],
        correction_masks=all_masks, rule_triggered_rows=int(frame.rule_triggered.sum()),
        against_v44_official=against, against_v45=against_previous, prediction_summary=frame.pred.describe().to_dict(), identity=identity,
        external_data=v44['external_data'] + '; ' + EXTERNAL, minutes=(time.monotonic() - started) / 60)
    write_json(manifest_path, manifest)
    print(f'Prepared {file_name(version)}: {len(frame):,} rows, SHA256 {manifest["sha256"]}', flush=True)
    for label, changes in (('v44', against), ('v45', against_previous)):
        for month, change in changes.items():
            print(f'  {month} against {label}: mean {change["mean_change"]:+.2f} s, mean |change| '
                  f'{change["mean_abs_change"]:.2f} s, corr {change["corr"]:.4f}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload', 'result'])
    parser.add_argument('--version', type=int, default=45)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        with threadpool_limits(limits=1, user_api='blas'):
            prepare(args.version, args.resume)
    else:
        configure(args.version)
        transfer.VERSION, transfer.NAME = args.version, NAME
        transfer.RUN = OUT / f'submission_v{args.version}'
        transfer.MANIFEST = transfer.RUN / f'submission_v{args.version}.json'
        (transfer.upload if args.action == 'upload' else transfer.result)()


if __name__ == '__main__':
    main()
