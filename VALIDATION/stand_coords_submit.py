"""Refit and submit v42 + the 3 stand-position inputs (row 46; v43 by default).

prepare [--resume] fits seed 42 on all of 2025 and writes the ranking file;
upload performs one conditional PUT; result reads the official score. The
validated CV run (`stand_coords_run.py --mode main`) is checked, not changed.
This follows `opdi_disruption_submit.py` (v42) step for step; the only recipe
change is the ordinary regressor's extra inputs.
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
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import RANKING, TEMPLATE, TRAIN_GLOB
import modeling
import opdi_disruption_submit as v42
import overnight_run
from records import data_identity, runtime_identity, sha256, source_snapshot, write_json
from rome_hybrid_submit import FREEZE, KIND, MONTHS, SEED, file_name, now, submission
import stand_coords_run as run
import stand_features as sf
import stand_day_submit as transfer
from submit import fill_template

cv, features, signals, weather, opdi = v42.cv, v42.features, v42.signals, v42.weather, v42.opdi
CV_DIR = run.DEFAULT / 'main'
SCORES = ROOT / 'VALIDATION/reports/stand_coords_experiment/main.json'
PARENT_MANIFEST = v42.PARENT_MANIFEST          # v40's: the same correction parameters v42 checked
NAME = 'v42-stand-coordinates'
ORDINARY = run.CANDIDATE
MODEL_KEYS = {'ordinary': 'refit2025__ordinary__42', 'classifier': 'refit2025__classifier__42'}
EXTERNAL = ('X-Plane Scenery Gateway airport data (GPL v2 or later) and OpenStreetMap '
            'aeroway=parking_position/runway (ODbL), fetched 2026-09-25')


def validated():
    cv.historical('v40')
    record, scores = cv.read(CV_DIR / 'result.json'), cv.read(SCORES)
    c = record['config']
    if (record['status'] != 'complete' or record['mode'] != 'main' or len(record['folds']) != 16
            or record['fits'] != 54 or c['smoke'] or c['sample'] != 1 or c['rounds'] != 800
            or c['seeds'] != [42, 142, 242] or c['recovery'] != 'off'
            or c['prediction_rule'] != 'rome_4h' or not c['exclude_rule_rows']
            or not c['late'] or c['dayshift'] or c['disruption'] != features.SETTINGS
            or c['stand'] != sf.SETTINGS or c['control_ordinary_extra'] != cv.EXTRA
            or c['candidate_ordinary_extra'] != ORDINARY or c['classifiers']['opdi'] != cv.CLASSIFIER):
        raise ValueError('Not the completed, validated stand-coordinates recipe')
    changed = [name for name, digest in record['signature'].items() if sha256(ROOT / name) != digest]
    if changed:
        raise ValueError(f'Validated sources changed: {changed}')
    if scores['run_record_sha256'] != sha256(CV_DIR / 'result.json'):
        raise ValueError('Scores do not belong to this validation run')
    change = scores['changes']['1']['weighted']['R_combined']
    if change['decision'] != 'improvement' or scores['changes']['3']['weighted']['R_combined']['decision'] != 'improvement':
        raise ValueError('Weighted CV gain is not supported')
    if data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]) != record['data_identity']:
        raise ValueError('Movement or weather data changed since validation')
    if data_identity([opdi.FEATURES_PATH]) != record['opdi_identity']:
        raise ValueError('OPDI cache differs from the validated run')
    if data_identity([sf.COORDS, sf.ENDS]) != record['stand_identity']:
        raise ValueError('Stand tables differ from the validated run')
    return record, scores


def predict(ordinary, classifier, V, raw, parameters, fitted):
    out = signals.predict(ordinary, classifier, V, raw, KIND, threads=run.THREADS,
        extra=ORDINARY, classifier_extra=cv.CLASSIFIER,
        late_mean=parameters['late_mean'], dayshift_weight=None)
    frame, masks = v42.parent.apply_corrections(out['final'], raw, V, fitted, out['p'])
    frame['ordinary'] = out['ordinary']
    frame['rule_triggered'] = out['rule_triggered']
    for column in features.FEATURES + sf.FEATURES:
        frame[column] = V[column].to_numpy()
    return frame, {name: int(mask.sum()) for name, mask in masks.items()}


def prepare(version, resume):
    started = time.monotonic()
    record, scores = validated()
    config = record['config'].copy()
    config.update(threads=run.THREADS, prediction_threads=run.THREADS, fit_workers=1)
    directory = run.DEFAULT / f'submission_v{version}'
    manifest_path = directory / f'submission_v{version}.json'
    if manifest_path.exists():
        raise ValueError('Submission is already prepared; use upload/result')
    if submission(version).exists():
        raise ValueError('Submission file already exists; inspect it before continuing')
    sources = {**record['signature'], **{name: sha256(ROOT / name) for name in (
        'VALIDATION/stand_coords_submit.py', 'VALIDATION/opdi_disruption_submit.py',
        'VALIDATION/opdi_submit.py', 'VALIDATION/stand_day_submit.py', 'VALIDATION/rome_hybrid_submit.py',
        'VALIDATION/overnight_submit.py', 'VALIDATION/submit.py')}}
    identity = dict(version=version, signature=sources, config=config,
        data_identity=record['data_identity'], opdi_identity=record['opdi_identity'],
        stand_identity=record['stand_identity'], template_sha256=sha256(TEMPLATE),
        anchors={str(p.relative_to(ROOT)): sha256(p) for p in (CV_DIR / 'result.json', SCORES, PARENT_MANIFEST)})
    if resume:
        saved = cv.read(directory / 'refit.json')
        if saved['identity'] != identity:
            raise ValueError('Cannot resume changed code, recipe, data or validation evidence')
        progress = cv.read(directory / 'progress.json')
        overnight_run.check_artifacts(directory, progress)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'models').mkdir()
        write_json(directory / 'refit.json', dict(identity=identity, runtime=runtime_identity(),
            started_at=now(), source=source_snapshot(directory)))
        progress = dict(models={}, predictions={})
        write_json(directory / 'progress.json', progress)
    previous = cv.read(PARENT_MANIFEST)
    expected = previous['model']['parameters']
    tables = sf.load()
    print(f'v{version}: verified CV, source and data hashes; loading local inputs', flush=True)
    reports = weather.load_reports()
    grid = features.snow_grid(pd.read_parquet(weather.METAR))
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1., ranking=True)
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    if months != MONTHS or len(rank) != transfer.EXPECTED_ROWS or rank.mvt_id.duplicated().any():
        raise ValueError('Ranking population changed')
    if not rank.mvt_id.isin(cv.recipe.table().index).all():
        raise ValueError('Ranking rows missing from OPDI cache')
    if set(progress['models']) != set(MODEL_KEYS.values()):
        print('Preparing full-year features and fitting two boosters', flush=True)
        fitted, correction_audit = v42.fit_corrections(dep, arr, config, expected['late_mean'])
        v42.check_parameters(fitted, previous['model']['corrections'])
        X, y, cats, raw, parameters, membership = cv.training_inputs(
            dep, arr, config, reports, grid, expected)
        if len(membership) != 2084659 or membership.excluded_by_rule.sum() != 127 or len(raw) != 2084532:
            raise ValueError('Full-year training population differs from v40/v42')
        X = sf.add_features(X, raw, tables)
        overnight_run.save_frame(directory / 'training_membership.parquet', membership)
        overnight_run.save_frame(directory / 'correction_audit.parquet', correction_audit)
        write_json(directory / 'categories.json', overnight_run.category_values(cats))
        progress['training'] = dict(parameters=parameters, corrections=fitted,
            rows_before_exclusion=len(membership), excluded=127, rows=len(raw),
            ordinary_rows=int(membership.used_by_ordinary.sum()),
            classifier_rows=int(membership.used_by_classifier.sum()), stand_coverage=sf.coverage(X),
            artifacts=overnight_run.artifact_hashes(directory, directory / 'training_membership.parquet',
                directory / 'correction_audit.parquet', directory / 'categories.json'))
        write_json(directory / 'progress.json', progress)
        for part, extra in [('ordinary', ORDINARY), ('classifier', cv.CLASSIFIER)]:
            cv.fitted_model(directory, progress, MODEL_KEYS[part], X, y, raw, part, extra, SEED, config)
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
            frame = pd.read_parquet(path)
            masks = progress['predictions'][month]['masks']
        else:
            print(f'v{version}: predicting ranking month {month}', flush=True)
            raw, ra = modeling.month_rows(rank, month), modeling.month_rows(rank_arr, month)
            V = modeling.validation_matrix(dep, arr, raw, ra, FREEZE, cats, config)
            arrivals = pd.concat([arr, ra], ignore_index=True)
            V = cv.recipe.add_inputs(V, raw, reports, pd.concat([dep, raw], ignore_index=True), arrivals)
            V = features.add_features(V, raw, arrivals, grid)
            V = sf.add_features(V, raw, tables)
            frame, masks = predict(boosters['ordinary'], boosters['classifier'], V, raw, parameters, fitted)
            frame['month'] = month
            overnight_run.save_frame(path, frame)
            progress['predictions'][month] = dict(path=path.name, rows=len(frame), masks=masks,
                stand_coverage=sf.coverage(V), artifacts=overnight_run.artifact_hashes(directory, path))
            write_json(directory / 'progress.json', progress)
            del V, arrivals, raw, ra
            gc.collect()
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
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    transfer.validate_submission(table, template)
    path = submission(version)
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix('.parquet.tmp')
    pq.write_table(table, temp)
    temp.replace(path)
    candidate = scores['scores']['candidate']
    manifest = dict(version=version, file=file_name(version), recipe=NAME, state='prepared',
        prepared_at=now(), sha256=sha256(path), rows=table.num_rows, seed=SEED, rounds=800,
        classifier_rounds=300, freeze=FREEZE, ranking_months=MONTHS,
        validation_run=str(CV_DIR.relative_to(ROOT)),
        cv={**{k: v['score'] for k, v in candidate['unweighted'].items()},
            'weighted': candidate['weighted']['R_combined']['score']},
        matched_control_cv=scores['scores']['control'],
        ordinary_extra=ORDINARY, classifier_extra=cv.CLASSIFIER, disruption=features.SETTINGS,
        stand=sf.SETTINGS, late=True, dayshift=False, exclude_rule_rows=True,
        model=progress['models'], training=progress['training'], predictions=progress['predictions'],
        correction_masks=all_masks, rule_triggered_rows=int(frame.rule_triggered.sum()),
        stand_located_ranking_rows=int(frame.sc_x_m.notna().sum()),
        prediction_summary=frame.pred.describe().to_dict(), identity=identity,
        external_data=f"{previous['external_data']}; {EXTERNAL}",
        minutes=(time.monotonic() - started) / 60)
    write_json(manifest_path, manifest)
    print(f'Prepared {file_name(version)}: {len(frame):,} rows, SHA256 {manifest["sha256"]}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload', 'result'])
    parser.add_argument('--version', type=int, default=43)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        with threadpool_limits(limits=1, user_api='blas'):
            prepare(args.version, args.resume)
    else:
        transfer.VERSION, transfer.NAME = args.version, NAME
        transfer.RUN = run.DEFAULT / f'submission_v{args.version}'
        transfer.MANIFEST = transfer.RUN / f'submission_v{args.version}.json'
        (transfer.upload if args.action == 'upload' else transfer.result)()


if __name__ == '__main__':
    main()
