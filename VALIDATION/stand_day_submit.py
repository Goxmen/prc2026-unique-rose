"""Full-year refit, ranking predictions and upload for v30, Stand-clock.

v30 is row 30 of SUBMISSIONS.md: row 29 (Rome-signals) plus the five stand clock
inputs of stand_day.py in both the ordinary regressor and the Rome classifier,
validated as candidate `stand` in run v3_20260918T151436598224Z:

    ordinary regressor   row 29's inputs (row 28's + airline) + STAND_FEATURES
    Rome classifier      row 29's inputs (+ airline, handler share and count,
                         flight-number format) + STAND_FEATURES
    late-row estimate    on, the late mean fitted on the 2025 training labels
    day-shift            off

    prepare  one seed-42, 800-round fit of both boosters and the late mean on all of 2025,
             then predictions for the ranking set. Each ranking month sees all of 2025 plus
             its own month only, as for v26-v29, so its stand clock reads that month's own
             arrivals (and 2025's before them).
    upload   one conditional PUT, refused if five files were already uploaded today (UTC).
    result   read the official result file once it exists.

Run: .venv/bin/python VALIDATION/stand_day_submit.py prepare|upload|result
"""

import argparse
import gc
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import RANKING, TEMPLATE, TRAIN_GLOB
import overnight_run
from overnight_submit import BUCKET, EXPECTED_ROWS, bucket_listing, uploads_today, validate_submission
from records import RUNS, data_identity, sha256, write_json
from rome_hybrid_run import save_booster
from rome_hybrid_submit import FREEZE, KIND, MONTHS, SEED, file_name, islands, now, remote_digest, submission
import rome_signals as signals
import rome_signals_submit as v29
import stand_day_run as run
from submit import fill_template
from tools.sweep_taxi_thresholds import mc
import weather

VERSION = 30
NAME = 'Stand-clock'
RUN = RUNS / 'v3_20260918T151436598224Z'
CANDIDATE = 'stand'
ORDINARY = run.ORDINARIES['stand']
CLASSIFIER = run.CLASSIFIERS['stand']
MANIFEST = RUN / f'submission_v{VERSION}.json'
PREVIOUS = v29.RUN / 'refit' / f'v{v29.VERSION}.parquet'
SOURCES = ['VALIDATION/stand_day_submit.py', 'VALIDATION/rome_signals_submit.py', 'VALIDATION/rome_hybrid_submit.py',
           'VALIDATION/overnight_submit.py', 'VALIDATION/submit.py']


def validated_run():
    record = json.loads((RUN / 'result.json').read_text())
    config = record['config']
    if (record['status'] != 'complete' or config['smoke'] or config['sample'] != 1.0
            or config['rounds'] != 800 or SEED not in config['seeds']
            or config['candidate_specs'].get(CANDIDATE) != dict(ordinary='stand', classifier='stand')
            or not config['late'] or config['dayshift']
            or config['ordinaries']['stand'] != list(ORDINARY)
            or config['classifiers']['stand'] != list(CLASSIFIER)):
        raise ValueError(f'{RUN.name} is not a completed full validation run of row 30')
    changed = sorted(name for name, digest in record['signature'].items() if sha256(ROOT / name) != digest)
    if changed:
        raise ValueError(f'Sources changed since validation: {changed}')
    if data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]) != record['data_identity']:
        raise ValueError('Training, ranking or METAR data changed since validation')
    return record, config


def fit(X, y, raw_kept, config, threads):
    parameters = signals.fit_parameters(X, y, raw_kept)
    print(f"v{VERSION}: late mean {parameters['late_mean']:.1f} s from {parameters['late_rows']} rows", flush=True)
    started = time.monotonic()
    ordinary = signals.fit_ordinary(X, y, raw_kept, KIND, SEED, rounds=config['rounds'],
                                    threads=threads, extra=ORDINARY)
    classifier = signals.fit_classifier(X, y, raw_kept, KIND, SEED, threads=threads, extra=CLASSIFIER)
    paths = {}
    for part, booster in [('ordinary', ordinary), ('classifier', classifier)]:
        path = RUN / 'models' / f'refit_2025__v{VERSION}__{SEED}.{part}.txt'
        save_booster(booster, path)
        paths[part] = str(path.relative_to(RUN))
    model = dict(minutes=round((time.monotonic() - started) / 60, 2), paths=paths,
                 artifacts={name: sha256(RUN / name) for name in paths.values()},
                 parameters=parameters, coverage=run.coverage(X, raw_kept))
    print(f"v{VERSION}: fitted in {model['minutes']} min", flush=True)
    return ordinary, classifier, model


def predict(ranking, dep, arr, rank, reports, ordinary, classifier, late_mean, threads):
    pieces = []
    for month, (part, part_arr, V) in ranking.items():
        print(f'v{VERSION}: predicting ranking island {month}: {len(part):,} departures', flush=True)
        V = run.add_inputs(V, part, reports, pd.concat([dep, part], ignore_index=True),
                           pd.concat([arr, part_arr], ignore_index=True))
        out = signals.predict(ordinary, classifier, V, part, KIND, threads=threads, extra=ORDINARY,
                              classifier_extra=CLASSIFIER, late_mean=late_mean, dayshift_weight=None)
        pieces.append(pd.DataFrame(dict(mvt_id=part.mvt_id.to_numpy(), month=month,
                                        ordinary=out['ordinary'], p=out['p'], blended=out['blended'],
                                        pred=out['final'], rule_triggered=out['rule_triggered'],
                                        late_rows=out['late_rows'],
                                        stand_clock_present=V['st_last_arr_sec'].notna().to_numpy())))
    frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(rank.mvt_id).reset_index()
    if len(frame) != EXPECTED_ROWS or not np.isfinite(frame.pred).all() or (frame.pred < 0).any():
        raise ValueError(f'v{VERSION}: ranking predictions are incomplete, nonfinite or negative')
    return frame


def against_previous(frame):
    """Row-by-row comparison with v29's ranking predictions; the Rome rule rows must not move."""
    old = pd.read_parquet(PREVIOUS, columns=['mvt_id', 'pred', 'rule_triggered']).set_index('mvt_id')
    old = old.reindex(frame.mvt_id)
    if old.pred.isna().any():
        raise ValueError(f'v{v29.VERSION} predictions do not cover every ranking row')
    rule = frame.rule_triggered.to_numpy()
    if not np.array_equal(rule, old.rule_triggered.to_numpy()) or not np.array_equal(
            frame.pred.to_numpy()[rule], old.pred.to_numpy()[rule]):
        raise ValueError(f'Rome rule rows differ from v{v29.VERSION}')
    change = frame.pred.to_numpy() - old.pred.to_numpy()
    return dict(version=v29.VERSION, predictions=str(PREVIOUS.relative_to(ROOT)),
                rule_rows_identical=int(rule.sum()), mean_change=round(float(change.mean()), 3),
                median_abs_change=round(float(np.median(np.abs(change))), 3),
                p99_abs_change=round(float(np.quantile(np.abs(change), 0.99)), 3),
                max_abs_change=round(float(np.abs(change).max()), 3))


def write(frame, model, record, config):
    predictions = RUN / 'refit' / f'v{VERSION}.parquet'
    predictions.parent.mkdir(exist_ok=True)
    frame.to_parquet(predictions, index=False)
    template = pq.read_table(TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    validate_submission(table, template)
    path = submission(VERSION)
    path.parent.mkdir(exist_ok=True)
    pq.write_table(table, path)
    manifest = dict(version=VERSION, file=file_name(VERSION), recipe=NAME, validation_run=RUN.name,
                    candidate=CANDIDATE, column=run.COLUMNS[CANDIDATE], late=True, dayshift=False,
                    ordinary_extra=list(ORDINARY), classifier_extra=list(CLASSIFIER), seed=SEED,
                    rounds=config['rounds'], freeze=FREEZE, ranking_months=MONTHS, rows=table.num_rows,
                    rule_triggered_rows=int(frame.rule_triggered.sum()),
                    late_rows=int(frame.late_rows.sum()),
                    stand_clock_coverage=round(float(frame.stand_clock_present.mean()), 5),
                    prediction_summary=frame.pred.describe().round(3).to_dict(),
                    against_previous=against_previous(frame),
                    model=model, predictions=str(predictions.relative_to(RUN)),
                    predictions_sha256=sha256(predictions), sha256=sha256(path),
                    template_sha256=sha256(TEMPLATE), validation_signature=record['signature'],
                    data_identity=record['data_identity'],
                    preparation_source={name: sha256(ROOT / name) for name in SOURCES},
                    state='prepared', prepared_at=now())
    write_json(MANIFEST, manifest)
    print(f"Prepared {path}: {table.num_rows:,} rows, {manifest['rule_triggered_rows']} Rome-rule rows, "
          f"{manifest['late_rows']} late-estimate rows; against v{v29.VERSION}: "
          f"{json.dumps(manifest['against_previous'])}", flush=True)


def prepare(threads):
    record, config = validated_run()
    if submission(VERSION).exists() or MANIFEST.exists():
        raise ValueError(f'v{VERSION} is already prepared')
    reports = weather.load_reports()
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1.0, ranking=True)
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    if months != MONTHS or len(rank) != EXPECTED_ROWS or rank.mvt_id.duplicated().any():
        raise ValueError('Expected the complete January/July 2026 ranking departures')
    X, y, cats, raw_kept = overnight_run.prepare_training(dep, arr, config)
    ranking = islands(dep, arr, rank, rank_arr, cats, config)
    # Training rows see all 2025 movements, as in each validation window's own training rows.
    X = run.add_inputs(X, raw_kept, reports, dep, arr)
    print(f'v{VERSION}: fitting on {len(X):,} departures, {X.shape[1]} prepared inputs', flush=True)
    ordinary, classifier, model = fit(X, y, raw_kept, config, threads)
    del X, y, raw_kept
    gc.collect()
    frame = predict(ranking, dep, arr, rank, reports, ordinary, classifier,
                    model['parameters']['late_mean'], threads)
    write(frame, model, record, config)


def upload():
    manifest, path = json.loads(MANIFEST.read_text()), submission(VERSION)
    if manifest['state'] != 'prepared' or sha256(path) != manifest['sha256']:
        raise ValueError(f'v{VERSION} is not in its prepared, fingerprinted state')
    validate_submission(pq.read_table(path), pq.read_table(TEMPLATE))
    remote = bucket_listing()
    if file_name(VERSION) in remote:
        raise ValueError(f'{file_name(VERSION)} already exists in the bucket')
    at = now()
    used = uploads_today(remote, at)
    manifest['quota_check'] = dict(at=at, existing_uploads_today=used)
    if len(used) >= 5:
        write_json(MANIFEST, manifest)
        raise ValueError('Five uploads already exist today (UTC)')
    attrs = f"candidate={NAME.lower()};cv-run={RUN.name};sha256={manifest['sha256']}"
    # Conditional single PUT prevents a racing writer from being overwritten.
    mc('cp', '--disable-multipart', '--custom-header', 'If-None-Match:*',
       '--attr', attrs, path, BUCKET + file_name(VERSION))
    digest = remote_digest(VERSION)
    if digest != manifest['sha256']:
        raise ValueError(f'Uploaded v{VERSION} differs from the prepared file')
    manifest.update(state='uploaded', uploaded_at=now(), remote_sha256=digest)
    write_json(MANIFEST, manifest)
    print(f'Uploaded and verified {file_name(VERSION)}', flush=True)


def result():
    manifest = json.loads(MANIFEST.read_text())
    if manifest['state'] not in ('uploaded', 'scored'):
        raise ValueError(f'Upload v{VERSION} first')
    key = file_name(VERSION) + '_result.json'
    if key not in bucket_listing():
        print(f'v{VERSION}: pending', flush=True)
        return
    official = json.loads(mc('cat', BUCKET + key))
    if official.get('status') == 'Succeeded' and official.get('used_pairs') != EXPECTED_ROWS:
        raise ValueError(f'v{VERSION}: official result scored an incomplete set of rows')
    manifest.update(official_result=official, score_checked_at=now())
    if official.get('status') == 'Succeeded':
        manifest.update(state='scored', score=float(official['score']))
    write_json(MANIFEST, manifest)
    print(f'v{VERSION}: ' + json.dumps(official), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['prepare', 'upload', 'result'])
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        prepare(args.threads)
    elif args.action == 'upload':
        upload()
    else:
        result()


if __name__ == '__main__':
    main()
