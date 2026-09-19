"""Full-year refits, ranking predictions and uploads for the Rome hybrids v26–v28.

Each version is a row of SUBMISSIONS.md and one candidate of a completed
validation run whose sources and data must still match the working tree:

    v26  Rome-hybrid-4h-clock                 no extra inputs
    v27  Rome-hybrid-4h-clock-weather         + 21 METAR weather inputs
    v28  Rome-hybrid-4h-clock-weather-queue   + weather and 11 runway queue inputs

v26 and v27 are checked against the weather run v3_20260917T155106873031Z.
Its control is v26's recipe and reproduced every saved prediction of row 26's
own run v3_20260917T124103912051Z. That run predates the backward-compatible
`extra` argument of rome_hybrid.py, so its source check no longer passes.

    prepare  per version, one seed-42, 800-round fit of the ordinary regressor
             and the Rome classifier on all of 2025, then predictions for the
             ranking set. Training inputs see all 2025 movements. Each ranking
             month is an isolated context island: it sees all of 2025 plus its
             own month only, as in overnight_run.execute_refit.
    upload   one conditional PUT per file, refused if five files were already
             uploaded today (UTC).
    result   read the official result file once it exists.

Run: .venv/bin/python VALIDATION/rome_hybrid_submit.py prepare|upload|result 26 27
"""

import argparse
from datetime import datetime, timezone
import gc
import glob
import hashlib
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
import modeling
import overnight_models as models
import overnight_run
from overnight_submit import BUCKET, EXPECTED_ROWS, bucket_listing, uploads_today, validate_submission
from records import RUNS, data_identity, sha256, write_json
import rome_hybrid
from rome_hybrid_run import save_booster
import runway_queue_run as queue_run
from submit import fill_template
from tools.sweep_taxi_thresholds import mc
import weather
import weather_run

WEATHER_RUN, QUEUE_RUN = 'v3_20260917T155106873031Z', 'v3_20260917T172934966809Z'
RECIPES = {
    26: dict(name='Rome-hybrid-4h-clock', run=WEATHER_RUN, model='rome_hybrid_4h_clock',
             column='pred', inputs='none', extra=weather_run.HYBRIDS['rome_hybrid_4h_clock']),
    27: dict(name='Rome-hybrid-4h-clock-weather', run=WEATHER_RUN, model='rome_hybrid_4h_clock_weather',
             column='pred_weather', inputs='weather', extra=weather_run.HYBRIDS['rome_hybrid_4h_clock_weather']),
    28: dict(name='Rome-hybrid-4h-clock-weather-queue', run=QUEUE_RUN, model='queue_all',
             column='pred_queue_all', inputs='queue', extra=queue_run.CANDIDATES['queue_all']),
}
KIND = 'clock'
SEED = 42
FREEZE = '2025-12-31'
MONTHS = ['2026-01', '2026-07']
SOURCES = ['VALIDATION/rome_hybrid_submit.py', 'VALIDATION/overnight_submit.py', 'VALIDATION/submit.py']


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def file_name(version):
    return f'unique-rose_v{version}.parquet'


def submission(version):
    return ROOT / 'submissions' / file_name(version)


def run_dir(version):
    return RUNS / RECIPES[version]['run']


def manifest_path(version):
    return run_dir(version) / f'submission_v{version}.json'


def validated_run(version):
    recipe = RECIPES[version]
    record = json.loads((run_dir(version) / 'result.json').read_text())
    config = record['config']
    if (record['status'] != 'complete' or config['smoke'] or config['sample'] != 1.0
            or config['rounds'] != 800 or SEED not in config['seeds']
            or config['candidates'].get(recipe['model']) != recipe['column']):
        raise ValueError(f"{recipe['run']} is not a completed full validation run of v{version}")
    changed = sorted(name for name, digest in record['signature'].items() if sha256(ROOT / name) != digest)
    if changed:
        raise ValueError(f'v{version}: sources changed since validation: {changed}')
    if data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]) != record['data_identity']:
        raise ValueError(f'v{version}: training, ranking or METAR data changed since validation')
    return record, config


def add_inputs(version, X, raw, reports, dep_events, arr_events):
    inputs = RECIPES[version]['inputs']
    if inputs == 'weather':
        return weather.add_weather(X, raw, reports)
    if inputs == 'queue':
        return queue_run.add_inputs(X, raw, reports, dep_events, arr_events)
    return X


def fit(version, X, y, raw_kept, dep, arr, reports, config, threads):
    recipe, directory = RECIPES[version], run_dir(version)
    X = add_inputs(version, X, raw_kept, reports, dep, arr)
    print(f'v{version}: fitting on {len(X):,} departures, {X.shape[1]} prepared inputs', flush=True)
    started = time.monotonic()
    ordinary, classifier, details = rome_hybrid.fit_components(
        X, y, raw_kept, KIND, SEED, rounds=config['rounds'], threads=threads, extra=recipe['extra'])
    paths = {}
    for part, booster in [('ordinary', ordinary), ('classifier', classifier)]:
        path = directory / 'models' / f"refit_2025__{recipe['model']}__{SEED}.{part}.txt"
        save_booster(booster, path)
        paths[part] = str(path.relative_to(directory))
    details.update(minutes=round((time.monotonic() - started) / 60, 2), paths=paths,
                   artifacts={name: sha256(directory / name) for name in paths.values()})
    if recipe['inputs'] == 'queue':
        details['coverage'] = queue_run.coverage(X)
    print(f"v{version}: fitted in {details['minutes']} min", flush=True)
    return ordinary, classifier, details


def islands(dep, arr, rank, rank_arr, cats, config):
    """Shared ranking matrices, one per month, before any version's extra inputs."""
    out = {}
    for month in MONTHS:
        part, part_arr = modeling.month_rows(rank, month), modeling.month_rows(rank_arr, month)
        V = modeling.validation_matrix(dep, arr, part, part_arr, FREEZE, cats, config)
        out[month] = (part, part_arr, V)
    return out


def predict(version, ranking, dep, arr, rank, reports, ordinary, classifier, threads):
    pieces = []
    for month, (part, part_arr, V) in ranking.items():
        print(f'v{version}: predicting ranking island {month}: {len(part):,} departures', flush=True)
        V = add_inputs(version, V, part, reports, pd.concat([dep, part], ignore_index=True),
                       pd.concat([arr, part_arr], ignore_index=True))
        out = rome_hybrid.predict(ordinary, classifier, V, part, KIND, threads=threads,
                                  extra=RECIPES[version]['extra'])
        pieces.append(pd.DataFrame(dict(mvt_id=part.mvt_id.to_numpy(), month=month,
                                        ordinary=out['ordinary'], p=out['p'], blended=out['blended'],
                                        pred=out['final'], rule_triggered=out['rule_triggered'])))
    frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(rank.mvt_id).reset_index()
    if len(frame) != EXPECTED_ROWS or not np.isfinite(frame.pred).all() or (frame.pred < 0).any():
        raise ValueError(f'v{version}: ranking predictions are incomplete, nonfinite or negative')
    return frame


def write(version, frame, model, record, config):
    recipe, directory = RECIPES[version], run_dir(version)
    predictions = directory / 'refit' / f"{recipe['model']}.parquet"
    predictions.parent.mkdir(exist_ok=True)
    frame.to_parquet(predictions, index=False)
    template = pq.read_table(TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    validate_submission(table, template)
    path = submission(version)
    path.parent.mkdir(exist_ok=True)
    pq.write_table(table, path)
    manifest = dict(version=version, file=file_name(version), recipe=recipe['name'],
                    validation_run=recipe['run'], candidate=recipe['model'], column=recipe['column'],
                    extra_features=list(recipe['extra']), seed=SEED, rounds=config['rounds'],
                    freeze=FREEZE, ranking_months=MONTHS, rows=table.num_rows,
                    rule_triggered_rows=int(frame.rule_triggered.sum()),
                    prediction_summary=frame.pred.describe().round(3).to_dict(),
                    model=model, predictions=str(predictions.relative_to(directory)),
                    predictions_sha256=sha256(predictions), sha256=sha256(path),
                    template_sha256=sha256(TEMPLATE), validation_signature=record['signature'],
                    data_identity=record['data_identity'],
                    preparation_source={name: sha256(ROOT / name) for name in SOURCES},
                    state='prepared', prepared_at=now())
    write_json(manifest_path(version), manifest)
    print(f'Prepared {path}: {table.num_rows:,} rows, {manifest["rule_triggered_rows"]} Rome-rule rows', flush=True)


def prepare(versions, threads):
    checked = {version: validated_run(version) for version in versions}
    existing = [str(submission(v)) for v in versions if submission(v).exists() or manifest_path(v).exists()]
    if existing:
        raise ValueError(f'Already prepared: {existing}')
    reports = weather.load_reports()
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1.0, ranking=True)
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    if months != MONTHS or len(rank) != EXPECTED_ROWS or rank.mvt_id.duplicated().any():
        raise ValueError('Expected the complete January/July 2026 ranking departures')
    base = checked[versions[0]][1]
    if any(config[key] != base[key] for _, config in checked.values() for key in models.BASE_CONFIG):
        raise ValueError('Versions prepared together must share the base training configuration')
    X, y, cats, raw_kept = overnight_run.prepare_training(dep, arr, base)
    ranking = islands(dep, arr, rank, rank_arr, cats, base)
    for version in versions:
        record, config = checked[version]
        ordinary, classifier, model = fit(version, X, y, raw_kept, dep, arr, reports, config, threads)
        frame = predict(version, ranking, dep, arr, rank, reports, ordinary, classifier, threads)
        write(version, frame, model, record, config)
        del ordinary, classifier, frame
        gc.collect()


def remote_digest(version):
    return hashlib.sha256(mc('cat', BUCKET + file_name(version), binary=True)).hexdigest()


def upload(version):
    manifest, path = json.loads(manifest_path(version).read_text()), submission(version)
    if manifest['state'] != 'prepared' or sha256(path) != manifest['sha256']:
        raise ValueError(f'v{version} is not in its prepared, fingerprinted state')
    validate_submission(pq.read_table(path), pq.read_table(TEMPLATE))
    remote = bucket_listing()
    if file_name(version) in remote:
        raise ValueError(f'{file_name(version)} already exists in the bucket')
    at = now()
    used = uploads_today(remote, at)
    manifest['quota_check'] = dict(at=at, existing_uploads_today=used)
    if len(used) >= 5:
        write_json(manifest_path(version), manifest)
        raise ValueError('Five uploads already exist today (UTC)')
    attrs = f"candidate={manifest['recipe'].lower()};cv-run={manifest['validation_run']};sha256={manifest['sha256']}"
    # Conditional single PUT prevents a racing writer from being overwritten.
    mc('cp', '--disable-multipart', '--custom-header', 'If-None-Match:*',
       '--attr', attrs, path, BUCKET + file_name(version))
    digest = remote_digest(version)
    if digest != manifest['sha256']:
        raise ValueError(f'Uploaded v{version} differs from the prepared file')
    manifest.update(state='uploaded', uploaded_at=now(), remote_sha256=digest)
    write_json(manifest_path(version), manifest)
    print(f'Uploaded and verified {file_name(version)}', flush=True)


def result(version):
    manifest = json.loads(manifest_path(version).read_text())
    if manifest['state'] not in ('uploaded', 'scored'):
        raise ValueError(f'Upload v{version} first')
    key = file_name(version) + '_result.json'
    if key not in bucket_listing():
        print(f'v{version}: pending', flush=True)
        return
    official = json.loads(mc('cat', BUCKET + key))
    if official.get('status') == 'Succeeded' and official.get('used_pairs') != EXPECTED_ROWS:
        raise ValueError(f'v{version}: official result scored an incomplete set of rows')
    manifest.update(official_result=official, score_checked_at=now())
    if official.get('status') == 'Succeeded':
        manifest.update(state='scored', score=float(official['score']))
    write_json(manifest_path(version), manifest)
    print(f'v{version}: ' + json.dumps(official), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['prepare', 'upload', 'result'])
    parser.add_argument('versions', type=int, nargs='+', choices=sorted(RECIPES))
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args(argv)
    if len(set(args.versions)) != len(args.versions):
        parser.error('Versions must be unique')
    if args.action == 'prepare':
        prepare(args.versions, args.threads)
    else:
        for version in args.versions:
            (upload if args.action == 'upload' else result)(version)


if __name__ == '__main__':
    main()
