"""Ranking file and upload for v37, Dayshift-hedge.

v37 is row 37 of SUBMISSIONS.md: v34's ranking predictions (row 34, No-ID-cap) with
dayshift_hedge.hedge applied, validated in run v3_20260919T112057144323Z. Nothing is
refitted. The airline and format rates and the wrong-date taxi come from all 2025
training labels. v34's saved ranking predictions are the input, and the script
refuses to run unless they still reproduce the scored v34 file byte for byte.

    prepare  apply the hedge to v34's ranking predictions and write the file.
    upload   one conditional PUT, refused if five files were already uploaded today (UTC).
    result   read the official result file once it exists.

Run: .venv/bin/python VALIDATION/dayshift_hedge_submit.py prepare|upload|result
"""

import argparse
import glob
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import RANKING, TEMPLATE, TRAIN_GLOB
import dayshift_hedge
from overnight_submit import BUCKET, EXPECTED_ROWS, bucket_listing, uploads_today, validate_submission
import plan_window_submit as v33
import record_fixes_submit as v34
from records import RUNS, data_identity, sha256, write_json
from rome_hybrid_submit import file_name, now, remote_digest, submission
from submit import fill_template
from tools.sweep_taxi_thresholds import mc
import weather

VERSION = 37
NAME = 'Dayshift-hedge'
PREVIOUS = 34
RUN = RUNS / 'v3_20260919T112057144323Z'
MANIFEST = RUN / f'submission_v{VERSION}.json'
SOURCES = ['VALIDATION/dayshift_hedge_submit.py', 'VALIDATION/record_fixes_submit.py',
           'VALIDATION/plan_window_submit.py', 'VALIDATION/overnight_submit.py', 'VALIDATION/submit.py']
COLUMNS = "MVT_ID_mvt AS mvt_id, ADEP_mvt, FLIGHT_mvt, FLIGHT_ID_mvt, MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt"


def validated_run():
    record = json.loads((RUN / 'result.json').read_text())
    if (record['status'] != 'complete' or record['label'] != 'dayshift-hedge'
            or record['derived_from'] != dict(run=v34.RUN.name, column='pred_noid_cap')):
        raise ValueError(f'{RUN.name} is not the completed validation of row 37')
    changed = sorted(name for name, digest in record['signature'].items() if sha256(ROOT / name) != digest)
    if changed:
        raise ValueError(f'Sources changed since validation: {changed}')
    if data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]) != record['data_identity']:
        raise ValueError('Training, ranking or METAR data changed since validation')
    return record


def previous_predictions():
    """v34's saved ranking predictions, checked against its scored file."""
    manifest = json.loads(v34.manifest_path(PREVIOUS).read_text())
    path = v34.RUN / manifest['predictions']
    if manifest['state'] != 'scored' or sha256(path) != manifest['predictions_sha256']:
        raise ValueError(f'v{PREVIOUS} predictions are not in their scored, fingerprinted state')
    frame = pd.read_parquet(path)
    table = fill_template(pq.read_table(TEMPLATE), frame.mvt_id, frame.pred.to_numpy())
    if not table.equals(pq.read_table(submission(PREVIOUS))) or sha256(submission(PREVIOUS)) != manifest['sha256']:
        raise ValueError(f'v{PREVIOUS} predictions no longer reproduce its scored file')
    return frame, manifest


def training_history():
    dep = duckdb.sql(f"SELECT {COLUMNS}, TAXITIME_SEC_mvt AS y FROM read_parquet(?) WHERE PHASE_mvt = 'DEP'",
                     params=[sorted(glob.glob(TRAIN_GLOB))]).df()
    return dayshift_hedge.history(dep, dep.y.to_numpy(float))


def ranking_departures(ids):
    raw = duckdb.sql(f"SELECT {COLUMNS} FROM read_parquet(?) WHERE PHASE_mvt = 'DEP'", params=[str(RANKING)]).df()
    if raw.mvt_id.duplicated().any() or len(raw) != EXPECTED_ROWS:
        raise ValueError('Expected the complete ranking departures with unique movement IDs')
    raw = raw.set_index('mvt_id').reindex(ids)
    if raw.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError(f'v{PREVIOUS} predictions include movements missing from the ranking departures')
    return raw.reset_index()


def prepare():
    record = validated_run()
    if submission(VERSION).exists() or MANIFEST.exists():
        raise ValueError(f'v{VERSION} is already prepared')
    old, old_manifest = previous_predictions()
    raw = ranking_departures(old.mvt_id)
    hist = training_history()
    p = dayshift_hedge.weights(raw, hist)
    frame = pd.DataFrame(dict(mvt_id=old.mvt_id, month=old.month, pred_previous=old.pred,
                              pred=dayshift_hedge.hedge(old.pred.to_numpy(), raw, hist), p_wrong_date=p,
                              rule_triggered=old.rule_triggered))
    if len(frame) != EXPECTED_ROWS or not np.isfinite(frame.pred).all() or (frame.pred < 0).any():
        raise ValueError(f'v{VERSION}: ranking predictions are incomplete, nonfinite or negative')
    comparison = v33.against_previous(frame, raw)
    comparison['version'] = PREVIOUS
    moved = frame.pred.to_numpy() != frame.pred_previous.to_numpy()
    if (moved & ~frame.rule_triggered.to_numpy()).any():
        raise ValueError('The hedge moved a row the Rome rule did not fire on')
    predictions = RUN / 'refit' / f'v{VERSION}.parquet'
    predictions.parent.mkdir(exist_ok=True)
    frame.to_parquet(predictions, index=False)
    template = pq.read_table(TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    validate_submission(table, template)
    path = submission(VERSION)
    pq.write_table(table, path)
    rows = [dict(flight=raw.FLIGHT_mvt[i], p=round(float(p[i]), 3), previous=round(float(frame.pred_previous[i]), 1),
                 new=round(float(frame.pred[i]), 1)) for i in np.flatnonzero(moved)]
    manifest = dict(version=VERSION, file=file_name(VERSION), recipe=NAME, validation_run=RUN.name,
                    column='pred_dayshift', built_on=dict(version=PREVIOUS, file=old_manifest['file'],
                                                          sha256=old_manifest['sha256'], score=old_manifest['score']),
                    history=hist, moved_rows=rows, rows=table.num_rows, against_previous=comparison,
                    predictions=str(predictions.relative_to(RUN)), predictions_sha256=sha256(predictions),
                    sha256=sha256(path), template_sha256=sha256(TEMPLATE),
                    validation_signature=record['signature'], data_identity=record['data_identity'],
                    preparation_source={name: sha256(ROOT / name) for name in SOURCES},
                    state='prepared', prepared_at=now())
    write_json(MANIFEST, manifest)
    print(f"Prepared {path}: {table.num_rows:,} rows; {int(moved.sum())} rows moved against v{PREVIOUS}", flush=True)
    for item in rows:
        print(f"  {item['flight']}: p {item['p']:.2f}, {item['previous']:.0f} s -> {item['new']:.0f} s", flush=True)


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
    args = parser.parse_args(argv)
    {'prepare': prepare, 'upload': upload, 'result': result}[args.action]()


if __name__ == '__main__':
    main()
