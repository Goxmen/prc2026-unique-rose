"""Ranking file and upload for v33, Plan-window.

v33 is row 33 of SUBMISSIONS.md: v30's ranking predictions (row 30, Stand-clock)
with the flight-plan window of plan_window.py applied, validated in run
v3_20260918T181924599658Z. Nothing is refitted; v30's saved ranking predictions
are the input, and the script refuses to run unless they still reproduce the
scored v30 file byte for byte.

    prepare  apply the window to v30's ranking predictions and write the file.
    upload   one conditional PUT, refused if five files were already uploaded today (UTC).
    result   read the official result file once it exists.

Run: .venv/bin/python VALIDATION/plan_window_submit.py prepare|upload|result
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
from overnight_submit import BUCKET, EXPECTED_ROWS, bucket_listing, uploads_today, validate_submission
import plan_window
from records import RUNS, data_identity, sha256, write_json
from rome_hybrid_submit import file_name, now, remote_digest, submission
from submit import fill_template
from tools.sweep_taxi_thresholds import mc
import weather

VERSION = 33
NAME = 'Plan-window'
PREVIOUS = 30
RUN = RUNS / 'v3_20260918T181924599658Z'
PREVIOUS_RUN = RUNS / 'v3_20260918T151436598224Z'
MANIFEST = RUN / f'submission_v{VERSION}.json'
SOURCES = ['VALIDATION/plan_window_submit.py', 'VALIDATION/overnight_submit.py', 'VALIDATION/submit.py']


def validated_run():
    record = json.loads((RUN / 'result.json').read_text())
    if (record['status'] != 'complete' or record['label'] != 'plan-window'
            or record['derived_from'] != dict(run=PREVIOUS_RUN.name, column='pred_stand')
            or record['plan_window']['margin_sec'] != plan_window.MARGIN_SEC):
        raise ValueError(f'{RUN.name} is not the completed validation of row 33')
    changed = sorted(name for name, digest in record['signature'].items() if sha256(ROOT / name) != digest)
    if changed:
        raise ValueError(f'Sources changed since validation: {changed}')
    if data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]) != record['data_identity']:
        raise ValueError('Training, ranking or METAR data changed since validation')
    return record


def previous_predictions():
    """v30's saved ranking predictions, checked against its scored file."""
    manifest = json.loads((PREVIOUS_RUN / f'submission_v{PREVIOUS}.json').read_text())
    path = PREVIOUS_RUN / manifest['predictions']
    if manifest['state'] != 'scored' or sha256(path) != manifest['predictions_sha256']:
        raise ValueError(f'v{PREVIOUS} predictions are not in their scored, fingerprinted state')
    frame = pd.read_parquet(path)
    table = fill_template(pq.read_table(TEMPLATE), frame.mvt_id, frame.pred.to_numpy())
    if not table.equals(pq.read_table(submission(PREVIOUS))) or sha256(submission(PREVIOUS)) != manifest['sha256']:
        raise ValueError(f'v{PREVIOUS} predictions no longer reproduce its scored file')
    return frame, manifest


def ranking_departures(ids):
    raw = duckdb.sql("SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt, FLIGHT_mvt, FLIGHT_ID_mvt, MVT_TIME_UTC_mvt, LOBT_flt "
                     "FROM read_parquet(?) WHERE PHASE_mvt = 'DEP'", params=[str(RANKING)]).df()
    if raw.mvt_id.duplicated().any():
        raise ValueError('Ranking departures must have unique movement IDs')
    raw = raw.set_index('mvt_id').reindex(ids)
    if raw.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError(f'v{PREVIOUS} predictions include movements missing from the ranking departures')
    return raw.reset_index()


def against_previous(frame, raw):
    moved = frame.pred.to_numpy() != frame.pred_previous.to_numpy()
    change = (frame.pred - frame.pred_previous).to_numpy()
    largest = frame.assign(ADEP_mvt=raw.ADEP_mvt, FLIGHT_mvt=raw.FLIGHT_mvt, change=change)[moved]
    largest = largest.reindex(largest.change.abs().sort_values(ascending=False).index).head(10)
    return dict(version=PREVIOUS, moved=int(moved.sum()), raised=int((change > 0).sum()),
                lowered=int((change < 0).sum()), rule_rows_moved=int((moved & frame.rule_triggered).sum()),
                moved_by_airport=raw.ADEP_mvt[moved].value_counts().sort_index().to_dict(),
                median_abs_change_moved=round(float(np.median(np.abs(change[moved]))), 1) if moved.any() else 0.0,
                largest=[dict(mvt_id=int(r.mvt_id), airport=r.ADEP_mvt, flight=r.FLIGHT_mvt,
                              previous=round(float(r.pred_previous), 1), new=round(float(r.pred), 1))
                         for r in largest.itertuples()])


def prepare():
    record = validated_run()
    if submission(VERSION).exists() or MANIFEST.exists():
        raise ValueError(f'v{VERSION} is already prepared')
    old, old_manifest = previous_predictions()
    raw = ranking_departures(old.mvt_id)
    low, high = plan_window.window(raw)
    frame = pd.DataFrame(dict(mvt_id=old.mvt_id, month=old.month, pred_previous=old.pred,
                              pred=plan_window.apply(old.pred, raw), window_low=low, window_high=high,
                              rule_triggered=old.rule_triggered))
    if len(frame) != EXPECTED_ROWS or not np.isfinite(frame.pred).all() or (frame.pred < 0).any():
        raise ValueError(f'v{VERSION}: ranking predictions are incomplete, nonfinite or negative')
    comparison = against_previous(frame, raw)
    if comparison['rule_rows_moved']:
        raise ValueError('A Rome rule row moved')
    predictions = RUN / 'refit' / f'v{VERSION}.parquet'
    predictions.parent.mkdir(exist_ok=True)
    frame.to_parquet(predictions, index=False)
    template = pq.read_table(TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    validate_submission(table, template)
    path = submission(VERSION)
    pq.write_table(table, path)
    manifest = dict(version=VERSION, file=file_name(VERSION), recipe=NAME, validation_run=RUN.name,
                    column='pred_window', built_on=dict(version=PREVIOUS, file=old_manifest['file'],
                                                        sha256=old_manifest['sha256'], score=old_manifest['score']),
                    margin_sec=plan_window.MARGIN_SEC, rows=table.num_rows,
                    bounded_rows=int(np.isfinite(low).sum()), against_previous=comparison,
                    predictions=str(predictions.relative_to(RUN)), predictions_sha256=sha256(predictions),
                    sha256=sha256(path), template_sha256=sha256(TEMPLATE),
                    validation_signature=record['signature'], data_identity=record['data_identity'],
                    preparation_source={name: sha256(ROOT / name) for name in SOURCES},
                    state='prepared', prepared_at=now())
    write_json(MANIFEST, manifest)
    print(f"Prepared {path}: {table.num_rows:,} rows, {manifest['bounded_rows']:,} bounded; against "
          f"v{PREVIOUS}: {json.dumps({k: v for k, v in comparison.items() if k != 'largest'})}", flush=True)
    for item in comparison['largest']:
        print(f"  {item['airport']} {item['flight']}: {item['previous']:.0f} s -> {item['new']:.0f} s", flush=True)


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
