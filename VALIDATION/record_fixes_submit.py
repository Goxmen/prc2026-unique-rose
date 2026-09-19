"""Ranking file and upload for v34 (No-ID-cap) or v35 (Twin-records).

Each is v33's ranking predictions (row 33, Plan-window) with one fix of
record_fixes.py applied, validated as rows 34 and 35 in run
v3_20260919T104157403241Z. Nothing is refitted; v33's saved ranking predictions
are the input, and the script refuses to run unless they still reproduce the
scored v33 file byte for byte.

    prepare  apply the fix to v33's ranking predictions and write the file.
    upload   one conditional PUT, refused if five files were already uploaded today (UTC).
    result   read the official result file once it exists.

Run: .venv/bin/python VALIDATION/record_fixes_submit.py 34|35 prepare|upload|result
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
import plan_window_submit as v33
import record_fixes
from records import RUNS, data_identity, sha256, write_json
from rome_hybrid_submit import file_name, now, remote_digest, submission
from submit import fill_template
from tools.sweep_taxi_thresholds import mc
import weather

RECIPES = {
    34: dict(name='No-ID-cap', column='pred_noid_cap', only_lowers=True,
             fix=lambda pred, raw: record_fixes.noid_cap(pred, raw)),
    35: dict(name='Twin-records', column='pred_twins', only_lowers=False,
             fix=lambda pred, raw: record_fixes.twin_records(pred, raw)),
}
PREVIOUS = v33.VERSION
RUN = RUNS / 'v3_20260919T104157403241Z'
SOURCES = ['VALIDATION/record_fixes_submit.py', 'VALIDATION/plan_window_submit.py',
           'VALIDATION/overnight_submit.py', 'VALIDATION/submit.py']


def manifest_path(version):
    return RUN / f'submission_v{version}.json'


def validated_run():
    record = json.loads((RUN / 'result.json').read_text())
    if (record['status'] != 'complete' or record['label'] != 'record-fixes'
            or record['derived_from'] != dict(run=v33.RUN.name, column='pred_window')):
        raise ValueError(f'{RUN.name} is not the completed validation of rows 34-36')
    changed = sorted(name for name, digest in record['signature'].items() if sha256(ROOT / name) != digest)
    if changed:
        raise ValueError(f'Sources changed since validation: {changed}')
    if data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]) != record['data_identity']:
        raise ValueError('Training, ranking or METAR data changed since validation')
    return record


def previous_predictions():
    """v33's saved ranking predictions, checked against its scored file."""
    manifest = json.loads(v33.MANIFEST.read_text())
    path = v33.RUN / manifest['predictions']
    if manifest['state'] != 'scored' or sha256(path) != manifest['predictions_sha256']:
        raise ValueError(f'v{PREVIOUS} predictions are not in their scored, fingerprinted state')
    frame = pd.read_parquet(path)
    table = fill_template(pq.read_table(TEMPLATE), frame.mvt_id, frame.pred.to_numpy())
    if not table.equals(pq.read_table(submission(PREVIOUS))) or sha256(submission(PREVIOUS)) != manifest['sha256']:
        raise ValueError(f'v{PREVIOUS} predictions no longer reproduce its scored file')
    return frame, manifest


def ranking_departures(ids):
    raw = duckdb.sql("SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt, FLIGHT_mvt, FLIGHT_ID_mvt, MVT_TIME_UTC_mvt, "
                     "SCHED_TIME_UTC_mvt, LOBT_flt FROM read_parquet(?) WHERE PHASE_mvt = 'DEP'",
                     params=[str(RANKING)]).df()
    if raw.mvt_id.duplicated().any() or len(raw) != EXPECTED_ROWS:
        raise ValueError('Expected the complete ranking departures with unique movement IDs')
    raw = raw.set_index('mvt_id').reindex(ids)
    if raw.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError(f'v{PREVIOUS} predictions include movements missing from the ranking departures')
    return raw.reset_index()


def prepare(version):
    recipe, manifest_file = RECIPES[version], manifest_path(version)
    record = validated_run()
    if submission(version).exists() or manifest_file.exists():
        raise ValueError(f'v{version} is already prepared')
    old, old_manifest = previous_predictions()
    raw = ranking_departures(old.mvt_id)
    frame = pd.DataFrame(dict(mvt_id=old.mvt_id, month=old.month, pred_previous=old.pred,
                              pred=recipe['fix'](old.pred.to_numpy(), raw), rule_triggered=old.rule_triggered))
    if len(frame) != EXPECTED_ROWS or not np.isfinite(frame.pred).all() or (frame.pred < 0).any():
        raise ValueError(f'v{version}: ranking predictions are incomplete, nonfinite or negative')
    comparison = v33.against_previous(frame, raw)
    comparison['version'] = PREVIOUS
    if comparison['rule_rows_moved']:
        raise ValueError('A Rome rule row moved')
    if recipe['only_lowers'] and comparison['raised']:
        raise ValueError(f'v{version} may only lower predictions')
    predictions = RUN / 'refit' / f'v{version}.parquet'
    predictions.parent.mkdir(exist_ok=True)
    frame.to_parquet(predictions, index=False)
    template = pq.read_table(TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    validate_submission(table, template)
    path = submission(version)
    pq.write_table(table, path)
    constants = record['record_fixes']['constants']
    manifest = dict(version=version, file=file_name(version), recipe=recipe['name'], validation_run=RUN.name,
                    column=recipe['column'], built_on=dict(version=PREVIOUS, file=old_manifest['file'],
                                                           sha256=old_manifest['sha256'], score=old_manifest['score']),
                    constants=constants, rows=table.num_rows, against_previous=comparison,
                    predictions=str(predictions.relative_to(RUN)), predictions_sha256=sha256(predictions),
                    sha256=sha256(path), template_sha256=sha256(TEMPLATE),
                    validation_signature=record['signature'], data_identity=record['data_identity'],
                    preparation_source={name: sha256(ROOT / name) for name in SOURCES},
                    state='prepared', prepared_at=now())
    write_json(manifest_file, manifest)
    print(f"Prepared {path}: {table.num_rows:,} rows; against v{PREVIOUS}: "
          f"{json.dumps({k: v for k, v in comparison.items() if k != 'largest'})}", flush=True)
    for item in comparison['largest']:
        print(f"  {item['airport']} {item['flight']}: {item['previous']:.0f} s -> {item['new']:.0f} s", flush=True)


def upload(version):
    manifest_file, path = manifest_path(version), submission(version)
    manifest = json.loads(manifest_file.read_text())
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
        write_json(manifest_file, manifest)
        raise ValueError('Five uploads already exist today (UTC)')
    attrs = f"candidate={RECIPES[version]['name'].lower()};cv-run={RUN.name};sha256={manifest['sha256']}"
    # Conditional single PUT prevents a racing writer from being overwritten.
    mc('cp', '--disable-multipart', '--custom-header', 'If-None-Match:*',
       '--attr', attrs, path, BUCKET + file_name(version))
    digest = remote_digest(version)
    if digest != manifest['sha256']:
        raise ValueError(f'Uploaded v{version} differs from the prepared file')
    manifest.update(state='uploaded', uploaded_at=now(), remote_sha256=digest)
    write_json(manifest_file, manifest)
    print(f'Uploaded and verified {file_name(version)}', flush=True)


def result(version):
    manifest_file = manifest_path(version)
    manifest = json.loads(manifest_file.read_text())
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
    write_json(manifest_file, manifest)
    print(f'v{version}: ' + json.dumps(official), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('version', type=int, choices=sorted(RECIPES))
    parser.add_argument('action', choices=['prepare', 'upload', 'result'])
    args = parser.parse_args(argv)
    {'prepare': prepare, 'upload': upload, 'result': result}[args.action](args.version)


if __name__ == '__main__':
    main()
