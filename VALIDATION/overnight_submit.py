"""Prepare five authorized uploads and reveal discovery/confirmation scores separately.

No training is performed here. Preparation only creates local submission files;
``upload`` is explicit and stops at the current UTC day's quota. Result reads are
restricted to the selected stage, without any leaderboard-wide request.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import overnight_run as runner
from overnight_models import RECIPES
from records import sha256, write_json
from submit import fill_template
from tools.sweep_taxi_thresholds import mc

BUCKET = 'opensky/prc-2026-unique-rose/'
EXPECTED_ROWS = 344_841
DISCOVERY = ['no_month', 'no_history', 'clock_features']
CONFIRMATION = ['small_trees', 'rule_routing']
SELECTED = DISCOVERY + CONFIRMATION
DEFAULT_RUN = ROOT / 'VALIDATION/runs/overnight_20260917'
MANIFEST = 'submission_manifest.json'
VERSION_RE = re.compile(r'^unique-rose_v(\d+)\.')
PREPARATION_SOURCES = ['VALIDATION/overnight_submit.py', 'VALIDATION/submit.py',
                       'VALIDATION/tools/sweep_taxi_thresholds.py']


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def read_json(path):
    return json.loads(Path(path).read_text())


def bucket_listing():
    rows = [json.loads(line) for line in mc('ls', '--json', BUCKET).splitlines() if line.strip()]
    if any(row.get('status') == 'error' for row in rows):
        raise RuntimeError('MinIO submission listing failed')
    return {row['key']: row for row in rows if row.get('type') == 'file'}


def contained(directory, relative):
    path = (Path(directory) / relative).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError('Artifact path escapes its recorded root')
    return path


def verify_hashes(directory, values):
    if not values:
        raise ValueError('Missing artifact/source fingerprints')
    for name, digest in values.items():
        path = contained(directory, name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Missing/changed fingerprinted file: {name}')


def validate_run(directory, *, all_artifacts=True):
    """Accept only the completed full benchmark and its exact validated refits."""
    directory = Path(directory)
    record, progress = read_json(directory / 'result.json'), read_json(directory / 'progress.json')
    config, signature = record['config'], record['signature']
    if (record.get('status') != 'complete' or config.get('smoke')
            or config.get('sample') != 1. or config.get('rounds') != 800
            or config.get('seeds') != [42, 142, 242]
            or config.get('recovery') != 'off' or config.get('target_mode') != 'direct'
            or config.get('prediction_rule') != 'rome_4h'
            or config.get('clock_features') is not True
            or config.get('max_taxi_sec') != 200_000):
        raise ValueError('Submissions require the complete nonsmoke frozen 800-round benchmark')
    if (signature.get('protocol') != runner.PROTOCOL or signature.get('config') != config
            or signature.get('jobs') != record['jobs'] or record['jobs'] != runner.schedule()
            or any(recipe not in config['recipes'] for recipe in SELECTED)
            or not runner.validation_complete(progress, record['jobs'], SELECTED, config['seeds'])):
        raise ValueError('Validation roster/configuration is incomplete or changed')
    if signature['source'] != runner.source_signature():
        raise ValueError('Model/rule/runner source changed since validation')
    verify_hashes(directory / 'source', signature['source'])
    data = signature['data_identity']
    if len(data) != 13 or len({item['file'] for item in data}) != 13:
        raise ValueError('Expected twelve original training months and the ranking data')
    verify_hashes(ROOT / 'data', {item['file']: item['sha256'] for item in data})
    if all_artifacts:
        runner.check_artifacts(directory, progress)
    for recipe in SELECTED:
        entry = progress['refits'].get(recipe)
        if not entry or entry.get('recipe') != recipe or entry.get('rows') != EXPECTED_ROWS:
            raise ValueError(f'Missing full-year refit: {recipe}')
        verify_hashes(directory, entry['artifacts'])
        meta = read_json(contained(directory, entry['metadata']))
        model = progress['refit_models'].get(f'refit_2025__{recipe}__42')
        if (meta.get('recipe') != recipe or meta.get('seed') != 42 or meta.get('rounds') != 800
                or meta.get('freeze') != '2025-12-31' or meta.get('source_validated') is not True
                or meta.get('signature_sha256') != runner.signature_digest(signature)
                or meta.get('validation_jobs') != [job['id'] for job in record['jobs']]
                or meta.get('validation_seeds') != [42, 142, 242]
                or meta.get('config') != config or meta.get('model') != model
                or meta.get('ranking_rows') != EXPECTED_ROWS
                or meta.get('ranking_months') != ['2026-01', '2026-07']
                or meta.get('prediction_rule') != 'rome_4h'
                or meta.get('predictions_sha256') != sha256(contained(directory, entry['path']))
                or meta.get('model_source') != signature['source']
                or meta.get('data_identity') != data):
            raise ValueError(f'Full-year refit provenance mismatch: {recipe}')
        verify_hashes(directory, model['artifacts'])
    return record, progress


def validate_submission(table, template):
    if table.num_rows != EXPECTED_ROWS or template.num_rows != EXPECTED_ROWS:
        raise ValueError('Submission/template row count differs from the complete ranking set')
    if not table.schema.equals(template.schema) or table.schema.field('TAXITIME_SEC_mvt').type != pa.int32():
        raise ValueError('Submission/template schema or int32 target type differs')
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt' and not table[name].equals(template[name]):
            raise ValueError('Submission changed template IDs/order or other original fields')
    ids = pd.Index(table['MVT_ID_mvt'].to_pandas())
    values = table['TAXITIME_SEC_mvt'].to_numpy()
    if ids.hasnans or not ids.is_unique or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Submission IDs/predictions are missing, duplicated, nonfinite or negative')


def _filenames(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _filenames(child)
    elif isinstance(value, list):
        for child in value:
            yield from _filenames(child)
    elif isinstance(value, str) and VERSION_RE.match(Path(value).name):
        yield Path(value).name


def allocate_versions(remote):
    names = set(remote) | {path.name for path in (ROOT / 'submissions').glob('unique-rose_v*')}
    # Preserve IDs reserved by earlier queues, including incomplete runs.
    manifests = list((ROOT / 'VALIDATION/reports').glob('*.json'))
    manifests += list((ROOT / 'VALIDATION/reports/sweeps').glob('*.json'))
    manifests += list((ROOT / 'VALIDATION/runs').glob('*/submission_manifest.json'))
    for path in manifests:
        names.update(_filenames(read_json(path)))
    first = max((int(match[1]) for name in names if (match := VERSION_RE.match(name))), default=0) + 1
    return list(range(first, first + len(SELECTED)))


def check_queue(batch):
    if [item.get('recipe') for item in batch['submissions']] != SELECTED:
        raise ValueError('Queue differs from the five explicitly selected recipes')
    files = [item['file'] for item in batch['submissions']]
    if len(set(files)) != 5 or any(name != f"unique-rose_v{item['version']}.parquet"
                                   for name, item in zip(files, batch['submissions'])):
        raise ValueError('Invalid or duplicate reserved submission versions')
    if any(item['role'] != RECIPES[item['recipe']]['role'] for item in batch['submissions']):
        raise ValueError('Discovery/confirmation assignment changed')


def verify_batch(directory, batch, *, validate=True):
    check_queue(batch)
    verify_hashes(ROOT, batch['preparation_source'])
    verify_hashes(directory, batch['run_records'])
    if sha256(ROOT / 'data/submitting.parquet') != batch['template_sha256']:
        raise ValueError('Submission template changed')
    if validate:
        record, progress = validate_run(directory, all_artifacts=False)
        if batch['signature_sha256'] != runner.signature_digest(record['signature']):
            raise ValueError('Prepared queue no longer matches the validated run signature')
        for item in batch['submissions']:
            if item.get('state') != 'planned' and (
                    item.get('refit') != progress['refits'][item['recipe']]
                    or item.get('model') != progress['refit_models'][f"refit_2025__{item['recipe']}__42"]):
                raise ValueError('Prepared queue refit/model link differs from completed provenance')
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    for item in batch['submissions']:
        if item.get('state') == 'planned':
            continue
        path = ROOT / 'submissions' / item['file']
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError(f"Prepared submission changed: {item['file']}")
        validate_submission(pq.read_table(path), template)


def prepare(directory):
    directory = Path(directory)
    record, progress = validate_run(directory)
    manifest_path = directory / MANIFEST
    if manifest_path.exists():
        batch = read_json(manifest_path)
        verify_batch(directory, batch, validate=False)
    else:
        versions = allocate_versions(bucket_listing())
        batch = dict(prepared_at=now(), signature_sha256=runner.signature_digest(record['signature']),
                     template_sha256=sha256(ROOT / 'data/submitting.parquet'),
                     run_records={name: sha256(directory / name) for name in ('result.json', 'progress.json')},
                     preparation_source={name: sha256(ROOT / name) for name in PREPARATION_SOURCES},
                     policy='Five frozen recipes only; no automatic quota reset retry; confirmation sealed',
                     submissions=[dict(recipe=recipe, role=RECIPES[recipe]['role'], version=version,
                                       file=f'unique-rose_v{version}.parquet', state='planned', score=None)
                                  for recipe, version in zip(SELECTED, versions)])
        write_json(manifest_path, batch)  # Reserve all versions before creating any output.
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    (ROOT / 'submissions').mkdir(exist_ok=True)
    for item in batch['submissions']:
        if item['state'] != 'planned':
            continue
        raw = pd.read_parquet(contained(directory, progress['refits'][item['recipe']]['path']))
        if not np.isfinite(raw[['pred_raw', 'pred']].to_numpy()).all() or (raw.pred < 0).any():
            raise ValueError('Refit predictions must be finite and nonnegative before template filling')
        out = fill_template(template, raw.mvt_id, raw.pred.to_numpy())
        validate_submission(out, template)
        path = ROOT / 'submissions' / item['file']
        if path.exists():
            if not pq.read_table(path).equals(out):
                raise ValueError(f'Local version collision: {path.name}')
        else:
            pq.write_table(out, path)
        item.update(state='prepared', sha256=sha256(path), rows=out.num_rows,
                    triggered_rows=int(raw.rule_triggered.sum()), refit=progress['refits'][item['recipe']],
                    model=progress['refit_models'][f"refit_2025__{item['recipe']}__42"])
        write_json(manifest_path, batch)
        print(f"Prepared {item['file']}: {item['recipe']}, {out.num_rows:,} rows", flush=True)
    return batch


def uploads_today(remote, at):
    today = datetime.fromisoformat(at).astimezone(timezone.utc).date()
    return sorted(name for name, row in remote.items() if name.endswith('.parquet')
                  and datetime.fromisoformat(row['lastModified'].replace('Z', '+00:00'))
                  .astimezone(timezone.utc).date() == today)


def remote_digest(filename):
    return hashlib.sha256(mc('cat', BUCKET + filename, binary=True)).hexdigest()


def upload(directory):
    directory = Path(directory)
    batch = read_json(directory / MANIFEST)
    verify_batch(directory, batch)
    if any(item['state'] == 'planned' for item in batch['submissions']):
        raise ValueError('Finish preparation before uploading')
    for item in batch['submissions']:
        # A separate guard before each file catches changes during a long upload.
        verify_batch(directory, batch)
        remote = bucket_listing()
        if item['file'] in remote:
            digest = remote_digest(item['file'])
            if digest != item['sha256']:
                raise ValueError(f"Remote version collision: {item['file']}")
            if not item.get('uploaded_at'):
                modified = remote[item['file']]['lastModified']
                item['uploaded_at'] = datetime.fromisoformat(modified.replace('Z', '+00:00')) \
                    .astimezone(timezone.utc).isoformat(timespec='seconds')
                item['uploaded_at_source'] = 'remote_last_modified'
                item['remote_last_modified'] = modified
            item.update(state='uploaded', remote_sha256=digest, verified_at=now())
            write_json(directory / MANIFEST, batch)
            continue
        at = now()
        used = uploads_today(remote, at)
        batch.setdefault('quota_checks', []).append(dict(at=at, existing_uploads_today=used))
        if len(used) >= 5:
            batch['blocked'] = 'Five uploads already exist today; no automatic quota reset retry'
            write_json(directory / MANIFEST, batch)
            print(batch['blocked'], flush=True)
            return False
        item.setdefault('upload_intents', []).append(at)
        write_json(directory / MANIFEST, batch)
        path = ROOT / 'submissions' / item['file']
        attrs = f"candidate={item['recipe']};cv-study=overnight-20260917;sha256={item['sha256']}"
        # Conditional single PUT prevents a racing writer from being overwritten.
        mc('cp', '--disable-multipart', '--custom-header', 'If-None-Match:*',
           '--attr', attrs, path, BUCKET + item['file'])
        digest = remote_digest(item['file'])
        if digest != item['sha256']:
            raise ValueError('Uploaded object differs from the prepared artifact')
        item.update(state='uploaded', uploaded_at=now(), remote_sha256=digest, verified_at=now())
        batch.pop('blocked', None)
        write_json(directory / MANIFEST, batch)
        print(f"Uploaded and verified {item['file']}: {item['recipe']}", flush=True)
    return True


def validate_confirmation_seal(directory, batch):
    path = directory / 'cv_selection_frozen.json'
    if not path.is_file():
        raise ValueError('Freeze CV selection before revealing confirmation scores')
    frozen = read_json(path)
    if (frozen.get('status') != 'frozen' or not frozen.get('created_at')
            or not frozen.get('selected_validation')
            or frozen.get('expected_confirmation') != CONFIRMATION):
        raise ValueError('CV selection file does not contain a complete frozen decision')
    discovery_path = directory / 'discovery_scores.json'
    if (not discovery_path.is_file() or set(read_json(discovery_path)) != set(DISCOVERY)
            or frozen.get('discovery_scores_sha256') != sha256(discovery_path)):
        raise ValueError('CV selection is not bound to the completed discovery scores')
    inputs = frozen.get('input_hashes', {})
    if (inputs.get('discovery_scores_sha256') != sha256(discovery_path)
            or inputs.get('cv_scores_sha256') != sha256(directory / 'cv_scores.json')):
        raise ValueError('Frozen CV/discovery inputs changed')
    verify_hashes(ROOT, frozen.get('source_hashes', {}))
    digest = sha256(path)
    if batch.get('confirmation_seal_sha256', digest) != digest:
        raise ValueError('Frozen CV selection changed after confirmation reveal was authorized')
    if 'confirmation_seal_sha256' not in batch:
        batch['confirmation_seal_sha256'] = digest
        batch['confirmation_reveal_authorized_at'] = now()
        write_json(directory / MANIFEST, batch)
    return frozen


def refresh(directory, stage):
    directory = Path(directory)
    if stage not in ('discovery', 'confirmation'):
        raise ValueError('Result stage must be discovery or confirmation')
    batch = read_json(directory / MANIFEST)
    verify_batch(directory, batch)
    if stage == 'confirmation':
        validate_confirmation_seal(directory, batch)
    selected = DISCOVERY if stage == 'discovery' else CONFIRMATION
    remote = bucket_listing()  # Object names only; confirmation payloads stay unread.
    scores = {}
    for item in batch['submissions']:
        if item['recipe'] not in selected:
            continue
        if item.get('state') != 'uploaded' or item.get('remote_sha256') != item.get('sha256'):
            raise ValueError(f"Cannot score an unverified upload: {item['recipe']}")
        if item['file'] not in remote or remote_digest(item['file']) != item['sha256']:
            raise ValueError(f"Remote submitted content changed: {item['file']}")
        key = item['file'] + '_result.json'
        if key not in remote:
            print(f"{item['recipe']}: pending", flush=True)
            continue
        result = json.loads(mc('cat', BUCKET + key))
        status = result.get('status')
        if (result.get('file', item['file']) != item['file']
                or result.get('bucket', 'prc-2026-unique-rose') != 'prc-2026-unique-rose'):
            raise ValueError('Official result filename/bucket differs from verified upload')
        item.update(official_result=result, official_status=status, score_checked_at=now())
        if status != 'Succeeded':
            write_json(directory / MANIFEST, batch)
            if status in ('Failed', 'Error', 'Rejected'):
                raise ValueError(f"Official scoring failed for {item['recipe']}: {status}")
            print(f"{item['recipe']}: {status or 'pending'}", flush=True)
            continue
        score = result.get('score')
        if (result.get('file') != item['file'] or result.get('bucket') != 'prc-2026-unique-rose'
                or result.get('used_pairs') != EXPECTED_ROWS or isinstance(score, bool)
                or not isinstance(score, (float, int)) or not np.isfinite(score) or score < 0):
            raise ValueError('Official result has invalid score or incomplete used_pairs')
        score = float(score)
        if item.get('score') is not None and item['score'] != score:
            raise ValueError('Previously revealed official score changed')
        item['score'] = scores[item['recipe']] = score
        write_json(directory / MANIFEST, batch)
        print(f"{item['recipe']}: {score:.4f}", flush=True)
    # Never replace an already complete score file with a transient partial view.
    output = directory / f'{stage}_scores.json'
    previous = read_json(output) if output.exists() else {}
    if any(name not in selected or name in scores and scores[name] != score for name, score in previous.items()):
        raise ValueError('Previously revealed stage scores differ')
    scores = {**previous, **scores}
    write_json(output, scores)
    return len(scores) == len(selected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload', 'discovery', 'confirmation'])
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args(argv)
    directory = args.run_root.resolve()
    if args.action == 'prepare':
        prepare(directory)
    elif args.action == 'upload':
        upload(directory)
    else:
        refresh(directory, args.action)


if __name__ == '__main__':
    main()
