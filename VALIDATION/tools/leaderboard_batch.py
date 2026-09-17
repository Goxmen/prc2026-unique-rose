"""Prepare the five frozen candidates, or explicitly upload their saved artifacts.

Prepare never uploads; upload checks current-day usage and never waits for reset.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline')]
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from records import read_run, sha256, write_json
from registered_rules import RULE_FILES, apply_rule
from submit import fill_template
from tools.sweep_taxi_thresholds import mc, remote_files

BASELINE = 'v3_20260916T095948645262Z'
RECOVERY = 'v3_20260916T113256202842Z'
MANIFEST = ROOT/'VALIDATION/reports/leaderboard_selected_20260916.json'
FITS = ROOT/'submissions/selected-20260916'
BUCKET = 'opensky/prc-2026-unique-rose/'


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def prepare(args):
    if MANIFEST.exists():
        raise ValueError('Selected queue already prepared; preserve the frozen manifest')
    queue = [
        ('B0-800', BASELINE, 'baseline', 'off'),
        ('Q0', 'v3_20260916T123230629423Z', 'baseline', 'rome_6h'),
        ('Q1', 'v3_20260916T123433565511Z', 'recovery', 'rome_6h'),
        ('Q0-4h', args.four_hour_baseline, 'baseline', 'rome_4h'),
        ('Q1-4h', args.four_hour_recovery, 'recovery', 'rome_4h'),
    ]
    remote = remote_files()
    local = {p.name for p in (ROOT/'submissions').glob('unique-rose_v*')}
    for path in (ROOT/'VALIDATION/reports/sweeps').glob('*.json'):
        local.update(row['file'] for row in json.loads(path.read_text())['submissions'])
    versions = [int(m[1]) for name in remote|local
                if (m := re.match(r'^unique-rose_v(\d+)\.', name))]
    first = max(versions, default=0)+1
    template_path = ROOT/'data/submitting.parquet'
    template = pq.read_table(template_path)
    batch = {'selected_at': '2026-09-16', 'prepared_at': now(), 'seed': 42,
             'selection_source': 'VALIDATION/docs/LEADERBOARD_CANDIDATES.md',
             'selection_sha256': sha256(ROOT/'VALIDATION/docs/LEADERBOARD_CANDIDATES.md'),
             'template_sha256': sha256(template_path), 'new_cv_fits': 0,
             'full_year_fits': 2, 'submissions': [],
             'upload_policy': 'Five selected files only; no automatic quota-reset retry',
             'preparation_source': {name: sha256(ROOT/name) for name in [
                 'VALIDATION/tools/leaderboard_batch.py', 'VALIDATION/tools/refit_frozen.py',
                 'VALIDATION/submit.py']}}
    fits = {}
    for name, parent_id in [('baseline', BASELINE), ('recovery', RECOVERY)]:
        folder = FITS/name
        meta = json.loads((folder/'metadata.json').read_text())
        parent, parent_dir = read_run(parent_id)
        if (meta['parent'] != parent_id or meta['seed'] != 42 or meta['rounds'] != 800
                or meta['parent_record_sha256'] != sha256(parent_dir/'result.json')
                or meta['data_identity'] != parent['data_identity']
                or meta['model_sha256'] != sha256(folder/'model.txt')
                or meta['predictions_sha256'] != sha256(folder/'predictions.parquet')
                or any(parent['source']['files'].get(k) != v for k,v in meta['model_source'].items())):
            raise ValueError(f'Fit provenance mismatch: {name}')
        fits[name] = (meta, pd.read_parquet(folder/'predictions.parquet'))
    batch['models'] = {name: {**meta, 'directory': str((FITS/name).relative_to(ROOT)),
                              'metadata_sha256': sha256(FITS/name/'metadata.json')}
                       for name, (meta, _) in fits.items()}
    for index, (code, run_id, fit, rule) in enumerate(queue):
        record, directory = read_run(run_id)
        meta, raw = fits[fit]
        model_source = record.get('model_source', record['source'])['files']
        expected_parent = BASELINE if fit == 'baseline' else RECOVERY
        if (record['status'] != 'complete' or not record['full_benchmark']
                or record.get('model_parent_id', record['run_id']) != expected_parent
                or record['config'].get('prediction_rule', 'off') != rule
                or record['data_identity'] != meta['data_identity']
                or any(model_source.get(k) != v for k,v in meta['model_source'].items())):
            raise ValueError(f'Candidate does not match full-year model: {code}')
        rule_source = {name: sha256(ROOT/name) for name in RULE_FILES[rule]}
        if rule_source != record.get('rule_source', {}):
            raise ValueError(f'Rule changed since validation: {code}')
        pred, audit = apply_rule(raw.pred, raw, rule)
        out = fill_template(template, raw.mvt_id, pred)
        version = first+index
        path = ROOT/'submissions'/f'unique-rose_v{version}.parquet'
        pq.write_table(out, path)
        scores = record['report']['scores']
        cv = {k: scores[k] for k in ['R_A', 'R_B_long', 'R_combined']}
        seed_cv = {k: record['report']['groups'][k]['rmse_by_seed']['42'] for k in ['R_A', 'R_B_long']}
        w = record['ranking_weights']['jan_fraction']
        seed_cv['R_combined'] = float(np.sqrt(w*seed_cv['R_A']**2+(1-w)*seed_cv['R_B_long']**2))
        item = {'codename': code, 'version': version, 'file': path.name,
                'state': 'prepared', 'score': None, 'validation_run': run_id,
                'validation_record_sha256': sha256(directory/'result.json'),
                'model': fit, 'model_parent': expected_parent, 'prediction_rule': rule,
                'rule_source': rule_source, 'model_source': meta['model_source'],
                'model_sha256': meta['model_sha256'], 'config': meta['config'],
                'data_identity': meta['data_identity'], 'seed': 42,
                'raw_predictions_sha256': meta['predictions_sha256'],
                'cv_standard': cv, 'cv_seed42': seed_cv, 'rows': out.num_rows,
                'triggered_rows': int(audit.rule_triggered.sum()),
                'sha256': sha256(path), 'template_sha256': batch['template_sha256']}
        write_json(path.with_suffix('.meta.json'), item)
        item['metadata_sha256'] = sha256(path.with_suffix('.meta.json'))
        batch['submissions'].append(item)
        print(f"Prepared {path.name}: {code}, {out.num_rows:,} rows, {item['triggered_rows']} rule rows", flush=True)
    write_json(MANIFEST, batch)


def bucket_listing():
    rows = [json.loads(line) for line in mc('ls', '--json', BUCKET).splitlines() if line.strip()]
    if any(row.get('status') == 'error' for row in rows):
        raise RuntimeError('Bucket listing failed')
    return {r['key']: r for r in rows if r.get('type') == 'file'}


def upload():
    batch = json.loads(MANIFEST.read_text())
    expected = ['B0-800', 'Q0', 'Q1', 'Q0-4h', 'Q1-4h']
    if [r['codename'] for r in batch['submissions']] != expected:
        raise ValueError('Upload queue differs from selected candidates')
    for item in batch['submissions']:
        path = ROOT/'submissions'/item['file']
        if (sha256(path) != item['sha256']
                or sha256(path.with_suffix('.meta.json')) != item['metadata_sha256']):
            raise ValueError(f"Artifact changed after preparation: {item['file']}")
    for item in batch['submissions']:
        remote = bucket_listing()
        if item['file'] in remote:
            digest = hashlib.sha256(mc('cat', BUCKET+item['file'], binary=True)).hexdigest()
            if digest != item['sha256']:
                raise ValueError(f"Remote version collision: {item['file']}")
            item.update(state='uploaded', remote_sha256=digest)
            write_json(MANIFEST, batch)
            continue
        today = datetime.now(timezone.utc).date()
        today_uploads = [name for name,row in remote.items() if name.endswith('.parquet')
                         and datetime.fromisoformat(row['lastModified'].replace('Z','+00:00')).astimezone(timezone.utc).date() == today]
        batch.setdefault('quota_checks', []).append({'at': now(), 'utc_day': str(today),
                                                    'existing_uploads_today': today_uploads})
        if len(today_uploads) >= 5:
            batch['blocked'] = 'Five uploads already exist today; no automatic reset retry'
            write_json(MANIFEST, batch)
            print(batch['blocked'], flush=True)
            return
        path = ROOT/'submissions'/item['file']
        attrs = f"candidate={item['codename']};validation-run={item['validation_run']};sha256={item['sha256']}"
        item['upload_started_at'] = now()
        write_json(MANIFEST, batch)
        mc('cp', '--attr', attrs, path, BUCKET+item['file'])
        item.update(state='uploaded', uploaded_at=now())
        write_json(MANIFEST, batch)
        digest = hashlib.sha256(mc('cat', BUCKET+item['file'], binary=True)).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Uploaded object differs from frozen artifact')
        item['remote_sha256'] = digest
        write_json(MANIFEST, batch)
        print(f"Uploaded and verified {item['file']}: {item['codename']}", flush=True)


def refresh():
    batch = json.loads(MANIFEST.read_text())
    remote = bucket_listing()
    for item in batch['submissions']:
        key = item['file']+'_result.json'
        if item['state'] != 'uploaded' or key not in remote:
            continue
        if item.get('remote_sha256') != item['sha256']:
            raise ValueError('Cannot attach a score without verified uploaded content')
        result = json.loads(mc('cat', BUCKET+key))
        item['official_result'] = result
        item['official_status'] = result.get('status', 'Unknown')
        item['score'] = result.get('score') if result.get('status') == 'Succeeded' else None
        print(item['codename'], item['official_status'], item['score'], flush=True)
    batch['scores_checked_at'] = now()
    write_json(MANIFEST, batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload', 'refresh'])
    parser.add_argument('--four-hour-baseline')
    parser.add_argument('--four-hour-recovery')
    args = parser.parse_args()
    if args.action == 'prepare':
        if not args.four_hour_baseline or not args.four_hour_recovery:
            parser.error('prepare requires both registered four-hour run IDs')
        prepare(args)
    elif args.action == 'upload':
        upload()
    else:
        refresh()


if __name__ == '__main__':
    main()
