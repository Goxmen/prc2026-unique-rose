"""Create the uniquely named final object once after final-selection checks.

Uses the separately reviewed four-month validator and the same reserved-slot
policy. A delivery receipt means exact remote bytes, not an accepted score.
An interrupted attempt may only be reconciled by reading the existing object;
this runner never retries a PUT or overwrites a remote final submission.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import final_candidate_upload_preflight_20261013 as preflight
import scored_blend_submit as transport

ROOT = preflight.ROOT


def write_new(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def save_manifest(path, value):
    temp = path.with_suffix('.json.tmp')
    with temp.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load(path):
    path = Path(path).resolve()
    assert path.is_relative_to(ROOT / 'VALIDATION/runs')
    manifest = preflight.read(path)
    assert manifest['submission_kind'] == 'final' and manifest['file'] == preflight.EXPECTED_FILE
    for helper in [Path(__file__), Path(transport.__file__)]:
        preflight.pinned(manifest, preflight.artifact(helper))
    return path, manifest


def verify_existing(path):
    path, manifest = load(path)
    attempt_path = path.parent / 'final_upload_attempt.json'
    attempt = preflight.read(attempt_path)
    assert manifest['state'] in ['prepared', 'delivered_final']
    if manifest['state'] == 'prepared':
        assert attempt['manifest'] == preflight.artifact(path), 'Prepared manifest changed after upload attempt'
    assert attempt['file'] == dict(path=manifest['file'], sha256=manifest['sha256'])
    assert attempt['uploader'] == preflight.artifact(__file__)
    output = preflight.verify(attempt['file'])
    remote = hashlib.sha256(transport.mc('cat', transport.live.BUCKET + output.name, binary=True)).hexdigest()
    assert remote == manifest['sha256'], 'Remote final differs; never overwrite'
    receipt_path = path.parent / 'final_delivery.json'
    if not receipt_path.exists():
        write_new(receipt_path, dict(status='delivered_hash_verified',
            checked_at=datetime.now(timezone.utc).isoformat(),
            attempted_at=attempt['at'], attempt=preflight.artifact(attempt_path),
            file=attempt['file'], remote_sha256=remote, rows=670790,
            score_acceptance_unverified=True, no_reupload=True))
    receipt = preflight.read(receipt_path)
    assert receipt['remote_sha256'] == remote and receipt['file'] == attempt['file']
    manifest.update(state='delivered_final', remote_sha256=remote,
                    delivered_at=receipt['checked_at'], delivery=preflight.artifact(receipt_path),
                    score_acceptance_unverified=True)
    save_manifest(path, manifest)
    print(json.dumps(dict(state=manifest['state'], rows=670790, remote_hash_exact=True,
                          score_acceptance_unverified=True)))


def upload(path):
    path, manifest = load(path)
    assert manifest['state'] == 'prepared'
    attempt_path = path.parent / 'final_upload_attempt.json'
    assert not attempt_path.exists(), 'Attempt already recorded; use verify-existing, never repeat upload'
    validated, checks = preflight.validate_manifest(path)
    assert validated == manifest
    uploads = preflight.dashboard_data.fetch_bucket(strict=True)
    at = datetime.now(timezone.utc)
    slots = preflight.check_live_slots(manifest, uploads, at.date())
    selection = preflight.check_final_selection(manifest, at.date())
    attempt = dict(at=at.isoformat(), manifest=preflight.artifact(path),
                   uploader=preflight.artifact(__file__), file=checks['file'],
                   validation=checks, live_slots=slots, final_selection=selection,
                   authorization=manifest['authorization'], atomic_create_only=True)
    write_new(attempt_path, attempt)
    output = ROOT / manifest['file']
    try:
        message = transport.mc('cp', '--disable-multipart', '--custom-header', 'If-None-Match:*',
            '--attr', f'sha256={manifest["sha256"]}', output, transport.live.BUCKET + output.name)
        (path.parent / 'final_mc_upload.log').write_text(message)
    except BaseException as error:
        write_new(path.parent / 'final_upload_error.json', dict(error=repr(error),
            at=datetime.now(timezone.utc).isoformat(), action='Inspect remote bytes; no automatic retry'))
        raise
    verify_existing(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['upload', 'verify-existing'])
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    if args.stage == 'upload':
        upload(args.manifest)
    else:
        verify_existing(args.manifest)
