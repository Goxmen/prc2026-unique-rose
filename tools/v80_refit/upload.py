"""Single, explicitly user-launched replacement of the accepted final object.

A recorded uncertain PUT is never repeated automatically. Resume reconciles its
remote hash and waits for a fresh official acceptance instead. No score is inferred.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from refit_io import ROOT, DEFAULT, DEADLINE, artifact, now, read, save, sha, verify

sys.path[:0] = [str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION')]
import dashboard_data
from final_candidate_upload_preflight_20261013 import check_tables, TEMPLATE_SHA

BUCKET = 'opensky/prc-2026-unique-rose/'
NAME = 'unique-rose_final.parquet'
PRIOR_SHA = 'f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f'
MC = str(Path.home() / '.local/bin/mc')

def timestamp(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(timezone.utc)

def mc(*args):
    result = subprocess.run([MC, *args], env=dashboard_data.mc_environment(),
                            capture_output=True, timeout=180)
    if result.returncode:
        # Do not echo credential-bearing diagnostics; outputs are not needed here.
        raise RuntimeError(f'mc {args[0]} failed (exit {result.returncode})')
    return result.stdout

def parse_stat(data, name):
    # mc stat on a prefix can emit both parquet and its result JSON.
    items = [json.loads(line) for line in data.splitlines() if line.strip()]
    matches = [r for r in items if r.get('name') == name and r.get('status') == 'success']
    if len(matches) != 1:
        raise ValueError('Exact remote object stat missing or ambiguous: ' + name)
    return matches[0]

def stat(name):
    return parse_stat(mc('stat','--json', BUCKET + name), name)

def fetch(name, destination):
    data = mc('cat', BUCKET + name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + '.tmp')
    tmp.write_bytes(data); tmp.replace(destination)
    return artifact(destination)

def validate(out):
    import pyarrow.parquet as pq
    audit = read(out / 'audit.json')
    if audit['status'] != 'pass':
        raise ValueError('Assembly audit must pass')
    candidate = verify(audit['candidate'])
    if candidate != out / 'final/candidate.parquet':
        raise ValueError('Unexpected final source')
    template_path = ROOT / 'data/final_submitting.parquet'
    if sha(template_path) != TEMPLATE_SHA:
        raise ValueError('Final template hash changed')
    template, actual = pq.read_table(template_path), pq.read_table(candidate)
    table = check_tables(template, actual, actual)
    if candidate.stat().st_size > 50_000_000:
        raise ValueError('Unexpected final file size')
    return dict(candidate=artifact(candidate), audit=artifact(out/'audit.json'), checks=table,
                recipe='v80 fixed ensemble; missing components freshly refitted',
                selected_by='Explicit user request to refit v80 and submit its predictions as final',
                original_leaderboard_score_reference_only=218.8525,
                fresh_refit_cv=None, original_weights_recovered_exactly=False)

def check_quota(current, entries):
    used = sum(bool(e.get('uploadedAt')) and timestamp(e['uploadedAt']).date() == current.date()
               for e in entries)
    if used >= 5:
        raise ValueError('All five daily submission slots are already used')
    return dict(utc_date=current.date().isoformat(), observed_uploaded_objects=used,
                reserved_final_slot_used_by_explicit_user_selection=True)

def fresh_acceptance(meta, result, attempt):
    return (timestamp(meta['lastModified']) >= timestamp(attempt['started_at']) and
            result.get('status') == 'Succeeded' and result.get('file') == NAME and
            result.get('bucket') == 'prc-2026-unique-rose' and
            result.get('used_pairs') == 670790 and
            result.get('inputs', {}).get('test_set_file') == 'final_submitting.parquet')

def put_arguments(candidate, etag, digest):
    if not etag or any(c in etag for c in '\r\n"'):
        raise ValueError('Invalid remote ETag')
    return ('cp','--disable-multipart','--custom-header', 'If-Match:"' + etag + '"',
            '--attr','sha256='+digest,str(candidate),BUCKET+NAME)

def deliver(out, wait_seconds=900):
    folder = out / 'delivery'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'upload.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = validate(out)
        if manifest.get('original_weights_recovered_exactly') is False:
            raise ValueError(
                'Approximate v80 refit delivery is disabled by the owner\'s latest instruction; '
                'only fully recovered, verified v80 may replace the selected final.')
        save(folder/'selection.json', manifest)
        candidate = verify(manifest['candidate'])
        attempt_path = folder/'attempt.json'
        accepted_path = folder/'acceptance.json'
        if accepted_path.exists():
            accepted = read(accepted_path)
            if accepted['candidate'] != manifest['candidate']:
                raise ValueError('Accepted final differs from this candidate')
            actual = fetch(NAME, folder/'remote_verified.parquet')
            if actual['sha256'] != manifest['candidate']['sha256']:
                raise ValueError('Remote accepted final changed')
            return accepted
        if attempt_path.exists():
            attempt = read(attempt_path)
            if attempt['candidate'] != manifest['candidate']:
                raise ValueError('Prior PUT attempt belongs to a different candidate')
        else:
            if now() >= DEADLINE:
                raise ValueError('Final submission deadline has passed; no upload attempted')
            # Keep the old bytes and stat. Local backups never touch the old receipts.
            before = stat(NAME)
            prior = fetch(NAME, folder/'previous_remote_final.parquet')
            if prior['sha256'] != PRIOR_SHA:
                raise ValueError('Remote final is not the reviewed v83 file; refusing overwrite')
            local = ROOT/'submissions'/NAME
            if local.exists():
                backup = folder/'previous_local_final.parquet'
                if not backup.exists():
                    shutil.copy2(local, backup)
            quota = check_quota(now(), dashboard_data.fetch_bucket(strict=True))
            # Recheck immediately before the conditional request.
            latest = stat(NAME)
            if latest['etag'] != before['etag'] or latest['lastModified'] != before['lastModified']:
                raise ValueError('Remote final changed during preflight')
            if now() >= DEADLINE:
                raise ValueError('Final submission deadline has passed; no upload attempted')
            attempt = dict(status='put_attempt_recorded', started_at=now().isoformat(),
                           candidate=manifest['candidate'], previous=prior, previous_stat=before,
                           quota=quota, deadline_utc=DEADLINE.isoformat(), no_automatic_put_retry=True)
            # Exclusive write makes a second invocation reconcile, never repeat PUT.
            with attempt_path.open('x') as f:
                json.dump(attempt, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
            try:
                mc(*put_arguments(candidate, before['etag'].strip('"'), manifest['candidate']['sha256']))
            except Exception as exc:
                save(folder/'put_error.json', dict(error=type(exc).__name__, uncertain=True,
                                                 action='Resume only reconciles remote bytes; no automatic retry'))
                # Continue to reconcile: the server may have accepted a timed-out PUT.
        remote = fetch(NAME, folder/'remote_verified.parquet')
        if remote['sha256'] != manifest['candidate']['sha256']:
            raise RuntimeError('Recorded PUT is not confirmed remotely. It will NOT be retried automatically; inspect delivery/ receipts.')
        save(folder/'delivery.json', dict(status='remote_hash_verified', candidate=manifest['candidate'],
                                         remote=remote, checked_at=now().isoformat()))
        # Change local final alias only after exact remote delivery verification.
        local = ROOT/'submissions'/NAME
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix('.parquet.tmp')
        shutil.copy2(candidate, temporary); temporary.replace(local)
        deadline = time.monotonic()+wait_seconds
        while True:
            try:
                name = NAME+'_result.json'
                meta = stat(name)
                body = json.loads(mc('cat',BUCKET+name))
                if fresh_acceptance(meta, body, attempt):
                    accepted_remote = fetch(NAME, folder/'remote_verified.parquet')
                    if accepted_remote['sha256'] != manifest['candidate']['sha256']:
                        raise ValueError('Remote final changed before acceptance was confirmed')
                    result = dict(status='accepted', candidate=manifest['candidate'], official=body,
                                  official_stat=meta, checked_at=now().isoformat(),
                                  numeric_final_score_published=isinstance(body.get('score'), (int,float)))
                    save(accepted_path, result)
                    return result
                if timestamp(meta['lastModified']) >= timestamp(attempt['started_at']) and body.get('status') in {'Failed','Rejected'}:
                    save(folder/'official_failure.json',dict(result=body,meta=meta))
                    raise ValueError('New official final was rejected; preserved delivery/official_failure.json')
            except (RuntimeError, json.JSONDecodeError):
                pass  # Temporary read failure; never another PUT.
            if time.monotonic() >= deadline:
                raise RuntimeError('Upload hash verified; fresh official acceptance still pending. Rerun the same command to reconcile without uploading again.')
            time.sleep(15)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['check','run'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    args=parser.parse_args()
    result=validate(args.out.resolve()) if args.command=='check' else deliver(args.out.resolve())
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
