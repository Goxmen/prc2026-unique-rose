#!/usr/bin/env python3
"""Publish the explicitly authorized repository/release, never before its deadline.

`check` performs read-only preflight. Only `publish` can change GitHub visibility.
No competition submission, model execution or credential logging is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import audit_repository

ROOT = Path(__file__).resolve().parents[2]
GH = Path.home() / '.local/bin/gh'


def now():
    return datetime.now(timezone.utc)


def deadline(policy):
    return datetime.fromisoformat(policy['not_before_utc'].replace('Z', '+00:00'))


def guard_time(policy, current=None):
    if (current or now()) < deadline(policy):
        raise ValueError('Publication is not authorized before the deadline.')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args],
                                   stderr=subprocess.DEVNULL, text=True).strip()


def api(endpoint, payload=None):
    command = [str(GH), 'api', '--method', 'GET' if payload is None else 'PATCH', endpoint]
    if payload is not None:
        command += ['--input', '-']
    try:
        result = subprocess.run(command, input=None if payload is None else json.dumps(payload),
                                capture_output=True, text=True, timeout=60, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        # Authentication errors can contain sensitive diagnostic context.
        raise RuntimeError(f'GitHub request failed: {type(exc).__name__}; inspect auth privately.') from None
    return json.loads(result.stdout)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_remote(policy, repository, release):
    require(repository['id'] == policy['repository_id'] and
            repository['full_name'] == policy['repository'], 'Repository identity changed.')
    require(repository.get('permissions', {}).get('admin'), 'Repository administration permission is required.')
    require(release['id'] == policy['release_id'] and release['tag_name'] == policy['release_tag'],
            'Release identity changed; update the reviewed publication policy.')
    require(release['target_commitish'] == policy['release_source_commit'], 'Release source commit changed.')
    actual = {asset['name']: asset for asset in release['assets']}
    require(set(actual) == {asset['name'] for asset in policy['assets']}, 'Release asset set changed.')
    for expected in policy['assets']:
        asset = actual[expected['name']]
        require(asset['id'] == expected['github_asset_id'] and asset['state'] == 'uploaded' and
                asset['size'] == expected['bytes'] and asset.get('digest') == 'sha256:' + expected['sha256'],
                'Release asset identity/digest changed.')


def verify_tag(policy, required=False):
    base = 'repos/' + policy['repository']
    reference = 'refs/tags/' + policy['release_tag']
    matches = [item for item in api(base + '/git/matching-refs/tags/' + policy['release_tag'])
               if item['ref'] == reference]
    if not matches:
        require(not required, 'Published release tag is missing.')
        return
    require(len(matches) == 1, 'Ambiguous release tag.')
    obj = matches[0]['object']
    for _ in range(10):
        if obj['type'] == 'commit':
            require(obj['sha'] == policy['release_source_commit'], 'Release tag points to different source.')
            return
        require(obj['type'] == 'tag', 'Release tag does not resolve to a commit.')
        obj = api(base + '/git/tags/' + obj['sha'])['object']
    raise ValueError('Release tag nesting exceeds the verification limit.')


def preflight(policy):
    require(policy.get('authorized') is True and policy['repository'] == 'Goxmen/PRC',
            'Missing explicit publication authorization.')
    require(deadline(policy).utcoffset().total_seconds() == 0, 'Deadline must be explicit UTC.')
    require(git('branch', '--show-current') == 'main', 'Review must run from main.')
    head = git('rev-parse', 'HEAD')
    report = audit_repository.audit(ROOT)
    require(not report['findings'], 'Repository publication audit needs review; no values are logged.')
    require(not report['tracked_artifact_paths'], 'Unexpected tracked model/data artifacts.')
    require(all(report['ignored_path_checks'].values()), 'Private file ignore rules changed.')
    base = 'repos/' + policy['repository']
    repository, release = api(base), api(base + '/releases/' + str(policy['release_id']))
    verify_remote(policy, repository, release)
    verify_tag(policy, required=not release['draft'])
    remote = api(base + '/git/ref/heads/main')['object']['sha']
    require(head == git('rev-parse', 'HEAD') == remote, 'Local and remote reviewed main must agree.')
    return dict(checked_at_utc=now().isoformat(), reviewed_main=head,
                repository_private=repository['private'], release_draft=release['draft'],
                release_url=release['html_url'], audit_findings=0,
                audit_coverage=report['coverage'], assets_verified=len(policy['assets']))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def publish(policy, receipt):
    guard_time(policy)  # No network call or mutation if accidentally started early.
    check = preflight(policy)
    base = 'repos/' + policy['repository']
    result = dict(status='preflight_passed', not_before_utc=policy['not_before_utc'],
                  policy_sha256=hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest(),
                  preflight=check, competition_submission_unchanged=True)
    save(receipt, result)
    # Recheck the audited remote commit immediately before exposing the repository.
    require(api(base + '/git/ref/heads/main')['object']['sha'] == check['reviewed_main'],
            'Remote main changed after the publication audit.')
    if check['repository_private']:
        guard_time(policy)
        api(base, {'private': False})
    require(api(base)['private'] is False, 'Repository public visibility not confirmed.')
    result.update(status='repository_public', repository_public_confirmed_at=now().isoformat())
    save(receipt, result)
    # A retry after partial success observes the public repository and finishes the release.
    if check['release_draft']:
        verify_tag(policy)
        guard_time(policy)
        api(base + '/releases/' + str(policy['release_id']), {'draft': False})
    repository, release = api(base), api(base + '/releases/' + str(policy['release_id']))
    verify_remote(policy, repository, release)
    require(not repository['private'] and not release['draft'], 'Publication did not reach requested state.')
    verify_tag(policy, required=True)
    result.update(status='published_verified', checked_at_utc=now().isoformat(),
                  repository_url=repository['html_url'], release_url=release['html_url'])
    save(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'publish'])
    parser.add_argument('--policy', type=Path, default=ROOT / 'release/publication_schedule.json')
    parser.add_argument('--receipt', type=Path, default=ROOT / 'release/artifacts/publication.json')
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = preflight(policy) if args.command == 'check' else publish(policy, args.receipt)
            if args.command == 'check':
                save(args.receipt, dict(status='read_only_preflight_passed', **result))
        except Exception as exc:
            failure = dict(status='failed_without_rollback', checked_at_utc=now().isoformat(),
                           exception=type(exc).__name__, message=str(exc),
                           competition_submission_unchanged=True)
            # Keep all failed attempts; don't overwrite evidence of partial publication.
            stamp = now().strftime('%Y%m%dT%H%M%S%fZ')
            save(args.receipt.with_name(args.receipt.stem + '.failure.' + stamp + '.json'), failure)
            print(json.dumps(failure))
            return 1
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
