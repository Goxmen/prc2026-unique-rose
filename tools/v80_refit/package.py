#!/usr/bin/env python3
"""Prepare private v80-refit replay release; activate only after final acceptance.

prepare exports and verifies a standalone replay, then uploads a private draft.
activate validates the exact final-delivery/acceptance receipts and switches the
already authorized deadline timer to that draft. Neither command publishes now
or uploads competition predictions. Only explicitly named generated release
metadata is committed. The historical v83 draft remains intact.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'VALIDATION/runs/v3_20261013_v80_refit_final'
GH = Path.home() / '.local/bin/gh'
REPO = 'Goxmen/PRC'
REPO_ID = 1357299087
POLICY = ROOT / 'release/publication_schedule.json'
GENERATED = ['release/publication_schedule.json', 'release/artifact_manifest.json',
             'release/RELEASE_NOTES.md', 'release/SELECTED_FINAL.md', 'release/delivery.json']
COMMIT_MESSAGE = 'Record accepted v80 refit final and switch deadline release'
CAVEAT = ('This is the fixed v80 recipe with fresh full-2025 fits for missing members and '
          'retained matching full-year models for the others. Original ensemble weights and '
          'calibrations are fixed. Historical v3/v5-v9 use the archived original27 implementation '
          'because their exact historical modeling.py is absent. New GPU fits can differ numerically. '
          'No fresh four-month CV or new leaderboard RMSE is claimed. The replay verifies retained '
          'component assembly, not a clean-clone/raw-data training rebuild.')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def artifact(path):
    p = Path(path).absolute()
    return dict(path=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                sha256=sha(p), bytes=p.stat().st_size)


def resolve(item):
    path = Path(item['path'])
    return path if path.is_absolute() else ROOT / path


def checked(item):
    path = resolve(item)
    if sha(path) != item['sha256']:
        raise ValueError(f"Artifact changed: {item['path']}")
    return path


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def command(args, *, input=None, timeout=120):
    try:
        result = subprocess.run(args, cwd=ROOT, input=input, text=True, capture_output=True,
                                check=True, timeout=timeout)
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(f'Command failed ({Path(str(args[0])).name}, {type(exc).__name__}); inspect privately') from None
    return result.stdout.strip()


def git(*args):
    return command(['git', *args])


def api(endpoint, payload=None, method=None):
    args = [str(GH), 'api', '--method', method or ('GET' if payload is None else 'POST'), endpoint]
    if payload is not None:
        args += ['--input', '-']
    return json.loads(command(args, input=None if payload is None else json.dumps(payload)))


def verify_repository(expected_head=None, *, allow_after_deadline=False):
    if git('branch', '--show-current') != 'main':
        raise ValueError('Release requires main')
    head = git('rev-parse', 'HEAD')
    remote = api(f'repos/{REPO}/git/ref/heads/main')['object']['sha']
    repo = api(f'repos/{REPO}')
    if repo['id'] != REPO_ID or repo['full_name'] != REPO or not repo.get('permissions', {}).get('admin'):
        raise ValueError('Repository identity/permission mismatch')
    if head != remote or (expected_head and head != expected_head):
        raise ValueError('Local and remote reviewed main must agree')
    deadline = datetime.fromisoformat(read(POLICY)['not_before_utc'].replace('Z', '+00:00'))
    after_deadline = datetime.now(timezone.utc) >= deadline
    if after_deadline and not allow_after_deadline:
        raise ValueError('New refit release preparation must finish before the authorized deadline')
    if not repo['private'] and not after_deadline:
        raise ValueError('Repository must remain private before the deadline')
    # These sources must be committed, not only present in a local workspace.
    for path in sorted((ROOT / 'tools/v80_refit').glob('*.py')):
        relative = str(path.relative_to(ROOT))
        if git('ls-files', '--error-unmatch', relative) != relative or git('diff', 'HEAD', '--', relative):
            raise ValueError(f'Uncommitted release source: {relative}')
    return head


def replay_module():
    spec = importlib.util.spec_from_file_location('v80_refit_standalone', Path(__file__).with_name('replay.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_audit(out):
    audit = read(out / 'audit.json')
    if audit.get('status') != 'pass' or audit.get('rows') != 670790:
        raise ValueError('Full final assembly audit is required')
    candidate = checked(audit['candidate'])
    identity = checked(audit['identity'])
    if identity != out / 'assembly_identity.json':
        raise ValueError('Unexpected assembly identity')
    return audit, candidate, read(identity)


def export_bundle(out, source_commit):
    audit, candidate, identity = validate_audit(out)
    digest = sha(candidate)
    bundle = ROOT / 'release/bundles' / f'v80-refit-{digest[:16]}'
    replay = replay_module()
    if (bundle / 'manifest.json').exists() and not (bundle / 'INCOMPLETE.json').exists():
        manifest = replay.verify(bundle)
        if manifest['candidate_sha256'] != digest or manifest['source_commit'] != source_commit:
            raise ValueError('Existing bundle identity differs')
        audit_replay(out, bundle, digest)
        return bundle, manifest
    if bundle.exists():
        raise ValueError('Interrupted bundle export exists; preserve it and inspect')
    bundle.mkdir(parents=True)
    save(bundle / 'INCOMPLETE.json', dict(started_at=now()))
    shutil.copyfile(out / 'final/basis51.parquet', bundle / 'basis51.parquet')
    stages = pd.read_parquet(out / 'final/stages.parquet', columns=['mvt_id', 'v78', 'tabm_v78', 'robust_v78'])
    stages.to_parquet(bundle / 'corrections.parquet', index=False)
    shutil.copyfile(ROOT / 'data/final_submitting.parquet', bundle / 'template.parquet')
    save(bundle / 'recipe.json', dict(versions=identity['versions'], weights=identity['weights'],
         arithmetic='v79=rint(max(P@w,0)); v80=rint(max(v79+(tabm_v78-v78)+(robust_v78-v78),0))', threads=1))
    provenance = [out / 'run_identity.json', out / 'assembly_identity.json', out / 'audit.json',
                  out / 'trees/identity.json', out / 'trees/result.json',
                  out / 'neural/identity.json', out / 'neural/result.json']
    provenance += sorted((out / 'trees/models').glob('*.json'))
    provenance += sorted((out / 'neural/models').glob('*/complete.json'))
    # Source repairs preserve original identities/receipts and their migration
    # record here. Include them so reviewers can distinguish pre-repair fits.
    provenance += sorted(p for p in (out / 'recovery').rglob('*') if p.is_file())
    for path in provenance:
        dest = bundle / 'provenance' / path.relative_to(out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    for path in sorted((ROOT / 'tools/v80_refit').glob('*.py')):
        dest = bundle / 'source' / path.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, dest)
    for source, name in [(ROOT / 'LICENSE', 'LICENSE'),
                         (ROOT / 'docs/DATA_SOURCES.md', 'DATA_SOURCES.md'),
                         (ROOT / 'docs/THIRD_PARTY.md', 'THIRD_PARTY.md'),
                         (ROOT / 'release/requirements-replay.txt', 'requirements-replay.txt'),
                         (Path(__file__).with_name('replay.py'), 'replay.py')]:
        shutil.copyfile(source, bundle / name)
    (bundle / 'README.md').write_text(
        '# v80 full-year refit: exact assembly replay\n\n' + CAVEAT + '\n\n'
        'Install `requirements-replay.txt`, then run:\n\n```bash\n'
        'python replay.py verify --bundle .\n'
        'python replay.py reproduce --bundle . --output unique-rose_final.parquet\n```\n\n'
        'No credentials, network, GPU, model training or submission is performed. '
        'Use the pinned runtime; all input files and final output bytes are SHA-256 verified.\n')
    files = [dict(path=str(p.relative_to(bundle)), sha256=sha(p), bytes=p.stat().st_size)
             for p in sorted(bundle.rglob('*')) if p.is_file() and p.name != 'INCOMPLETE.json']
    manifest = dict(format='prc-v80-refit-replay-v1', rows=670790, candidate_sha256=digest,
                    source_commit=source_commit, versions=identity['versions'], files=files,
                    caveat=CAVEAT, environment=dict(pyarrow=pa.__version__, pandas=pd.__version__),
                    expected_scope='final January/February/June/July 2026')
    save(bundle / 'manifest.json', manifest)
    (bundle / 'INCOMPLETE.json').unlink()
    replay.verify(bundle)
    audit_replay(out, bundle, digest)
    return bundle, manifest


def audit_replay(out, bundle, digest):
    replay = replay_module()
    manifest_hash = sha(bundle / 'manifest.json')
    marker = out / 'release/replay_audit.json'
    check_output = out / 'release/replayed_final.parquet'
    if marker.exists():
        prior = read(marker)
        if prior.get('status') != 'pass' or prior['manifest_sha256'] != manifest_hash or prior['candidate_sha256'] != digest:
            raise ValueError('Prior standalone replay audit differs')
        checked(prior['output'])
        return
    if check_output.exists():
        if sha(check_output) != digest:
            raise ValueError('Prior replay output differs')
    else:
        replay.reproduce(bundle, check_output, sha(bundle / 'manifest.json'))
    if sha(check_output) != digest:
        raise ValueError('Standalone replay did not reproduce candidate bytes')
    save(marker, dict(status='pass', manifest_sha256=manifest_hash, candidate_sha256=digest,
                      output=artifact(check_output), checked_at=now()))


def verify_archive(destination, bundle):
    expected = {str(p.relative_to(bundle)): p for p in bundle.rglob('*') if p.is_file()}
    seen = set()
    with tarfile.open(destination, 'r') as tar:
        for item in tar:
            if not item.isfile() or item.name not in expected or item.name in seen:
                raise ValueError('Archive contains unexpected or duplicate member')
            seen.add(item.name)
            stream = tar.extractfile(item)
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(8 << 20), b''):
                digest.update(block)
            if item.size != expected[item.name].stat().st_size or digest.hexdigest() != sha(expected[item.name]):
                raise ValueError('Archive member differs from verified replay bundle')
    if seen != set(expected):
        raise ValueError('Archive is missing bundle members')


def archive(bundle):
    destination = ROOT / 'release/artifacts' / (bundle.name + '.tar')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        tmp = destination.with_suffix('.tmp')
        with tarfile.open(tmp, 'w') as tar:
            for path in sorted(bundle.rglob('*')):
                if not path.is_file():
                    continue
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(bundle)))
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                info.mode = 0o644
                with path.open('rb') as f:
                    tar.addfile(info, f)
        tmp.replace(destination)
    verify_archive(destination, bundle)
    checksum = destination.with_name(destination.stem + '-SHA256SUMS')
    expected = f'{sha(destination)}  {destination.name}\n'
    if checksum.exists() and checksum.read_text() != expected:
        raise ValueError('Archive checksum receipt changed')
    checksum.write_text(expected)
    return [destination, checksum]


def release_notes(candidate_hash, manifest_hash, commit, accepted=False):
    return ('# v80 full-year refit final\n\n'
            + ('Accepted replacement final: ' if accepted else 'Prepared replacement final, pending competition acceptance: ')
            + f'670,790 predictions; SHA-256 `{candidate_hash}`.\n\n'
            + CAVEAT + '\n\n'
            + f'Source commit: `{commit}`. Bundle manifest: `{manifest_hash}`.\n\n'
            + 'The archive includes an offline replay script, the 51 integer basis vectors, fixed weights, '
              'three public-correction input vectors, blank template, source and provenance/model checksums. '
              'It excludes the final output and raw external datasets. Provider terms remain documented.\n\n'
            + 'The repository and this draft are authorized for publication only at '
              '**14 October 2026, 10:00 UTC / 12:00 Amsterdam**.\n')


def prepare(out):
    out = Path(out).absolute()
    target = out / 'release'
    target.mkdir(parents=True, exist_ok=True)
    resume_generated_push(out)
    prior = read(target / 'prepared.json') if (target / 'prepared.json').exists() else None
    current_head = verify_repository(allow_after_deadline=bool(prior and (out / 'delivery/attempt.json').exists()))
    audit, candidate, _ = validate_audit(out)
    if (target / 'result.json').exists() and read(target / 'result.json').get('status') == 'accepted_final_release_activated':
        acceptance(out, prior)
        return prior
    head = prior['source_commit'] if prior else current_head
    if prior:
        if prior['candidate']['sha256'] != sha(candidate):
            raise ValueError('Previously prepared release belongs to another candidate')
        git('merge-base', '--is-ancestor', head, current_head)
    bundle, manifest = export_bundle(out, head)
    assets = archive(bundle)
    tag = f'prc-final-v80-refit-{sha(candidate)[:12]}'
    notes = release_notes(sha(candidate), sha(bundle / 'manifest.json'), head)
    matches = [r for r in api(f'repos/{REPO}/releases?per_page=100') if r['tag_name'] == tag]
    if len(matches) > 1:
        raise ValueError('Ambiguous refit release tag')
    if matches:
        release = matches[0]
    else:
        release = api(f'repos/{REPO}/releases', dict(tag_name=tag, target_commitish=head,
                      name='unique-rose v80 full-year refit final', body=notes, draft=True, prerelease=False))
    if not release['draft'] or release['target_commitish'] != head:
        raise ValueError('Existing refit release is published or targets another source')
    save(target / 'draft.json', dict(id=release['id'], tag=tag, source_commit=head,
                                  candidate=artifact(candidate), created_or_reused_at=now()))
    for path in assets:
        release = api(f'repos/{REPO}/releases/{release["id"]}')
        existing = [a for a in release['assets'] if a['name'] == path.name]
        if not existing:
            command([str(GH), 'release', 'upload', tag, str(path), '--repo', REPO], timeout=900)
        elif len(existing) != 1 or existing[0].get('digest') != 'sha256:' + sha(path):
            raise ValueError('Existing draft asset differs; never overwrite silently')
    release = api(f'repos/{REPO}/releases/{release["id"]}')
    actual = {a['name']: a for a in release['assets']}
    if set(actual) != {p.name for p in assets}:
        raise ValueError('Unexpected draft asset set')
    records = []
    for p in assets:
        a = actual[p.name]
        if a['size'] != p.stat().st_size or a['state'] != 'uploaded' or a.get('digest') != 'sha256:' + sha(p):
            raise ValueError('Remote release asset checksum verification failed')
        records.append(dict(name=p.name, bytes=p.stat().st_size, sha256=sha(p),
                            github_asset_id=a['id'], github_digest_verified=True))
    result = dict(status='private_draft_ready', candidate=artifact(candidate), audit=artifact(out / 'audit.json'),
                  bundle_manifest=artifact(bundle / 'manifest.json'), source_commit=head,
                  release_id=release['id'], release_tag=tag, release_url=release['html_url'],
                  assets=records, prepared_at=now(), caveat=CAVEAT, existing_publication_policy_unchanged=True)
    save(target / 'prepared.json', result)
    save(target / 'result.json', result)
    print(json.dumps(dict(status=result['status'], release_url=result['release_url'])))
    return result


def acceptance(out, prepared):
    accepted = read(out / 'delivery/acceptance.json')
    delivered = read(out / 'delivery/delivery.json')
    attempt = read(out / 'delivery/attempt.json')
    digest = prepared['candidate']['sha256']
    if (accepted.get('status') != 'accepted' or delivered.get('status') != 'remote_hash_verified'
            or accepted['candidate']['sha256'] != digest or delivered['candidate']['sha256'] != digest
            or delivered['remote']['sha256'] != digest):
        raise ValueError('New final acceptance/delivery does not bind the prepared candidate')
    official = accepted['official']
    if (official.get('status') != 'Succeeded' or official.get('used_pairs') != 670790
            or official.get('file') != 'unique-rose_final.parquet'
            or official.get('bucket') != 'prc-2026-unique-rose'
            or official.get('inputs', {}).get('test_set_file') != 'final_submitting.parquet'):
        raise ValueError('Official acceptance population/route differs')
    stamp = lambda value: datetime.fromisoformat(value.replace('Z', '+00:00'))
    if (attempt['candidate']['sha256'] != digest or
            stamp(accepted['official_stat']['lastModified']) < stamp(attempt['started_at'])):
        raise ValueError('Official acceptance predates the exact final PUT attempt')
    checked(prepared['candidate'])
    return accepted, delivered


def verify_generated_commit(marker, head):
    """Recognize only our exact one-parent metadata commit, never arbitrary work."""
    if marker['paths'] != GENERATED:
        raise ValueError('Pending activation path allowlist changed')
    if git('rev-parse', head + '^') != marker['base_head']:
        raise ValueError('Pending activation commit parent differs')
    if git('log', '-1', '--format=%s', head) != COMMIT_MESSAGE:
        raise ValueError('Pending activation commit message differs')
    changes = git('diff-tree', '--no-commit-id', '--name-only', '-r', head).splitlines()
    if not changes or not set(changes) <= set(GENERATED):
        raise ValueError('Pending activation commit contains unrelated files')
    for item in marker['files']:
        path = checked(item)
        rel = str(path.relative_to(ROOT))
        if git('rev-parse', head + ':' + rel) != git('hash-object', str(path)):
            raise ValueError('Pending activation committed bytes differ')


def resume_generated_push(out):
    marker_path = out / 'release/activation_commit.json'
    if not marker_path.exists():
        return None
    marker = read(marker_path)
    if marker.get('status') == 'pushed':
        for item in marker['files']:
            checked(item)
        return marker
    head = git('rev-parse', 'HEAD')
    if head == marker['base_head'] and marker.get('status') == 'prepared':
        # No completed commit yet; normal activation can finish the transaction.
        for item in marker['files']:
            checked(item)
        return marker
    if marker.get('commit') and marker['commit'] != head:
        raise ValueError('Local HEAD changed after pending activation commit')
    verify_generated_commit(marker, head)
    remote = api(f'repos/{REPO}/git/ref/heads/main')['object']['sha']
    if remote not in [marker['base_head'], head]:
        raise ValueError('Remote main changed; never force-push activation metadata')
    marker.update(status='committed_pending_push', commit=head)
    save(marker_path, marker)
    if remote != head:
        git('push', 'origin', 'main')
    if api(f'repos/{REPO}/git/ref/heads/main')['object']['sha'] != head:
        raise ValueError('Pending activation push verification failed')
    marker.update(status='pushed')
    save(marker_path, marker)
    return marker


def commit_generated(paths, out):
    """Recoverably commit only the five known files; never a user's staging."""
    relative = [str(p.relative_to(ROOT)) for p in paths]
    if relative != GENERATED:
        raise ValueError('Generated release path allowlist differs')
    marker_path = out / 'release/activation_commit.json'
    existing = read(marker_path) if marker_path.exists() else None
    staged = git('diff', '--cached', '--name-only').splitlines()
    if staged and not (existing and existing.get('status') == 'prepared' and set(staged) <= set(GENERATED)):
        raise ValueError('Unrelated staged changes exist; preserve them before activation commit')
    marker = dict(status='prepared', base_head=git('rev-parse', 'HEAD'), paths=relative,
                  files=[artifact(p) for p in paths])
    save(marker_path, marker)
    git('add', '--', *relative)
    staged = git('diff', '--cached', '--name-only').splitlines()
    if not set(staged) <= set(relative):
        raise ValueError('Unexpected staged path before release commit')
    if not staged:
        raise ValueError('No generated change and no completed activation receipt')
    git('commit', '-m', COMMIT_MESSAGE, '--', *relative)
    marker.update(status='committed_pending_push', commit=git('rev-parse', 'HEAD'))
    save(marker_path, marker)
    resume_generated_push(out)


def finish_activation(out, prepared):
    result = dict(prepared, status='accepted_final_release_activated',
                  publication_policy=artifact(POLICY), main_commit=git('rev-parse', 'HEAD'),
                  acceptance=artifact(out / 'delivery/acceptance.json'), published_now=False)
    path = out / 'release/result.json'
    prior = read(path) if path.exists() else None
    if prior and prior.get('status') == 'accepted_final_release_activated':
        if prior['publication_policy'] != result['publication_policy'] or prior['acceptance'] != result['acceptance']:
            raise ValueError('Previously activated final publication identity changed')
        result = prior
    else:
        result['activated_at'] = now()
        save(path, result)
    return result


def publish_if_due(out):
    policy = read(POLICY)
    due = datetime.fromisoformat(policy['not_before_utc'].replace('Z', '+00:00'))
    if datetime.now(timezone.utc) >= due:
        # A final PUT may have been accepted after the timer already fired.
        # The original guarded publisher verifies and idempotently repairs it.
        command([str(ROOT / '.venv/bin/python'), str(ROOT / 'tools/release/publish_at_deadline.py'), 'publish'], timeout=240)
        save(out / 'release/deadline_publication_repair.json', dict(status='guarded_publisher_succeeded', at=now()))


def activate(out):
    out = Path(out).absolute()
    prepared = read(out / 'release/prepared.json')
    accepted, delivered = acceptance(out, prepared)
    pending = resume_generated_push(out)
    verify_repository(allow_after_deadline=True)
    if pending and pending.get('status') == 'pushed':
        policy = read(POLICY)
        if policy['release_id'] != prepared['release_id'] or policy['selected_final_sha256'] != prepared['candidate']['sha256']:
            raise ValueError('Completed activation policy differs from accepted final')
        result = finish_activation(out, prepared)
        publish_if_due(out)
        print(json.dumps(dict(status=result['status'], release_tag=result['release_tag'])))
        return result
    # Same lock as the deadline publisher: no policy switch can race publication.
    lockpath = ROOT / 'release/artifacts/publication.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        policy = read(POLICY)
        old_policy = out / 'release/previous_publication_schedule.json'
        if not old_policy.exists():
            save(old_policy, policy)
        release = api(f'repos/{REPO}/releases/{prepared["release_id"]}')
        if not release['draft'] or release['tag_name'] != prepared['release_tag'] or release['target_commitish'] != prepared['source_commit']:
            raise ValueError('Prepared draft identity changed')
        assets = {a['name']: a for a in release['assets']}
        for item in prepared['assets']:
            a = assets.get(item['name'], {})
            if a.get('id') != item['github_asset_id'] or a.get('digest') != 'sha256:' + item['sha256'] or a.get('size') != item['bytes']:
                raise ValueError('Prepared remote asset changed')
        if len(assets) != len(prepared['assets']):
            raise ValueError('Prepared asset set changed')
        notes = release_notes(prepared['candidate']['sha256'], prepared['bundle_manifest']['sha256'],
                              prepared['source_commit'], accepted=True)
        updated = api(f'repos/{REPO}/releases/{prepared["release_id"]}', dict(body=notes), method='PATCH')
        if not updated['draft']:
            raise ValueError('Draft unexpectedly published')
        policy.update(release_id=prepared['release_id'], release_tag=prepared['release_tag'],
                      release_source_commit=prepared['source_commit'], assets=prepared['assets'],
                      selected_final_sha256=prepared['candidate']['sha256'],
                      selected_final_acceptance=artifact(out / 'delivery/acceptance.json'))
        save(POLICY, policy)
        manifest = ROOT / 'release/artifact_manifest.json'
        save(manifest, dict(schema='prc_release_artifact_v2', tag=prepared['release_tag'],
             accepted_final_sha256=prepared['candidate']['sha256'], rows=670790,
             bundle_manifest_sha256=prepared['bundle_manifest']['sha256'],
             source_commit=prepared['source_commit'], assets=prepared['assets'],
             draft_url=updated['html_url'], scope=CAVEAT,
             distribution='Private draft; scheduled for authorized deadline publication'))
        notes_path = ROOT / 'release/RELEASE_NOTES.md'
        notes_path.write_text(notes)
        selected = ROOT / 'release/SELECTED_FINAL.md'
        selected.write_text(notes + '\n## Replay\n\nDownload and extract the release archive, then run:\n\n'
            '```bash\npython -m pip install -r requirements-replay.txt\n'
            'python replay.py reproduce --bundle . --output unique-rose_final.parquet\n```\n\n'
            f'Acceptance checked at {accepted["checked_at"]}. Historical v83 remains archived.\n')
        delivery_path = ROOT / 'release/delivery.json'
        save(delivery_path, dict(status='accepted_v80_refit_private_draft_scheduled',
             source_commit=prepared['source_commit'], accepted_final_sha256=prepared['candidate']['sha256'],
             release_id=prepared['release_id'], release_tag=prepared['release_tag'],
             draft_url=updated['html_url'], assets=prepared['assets'], checked_at=now(),
             final_acceptance=accepted, remote_delivery=delivered))
        commit_generated([POLICY, manifest, notes_path, selected, delivery_path], out)
        result = finish_activation(out, prepared)
    publish_if_due(out)
    print(json.dumps(dict(status=result['status'], release_tag=result['release_tag'])))
    return result


def plan():
    return dict(stages=['prepare', 'activate'], no_mutations=True,
                prepare='audit, standalone exact replay, private draft/asset upload; old policy retained',
                activate='only after exact new-final acceptance: switch deadline policy and commit generated release docs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'prepare', 'activate'])
    parser.add_argument('--out', type=Path, default=DEFAULT)
    args = parser.parse_args(argv)
    if args.command == 'plan':
        print(json.dumps(plan(), indent=2))
    else:
        globals()[args.command](args.out)


if __name__ == '__main__':
    main()
