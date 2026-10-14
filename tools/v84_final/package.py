#!/usr/bin/env python3
"""Export/verify v84 assembly replay and create its private GitHub draft.

No competition upload, publication, policy activation, commit, or push occurs.
The previous v83 draft and selected publication policy are never modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'VALIDATION/runs/v3_20261014_v84_final'
REPO='Goxmen/PRC'
REPO_ID=1357299087
FORMAT='prc-v84-final-replay-v1'
CAVEAT=('v84 retains the original v80 ensemble recipe, using recovered original tree models, '
        'a reconstruction from the recovered v3 source, retained later models and fresh neural '
        'fits where original weights were not saved. Historical v80 predictions are not reproduced '
        'exactly. Old-input agreement is reported separately and is not validation accuracy or a '
        'leaderboard score. This package reproduces retained-component assembly exactly; a full '
        'clean-clone raw-data training rebuild has not been demonstrated.')


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


legacy=module(ROOT/'tools/v80_refit/package.py','v84_release_helpers')
replay=module(Path(__file__).with_name('replay.py'),'v84_standalone_replay')
sha,read,save,artifact,checked=legacy.sha,legacy.read,legacy.save,legacy.artifact,legacy.checked


def now():return datetime.now(timezone.utc).isoformat()


def validate(out):
    audit=read(out/'final/audit.json')
    if audit.get('status')!='pass' or audit.get('rows')!=670790:
        raise ValueError('A passing complete v84 final audit is required')
    candidate=checked(audit['candidate'])
    identity_path=checked(audit['identity'])
    if identity_path!=out/'assembly_identity.json':
        raise ValueError('Unexpected assembly identity')
    identity=read(identity_path)
    if len(identity['versions'])!=51 or len(identity['weights'])!=51:
        raise ValueError('Expected fixed 51-component recipe')
    return audit,candidate,identity


def verify_source(commit,network=False):
    if legacy.git('branch','--show-current')!='main':raise ValueError('Source must be on main')
    if legacy.git('rev-parse','HEAD')!=commit:raise ValueError('Reviewed source commit differs from HEAD')
    for path in sorted((ROOT/'tools/v84_final').glob('*.py')):
        rel=str(path.relative_to(ROOT))
        if legacy.git('ls-files','--error-unmatch',rel)!=rel or legacy.git('diff','HEAD','--',rel):
            raise ValueError(f'Uncommitted v84 release source: {rel}')
    if network:
        repo=legacy.api(f'repos/{REPO}')
        if repo['id']!=REPO_ID or not repo['private'] or not repo.get('permissions',{}).get('admin'):
            raise ValueError('Expected private owned repository; never publish from packaging')
        if legacy.api(f'repos/{REPO}/git/ref/heads/main')['object']['sha']!=commit:
            raise ValueError('Reviewed source is not on remote main')


def copy_verified(path,destination,expected=None):
    if expected is not None and sha(path)!=expected:raise ValueError(f'Changed source: {path}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,destination)


def sources_and_provenance(out,bundle,identity):
    """Export inspectable source/receipts, retaining model hashes, not raw data."""
    exported=set()
    # Include the full tracked Python source so historical formula imports remain
    # available for review. Large external data and environment/credential files
    # cannot enter through this explicit suffix/path allowlist.
    tracked=legacy.git('ls-files','--','*.py').splitlines()
    for rel in tracked:
        p=ROOT/rel
        if p.is_file():
            copy_verified(p,bundle/'source'/rel);exported.add(rel)
    for rec in identity['sources']:
        p=legacy.resolve(rec)
        if not p.is_relative_to(ROOT):raise ValueError('Nonrepository assembly source')
        rel=str(p.relative_to(ROOT))
        if not computational_provenance(p):continue
        if p.suffix in {'.py','.json','.md'}:
            destination=bundle/('source' if p.suffix=='.py' else 'provenance')/rel
            copy_verified(p,destination,rec['sha256']);exported.add(rel)
    # Only computational receipts belong in reviewer provenance. Owner policies,
    # canceled research and later selection conversations are deliberately absent.
    for rel in ['assembly_identity.json','leaf_catalog.json','dependency_inventory.json',
                'independent_audit.json','assembly_complete.json','v3/identity.json',
                'v3/old.json','v3/final.json','v3/result.json','final/audit.json','final/result.json']:
        p=out/rel
        if not p.is_file():continue
        copy_verified(p,bundle/'provenance'/p.relative_to(ROOT))
    return sorted(exported)


def computational_provenance(path):
    name=Path(path).name
    return not (name.startswith(('owner_','current_owner_','research_status','current_disposition'))
                or name=='independent_artifact_disposition.json' or 'hpo_weights' in Path(path).parts)


def exact_replay(out,bundle,digest):
    target=out/'release/replayed_v84_final.parquet';marker=out/'release/replay_audit.json'
    manifest_hash=sha(bundle/'manifest.json')
    if marker.exists():
        rec=read(marker)
        if rec['manifest_sha256']!=manifest_hash or rec['candidate_sha256']!=digest or rec['status']!='pass':
            raise ValueError('Existing replay receipt differs')
        checked(rec['output']);return rec
    if target.exists():
        if sha(target)!=digest:raise ValueError('Preserve different existing replay output')
    else:replay.reproduce(bundle,target,manifest_hash)
    if sha(target)!=digest:raise ValueError('Exact standalone replay failed')
    result=dict(status='pass',manifest_sha256=manifest_hash,candidate_sha256=digest,
                output=artifact(target),checked_at=now(),rows=670790,no_training=True,no_upload=True)
    save(marker,result);return result


def export(out,commit):
    verify_source(commit)
    audit,candidate,identity=validate(out);digest=sha(candidate)
    bundle=ROOT/'release/bundles'/f'v84-final-{digest[:16]}'
    if (bundle/'manifest.json').exists() and not (bundle/'INCOMPLETE.json').exists():
        manifest=replay.verify(bundle)
        if manifest['candidate_sha256']!=digest or manifest['source_commit']!=commit:
            raise ValueError('Existing bundle belongs to another result/source')
        exact_replay(out,bundle,digest);return bundle,manifest
    if bundle.exists():raise ValueError('Preserve interrupted bundle export')
    bundle.mkdir(parents=True);save(bundle/'INCOMPLETE.json',dict(started_at=now()))
    copy_verified(out/'final/basis51.parquet',bundle/'basis51.parquet')
    pd.read_parquet(out/'final/stages.parquet',columns=['mvt_id','v78','tabm_v78','robust_v78']).to_parquet(bundle/'corrections.parquet',index=False)
    copy_verified(ROOT/'data/final_submitting.parquet',bundle/'template.parquet')
    save(bundle/'recipe.json',dict(versions=identity['versions'],weights=identity['weights'],threads=1,
        arithmetic='v79=rint(max(P@w,0)); v84=rint(max(v79+(tabm_v78-v78)+(robust_v78-v78),0))'))
    exported=sources_and_provenance(out,bundle,identity)
    for source,name in [(ROOT/'LICENSE','LICENSE'),(ROOT/'docs/DATA_SOURCES.md','DATA_SOURCES.md'),
                        (ROOT/'docs/THIRD_PARTY.md','THIRD_PARTY.md'),
                        (ROOT/'release/requirements-replay.txt','requirements-replay.txt'),
                        (Path(__file__).with_name('replay.py'),'replay.py')]:
        copy_verified(source,bundle/name)
    (bundle/'README.md').write_text('# v84: exact final assembly replay\n\n'+CAVEAT+'\n\n'
        'Install the pinned CPU replay dependencies and run offline:\n\n```bash\n'
        'python -m pip install -r requirements-replay.txt\n'
        'python replay.py verify --bundle .\n'
        'python replay.py reproduce --bundle . --output unique-rose_final.parquet\n```\n\n'
        'The final submission is not included. The replay computes all670,790 predictions '
        'from the51 integer basis vectors, fixed weights and three correction inputs. '
        'The manifest binds every exported byte and the expected regenerated Parquet hash. '
        'Source and model/input provenance receipts are included. Trained weight/data caches '
        'remain in the research archive; they are not needed for this assembly replay.\n')
    files=[dict(path=str(p.relative_to(bundle)),sha256=sha(p),bytes=p.stat().st_size)
           for p in sorted(bundle.rglob('*')) if p.is_file() and p.name!='INCOMPLETE.json']
    manifest=dict(format=FORMAT,rows=670790,candidate_sha256=digest,source_commit=commit,
        versions=identity['versions'],files=files,caveat=CAVEAT,
        environment=dict(pyarrow=pa.__version__,pandas=pd.__version__),
        source_files=len(exported),expected_scope='January/February/June/July2026 final rows')
    save(bundle/'manifest.json',manifest);(bundle/'INCOMPLETE.json').unlink()
    replay.verify(bundle);exact_replay(out,bundle,digest)
    save(out/'release/export.json',dict(status='verified',bundle=str(bundle.relative_to(ROOT)),
         manifest=artifact(bundle/'manifest.json'),candidate=artifact(candidate),source_commit=commit))
    return bundle,manifest


def notes(candidate_hash,manifest_hash,commit):
    return ('# v84 final: recovered trees and fixed v80 recipe\n\n'
        f'670,790 final predictions; SHA-256 `{candidate_hash}`.\n\n'+CAVEAT+'\n\n'
        f'Source commit: `{commit}`. Bundle manifest: `{manifest_hash}`.\n\n'
        'The archive contains a standalone offline replay, source and provenance. '
        'It excludes the final output and raw external datasets; provider terms remain applicable. '
        'Publication is authorized only at14October2026,10:00UTC /12:00Amsterdam. '
        'Draft preparation does not change the competition final or active publication selection.\n')


def draft(out,commit):
    verify_source(commit,network=True)
    bundle,manifest=export(out,commit);assets=legacy.archive(bundle)
    _,candidate,_=validate(out);digest=sha(candidate);tag=f'prc-final-v84-{digest[:12]}'
    matches=[r for r in legacy.api(f'repos/{REPO}/releases?per_page=100') if r['tag_name']==tag]
    if len(matches)>1:raise ValueError('Ambiguous v84 release tag')
    release=matches[0] if matches else legacy.api(f'repos/{REPO}/releases',dict(
        tag_name=tag,target_commitish=commit,name='unique-rose v84 final',
        body=notes(digest,sha(bundle/'manifest.json'),commit),draft=True,prerelease=False))
    if not release['draft'] or release['target_commitish']!=commit:
        raise ValueError('Release is published or binds another source')
    save(out/'release/draft.json',dict(id=release['id'],tag=tag,source_commit=commit,candidate=artifact(candidate)))
    for p in assets:
        current=legacy.api(f'repos/{REPO}/releases/{release["id"]}')
        old=[a for a in current['assets'] if a['name']==p.name]
        if not old:legacy.command([str(legacy.GH),'release','upload',tag,str(p),'--repo',REPO],timeout=900)
        elif len(old)!=1 or old[0].get('digest')!='sha256:'+sha(p):
            raise ValueError('Existing draft asset differs; no silent overwrite')
    release=legacy.api(f'repos/{REPO}/releases/{release["id"]}')
    actual={a['name']:a for a in release['assets']}
    if set(actual)!={p.name for p in assets}:raise ValueError('Unexpected draft asset set')
    records=[]
    for p in assets:
        a=actual[p.name]
        if a['size']!=p.stat().st_size or a['state']!='uploaded' or a.get('digest')!='sha256:'+sha(p):
            raise ValueError('Remote asset verification failed')
        records.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p),github_asset_id=a['id'],github_digest_verified=True))
    result=dict(status='private_draft_ready',candidate=artifact(candidate),audit=artifact(out/'final/audit.json'),
        bundle_manifest=artifact(bundle/'manifest.json'),source_commit=commit,release_id=release['id'],
        release_tag=tag,release_url=release['html_url'],assets=records,prepared_at=now(),caveat=CAVEAT,
        existing_publication_policy_unchanged=True,no_competition_upload=True)
    save(out/'release/prepared.json',result);print(json.dumps(result));return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['export','draft'])
    p.add_argument('--out',type=Path,default=DEFAULT);p.add_argument('--source-commit',required=True)
    a=p.parse_args();out=a.out.absolute()
    if a.stage=='export':
        bundle,manifest=export(out,a.source_commit)
        print(json.dumps(dict(status='verified',bundle=str(bundle),candidate_sha256=manifest['candidate_sha256'])))
    else:draft(out,a.source_commit)


if __name__=='__main__':main()
