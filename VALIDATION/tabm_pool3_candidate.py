"""Package the independently audited three-TabM/runway candidate; no upload."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'VALIDATION/runs'
COMPONENT = RUNS / 'v3_20261012_tabm_pool3_other_runway_combo_submit'
CV = RUNS / 'v3_20261012_tabm_pool3_other_runway_combo'
OUT = RUNS / 'v3_20261012_tabm_pool3_candidate/submission_v80'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def prepare():
    assert not OUT.exists(), 'Preserve an existing version package'
    identity, result, audit = [read(COMPONENT / name) for name in
        ['identity.json', 'result.json', 'independent_audit.json']]
    assert result['identity'] == artifact(COMPONENT / 'identity.json')
    assert audit['status'] == 'pass' and audit['result'] == artifact(COMPONENT / 'result.json')
    assert audit['rows'] == result['rows'] == 344841 and audit['cv_rows'] == 344419
    score = read(CV / 'result.json')
    assert result['cv_source'] == artifact(CV / 'result.json')
    assert score['paired']['control']['decision'] == 'improvement'
    assert score['paired']['one_tabm_runway']['decision'] == 'improvement'
    cv = score['cv']['three_tabm_runway']
    assert abs(cv - 289.7627782102906) < 1e-9
    sources = {a['path']:a['sha256'] for a in [*identity['sources'], *result['artifacts']]}
    for p in [Path(__file__), COMPONENT / 'identity.json', COMPONENT / 'result.json',
              COMPONENT / 'independent_audit.json', CV / 'result.json', CV / 'scored.parquet',
              ROOT / 'VALIDATION/audited_candidate_upload.py']:
        a = artifact(p); sources[a['path']] = a['sha256']
    for path, digest in sources.items():
        assert sha(ROOT / path) == digest, path
    candidate = pq.read_table(COMPONENT / 'candidate.parquet')
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    assert candidate.schema == template.schema and candidate.num_rows == template.num_rows == 344841
    assert candidate['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    pred = candidate['TAXITIME_SEC_mvt'].to_numpy()
    assert candidate['TAXITIME_SEC_mvt'].null_count == 0 and np.isfinite(pred).all() and (pred >= 0).all()
    # Canonical ID sorting catches duplicate vectors despite parquet encoding
    # or historical row-order differences. Never spend a slot on a replay.
    order = np.argsort(candidate['MVT_ID_mvt'].to_numpy())
    canonical_ids = candidate['MVT_ID_mvt'].to_numpy()[order]
    digest = hashlib.sha256(pred[order].astype('<f8').tobytes()).hexdigest()
    compared = []
    for path in sorted((ROOT / 'submissions').glob('unique-rose_v*.parquet')):
        if pq.ParquetFile(path).metadata.num_rows != len(pred):
            continue
        prior = pq.read_table(path)
        at = np.argsort(prior['MVT_ID_mvt'].to_numpy())
        if np.array_equal(prior['MVT_ID_mvt'].to_numpy()[at], canonical_ids):
            other = prior['TAXITIME_SEC_mvt'].to_numpy()[at].astype('<f8')
            assert hashlib.sha256(other.tobytes()).hexdigest() != digest, f'Duplicate of {path.name}'
            compared.append(path.name)
    output = ROOT / 'submissions/unique-rose_v80.parquet'
    assert not output.exists()
    OUT.mkdir(parents=True)
    shutil.copyfile(COMPONENT / 'candidate.parquet', output)
    assert sha(output) == sha(COMPONENT / 'candidate.parquet')
    value = dict(version=80, state='prepared', parent='unique-rose_v79.parquet',
        change='Selected equal three-TabM public neural pool plus robust same-date runway evidence',
        file=str(output.relative_to(ROOT)), sha256=sha(output), rows=len(pred), jan_jul_cv=cv,
        cv_months={m:v['three_tabm_runway'] for m,v in score['scores'].items()},
        cv_record=artifact(CV / 'result.json'), component=artifact(COMPONENT / 'result.json'),
        audit=artifact(COMPONENT / 'independent_audit.json'), sources=sources,
        authorization='User permits autonomous leaderboard submissions; active target205.00. Use one remaining12OctoberUTC slot after audited CV selection.',
        basis_scores={'unique-rose_v79.parquet':219.6130, 'unique-rose_v78.parquet':220.2933},
        prepared_at=datetime.now(timezone.utc).isoformat(),
        duplicate_check=dict(canonical_prediction_sha256=digest, compared=compared),
        no_official_gain_claimed=True)
    (OUT / 'manifest.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(version=80, state='prepared', rows=len(pred), cv=cv,
        sha256=value['sha256'], duplicates_compared=len(compared))))


if __name__ == '__main__':
    prepare()
