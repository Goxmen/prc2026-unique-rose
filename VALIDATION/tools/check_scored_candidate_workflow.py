"""Isolated full-shape synthetic integration check; never access a live API.

All data, manifests and outputs live under a temporary root. No competition
labels, official score, submission file or version are fabricated in the repo.
Only the final synthetic-check receipt is retained when explicitly requested.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'VALIDATION'))
from records import sha256, write_json


def check():
    with tempfile.TemporaryDirectory(prefix='scored_candidate_fixture_') as temporary:
        root = Path(temporary)
        geometry = root / 'VALIDATION/runs/v3_20261011_daily_models_geometry'
        geometry.mkdir(parents=True)
        (root / 'submissions').mkdir()
        (root / 'data').mkdir()
        def read(path): return json.loads(Path(path).read_text())
        def art(path):
            path = Path(path).resolve()
            # Production code paths are mirrored into the temporary fixture;
            # fixture source verification never reaches out to a real candidate.
            if not path.is_relative_to(root):
                destination = root / 'source' / path.name
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(path.read_bytes())
                path = destination
            return dict(path=str(path.relative_to(root)), sha256=sha256(path))
        def verify(record):
            path = root / record['path']
            assert sha256(path) == record['sha256']
            return path
        versions = list(range(1, 45)) + [64, 70, 72, 74, 75, 76, 77]
        columns = [f'v{version}' for version in versions]
        weights = np.zeros(51)
        for version, weight in [(75, .4), (76, .3), (77, .3)]: weights[versions.index(version)] = weight
        rng = np.random.default_rng(64829)
        n, ncv = 344841, 344419
        y = 5000 + rng.normal(0, 50, n)
        values = np.rint(np.maximum(y[:, None]+rng.normal(0, 1000, (n, 51)), 0))
        rank = pd.DataFrame(values, columns=columns)
        rank.insert(0, 'mvt_id', np.arange(1, n+1, dtype=np.int64))
        scores = np.round(np.sqrt(np.mean((values-y[:, None])**2, axis=0)), 4)
        raw = values @ weights
        serialized = np.rint(np.maximum(raw, 0))
        cv_y = 5000 + rng.normal(0, 50, ncv)
        held = pd.DataFrame(np.rint(np.maximum(cv_y[:, None]+rng.normal(0, 1000, (ncv, 51)), 0)), columns=columns)
        held.insert(0, 'mvt_id', np.arange(1, ncv+1, dtype=np.int64))
        held['y'] = cv_y
        held['month'] = np.where(np.arange(ncv) < 153706, '2025-01', '2025-07')
        held['day'] = held.month + '-01'
        rank.to_parquet(geometry / 'ranking_basis.parquet', index=False)
        held.to_parquet(geometry / 'cv_basis.parquet', index=False)
        cv = held[['mvt_id', 'month', 'day', 'y']].copy()
        cv['pred'] = np.rint(np.maximum(held[columns].to_numpy() @ weights, 0))
        cv['control'] = held.v64
        cv.to_parquet(geometry / 'diagnostic_cv.parquet', index=False)
        pd.DataFrame(dict(mvt_id=rank.mvt_id, raw=raw, submitted=serialized)).to_parquet(geometry / 'diagnostic_ranking.parquet', index=False)
        truth = root / 'VALIDATION/runs/v3_20261011_daily_delay/rows2025.parquet'
        truth.parent.mkdir(parents=True)
        held[['mvt_id', 'month', 'day', 'y']].to_parquet(truth, index=False)
        template = pa.table(dict(MVT_ID_mvt=rank.mvt_id.to_numpy(), TAXITIME_SEC_mvt=pa.array(np.zeros(n, dtype=np.int32))))
        pq.write_table(template, root / 'data/submitting.parquet')
        public, additions, official = [], {}, {}
        for version, score in zip(versions, scores):
            file = root / f'submissions/unique-rose_v{version}.parquet'
            pq.write_table(pa.table(dict(MVT_ID_mvt=rank.mvt_id.to_numpy(), TAXITIME_SEC_mvt=pa.array(rank[f'v{version}'].to_numpy().astype(np.int32)))), file)
            public.append(dict(teamName='synthetic-fixture', filename=file.name, usedPairs=n, score=float(score)))
            if version in [70, 72, 74, 75, 76, 77]:
                outcome = dict(status='Succeeded', used_pairs=n, score=float(score))
                manifest = dict(state='scored', file=str(file.relative_to(root)), sha256=sha256(file), remote_sha256=sha256(file), score=float(score), official_result=outcome)
                live_path = geometry / f'live_v{version}.json'
                snapshot = geometry / f'snapshot_v{version}.json'
                write_json(live_path, manifest); write_json(snapshot, manifest)
                additions[str(version)] = dict(snapshot=art(snapshot), live_manifest=str(live_path.relative_to(root)))
                official[str(version)] = outcome
        design = dict(scope='synthetic full-shape fixture; no competition result')
        prepared = dict(versions=versions, design=design, additions=additions,
            known_scores={str(v): float(s) for v, s in zip(versions, scores) if v not in [70, 72, 74, 75, 76, 77]},
            sources={}, ranking=art(geometry / 'ranking_basis.parquet'), cv=art(geometry / 'cv_basis.parquet'),
            historical_cv_caveat='Synthetic fixture only')
        write_json(geometry / 'prepared.json', prepared)
        write_json(geometry / 'audit.json', dict(status='pass', prepared=art(geometry / 'prepared.json')))
        mse = float(weights @ (scores**2) - weights @ np.mean(values**2, axis=0) + np.mean(raw**2))
        radius = float(np.abs(weights) @ (scores*1e-4+2.5e-9))
        distance = float(np.sqrt(np.mean((serialized-raw)**2)))
        cv_score = float(np.mean([np.sqrt(np.mean((g.pred-g.y)**2)) for _, g in cv.groupby('month')]))
        result = dict(prepared=art(geometry / 'prepared.json'), design=design, independent_mse_identity=True,
            official=official, solution=dict(weights={str(v): float(w) for v, w in zip(versions, weights)},
            constraints=dict(lower=-3., upper=4., ridge_mse=1.), projected_rmse=float(np.sqrt(mse))),
            conservative_rmse_interval=[float(np.sqrt(mse-radius)-distance), float(np.sqrt(mse+radius)+distance)],
            cv=cv_score, diagnostic_ranking=art(geometry / 'diagnostic_ranking.parquet'), diagnostic_cv=art(geometry / 'diagnostic_cv.parquet'))
        write_json(geometry / 'result.json', result)
        # Load exact production files from temporary copies, so their __file__
        # records also stay inside the isolated fixture root.
        source = root / 'VALIDATION'; tools = source / 'tools'; tools.mkdir(exist_ok=True)
        build_path = source / 'scored_basis_candidate.py'
        audit_path = tools / 'audit_scored_basis_candidate.py'
        build_path.write_bytes((REPO / 'VALIDATION/scored_basis_candidate.py').read_bytes())
        audit_path.write_bytes((REPO / 'VALIDATION/tools/audit_scored_basis_candidate.py').read_bytes())
        fake_path = source / 'daily_models_geometry.py'; fake_path.write_text('# synthetic provider only\n')
        provider = types.ModuleType('daily_models_geometry')
        provider.__dict__.update(ROOT=root, OUT=geometry, read=read, art=art, verify=verify,
            ADDITIONS=[70, 72, 74, 75, 76, 77], DESIGN=design, __file__=str(fake_path),
            parent=types.SimpleNamespace(transport=types.SimpleNamespace(live=types.SimpleNamespace(TEAM='synthetic-fixture', fetch_leaderboard=lambda: public))))
        sys.modules['daily_models_geometry'] = provider
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            return module
        builder = load('fixture_builder', build_path)
        auditor = load('fixture_auditor', audit_path)
        destination = builder.OUT / 'submission_v78'
        output = root / 'submissions/unique-rose_v78.parquet'
        live75 = geometry / 'live_v75.json'; original = read(live75)
        write_json(live75, dict(original, state='prepared'))
        try: builder.build(78)
        except AssertionError: pass
        else: raise AssertionError('Missing official score accepted')
        assert not destination.exists() and not output.exists()
        write_json(live75, original)
        builder.build(78)
        original_output = output.read_bytes()
        output.write_bytes(original_output+b'corruption')
        try: auditor.audit(78)
        except Exception: pass
        else: raise AssertionError('Corrupt output accepted')
        assert not (destination / 'audit.json').exists()
        output.write_bytes(original_output)
        auditor.audit(78)
        final = read(destination / 'manifest.json')
        assert final['audit']['independent_review']['status'] == 'pass'
        for path, digest in final['sources'].items(): assert sha256(root / path) == digest
        actual = float(np.sqrt(np.mean((pq.read_table(output)['TAXITIME_SEC_mvt'].to_numpy()-y)**2)))
        lo, hi = final['conservative_rmse_interval']
        assert lo <= actual <= hi
        try: builder.build(78)
        except AssertionError: pass
        else: raise AssertionError('Existing candidate overwritten')
        return dict(status='pass', synthetic_only=True, ranking_rows=n, cv_rows=ncv, basis_vectors=51,
            exact_production_sources=True, full_build_and_separate_sql_audit=True,
            actual_synthetic_error_within_conservative_interval=True, missing_score_rejected_before_output=True,
            corrupt_output_rejected=True, existing_candidate_preserved=True,
            no_live_api_called=True, no_real_submission_created=True,
            sources={str(path.relative_to(REPO)): sha256(path) for path in
                [Path(__file__), REPO / 'VALIDATION/scored_basis_candidate.py', REPO / 'VALIDATION/tools/audit_scored_basis_candidate.py']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    with threadpool_limits(1): result = check()
    result['checked_at'] = datetime.now(timezone.utc).isoformat()
    if args.report: write_json(args.report, result)
    print(json.dumps(result, indent=2))
