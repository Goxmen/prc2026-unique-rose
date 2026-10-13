#!/usr/bin/env python3
"""Offline exact final assembly from v80-refit component predictions.

The package holds the 51-member integer basis, the fixed historical weights,
and three independently computed correction inputs. It does not contain the
final submission. This replays assembly, not feature generation/model fitting.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe(bundle, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe bundle path')
    root = Path(bundle).resolve()
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Bundle path escapes root')
    return result


def verify(bundle, manifest_sha256=None):
    bundle = Path(bundle)
    if (bundle / 'INCOMPLETE.json').exists():
        raise ValueError('Incomplete bundle')
    if manifest_sha256 and sha(bundle / 'manifest.json') != manifest_sha256:
        raise ValueError('Bundle manifest changed')
    manifest = read(bundle / 'manifest.json')
    if manifest['format'] != 'prc-v80-refit-replay-v1' or manifest['rows'] != 670790:
        raise ValueError('Unexpected replay format or population')
    names = []
    for item in manifest['files']:
        name = item['path']
        path = safe(bundle, name)
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError(f'Changed bundle artifact: {name}')
        names.append(name)
    if len(names) != len(set(names)) or not {'basis51.parquet', 'corrections.parquet', 'template.parquet', 'recipe.json'} <= set(names):
        raise ValueError('Incomplete/duplicate replay manifest')
    return manifest


def align(frame, ids):
    if pd.Index(ids).has_duplicates or pd.Index(ids).hasnans or frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any() or len(frame) != len(ids) or set(frame.mvt_id) != set(ids):
        raise ValueError('Component/template ID mismatch')
    return frame.set_index('mvt_id').loc[ids].reset_index()


def integers(x):
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite prediction')
    out = np.rint(np.maximum(x, 0))
    if np.any(out > np.iinfo(np.int32).max):
        raise ValueError('Prediction exceeds int32')
    return out.astype(np.int32)


def calculate(basis, corrections, versions, weights):
    matrix = basis[[f'v{v}' for v in versions]].to_numpy(float)
    weights = np.asarray(weights, float)
    if matrix.shape[1] != 51 or len(weights) != 51 or not np.isfinite(weights).all():
        raise ValueError('Fixed 51-component recipe changed')
    if not np.isfinite(matrix).all() or not np.array_equal(matrix, integers(matrix)):
        raise ValueError('Basis must contain nonnegative integer predictions')
    for name in ['v78', 'tabm_v78', 'robust_v78']:
        values = corrections[name].to_numpy(float)
        if not np.array_equal(values, integers(values)):
            raise ValueError('Correction input predictions must be nonnegative integers')
    with threadpool_limits(1):
        v79 = integers(matrix @ weights)
        return integers(v79.astype(float)
                        + (corrections.tabm_v78.to_numpy(float) - corrections.v78.to_numpy(float))
                        + (corrections.robust_v78.to_numpy(float) - corrections.v78.to_numpy(float)))


def reproduce(bundle, output, manifest_sha256=None):
    bundle, output = Path(bundle), Path(output)
    manifest = verify(bundle, manifest_sha256)
    if output.exists():
        raise ValueError('Output already exists; preserve it')
    recipe = read(bundle / 'recipe.json')
    template = pq.read_table(bundle / 'template.parquet')
    ids = template['MVT_ID_mvt'].to_numpy()
    if len(ids) != manifest['rows'] or pd.Index(ids).has_duplicates:
        raise ValueError('Invalid template')
    basis = align(pd.read_parquet(bundle / 'basis51.parquet'), ids)
    corrections = align(pd.read_parquet(bundle / 'corrections.parquet'), ids)
    values = calculate(basis, corrections, recipe['versions'], recipe['weights'])
    field = template.schema.field('TAXITIME_SEC_mvt')
    table = template.set_column(template.schema.get_field_index(field.name), field,
                                pa.array(values, type=field.type))
    if not table.schema.equals(template.schema) or not table['MVT_ID_mvt'].equals(template['MVT_ID_mvt']):
        raise ValueError('Template schema/ID change')
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output)
    actual = sha(output)
    if actual != manifest['candidate_sha256']:
        raise ValueError('Reproduced bytes differ; preserve output and inspect pinned runtime/BLAS')
    return dict(status='pass', rows=len(ids), sha256=actual, scope='retained-component assembly replay')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['verify', 'reproduce'])
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--manifest-sha256')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(argv)
    if args.command == 'verify':
        manifest = verify(args.bundle, args.manifest_sha256)
        print(json.dumps(dict(status='pass', files=len(manifest['files']), rows=manifest['rows'])))
    else:
        if args.output is None:
            ap.error('--output is required for reproduce')
        print(json.dumps(reproduce(args.bundle, args.output, args.manifest_sha256)))


if __name__ == '__main__':
    main()
