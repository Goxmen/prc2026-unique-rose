"""Independently reload all fifteen production models on fixed ranking samples.

No training cache or departure labels are read. Preparation samples each original
ranking view independently and joins the six saved inputs by movement ID. The
audit runs after production completes, outside another heavy desktop job.
"""
import argparse
import gc
import os
from pathlib import Path
import sys

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import callsign_submit_fast as worker

parent = worker.parent
OUT = parent.OUT / 'model_reload_audit'
SAMPLE_ROWS = 512
ATOL_SEC = 0.01  # Different inference batch shapes/devices can change float32 GEMM rounding.


def prepare():
    prepared = parent.validate_prepared()
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for view in parent.VIEWS:
        data = parent.ranking_data(view)
        ids = np.asarray(data['rank_ids'])
        rng = np.random.default_rng(20261010)
        positions = np.sort(np.r_[0, rng.choice(np.arange(1, len(ids)-1), SAMPLE_ROWS-2, replace=False), len(ids)-1])
        matrix = data['Vn'].iloc[positions].copy()
        if list(matrix) != prepared['original_columns']:
            raise ValueError('Original ranking feature order changed')
        record = prepared['views'][view]['features']['artifact']
        table_path = ROOT / record['path']
        if parent.sha256(table_path) != record['sha256']:
            raise ValueError('New ranking inputs changed')
        features = pd.read_parquet(table_path).set_index('mvt_id')
        if not features.index.is_unique:
            raise ValueError('Duplicate feature IDs')
        selected = features.loc[ids[positions]]
        for name in parent.features.FEATURE_NAMES:
            matrix[name] = selected[name].to_numpy(np.float32)
        path = OUT / f'{view}.pkl'
        pd.to_pickle(dict(view=view, X=matrix, mvt_id=ids[positions], positions=positions), path)
        records[view] = dict(sample=parent.artifact(path), original_inputs=prepared['views'][view]['inputs'],
                             features=record, rows=len(matrix), original_rows=len(ids))
        del data, matrix, features, selected
        gc.collect()
    parent.write_json(OUT / 'samples.json', dict(source=parent.artifact(Path(__file__)),
        prepared=parent.artifact(parent.OUT/'prepared.json'), seed=20261010, views=records))


def audit(device):
    manifest = parent.read(OUT/'samples.json')
    if manifest['source'] != parent.artifact(Path(__file__)):
        raise ValueError('Sample auditor source changed')
    if manifest['prepared'] != parent.artifact(parent.OUT/'prepared.json'):
        raise ValueError('Prepared production contract changed')
    # The separate desktop worker audit checks all original sources. This audit
    # checks every parent-compatible model/prediction/preprocessing record again.
    worker.guard()
    records = {(g, k): worker.completed_member(g, k) for g in parent.GROUPS for k in parent.NETS}
    if any(record is None for record in records.values()):
        raise ValueError('All fifteen models must complete before the reload audit')
    prep = pd.read_pickle(parent.OUT/'preprocessing/full_year.pkl')
    parent.cv.shared.recipe()
    inputs, ids, samples = {}, {}, {}
    for view, record in manifest['views'].items():
        if record['sample'] != parent.artifact(ROOT/record['sample']['path']):
            raise ValueError('Ranking sample artifact changed')
        item = pd.read_pickle(ROOT/record['sample']['path'])
        if list(item['X']) != prep['columns'] or item['view'] != view:
            raise ValueError('Sample column/view alignment changed')
        cat, num = prep['encoder'].transform(item['X'])
        inputs[view] = (torch.from_numpy(np.ascontiguousarray(cat)).to(device),
            torch.from_numpy(np.ascontiguousarray(num[:, prep['ple']])).to(device),
            torch.from_numpy(np.ascontiguousarray(num[:, prep['plain']])).to(device))
        ids[view], samples[view] = item['mvt_id'], record['sample']
    checks = []
    for (group, k), record in records.items():
        saved = torch.load(ROOT/record['model']['path'], map_location=device, weights_only=False)
        if saved['design'] != parent.DESIGN or saved['config'] != record['config']:
            raise ValueError('Reloaded model recipe changed')
        model = parent.cv.shared.nni.PleNet(prep['cards'], prep['bins'], len(prep['plain'])).to(device).eval()
        model.load_state_dict(saved['state_dict'], strict=True)
        for view in parent.VIEWS:
            predictions = pd.read_parquet(ROOT/record['predictions'][view]['path'], columns=['mvt_id', 'base'])
            if not predictions.mvt_id.is_unique:
                raise ValueError('Duplicate prediction IDs')
            expected = predictions.set_index('mvt_id').loc[ids[view], 'base'].to_numpy(np.float32)
            with torch.no_grad():
                standardized = model(*inputs[view]).mean(1).cpu().numpy().astype(float)
            actual = np.maximum(standardized*prep['sd']+prep['mu'], 0).astype(np.float32)
            if not np.isfinite(actual).all():
                raise ValueError('Nonfinite reloaded predictions')
            delta = actual.astype(float)-expected.astype(float)
            np.testing.assert_allclose(actual, expected, rtol=0, atol=ATOL_SEC)
            checks.append(dict(group=group, k=k, view=view, rows=len(actual),
                exact=int(np.count_nonzero(delta == 0)), max_abs_sec=float(np.max(np.abs(delta))),
                mean_abs_sec=float(np.mean(np.abs(delta))), model=record['model'], predictions=record['predictions'][view]))
        del saved, model
    parent.write_json(OUT/'result.json', dict(source=parent.artifact(Path(__file__)),
        samples=samples, preprocessing=parent.artifact(parent.OUT/'preprocessing/full_year.pkl'),
        checks=checks, total_predictions=sum(r['rows'] for r in checks), tolerance_abs_sec=ATOL_SEC,
        device=str(device), passed=True, no_training_cache_or_labels_read=True))
    parent.log(f'Reloaded fifteen models; {sum(r["rows"] for r in checks)} sampled predictions pass; '
               f'max difference {max(r["max_abs_sec"] for r in checks):.6f}s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'audit'])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        prepare() if args.stage == 'prepare' else audit(args.device)


if __name__ == '__main__':
    main()
