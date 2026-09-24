"""Compact existing seasonal OOF losses for transfer from the desktop.

Run at the repository root. No fits or original files are changed. Transfer the
output directory once, then use analyze_train_ranking_cost.py entirely locally.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    root = Path.cwd()
    run = root / 'VALIDATION/runs/seasonal_jan_jul'
    out = root / 'tmp/train_ranking_step3_inputs'
    out.mkdir(parents=True, exist_ok=True)
    result = json.loads((run / 'result.json').read_text())
    progress = json.loads((run / 'progress.json').read_text())
    manifest = dict(result_sha256=sha(run / 'result.json'),
                    progress_sha256=sha(run / 'progress.json'),
                    recipe='Q0-4h / v15', files={})
    identity = ['mvt_id', 'mvt_time', 'airport', 'y']
    for strategy in ['11m', '10m']:
        for month in ['2025-01', '2025-07']:
            meta, sq, inputs = None, [], {}
            for seed in [42, 142, 242]:
                found = [p for p in progress['predictions'].values()
                         if (p['strategy'], p['parent'], p['month'], p['seed'])
                         == (strategy, 'baseline', month, seed)]
                assert len(found) == 1
                item = found[0]
                path = run / item['path']
                inputs[item['path']] = sha(path)
                assert inputs[item['path']] == item['artifacts'][item['path']]
                f = pd.read_parquet(path, columns=identity + ['seed', 'month', 'Q0-4h'])
                f = f.sort_values('mvt_id').reset_index(drop=True)
                assert f.mvt_id.is_unique and f.mvt_id.notna().all()
                assert len(f) == item['rows'] and f.seed.eq(seed).all() and f.month.eq(month).all()
                assert np.isfinite(f[['y', 'Q0-4h']].to_numpy()).all()
                if meta is None:
                    meta = f[identity].copy()
                else:
                    pd.testing.assert_frame_equal(meta, f[identity])
                sq.append((f.y.to_numpy() - f['Q0-4h'].to_numpy()) ** 2)
            meta['mean_sq'] = np.mean(sq, axis=0)
            expected = result['report']['strategies'][strategy]['modes']['standard']['levels']['Q0-4h']['scores'][month]
            score = float(np.sqrt(meta.mean_sq.mean()))
            np.testing.assert_allclose(score, expected, rtol=0, atol=1e-8)
            path = out / f'v15_{strategy}_{month}.parquet'
            meta.to_parquet(path, index=False, compression='zstd', compression_level=9)
            manifest['files'][path.name] = dict(sha256=sha(path), rows=len(meta),
                rmse=score, strategy=strategy, month=month, seeds=[42,142,242],
                prediction_sha256=inputs, job=next(j for j in result['signature']['jobs']
                    if j['strategy'] == strategy and month in j['val']))
            print(path.name, path.stat().st_size, 'bytes', flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
