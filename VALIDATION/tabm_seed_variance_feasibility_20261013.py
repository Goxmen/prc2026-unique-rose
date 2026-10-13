"""Label-free allocation diagnostic for enlarging the retained three-seed pool.

This estimates random-seed variance in the ordinary pre-correction component.
It is not a bound or estimate for the nonlinear complete model's score, and
does not validate a five-seed candidate. No labels or new predictions are read.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'VALIDATION/reports/tabm_seed_variance_20261013'
PARENT = ROOT / 'VALIDATION/runs/v3_20261013_nn_public_tabm_track_consistency/identity.json'
DESIGN = dict(current_seeds=3, proposed_seeds=5, ordinary_coefficient=.71*.5,
              variance='unbiased per-row sample variance over retained seeds42/1042/2042',
              expectation='squared ordinary coefficient * mean seed variance * (1/3 - 1/5)',
              assumptions=['Seeds are exchangeable samples of one fixed learning algorithm.',
                           'Expected risk is over hypothetical training seeds, not the realized selected pool.',
                           'Only ordinary linear component is represented; nonlinear corrections can change the effect.',
                           'Three seeds supply a noisy variance estimate; no independence claim for reused CV.'],
              no_labels=True, no_model_fit=True, no_candidate=True, no_complete_score_claim=True)


def art(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    path = ROOT / record['path']
    assert art(path) == {k:record[k] for k in ['path','sha256']}
    return path


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    parent = read(PARENT)
    controls = {month:[read(verify(r)) for r in records] for month,records in parent['controls'].items()}
    sources = [art(__file__), art(PARENT)]
    for records in parent['controls'].values():
        sources.extend(records)
    sources.extend(r['prediction'] for records in controls.values() for r in records)
    for record in sources:
        verify(record)
    write(OUT/'identity.json', dict(design=DESIGN, sources=sources,
                                   frozen_at=datetime.now(timezone.utc).isoformat()))
    months = {}
    for month, records in controls.items():
        frames = [pd.read_parquet(verify(r['prediction']), columns=['mvt_id','base']) for r in records]
        assert len(frames)==3 and frames[0].mvt_id.is_unique
        for f in frames[1:]:
            np.testing.assert_array_equal(f.mvt_id, frames[0].mvt_id)
        values = np.stack([f.base.to_numpy(dtype=np.float64) for f in frames])
        assert np.isfinite(values).all() and (values>=0).all()
        variance = values.var(axis=0, ddof=1)
        pairs = [(values[a]-values[b])**2 for a,b in [(0,1),(0,2),(1,2)]]
        np.testing.assert_allclose(variance, np.sum(pairs,axis=0)/6, atol=1e-8, rtol=1e-12)
        ordinary_variance = DESIGN['ordinary_coefficient']**2 * float(variance.mean())
        months[month] = dict(rows=values.shape[1], mean_seed_variance_seconds_squared=float(variance.mean()),
                             raw_seed_standard_deviation_rms_seconds=float(np.sqrt(variance.mean())),
                             pairwise_prediction_rms_seconds=[float(np.sqrt(p.mean())) for p in pairs],
                             ordinary_mean3_variance_seconds_squared=ordinary_variance/3,
                             ordinary_expected_variance_reduction_3_to_5_seconds_squared=ordinary_variance*(1/3-1/5),
                             ordinary_expected_variance_reduction_3_to_infinity_seconds_squared=ordinary_variance/3)
    result = dict(identity=art(OUT/'identity.json'), status='descriptive_label_free_diagnostic', months=months,
                  exact_pairwise_variance_identity=True, deployment_score_inference_not_supported=True,
                  no_training_allocation=True, no_submission_candidate=True)
    write(OUT/'result.json', result)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    run()
