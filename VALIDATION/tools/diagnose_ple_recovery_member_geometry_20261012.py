"""Label-free fixed comparison of the failed PLE member to its original pool.

Descriptive prediction distances only: no teacher/member selection, fitting,
accuracy estimate, acceptance-tolerance change or production substitution.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_ple_pilot'
TEACHER = ROOT / 'VALIDATION/runs/v3_20261003_ideas_submit/work/parts/nn_v46.parquet'


def art(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest())


def moments(delta):
    return dict(rows=len(delta), mean=float(np.mean(delta)),
                rms=float(np.sqrt(np.mean(delta**2))), max_abs=float(np.max(abs(delta))))


def main():
    target = RUN / 'member_geometry_diagnosis.json'
    assert not target.exists()
    result = json.loads((RUN / 'result.json').read_text())
    assert result['state'] == 'closed_nonexact_pilot'
    old = pd.read_parquet(TEACHER)
    pilot = pd.read_parquet(RUN / 'old_epoch14.parquet')
    assert old.mvt_id.is_unique and pilot.mvt_id.is_unique
    assert len(old) == len(pilot) == 344841 and set(old.mvt_id) == set(pilot.mvt_id)
    old = old.set_index('mvt_id').loc[pilot.mvt_id]
    names = [f'n{group}_{member}' for group in [42, 142, 242] for member in range(5)]
    assert list(old) == names
    original = old.to_numpy(float)
    current = pilot.pred.to_numpy().astype(np.float32).astype(float)
    delta = current-original[:,0]
    assert moments(delta)['rms'] == result['rms_difference']
    raw = pd.read_parquet(ROOT/'data/ranking.parquet', columns=['MVT_ID_mvt', 'MVT_TIME_UTC_mvt'])
    assert raw.MVT_ID_mvt.is_unique
    dates = pd.to_datetime(raw.set_index('MVT_ID_mvt').loc[pilot.mvt_id, 'MVT_TIME_UTC_mvt'], utc=True)
    months = dates.dt.strftime('%Y-%m').to_numpy()
    pairwise = [moments(original[:,i]-original[:,j])['rms'] for i in range(15) for j in range(i)]
    report = dict(status='label_free_descriptive_only', sources=[art(p) for p in [Path(__file__), TEACHER,
        RUN/'old_epoch14.parquet', RUN/'result.json', ROOT/'data/ranking.parquet']],
        fixed_identity='Seed42/member0/epoch14 remains the only recovery target; other members are descriptive context only',
        original_n42_0_difference=moments(delta), by_month={m:moments(delta[months==m]) for m in sorted(set(months))},
        pilot_to_original_members={name:moments(current-original[:,i]) for i,name in enumerate(names)},
        original_n42_0_to_other_members={name:moments(original[:,0]-original[:,i]) for i,name in enumerate(names) if i},
        original_all_pairwise_member_rms=dict(pairs=len(pairwise), minimum=min(pairwise),
            median=float(np.median(pairwise)), maximum=max(pairwise)),
        pilot_to_original_group42_mean=moments(current-original[:,:5].mean(axis=1)),
        original_n42_0_to_group42_mean=moments(original[:,0]-original[:,:5].mean(axis=1)),
        caveat='Distances provide scale only. They cannot establish GPU nondeterminism as cause, exact weights, validation gain or final-view fidelity.',
        no_labels_loaded=True, no_fit=True, no_inference=True, no_selection=True, no_tolerance_change=True)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], original_pool=report['original_all_pairwise_member_rms'],
                         pilot_target_rms=result['rms_difference'], by_month=report['by_month'], report=art(target))))


if __name__ == '__main__':
    with threadpool_limits(1):
        main()
