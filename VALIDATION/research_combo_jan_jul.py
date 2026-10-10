"""Combine the two fixed passing research components without fitting or tuning.

Full15 callsign networks and a fixed equal mean of three-seed CatBoost Rome
probabilities with R20. All other v58 parts/rules stay unchanged. These are
reused development months, not independent confirmation or official scores.
"""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline'), str(ROOT/'VALIDATION/tools')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import nn_callsign_pool15 as calls
import rome_catboost_bagging_jan_jul as cat
from records import sha256, write_json

OUT = ROOT/'VALIDATION/runs/v3_20261010_research_combo_jan_jul'
DESIGN = dict(parent='v58', nn='full15 callsign PLE',
    rome='0.5 three-seed CatBoost plus 0.5 original R20 probability',
    unchanged='all three tree parts, network share0.71, corrections and rules',
    arms=['control', 'callsign15', 'catboost_half', 'combined'],
    exploratory_reused_development_months=True, no_weight_search=True)


def main():
    calls.guard()
    sc, rome, pool, trees = calls.runner.baseline.baseline()
    source = calls.runner.OUT/'scored_g42_142_242_k0_1_2_3_4.parquet'
    frame = pd.read_parquet(source)
    errors = {a:{} for a in DESIGN['arms']}
    rows = []
    for month in calls.runner.ts.SPLITS:
        d = sc.d[month]
        f = frame[frame.month.eq(month)]
        np.testing.assert_array_equal(f.mvt_id, d['ids'])
        np.testing.assert_allclose(f.direct_nn, pool[month], atol=1e-9, rtol=0)
        nn = f.candidate_nn.to_numpy(float)
        p = .5*cat.probabilities(month, d['ids']) + .5*rome[month]
        arms = dict(control=(pool[month], rome[month]), callsign15=(nn, rome[month]),
                    catboost_half=(pool[month], p), combined=(nn, p))
        for name, (neural, probability) in arms.items():
            errors[name][month] = np.mean([
                sc.se(month, seed, nn=neural, p=probability,
                      **{calls.runner.hpo.MEMBER_OF[part]:trees[part][seed][month] for part in trees})
                for seed in calls.runner.hpo.SEEDS], axis=0)
        rows.append(pd.DataFrame(dict(mvt_id=d['ids'], month=month, day=d['day'],
            airport=d['held']['raw'].ADEP_mvt.to_numpy(),
            **{name:errors[name][month] for name in arms})))
    scores = {name:sc.cv(value) for name,value in errors.items()}
    expected = dict(control=295.7414415741871, callsign15=295.08866167274704,
                    catboost_half=295.1641971245211)
    for name, value in expected.items():
        if abs(scores[name]['cv']-value)>1e-6:
            raise ValueError(f'{name} does not reproduce its completed screen')
    scored = pd.concat(rows, ignore_index=True)
    comparisons = {f'combined_vs_{name}':calls.runner.ts.paired_bootstrap(
        scored.day, scored.month, scored[name], scored.combined)
        for name in ['control', 'callsign15', 'catboost_half']}
    inputs = [source, calls.MANIFEST]
    for month in calls.runner.ts.SPLITS:
        for seed in cat.SEEDS:
            path = (cat.parent_paths(month)[0] if seed == 42 else
                    cat.OUT/'predictions'/f'{month}_s{seed}.parquet')
            inputs.extend([path, path.with_suffix('.json')])
    report = dict(design=DESIGN, source_sha256=sha256(Path(__file__)),
        input_sha256={str(path.relative_to(ROOT)):sha256(path) for path in inputs},
        scores=scores, comparisons=comparisons)
    OUT.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUT/'scored_errors.parquet', index=False)
    write_json(OUT/'result.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
