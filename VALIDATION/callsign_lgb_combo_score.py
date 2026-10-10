"""Fixed callsign-LGB increment on callsign15 plus half CatBoost Rome.

Use only ordinary tree seed42 throughout, because the new main LightGBM has
only seed42 predictions. Compare against its same-laptop original156 control.
No fit, extra blend search, or copied candidate across other tree seeds.
"""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import callsign_lgb_control_jan_jul as control
import nn_callsign_pool15 as calls
import rome_catboost_bagging_jan_jul as cat
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261010_callsign_lgb_combo_score'
DESIGN = dict(parent='full15 callsign networks plus fixed half CatBoost Rome',
    changed_part='main LightGBM: original156 to original156 plus callsign6',
    ordinary_tree_seeds=[42], main_lgb_control='matched laptop original156 seed42',
    nn='full15 callsign PLE', rome='0.5 three-seed CatBoost plus 0.5 original R20',
    network_share=.71, other_parts_and_rules='unchanged', fitting=False,
    weight_search=False, evaluation='reused Jan/Jul development months')


def main():
    calls.guard()
    sc, rome, pool, trees = calls.runner.baseline.baseline()
    source = calls.runner.OUT / 'scored_g42_142_242_k0_1_2_3_4.parquet'
    frame = pd.read_parquet(source)
    names = ['matched_v58', 'combined_historical', 'combined_matched', 'combined_calls_lgb']
    errors = {name: {} for name in names}
    rows, inputs = [], [source, calls.MANIFEST]
    for month in calls.runner.ts.SPLITS:
        d = sc.d[month]
        neural_frame = frame[frame.month.eq(month)]
        np.testing.assert_array_equal(neural_frame.mvt_id, d['ids'])
        np.testing.assert_allclose(neural_frame.direct_nn, pool[month], atol=1e-9, rtol=0)
        nn = neural_frame.candidate_nn.to_numpy(float)
        p = .5 * cat.probabilities(month, d['ids']) + .5 * rome[month]
        main = {}
        for key, folder in [('matched', control.OUT), ('candidate', control.candidate.OUT)]:
            path = folder / 'predictions' / f'{month}_s42.parquet'
            record = control.ts.read(path.with_suffix('.json'))
            assert sha256(path) == record['prediction_sha256']
            f = pd.read_parquet(path)
            np.testing.assert_array_equal(f.mvt_id, d['ids'])
            main[key] = f.pred.to_numpy(float)
            inputs += [path, path.with_suffix('.json')]
        base = {calls.runner.hpo.MEMBER_OF[part]: trees[part][42][month] for part in trees}
        lgb_name = calls.runner.hpo.MEMBER_OF['lgb']
        arms = dict(matched_v58=(pool[month], rome[month], main['matched']),
            combined_historical=(nn, p, base[lgb_name]),
            combined_matched=(nn, p, main['matched']),
            combined_calls_lgb=(nn, p, main['candidate']))
        for name, (neural, probability, main_lgb) in arms.items():
            members = dict(base)
            members[lgb_name] = main_lgb
            errors[name][month] = sc.se(month, 42, nn=neural, p=probability, **members)
        rows.append(pd.DataFrame(dict(mvt_id=d['ids'], month=month, day=d['day'],
            **{name: errors[name][month] for name in names})))
        for seed in cat.SEEDS:
            path = (cat.parent_paths(month)[0] if seed == 42 else cat.OUT / 'predictions' / f'{month}_s{seed}.parquet')
            inputs += [path, path.with_suffix('.json')]
    scored = pd.concat(rows, ignore_index=True)
    scores = {name: sc.cv(values) for name, values in errors.items()}
    expected = control.ts.read(control.OUT / 'result.json')['scores']['laptop_control']['cv']
    assert abs(scores['matched_v58']['cv'] - expected) < 1e-9
    comparisons = {}
    for a, b in [('combined_calls_lgb', 'combined_matched'),
                 ('combined_calls_lgb', 'combined_historical'),
                 ('combined_calls_lgb', 'matched_v58'),
                 ('combined_matched', 'matched_v58')]:
        comparisons[f'{a}_vs_{b}'] = calls.runner.ts.paired_bootstrap(
            scored.day, scored.month, scored[b], scored[a])
    report = dict(design=DESIGN, source_sha256=sha256(Path(__file__)),
        input_sha256={str(p.relative_to(ROOT)): sha256(p) for p in inputs},
        scores=scores, comparisons=comparisons,
        caution='Bootstrap describes the reused development months, not independent confirmation or an official score.')
    OUT.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUT / 'scored_errors.parquet', index=False)
    write_json(OUT / 'result.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
