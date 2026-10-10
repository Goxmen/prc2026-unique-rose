"""Estimate remaining raw-track seed variance from saved predictions only.

This is an IID-seed extrapolation, not a measured improvement or a confidence
bound. Keep v64's weights, trees, classifier and rules fixed. Compare exact
post-rule member predictions and quantify their nonlinearity at the pool mean.
"""
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'VALIDATION/tools'), str(ROOT / 'pipeline')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import sparse_track_combo_score as combo
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261010_sparse_seed_variance'


def main():
    selected = combo.parent.control.ts.read(combo.OUT / 'result.json')
    for path, digest in selected['inputs'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Selected evidence changed: ' + path)
    sc, rome, _, trees = combo.parent.calls.runner.baseline.baseline()
    calls_path = combo.parent.calls.runner.OUT / 'scored_g42_142_242_k0_1_2_3_4.parquet'
    calls = pd.read_parquet(calls_path).set_index('mvt_id')
    recorded = pd.read_parquet(combo.OUT / 'scored_errors.parquet').set_index('mvt_id')
    summaries, estimates = {}, {str(n): {} for n in [5, 9, 15, 30]}
    estimates['infinity'] = {}
    inputs = {str(calls_path.relative_to(ROOT)): sha256(calls_path)}
    for month in combo.sequence.base.STOP:
        ids = sc.d[month]['ids']
        members = []
        for k in combo.sequence.NETS:
            path = combo.sequence.paths(month, k)[0]
            frame = pd.read_parquet(path)
            np.testing.assert_array_equal(frame.mvt_id, ids)
            members.append(frame.base.to_numpy(float))
            inputs[str(path.relative_to(ROOT))] = sha256(path)
        members = np.asarray(members)
        nn_calls = calls.loc[ids, 'candidate_nn'].to_numpy(float)
        p = .5 * combo.parent.cat.probabilities(month, ids) + .5 * rome[month]
        tree_parts = {combo.parent.calls.runner.hpo.MEMBER_OF[key]: trees[key][42][month]
                      for key in trees}
        path = combo.parent.control.candidate.OUT / 'predictions' / f'{month}_s42.parquet'
        main_tree = pd.read_parquet(path)
        np.testing.assert_array_equal(main_tree.mvt_id, ids)
        tree_parts[combo.parent.calls.runner.hpo.MEMBER_OF['lgb']] = main_tree.pred.to_numpy(float)
        tree = np.mean(list(tree_parts.values()), axis=0)

        def final(track):
            return sc.final(month, .71 * .5 * (nn_calls + track) + .29 * tree, p)

        pooled = final(members.mean(0))
        se = (sc.d[month]['y'] - pooled) ** 2
        np.testing.assert_allclose(se, recorded.loc[ids, 'half_sparse3_combination'], rtol=1e-10, atol=1e-6)
        individual = np.asarray([final(member) for member in members])
        variance = individual.var(0, ddof=1)
        pairwise = sum((individual[i] - individual[j]) ** 2
                       for i in range(3) for j in range(i)) / 6
        np.testing.assert_allclose(variance, pairwise, rtol=1e-10, atol=1e-8)
        nonlinear = pooled - individual.mean(0)
        summary = dict(rows=len(ids), current_rmse=float(np.sqrt(se.mean())),
                       post_rule_seed_variance=float(variance.mean()),
                       member_rmses=[float(np.sqrt(((sc.d[month]['y'] - x) ** 2).mean()))
                                     for x in individual],
                       nonlinearity_max_abs=float(abs(nonlinear).max()),
                       nonlinearity_rms=float(np.sqrt((nonlinear ** 2).mean())),
                       nonlinearity_rows_gt_1sec=int((abs(nonlinear) > 1).sum()))
        for count in estimates:
            reduction = variance.mean() * (1 / 3 - (0 if count == 'infinity' else 1 / int(count)))
            estimates[count][month] = float(np.sqrt(max(se.mean() - reduction, 0)))
        summaries[month] = summary
    current = np.mean([x['current_rmse'] for x in summaries.values()])
    for values in estimates.values():
        values['cv'] = float(np.mean(list(values.values())))
        values['estimated_change'] = values['cv'] - current
    result = dict(status='diagnostic only', current_cv=float(current), months=summaries,
                  estimates=estimates, inputs=inputs, source_sha256=sha256(Path(__file__)),
                  assumptions=['three seeds are exchangeable independent draws around one fixed recipe',
                               'same training distribution and stopping policy for future seeds',
                               'local ensemble averaging remains effectively linear through rules'],
                  limitations=['three seeds give imprecise variance estimates',
                               'does not reduce systematic bias or establish 2026 transfer',
                               'not a measured gain, upper bound, or independent validation'],
                  new_fit=False, weight_search=False, upload=False)
    write_json(OUT / 'result.json', result)
    print({k: result[k] for k in ['current_cv', 'months', 'estimates']}, flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
