"""Fixed raw-track increment on the callsign/main-tree/Rome combination.

Declared before new sparse pool3 fits finish. Half of the ordinary neural
prediction comes from callsign15 and half from sparse3; all other parts are
the completed callsign-LGB combination. Retain an identical half historical3
control to distinguish representation gain from pooling variance. No fit,
weight search, availability gate or additional correction is introduced.
"""
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

import callsign_lgb_combo_score as parent
import nn_sparse_track_pool3 as sequence
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261010_sparse_track_combo_score'
DESIGN = dict(parent='callsign15 plus callsign main LightGBM plus half CatBoost/R20 Rome',
    change='ordinary neural prediction becomes fixed equal mean of callsign15 and sparse3',
    network_share=.71, ordinary_tree_seeds=[42], neural_pool_sizes=[15, 3],
    fair_control='same equal blend, with matched historical group42 k0/k1/k2 replacing sparse3',
    rome='fixed half three-seed CatBoost plus half original R20',
    main_lgb='callsign162 LB1 seed42', other_trees_and_rules='unchanged',
    declaration='before new sparse3 k1/k2 outcomes; after original k0 screen',
    weight_search=False, new_availability_gate=False, fitting=False,
    evaluation='reused Jan/Jul development months')


def main():
    parent.calls.guard()
    cache = sequence.base.cache_identity()
    sequence.guard(cache)
    reference = parent.control.ts.read(parent.OUT / 'result.json')
    for name, digest in reference['input_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Completed combination evidence changed: ' + name)
    sequence_score = sequence.OUT / 'score_pool3.json'
    sequence_predictions = sequence.OUT / 'scored_pool3.parquet'
    finished = parent.control.ts.read(sequence_score)
    if finished['design'] != sequence.DESIGN:
        raise ValueError('Completed sequence pool differs from fixed design')
    sc, rome, pool, trees = parent.calls.runner.baseline.baseline()
    calls_path = parent.calls.runner.OUT / 'scored_g42_142_242_k0_1_2_3_4.parquet'
    calls_frame = pd.read_parquet(calls_path).set_index('mvt_id')
    seq_frame = pd.read_parquet(sequence_predictions).set_index('mvt_id')
    if not calls_frame.index.is_unique or not seq_frame.index.is_unique:
        raise ValueError('Duplicate reference prediction IDs')
    names = ['matched_v58', 'callsign_combination', 'half_direct3_combination', 'half_sparse3_combination']
    errors = {name: {} for name in names}
    rows = []
    sources = [parent.OUT / 'result.json', calls_path, parent.calls.MANIFEST,
               sequence_score, sequence_predictions, sequence.MANIFEST]
    for month in sequence.base.STOP:
        d = sc.d[month]
        ids = d['ids']
        old_neural = calls_frame.loc[ids, 'direct_nn'].to_numpy(float)
        np.testing.assert_allclose(old_neural, pool[month], atol=1e-9, rtol=0)
        calls_nn = calls_frame.loc[ids, 'candidate_nn'].to_numpy(float)
        # Verify every scored mean against its three immutable member files.
        direct, sparse = [], []
        for k in sequence.NETS:
            if sequence.validate_member(month, k, cache) is None:
                raise ValueError('Incomplete sparse continuation')
            path = sequence.paths(month, k)[0]
            f = pd.read_parquet(path)
            np.testing.assert_array_equal(f.mvt_id, ids)
            sparse.append(f.base.to_numpy(float))
            direct.append(sequence.shared.direct_member(sc, month, 42, k))
            sources.append(path)
        sparse_nn, direct_nn = np.mean(sparse, axis=0), np.mean(direct, axis=0)
        np.testing.assert_array_equal(sparse_nn, seq_frame.loc[ids, 'candidate_nn'])
        np.testing.assert_array_equal(direct_nn, seq_frame.loc[ids, 'direct_nn'])
        main = {}
        for key, folder in [('matched', parent.control.OUT), ('candidate', parent.control.candidate.OUT)]:
            path = folder / 'predictions' / f'{month}_s42.parquet'
            record = parent.control.ts.read(path.with_suffix('.json'))
            if sha256(path) != record['prediction_sha256']:
                raise ValueError('Main tree prediction changed')
            f = pd.read_parquet(path)
            np.testing.assert_array_equal(f.mvt_id, ids)
            main[key] = f.pred.to_numpy(float)
            sources.extend([path, path.with_suffix('.json')])
        p = .5 * parent.cat.probabilities(month, ids) + .5 * rome[month]
        base = {parent.calls.runner.hpo.MEMBER_OF[part]: trees[part][42][month] for part in trees}
        main_name = parent.calls.runner.hpo.MEMBER_OF['lgb']
        arms = dict(matched_v58=(pool[month], rome[month], main['matched']),
            callsign_combination=(calls_nn, p, main['candidate']),
            half_direct3_combination=(.5 * (calls_nn + direct_nn), p, main['candidate']),
            half_sparse3_combination=(.5 * (calls_nn + sparse_nn), p, main['candidate']))
        for name, (neural, probability, main_lgb) in arms.items():
            members = dict(base)
            members[main_name] = main_lgb
            errors[name][month] = sc.se(month, 42, nn=neural, p=probability, **members)
        rows.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=d['day'],
                                     **{name: errors[name][month] for name in names})))
    scores = {name: sc.cv(values) for name, values in errors.items()}
    for name, reference_name in [('matched_v58', 'matched_v58'), ('callsign_combination', 'combined_calls_lgb')]:
        if abs(scores[name]['cv'] - reference['scores'][reference_name]['cv']) > 1e-9:
            raise ValueError('Completed combination control did not reproduce')
    scored = pd.concat(rows, ignore_index=True)
    comparisons = {}
    for control in ['matched_v58', 'callsign_combination', 'half_direct3_combination']:
        comparisons['half_sparse3_combination_vs_' + control] = parent.calls.runner.ts.paired_bootstrap(
            scored.day, scored.month, scored[control], scored.half_sparse3_combination)
    OUT.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUT / 'scored_errors.parquet', index=False)
    result = dict(design=DESIGN, source_sha256=sha256(Path(__file__)),
        inputs={str(p.relative_to(ROOT)): sha256(p) for p in sources}, scores=scores, comparisons=comparisons,
        caution='Development comparisons, not independent validation or an official score.')
    write_json(OUT / 'result.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
