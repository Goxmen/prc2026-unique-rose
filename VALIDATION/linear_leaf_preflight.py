"""Bounded resource probe for a proposed matched linear-leaf comparison.

No validation predictions, scores, hyperparameter selection or production fit.
Run only in an allocated desktop CPU slot after the phase experiment finishes.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_daily_taxi_lgb as base
from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tmp/research_20261012/signal_scout/linear_leaf_preflight'
DESIGN = dict(
    proposal='fixed original184 main-model linear-leaf comparison',
    original_rows_targets=True, original_categories=True,
    scaling='training-only finite mean/population SD; preserve NaNs; constant scale1; no clipping/imputation',
    candidate_change={'linear_tree': True, 'linear_lambda': 1.0},
    proposed_control='fresh constant-leaf model on identical standardized inputs',
    proposed_other_params='original LB1 seed42,127leaves,learningrate.02,10000rounds,4threads',
    proposed_folds=['2025-01', '2025-07'],
    proposed_score='replace only equal-three-tree main member through exactv78; frozenv79 transport secondary',
    probe_fold='2025-01', probe_rounds=64, probe_sample_rows=65536,
    probe_timeout_seconds=240, probe_rss_limit_gib=16,
    no_validation_prediction_or_score=True,
    no_round_or_parameter_search=True,
    sources=[
        'https://github.com/radekacar/joyous-rainbow',
        'https://lightgbm.readthedocs.io/en/stable/Parameters.html#linear_tree',
        'https://github.com/microsoft/LightGBM/blob/v4.7.0/src/treelearner/linear_tree_learner.cpp',
    ],
    caveats=[
        'Competitor gain used different leaves, population and pipeline; not transferable evidence.',
        'Resource extrapolation from64 rounds is approximate, not a runtime guarantee.',
        'NaN on any selected linear-leaf feature can force constant prediction fallback.',
        'Constant-leaf boosted ensembles are not universally bounded by maximum training target.',
    ],
)


def scaler_fit(frame, categorical):
    """Never read heldout data; preserve missingness and category semantics."""
    assert len(set(frame)) == len(frame.columns)
    result = {}
    for name in frame:
        if name in categorical:
            assert isinstance(frame[name].dtype, pd.CategoricalDtype)
            continue
        a = frame[name].to_numpy(dtype=np.float64)
        assert not np.isinf(a).any(), name
        finite = a[np.isfinite(a)]
        mean = float(finite.mean()) if len(finite) else 0.
        std = float(finite.std()) if len(finite) else 1.
        result[name] = dict(mean=mean, scale=std if std > 0 else 1., finite=int(len(finite)))
    return result


def scaler_apply(frame, scaler):
    cols = {}
    for name in frame:
        if name not in scaler:
            cols[name] = frame[name].copy()
            continue
        spec = scaler[name]
        values = (frame[name].to_numpy(dtype=np.float64)-spec['mean'])/spec['scale']
        cols[name] = pd.Series(values.astype('float32'), index=frame.index)
        np.testing.assert_array_equal(cols[name].isna(), frame[name].isna())
        assert not np.isinf(values).any()
    out = pd.DataFrame(cols, index=frame.index)
    assert list(out) == list(frame)
    for name in frame:
        if name not in scaler:
            pd.testing.assert_series_equal(frame[name], out[name], check_exact=True)
    return out


def check():
    x = pd.DataFrame(dict(a=[1., 2., np.nan, 3.], constant=[9.]*4,
        absent=[np.nan]*4, category=pd.Categorical(['B', 'A', None, 'B'], categories=['B', 'A', 'C'])),
        index=[10, 20, 30, 40])
    stats = scaler_fit(x, ['category']); out = scaler_apply(x, stats)
    assert stats['a']['mean'] == 2. and stats['constant']['scale'] == 1.
    assert stats['absent']['mean'] == 0. and stats['absent']['scale'] == 1.
    assert out.index.equals(x.index) and out.absent.isna().all()
    assert np.isclose(out.a.dropna().std(ddof=0), 1.)
    held = x.copy(); held.loc[10, 'a'] = 1000.
    before = json.dumps(stats, sort_keys=True); scaler_apply(held, stats)
    assert before == json.dumps(stats, sort_keys=True)
    print('Training-only scaling, NaNs, constants, categories, index and unchanged saved statistics pass.', flush=True)


def rss_gib():
    # Linux ru_maxrss is KiB. Report cumulative process peak honestly.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def linear_coverage(model, frame):
    """Input-only linear-active share, sampled training row x tree pairs."""
    leaves = model.predict(frame, pred_leaf=True, num_threads=1)
    if leaves.ndim == 1:
        leaves = leaves[:, None]
    active = total = nonzero_model = 0
    def visit(node):
        if 'leaf_index' in node:
            yield node
        else:
            yield from visit(node['left_child'])
            yield from visit(node['right_child'])
    for k, tree in enumerate(model.dump_model()['tree_info']):
        for leaf in visit(tree['tree_structure']):
            selected = leaves[:, k] == leaf['leaf_index']
            n = int(selected.sum()); total += n
            indices = leaf.get('leaf_features', [])
            if indices:
                nonzero_model += n
                active += int(frame.iloc[selected, indices].notna().all(axis=1).sum())
    return dict(training_rows=len(frame), row_tree_pairs=total,
        pairs_with_linear_features=nonzero_model, pairs_without_missing_linear_features=active,
        active_fraction=active/total)


def probe(scope):
    started = time.monotonic(); OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT/f'{scope}.json'
    progress_path = OUT/f'{scope}_progress.json'
    assert not destination.exists(), 'Preserve completed resource probe'
    assert not progress_path.exists(), 'Preserve partial probe; inspect/archive before retry'
    progress = dict(scope=scope, design=DESIGN, source_sha256=sha256(Path(__file__)),
        stage='input_preparation', completed_arms=[])
    def persist():
        progress['elapsed_seconds'] = time.monotonic()-started
        progress['cumulative_peak_rss_gib'] = rss_gib()
        write_json(progress_path, progress)
    persist()
    design_path = OUT/'design.json'
    if design_path.exists():
        assert json.loads(design_path.read_text()) == DESIGN
    else:
        write_json(design_path, DESIGN)
    # This is the existing audited184-column input assembly, not a new cohort.
    base.guard()
    tables = base.tables(); d = base.inputs(DESIGN['probe_fold'], *tables)
    control = base.completed(DESIGN['probe_fold'])
    assert control and list(d['X']) == control['columns'] and len(control['columns']) == 184
    assert control['params']['num_leaves'] == 127 and control['params']['num_threads'] == 4
    X = d.pop('X'); y = d.pop('y'); ids = d.pop('train_ids')
    del d, tables; gc.collect()
    scaler = scaler_fit(X, control['categoricals']); scaled = scaler_apply(X, scaler)
    full_rows = len(X); del X; gc.collect()
    if scope == 'sample':
        # Input-only deterministic selection; no target stratification.
        picked = np.argsort(pd.util.hash_array(np.asarray(ids)), kind='stable')[:DESIGN['probe_sample_rows']]
        scaled = scaled.iloc[picked].copy(); y = np.asarray(y)[picked]; ids = np.asarray(ids)[picked]
    progress.update(stage='inputs_ready', rows=len(scaled), original_training_rows=full_rows)
    persist()
    assert rss_gib() < DESIGN['probe_rss_limit_gib']
    arm_records = []
    for linear in [False, True]:
        params = dict(control['params'], linear_tree=linear)
        if linear:
            params['linear_lambda'] = 1.
        dataset = lgb.Dataset(scaled, y, params=params, categorical_feature=control['categoricals'])
        times = []; fit_start = time.monotonic()
        progress.update(stage='fitting', current_linear=linear, current_arm_iterations=0,
            current_arm_elapsed_seconds=0.)
        persist()
        def record(env):
            times.append(time.monotonic())
            elapsed = time.monotonic()-fit_start
            progress.update(current_arm_iterations=env.iteration+1,
                current_arm_elapsed_seconds=elapsed, current_arm_timestamps=times)
            if (env.iteration+1)%8 == 0 or elapsed >= DESIGN['probe_timeout_seconds']:
                persist()
            assert elapsed < DESIGN['probe_timeout_seconds'], 'Per-arm resource limit; no scored model'
            assert rss_gib() < DESIGN['probe_rss_limit_gib'], 'Memory limit; no scored model'
        try:
            model = lgb.train(params, dataset, num_boost_round=DESIGN['probe_rounds'], callbacks=[record])
        except BaseException as error:
            progress.update(stage='aborted', error_type=type(error).__name__, error=str(error))
            persist()
            raise
        assert model.current_iteration() == DESIGN['probe_rounds']
        increments = np.diff(times)
        training_check = model.predict(scaled.iloc[:8192], num_threads=1)
        assert np.isfinite(training_check).all(), 'Non-finite training output; resource probe only'
        record_data = dict(linear=linear, rows=len(scaled), numeric_columns=len(scaler), params=params,
            cumulative_peak_rss_gib=rss_gib(), elapsed_seconds=time.monotonic()-fit_start,
            finite_training_output_check=True,
            seconds_per_tree_last32=float(np.mean(increments[-32:])),
            max_seconds_per_tree_last32=float(np.max(increments[-32:])))
        if linear:
            record_data['input_only_linear_coverage'] = linear_coverage(model, scaled.iloc[:8192])
        arm_records.append(record_data)
        progress.update(stage='arm_complete', completed_arms=arm_records)
        persist()
        print(json.dumps(record_data), flush=True)
        del model, dataset; gc.collect()
    write_json(destination, dict(design=DESIGN, scope=scope, original_training_rows=full_rows,
        no_heldout_predictions_or_scores=True, arms=arm_records, scaler=scaler,
        source_sha256=sha256(Path(__file__)), original_input=control['input'],
        original_control_metadata=base.art(base.OUT/'predictions'/'2025-01_s42.json'),
        original_identity=base.art(base.MANIFEST), lightgbm_version=lgb.__version__,
        training_ids_sha256=hashlib.sha256(np.asarray(ids).tobytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started))
    progress.update(stage='complete', result_sha256=sha256(destination))
    persist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'sample', 'full'])
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == 'check':
            check()
        else:
            probe(args.stage)
