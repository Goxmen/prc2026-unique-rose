"""Frozen model interventions for the five-submission CV investigation.

Feature/history construction remains in ``modeling``.  The caller builds the
same 45-column matrix once per training window, before any routing filter, and
passes the original retained raw rows alongside it.  This adapter only selects
model inputs, optionally routes training rows, and fits ordinary LightGBM.
The shared registered four-hour rule is applied by the caller after prediction.
"""

from collections import OrderedDict
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modeling
from common import CATEGORICAL
from clock_features import CLOCKS, PROVENANCE
from registered_rules import apply_rule


RECIPES = OrderedDict([
    ('control', dict(name='Q0-4h control', role='control',
                     hypothesis='Reproduce the submitted baseline under the shared matrix path.')),
    ('no_month', dict(name='No calendar month', role='discovery',
                      hypothesis='Calendar extrapolation may transfer differently in chronological CV and 2026.')),
    ('no_history', dict(name='No departure target history', role='discovery',
                        hypothesis='Removing stale departure-label summaries may reduce temporal transfer error.')),
    ('clock_features', dict(name='Explicit clock features', role='discovery',
                            hypothesis='Observable clock disagreements may improve ordinary predictions.')),
    ('small_trees', dict(name='63-leaf trees', role='confirmation',
                         hypothesis='Lower model capacity may generalize differently from the 255-leaf control.')),
    ('rule_routing', dict(name='Train on model-handled rows', role='confirmation',
                          hypothesis='Removing rule-handled training rows may improve the remaining predictions.')),
])

BASE_CONFIG = dict(
    recovery='off', target_mode='direct', model_mode='single',
    clock_features=True, fpl_flag=False, max_taxi_sec=200_000,
    lag_train_days=0, rounds=800, early_stopping_rounds=0,
    threads=8, prediction_threads=8, fit_workers=1,
    prediction_rule='rome_4h', seeds=[42, 142, 242],
    params=dict(modeling.PARAMS),
)

# Explicit allowlists prevent an accidental target/audit column in the shared
# matrix from becoming a model input.  Order matches modeling.matrix exactly.
ORIGINAL_FEATURES = [
    *CATEGORICAL, 'hour', 'dow', 'month',
    *[f'mvt_minus_{name}_sec' for name in ('sched', 'lobt', 'iobt', 'eobt1', 'aobt3')],
    'op_taxi_30d', 'op_taxi_30d_n', 'rwy_taxi_30d', 'stand_taxi_90d',
    'aphour_taxi_30d', 'arr_taxiin_7d', 'dep_actual_30m_before',
    'arr_actual_30m_before', 'dep_sched_60m_around',
]
CLOCK_FEATURES = [
    *[f'clock_{left}_minus_{right}_sec' for left, right in
      [('aobt3', 'schedule'), ('aobt3', 'eobt1'), ('eobt1', 'iobt'), ('iobt', 'lobt')]],
    *[f'clock_{name}_available' for name in CLOCKS],
    *['clock_' + name for name in PROVENANCE],
    'clock_available_count', 'clock_span_sec', 'clock_std_sec',
    'clock_median_sec', 'clock_baseline_sec', 'clock_baseline_source',
]


def _check_recipe(recipe):
    if recipe not in RECIPES:
        raise ValueError(f'Unknown overnight recipe: {recipe}')


def select_features(X, recipe):
    """Return the fixed, ordered model inputs without changing rows or categories."""
    _check_recipe(recipe)
    names = list(ORIGINAL_FEATURES)
    if recipe == 'clock_features':
        names += CLOCK_FEATURES
    elif recipe == 'no_month':
        names.remove('month')
    elif recipe == 'no_history':
        names = [name for name in names if name not in modeling.HISTORY_FEATURES]
    missing = [name for name in names if name not in X.columns]
    if missing:
        raise ValueError(f'Missing required {recipe} features: {missing}')
    if not X.columns.is_unique:
        raise ValueError('Shared feature matrix contains duplicate columns')
    return X.loc[:, names]


def training_mask(raw_kept, recipe):
    """Mask only the booster fit; the caller retains complete history and scoring."""
    _check_recipe(recipe)
    if recipe != 'rule_routing':
        return np.ones(len(raw_kept), dtype=bool)
    # The registered rule explicitly restricts itself to original observable
    # movement fields; neither the label nor any recovered flight ID is used.
    _, audit = apply_rule(np.zeros(len(raw_kept)), raw_kept, 'rome_4h')
    return ~audit.rule_triggered.to_numpy(bool)


def model_params(recipe, seed, threads=8):
    """Exact fit parameters, also available to source/config manifests."""
    _check_recipe(recipe)
    if threads < 1:
        raise ValueError('Use an explicit positive thread budget')
    params = dict(modeling.PARAMS, seed=int(seed), bagging_seed=int(seed) + 1,
                  feature_fraction_seed=int(seed) + 2, num_threads=int(threads))
    if recipe == 'small_trees':
        params['num_leaves'] = 63
    return params


def fit_model(X45, y, raw_kept, recipe, seed, rounds=800, threads=8):
    """Fit one controlled recipe using an already scoped common matrix."""
    if rounds < 1:
        raise ValueError('rounds must be positive')
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) != len(X45) or len(raw_kept) != len(y):
        raise ValueError('Training matrix, raw rows and targets must align')
    if not X45.index.equals(raw_kept.index):
        raise ValueError('Training matrix/raw index alignment differs')
    if not np.isfinite(y).all() or not ((y >= 1) & (y <= 200_000)).all():
        raise ValueError('Apply the registered 1..200000 training-label filter first')
    if 'taxi_sec' in raw_kept and not np.array_equal(y, raw_kept.taxi_sec.to_numpy(float)):
        raise ValueError('Training targets differ from aligned raw rows')
    X = select_features(X45, recipe)
    mask = training_mask(raw_kept, recipe)
    if not mask.any():
        raise ValueError('No training rows survive rule routing')
    params = model_params(recipe, seed, threads)
    dataset = lgb.Dataset(X.loc[mask], y[mask], categorical_feature=CATEGORICAL)
    history = {}
    with tqdm(total=rounds, desc=f'{recipe}/{seed}', unit='rnd', leave=False) as bar:
        def progress(env):
            for name, metric, value, *_ in env.evaluation_result_list or []:
                if name == 'training' and metric == 'rmse':
                    bar.set_postfix(train_rmse=f'{value:.2f}', refresh=False)
            bar.update(env.iteration + 1 - bar.n)

        booster = lgb.train(
            params, dataset, num_boost_round=rounds,
            valid_sets=[dataset], valid_names=['training'],
            callbacks=[progress, lgb.record_evaluation(history)],
        )
    return booster, history


def predict_model(booster, X45, recipe, threads=8):
    """Return clipped model predictions for every row; caller applies the rule."""
    X = select_features(X45, recipe)
    if booster.feature_name() != X.columns.tolist():
        raise ValueError('Saved model feature identity differs from the requested recipe')
    if len(X) == 0:
        return np.array([], dtype=float)
    values = modeling.predict(booster, X, threads=threads, target_mode='direct')
    if values.shape != (len(X),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Expected one finite nonnegative prediction per evaluation row')
    return values
