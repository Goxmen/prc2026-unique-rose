"""Frozen Stage 4 probability mixture; targets are used only during fitting."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from common import CATEGORICAL

CLASSES = ('ordinary', 'schedule', 'day')
ORDINARY_MAX = 7200
DAY_SEC = 86400
SCHEDULE_TOLERANCE = 60
BUNDLE_VERSION = 'timestamp_mixture_v1'


def training_classes(X, y):
    """Operational patterns, with schedule precedence; never called at inference."""
    y = np.asarray(y, dtype=float)
    if len(y) != len(X) or not np.isfinite(y).all():
        raise ValueError('mixture training targets must be finite and aligned')
    gap = X['mvt_minus_sched_sec'].to_numpy(float)
    result = np.zeros(len(y), dtype=np.int8)
    result[(y - DAY_SEC >= 1) & (y - DAY_SEC <= ORDINARY_MAX)] = 2
    result[(y > ORDINARY_MAX) & np.isfinite(gap) &
           (np.abs(y-gap) <= SCHEDULE_TOLERANCE)] = 1
    return result


def _train(X, y, params, rounds, desc, history):
    dataset = lgb.Dataset(X, y, categorical_feature=CATEGORICAL)
    with tqdm(total=rounds, desc=desc, unit='rnd', leave=False) as bar:
        def progress(env):
            for _, metric, value, *_ in env.evaluation_result_list or []:
                bar.set_postfix({metric: f'{value:.5f}'}, refresh=False)
            bar.update(env.iteration+1-bar.n)

        return lgb.train(params, dataset, num_boost_round=rounds,
                         valid_sets=[dataset], valid_names=['training'],
                         callbacks=[progress, lgb.record_evaluation(history)])


def fit_mixture(X, y, seed, rounds, config, params, desc, training_history):
    labels = training_classes(X, y)
    ordinary = labels == 0
    if not ordinary.any():
        raise ValueError('mixture requires ordinary/other training rows')
    history = training_history if training_history is not None else {}
    common = dict(params, seed=seed, bagging_seed=seed+1,
                  feature_fraction_seed=seed+2, num_threads=config.get('threads', 0))
    history['gate'], history['ordinary'] = {}, {}
    gate = _train(X, labels, dict(common, objective='multiclass', metric='multi_logloss', num_class=3),
                  rounds, desc+' probabilities', history['gate'])
    regressor = _train(X.loc[ordinary], np.asarray(y)[ordinary], common,
                       rounds, desc+' ordinary', history['ordinary'])
    counts = {name: int((labels == code).sum()) for code, name in enumerate(CLASSES)}
    return MixtureBooster(gate, regressor, counts, config.get('threads', 0))


class MixtureBooster:
    """One atomic bundle contains both boosters and the frozen component counts."""

    def __init__(self, gate, ordinary, counts, threads):
        self.gate, self.ordinary = gate, ordinary
        self.training_summary = {'class_counts': counts,
                                 'classifier_rows': sum(counts.values()),
                                 'ordinary_regressor_rows': counts['ordinary']}
        self.params = {'num_threads': threads}

    def components(self, X, num_iteration=None, num_threads=None):
        threads = self.params['num_threads'] if num_threads is None else num_threads
        kwargs = dict(num_iteration=num_iteration, num_threads=threads)
        probabilities = np.asarray(self.gate.predict(X, **kwargs))
        if (probabilities.shape != (len(X), 3) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8, rtol=0)):
            raise ValueError('invalid mixture probabilities')
        ordinary = np.maximum(self.ordinary.predict(X, **kwargs), 0)
        gap = X['mvt_minus_sched_sec'].to_numpy(float)
        schedule = np.where(np.isfinite(gap), np.maximum(gap, 0), ordinary)
        day = ordinary + DAY_SEC
        component = np.column_stack([ordinary, schedule, day])
        prediction = np.sum(probabilities * component, axis=1)
        if not np.isfinite(prediction).all():
            raise ValueError('nonfinite mixture predictions')
        return pd.DataFrame({
            **{f'mixture_p_{name}': probabilities[:, code] for code, name in enumerate(CLASSES)},
            'mixture_ordinary_sec': ordinary, 'mixture_schedule_sec': schedule,
            'mixture_day_sec': day, 'mixture_pred': prediction,
        }, index=X.index)

    def predict(self, X, num_iteration=None, num_threads=None):
        return self.components(X, num_iteration, num_threads)['mixture_pred'].to_numpy()

    def save_model(self, filename):
        path = Path(filename)
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps({
            'format': BUNDLE_VERSION, 'counts': self.training_summary['class_counts'],
            'threads': self.params['num_threads'], 'gate': self.gate.model_to_string(),
            'ordinary': self.ordinary.model_to_string(),
        }))
        temp.replace(path)

    @classmethod
    def load(cls, filename):
        record = json.loads(Path(filename).read_text())
        if record['format'] != BUNDLE_VERSION:
            raise ValueError('unknown mixture bundle format')
        return cls(lgb.Booster(model_str=record['gate']), lgb.Booster(model_str=record['ordinary']),
                   record['counts'], record['threads'])


def validate_recipe(config):
    if config.get('model_mode', 'single') == 'timestamp_mixture':
        if not config.get('clock_features', False) or config.get('target_mode', 'direct') != 'direct':
            raise ValueError('timestamp mixture requires clock features and direct target mode')
        if config.get('early_stopping_rounds', 0):
            raise ValueError('timestamp mixture requires fixed rounds; inner early stopping is unsupported')
