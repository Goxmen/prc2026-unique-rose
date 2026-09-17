"""Rome hybrid: substitution classifier blend plus the registered four-hour rule.

About a fifth of LIRF departures log off-block at the scheduled time, so their
label is takeoff minus schedule (delay plus taxi). Because
`TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` exactly in 2025, the
training label identifies these rows without loading block times:

    schedule-recorded = LIRF and |(takeoff - schedule) - label| <= 60 s

Per training window and seed, two components are fitted:

- an *ordinary* regressor: the Q0-4h LightGBM recipe on every training row
  except schedule-recorded ones;
- a LIRF-only binary classifier for P(schedule-recorded | observable inputs).

Prediction, using observable fields only:

    LIRF:   p * (takeoff - schedule) + (1 - p) * ordinary, clipped at zero
    other:  ordinary
    then:   the registered `rome_4h` rule overrides its eligible flights.

`original` components use the 27 Q0-4h inputs (the classifier adds the raw
flight-ID and plan-block missing flags); `clock` components use those plus the
18 clock inputs of `clock_features`, which already carry the same flags.
"""

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

import modeling
import overnight_models as models
from common import CATEGORICAL
from registered_rules import apply_rule

AIRPORT = 'LIRF'
MATCH_SEC = 60
GAP = 'mvt_minus_sched_sec'
RULE = 'rome_4h'
FLAGS = ['raw_flight_id_missing', 'raw_plan_block_missing']
CLASSIFIER_PARAMS = {
    'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': 0.05,
    'num_leaves': 31, 'min_data_in_leaf': 100, 'feature_fraction': 0.9,
    'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbosity': -1,
}
CLASSIFIER_ROUNDS = 300
KINDS = ('original', 'clock')
SETTINGS = dict(airport=AIRPORT, match_sec=MATCH_SEC, rule=RULE, flags=FLAGS,
                classifier_params=CLASSIFIER_PARAMS, classifier_rounds=CLASSIFIER_ROUNDS)


def regressor_features(kind):
    if kind not in KINDS:
        raise ValueError(f'Unknown feature kind: {kind}')
    return list(models.ORIGINAL_FEATURES) + (list(models.CLOCK_FEATURES) if kind == 'clock' else [])


def classifier_matrix(X45, raw, kind):
    X = X45.loc[:, regressor_features(kind)]
    if kind == 'original':
        X = X.assign(**{flag: raw[flag].to_numpy().astype('int8') for flag in FLAGS})
    return X


def substituted(X45, y, raw):
    """Training outcome only: the label equals takeoff minus schedule at LIRF."""
    gap = X45[GAP].to_numpy(float)
    lirf = raw['ADEP_mvt'].to_numpy() == AIRPORT
    return lirf & (np.abs(gap - np.asarray(y, float)) <= MATCH_SEC)


def _train(params, X, y, rounds, desc, history=None):
    dataset = lgb.Dataset(X, y, categorical_feature=CATEGORICAL)
    kwargs = {}
    callbacks = []
    if history is not None:
        kwargs.update(valid_sets=[dataset], valid_names=['training'])
        callbacks.append(lgb.record_evaluation(history))
    with tqdm(total=rounds, desc=desc, unit='rnd', leave=False) as bar:
        callbacks.append(lambda env: bar.update(env.iteration + 1 - bar.n))
        return lgb.train(params, dataset, num_boost_round=rounds, callbacks=callbacks, **kwargs)


def fit_components(X45, y, raw, kind, seed, rounds=800, threads=8):
    """Return (ordinary regressor, classifier, details) for one window and seed."""
    y = np.asarray(y, float)
    if len(y) != len(X45) or not X45.index.equals(raw.index):
        raise ValueError('Training matrix, targets and raw rows must align')
    s = substituted(X45, y, raw)
    lirf = raw['ADEP_mvt'].to_numpy() == AIRPORT
    if not s.any() or s[lirf].all():
        raise ValueError('Need both schedule-recorded and ordinary LIRF training rows')
    history = {}
    ordinary = _train(models.model_params('control', seed, threads),
                      X45.loc[~s, regressor_features(kind)], y[~s], rounds,
                      f'ordinary-{kind}/{seed}', history)
    params = dict(CLASSIFIER_PARAMS, seed=int(seed), bagging_seed=int(seed) + 1,
                  feature_fraction_seed=int(seed) + 2, num_threads=int(threads))
    classifier = _train(params, classifier_matrix(X45.loc[lirf], raw.loc[lirf], kind),
                        s[lirf].astype(float), CLASSIFIER_ROUNDS, f'classifier-{kind}/{seed}')
    details = dict(kind=kind, ordinary_rows=int((~s).sum()), schedule_recorded_rows=int(s.sum()),
                   classifier_rows=int(lirf.sum()),
                   ordinary_training_rmse_final=history['training']['rmse'][-1])
    return ordinary, classifier, details


def predict(ordinary, classifier, V45, raw, kind, threads=8):
    """Final hybrid predictions and audit columns for every evaluation row."""
    base = modeling.predict(ordinary, V45.loc[:, regressor_features(kind)], threads=threads)
    gap = V45[GAP].to_numpy(float)
    lirf = (raw['ADEP_mvt'].to_numpy() == AIRPORT) & np.isfinite(gap)
    p = np.zeros(len(V45))
    if lirf.any():
        p[lirf] = classifier.predict(classifier_matrix(V45.loc[lirf], raw.loc[lirf], kind),
                                     num_threads=threads)
    blended = np.where(lirf, np.clip(p * np.where(lirf, gap, 0) + (1 - p) * base, 0, None), base)
    final, audit = apply_rule(blended, raw, RULE)
    return dict(ordinary=base, p=p, blended=blended, final=final,
                rule_triggered=audit.rule_triggered.to_numpy())
