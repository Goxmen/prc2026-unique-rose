"""Rome schedule-substitution mixture (improvement-plan candidate M1).

About a fifth of LIRF departures record off-block at the scheduled time, so
their label is takeoff minus schedule (delay plus taxi) rather than a taxi-out.
For every 2025 departure `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt`
exactly, so the training label alone identifies these rows:

    substituted = airport is LIRF and |(takeoff - schedule) - label| <= 60 s

Hidden block times are never loaded. The component outcome uses training labels
only; prediction uses observable fields:

    p       = P(substituted | x), from a LIRF-only classifier
    mixture = p * (takeoff - schedule) + (1 - p) * ordinary

`ordinary` is either the unchanged baseline regressor (`pred_mix_base`) or a
regressor with the baseline recipe fitted without the substituted rows
(`pred_mix_ordinary`). Other airports keep the baseline prediction. The Rome
six-hour rule (Q0) is saved alongside as `pred_rule6h` for a paired comparison.
"""

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from tqdm import tqdm

from modeling import fit_booster, predict  # also puts pipeline/ on sys.path
from common import CATEGORICAL
from records import write_json

MODES = ["off", "lirf-schedule"]
AIRPORT = "LIRF"
MATCH_SEC = 60
RULE_GAP_SEC = 21_600
GAP = "mvt_minus_sched_sec"
FLAGS = ["raw_flight_id_missing", "raw_plan_block_missing"]
FRAME_COLUMNS = ["ADEP_mvt"] + FLAGS

CLASSIFIER_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
    "num_leaves": 31, "min_data_in_leaf": 100, "feature_fraction": 0.9,
    "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1,
}
CLASSIFIER_ROUNDS = 300
SETTINGS = {"airport": AIRPORT, "match_sec": MATCH_SEC, "rule_gap_sec": RULE_GAP_SEC,
            "classifier_params": CLASSIFIER_PARAMS, "classifier_rounds": CLASSIFIER_ROUNDS,
            "classifier_extra_features": FLAGS}
COLUMNS = ["p_substituted", "gap_sec", "pred_ordinary", "pred_mix_base",
           "pred_mix_ordinary", "pred_rule6h"]


def substituted(X, y, frame):
    """Training outcome: the label equals takeoff minus schedule at LIRF."""
    gap = X[GAP].to_numpy(float)
    lirf = frame["ADEP_mvt"].to_numpy() == AIRPORT
    return lirf & (np.abs(gap - np.asarray(y, float)) <= MATCH_SEC)


def classifier_matrix(X, frame):
    return X.assign(**{flag: frame[flag].to_numpy().astype("int8") for flag in FLAGS})


def fit_classifier(X, s, seed, config, desc):
    params = dict(CLASSIFIER_PARAMS, seed=seed, bagging_seed=seed + 1,
                  feature_fraction_seed=seed + 2, num_threads=config.get("threads", 0))
    train = lgb.Dataset(X, s.astype(float), categorical_feature=CATEGORICAL)
    with tqdm(total=CLASSIFIER_ROUNDS, desc=desc, unit="rnd", leave=False) as bar:
        return lgb.train(params, train, num_boost_round=CLASSIFIER_ROUNDS,
                         callbacks=[lambda env: bar.update(env.iteration + 1 - bar.n)])


def fit_components(X, y, frame, seed, rounds, config, desc, stem, outdir):
    """Fit the ordinary regressor and the substitution classifier for one seed."""
    s = substituted(X, y, frame)
    lirf = frame["ADEP_mvt"].to_numpy() == AIRPORT
    if not lirf.any() or s[lirf].all() or not s.any():
        raise ValueError("mixture needs both substituted and ordinary LIRF training rows")
    t0 = time.monotonic()
    history = {}
    ordinary = fit_booster(X.loc[~s], y[~s], seed, rounds, config, f"{desc} ordinary",
                           training_history=history)
    t1 = time.monotonic()
    classifier = fit_classifier(classifier_matrix(X.loc[lirf], frame.loc[lirf]), s[lirf],
                                seed, config, f"{desc} substitution")
    t2 = time.monotonic()
    paths = {"ordinary": Path(f"{stem}.ordinary.txt"),
             "substitution": Path(f"{stem}.substitution.txt")}
    ordinary.save_model(str(paths["ordinary"]))
    classifier.save_model(str(paths["substitution"]))
    loss_path = Path(f"{stem}.ordinary.training_loss.json")
    write_json(loss_path, {"seed": seed, "rounds": rounds, "train_rows": int((~s).sum()),
                           "rmse": history["training"]["rmse"]})
    details = {
        "models": {k: str(v.relative_to(outdir)) for k, v in paths.items()},
        "ordinary_training_loss": str(loss_path.relative_to(outdir)),
        "ordinary_train_rows": int((~s).sum()), "ordinary_training_rmse_final": history["training"]["rmse"][-1],
        "classifier_train_rows": int(lirf.sum()), "classifier_positives": int(s.sum()),
        "ordinary_fit_seconds": round(t1 - t0, 1), "classifier_fit_seconds": round(t2 - t1, 1)}
    return {"ordinary": ordinary, "substitution": classifier}, details


def load_components(details, outdir, threads):
    boosters = {}
    for name, path in details["models"].items():
        boosters[name] = lgb.Booster(model_file=str(Path(outdir) / path))
        boosters[name].params["num_threads"] = threads
    return boosters


def predict_components(boosters, V, frame, base):
    """Saved prediction columns; `base` is the baseline regressor's prediction."""
    base = np.asarray(base, float)
    gap = V[GAP].to_numpy(float)
    airport = frame["ADEP_mvt"].to_numpy()
    lirf = (airport == AIRPORT) & np.isfinite(gap)
    p = np.zeros(len(V))
    if lirf.any():
        rows = V.loc[lirf]
        p[lirf] = boosters["substitution"].predict(classifier_matrix(rows, frame.loc[lirf]))
    ordinary = predict(boosters["ordinary"], V)
    mix = lambda normal: np.where(lirf, np.clip(p * gap + (1 - p) * normal, 0, None), base)
    rule = lirf & frame["raw_flight_id_missing"].to_numpy(bool) & (gap >= RULE_GAP_SEC)
    return {"p_substituted": p, "gap_sec": gap, "pred_ordinary": ordinary,
            "pred_mix_base": mix(base), "pred_mix_ordinary": mix(ordinary),
            "pred_rule6h": np.where(rule, gap, base)}
