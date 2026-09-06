"""Seeded experiment behind viz/ch07-prc-validation.html.

Synthetic taxi-out world shaped like the PRC Data Challenge 2026:
train = calendar 2025, test = Jan + Jul 2026, metric = RMSE (seconds).
Mechanisms baked in:
  - operator effects drift month to month  -> rolling operator means help, but decay when stale
  - shared per-airport-day congestion      -> same-day features leak under random K-fold
  - smooth seasonality + rush-hour peaks

Pipelines compared:
  naive : rolling features computed "as of today" everywhere, random 5-fold CV
  honest: training rows see only yesterday and earlier; validation/test rows get
          features frozen at the training cutoff (what deployment actually allows)

Prints a JSON blob consumed by the HTML page.
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

rng = np.random.default_rng(42)

# ---------------------------------------------------------------- world
dates = pd.date_range("2025-01-01", "2026-07-31", freq="D")
n_days = len(dates)
A, O, PER_DAY = 6, 30, 35
month_of_day = ((dates.year - 2025) * 12 + dates.month - 1).to_numpy()  # 0..18
cal_month = dates.month.to_numpy()
dow_of_day = dates.dayofweek.to_numpy()

op_effect = rng.normal(0, 60, O)[:, None] + np.cumsum(rng.normal(0, 25, (O, 19)), axis=1)
airport_base = rng.uniform(-150, 150, A)
day_cong = rng.normal(0, 90, (A, n_days))

parts = []
for a in range(A):
    ops_a = rng.choice(O, 12, replace=False)
    w = rng.dirichlet(np.ones(12) * 2)
    n = PER_DAY * n_days
    day_ix = np.repeat(np.arange(n_days), PER_DAY)
    op = rng.choice(ops_a, n, p=w)
    morning = rng.random(n) < 0.5
    hour = np.clip(
        np.where(morning, rng.normal(8, 2, n), rng.normal(17, 3, n)).round(), 5, 23
    ).astype(int)
    season = 50 * np.cos(2 * np.pi * (cal_month[day_ix] - 7) / 12)
    hour_eff = 40 * np.exp(-(((hour - 8) / 2.5) ** 2)) + 45 * np.exp(-(((hour - 18) / 3) ** 2))
    y = (
        600.0
        + airport_base[a]
        + op_effect[op, month_of_day[day_ix]]
        + season
        + day_cong[a, day_ix]
        + hour_eff
        + rng.normal(0, 120, n)
    )
    parts.append(
        pd.DataFrame(
            dict(airport=a, op=op, day=day_ix, hour=hour, y=np.maximum(y, 120.0))
        )
    )
df = pd.concat(parts, ignore_index=True)
df["month"] = cal_month[df["day"]]
df["dow"] = dow_of_day[df["day"]]

# ------------------------------------------- day-level aggregate tables
op_sum = np.zeros((O, n_days)); op_cnt = np.zeros((O, n_days))
ap_sum = np.zeros((A, n_days)); ap_cnt = np.zeros((A, n_days))
np.add.at(op_sum, (df["op"], df["day"]), df["y"]); np.add.at(op_cnt, (df["op"], df["day"]), 1)
np.add.at(ap_sum, (df["airport"], df["day"]), df["y"]); np.add.at(ap_cnt, (df["airport"], df["day"]), 1)
op_cs = np.hstack([np.zeros((O, 1)), np.cumsum(op_sum, 1)])
op_cc = np.hstack([np.zeros((O, 1)), np.cumsum(op_cnt, 1)])

def day_index(datestr):
    return int(dates.get_loc(pd.Timestamp(datestr)))

def features(sub, cutoff):
    """Feature matrix; `cutoff` = last day whose taxi-out actuals are visible.
    'today' -> row's own day (self excluded), 'causal' -> row's day - 1,
    int -> frozen at that day index."""
    d = sub["day"].to_numpy()
    if isinstance(cutoff, str):
        cut = d if cutoff == "today" else d - 1
    else:
        cut = np.full(len(sub), cutoff)
    lo = np.maximum(cut - 29, 0)
    s = op_cs[sub["op"], cut + 1] - op_cs[sub["op"], lo]
    c = op_cc[sub["op"], cut + 1] - op_cc[sub["op"], lo]
    ap_s = ap_sum[sub["airport"], np.maximum(cut, 0)].astype(float)
    ap_c = ap_cnt[sub["airport"], np.maximum(cut, 0)].astype(float)
    own_day = cut >= d
    s = s - np.where(own_day, sub["y"], 0); c = c - own_day
    ap_s = ap_s - np.where(own_day, sub["y"], 0); ap_c = ap_c - own_day
    with np.errstate(invalid="ignore", divide="ignore"):
        op30 = np.where(c > 0, s / c, np.nan)
        ap1 = np.where(ap_c > 0, ap_s / ap_c, np.nan)
    X = np.column_stack(
        [sub["airport"], sub["op"], sub["hour"], sub["dow"], sub["month"], op30, ap1]
    )
    return np.where(np.isnan(X), 660.0, X)

def fit(X, y):
    m = HistGradientBoostingRegressor(
        max_iter=250, categorical_features=[0, 1], random_state=0
    )
    m.fit(X, y)
    return m

def rmse(model, X, y):
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))

D = {k: day_index(k) for k in
     ["2025-01-31", "2025-06-30", "2025-11-30", "2025-12-31",
      "2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30",
      "2026-05-31", "2026-06-30"]}
tr25 = df[df["day"] <= D["2025-12-31"]]
jan26 = df[(df["day"] > D["2025-12-31"]) & (df["day"] <= D["2026-01-31"])]
jul26 = df[df["day"] > D["2026-06-30"]]
res = {"n_train": len(tr25), "n_jan": len(jan26), "n_jul": len(jul26)}

# ------------------------------------------------- naive pipeline
Xn, yn = features(tr25, "today"), tr25["y"].to_numpy()
cv = []
for tr_ix, va_ix in KFold(5, shuffle=True, random_state=0).split(Xn):
    cv.append(rmse(fit(Xn[tr_ix], yn[tr_ix]), Xn[va_ix], yn[va_ix]))
res["naive_cv"] = float(np.mean(cv))
naive_full = fit(Xn, yn)
res["naive_jan"] = rmse(naive_full, features(jan26, D["2025-12-31"]), jan26["y"])
res["naive_jul"] = rmse(naive_full, features(jul26, D["2025-12-31"]), jul26["y"])

# ------------------------------------------------- honest harness
tr_nov = df[df["day"] <= D["2025-11-30"]]
dec25 = df[(df["day"] > D["2025-11-30"]) & (df["day"] <= D["2025-12-31"])]
res["honest_dec_fold"] = rmse(
    fit(features(tr_nov, "causal"), tr_nov["y"]),
    features(dec25, D["2025-11-30"]), dec25["y"])

tr_jun = df[df["day"] <= D["2025-06-30"]]
jul25 = df[(df["day"] > D["2025-06-30"]) & (df["day"] <= day_index("2025-07-31"))]
res["honest_jul_fold"] = rmse(
    fit(features(tr_jun, "causal"), tr_jun["y"]),
    features(jul25, D["2025-01-31"]), jul25["y"])

honest_full = fit(features(tr25, "causal"), tr25["y"].to_numpy())
res["honest_jan"] = rmse(honest_full, features(jan26, D["2025-12-31"]), jan26["y"])
res["honest_jul"] = rmse(honest_full, features(jul26, D["2025-12-31"]), jul26["y"])

# July model trained with staleness-matched (180-day lagged) features
lag_cut = np.maximum(tr25["day"].to_numpy() - 180, -1)
Xlag = features(tr25, 0)  # shape/base; overwrite rolling cols with lagged cutoffs
sub = tr25
lo = np.maximum(lag_cut - 29, 0)
s = op_cs[sub["op"], lag_cut + 1] - op_cs[sub["op"], lo]
c = op_cc[sub["op"], lag_cut + 1] - op_cc[sub["op"], lo]
ap_s = ap_sum[sub["airport"], np.maximum(lag_cut, 0)]
ap_c = ap_cnt[sub["airport"], np.maximum(lag_cut, 0)]
with np.errstate(invalid="ignore", divide="ignore"):
    Xlag[:, 5] = np.where(c > 0, s / c, 660.0)
    Xlag[:, 6] = np.where((ap_c > 0) & (lag_cut >= 0), ap_s / np.maximum(ap_c, 1), 660.0)
matched = fit(Xlag, tr25["y"].to_numpy())
res["matched_jul"] = rmse(matched, features(jul26, D["2025-12-31"]), jul26["y"])

# ------------------------------------------------- staleness curve (Jul 2026)
# Isolate the operator-rolling feature: the same-day airport feature is pinned
# to its fill value at every point, so only op30's cutoff moves.
def feat_op_only(sub, cutoff):
    X = features(sub, cutoff)
    X[:, 6] = 660.0
    return X

curve = []
for label in ["2025-12-31", "2026-01-31", "2026-02-28", "2026-03-31",
              "2026-04-30", "2026-05-31", "2026-06-30"]:
    curve.append({"freeze": label,
                  "rmse": rmse(honest_full, feat_op_only(jul26, D[label]), jul26["y"])})
curve.append({"freeze": "fresh",
              "rmse": rmse(honest_full, feat_op_only(jul26, "causal"), jul26["y"])})
res["staleness"] = curve

json.dump({k: (round(v, 1) if isinstance(v, float) else v) for k, v in res.items()},
          sys.stdout, indent=1, default=str)
print()
