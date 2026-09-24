"""Day-blocked domain classification and exact mixed-feature nearest neighbours.

These are population diagnostics, not taxi-out models or validation weights.
All preprocessing is learned from the permitted reference/training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit
from sklearn.metrics import roc_auc_score

GROUPS = {
    'route_fleet': ['dest', 'operator', 'airline', 'aircraft', 'market', 'wake'],
    'allocation': ['stand', 'runway'],
    'clocks': ['no_flight_id', 'takeoff_minus_sched_min', 'takeoff_minus_lobt_min',
               'takeoff_minus_iobt_min', 'takeoff_minus_eobt1_min', 'takeoff_minus_aobt3_min'],
    'schedule': ['hour', 'dow'],
    'weather': ['wx_temp_c', 'wx_dewpoint_spread_c', 'wx_wind_kt', 'wx_gust_kt',
                'wx_visibility_m', 'wx_ceiling_ft', 'wx_snow', 'wx_freezing',
                'wx_fog', 'wx_rain', 'wx_thunder', 'wx_deicing', 'wind_sector'],
    'traffic': ['q_apt_takeoffs_prev30', 'q_rwy_takeoffs_prev30', 'q_rwy_landings_prev30'],
}
RAW_GROUPS = ['route_fleet', 'allocation', 'clocks', 'schedule']
CATEGORICAL = ['airport', *GROUPS['route_fleet'], *GROUPS['allocation'],
               *GROUPS['schedule'], 'wind_sector']
EXCLUDED = {'mvt_id', 'date', 'month', 'year', 'day', 'label',
            'TAXITIME_SEC_mvt', 'BLOCK_TIME_UTC_mvt', 'wx_age_min'}
assert not set(sum(GROUPS.values(), [])) & EXCLUDED
BLOCK_STARTS = [1, 9, 17, 25]
SEED = 42
ROUNDS = 200
MODEL_PARAMS = dict(objective='binary', learning_rate=.05, num_leaves=31,
                    min_data_in_leaf=100, min_data_per_group=100, cat_smooth=10,
                    verbosity=-1, num_threads=6, deterministic=True,
                    force_col_wise=True, seed=SEED)


def roles(dates, fold):
    """Four contiguous date blocks: test=i, calibrate=i+1, other two fit.

    Remove calibration days adjacent to a test day and training days adjacent
    to any test/calibration day. The embargo is global, including month edges.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    unique = dates.unique().sort_values()
    blocks = np.minimum((unique.day.to_numpy()-1)//8, 3)
    test_days = unique[blocks == fold]
    initial_cal = unique[blocks == (fold+1) % 4]
    train_days = unique[(blocks != fold) & (blocks != (fold+1) % 4)]
    near_test = set(test_days) | set(test_days-pd.Timedelta(days=1)) | set(test_days+pd.Timedelta(days=1))
    cal_days = initial_cal[~initial_cal.isin(near_test)]
    held = test_days.union(initial_cal)
    near_held = set(held) | set(held-pd.Timedelta(days=1)) | set(held+pd.Timedelta(days=1))
    train_days = train_days[~train_days.isin(near_held)]
    result = np.full(len(dates), 'embargo', dtype=object)
    for name, days in [('fit', train_days), ('cal', cal_days), ('test', test_days)]:
        result[dates.isin(days)] = name
    assert all(np.any(result == name) for name in ['fit', 'cal', 'test'])
    return result


def balanced_weights(airport, y):
    """Equal airport contribution and equal classes within every airport."""
    f = pd.DataFrame({'airport': np.asarray(airport), 'y': np.asarray(y)})
    counts = f.groupby(['airport', 'y']).y.transform('size').to_numpy()
    assert f.groupby('airport').y.nunique().eq(2).all()
    return len(f)/(2*f.airport.nunique()*counts)


def model_matrix(frame, columns, categories=None):
    out = frame[columns].copy()
    mapping = {} if categories is None else categories
    for c in columns:
        if c in CATEGORICAL:
            values = frame[c].astype('string')
            if c in ['stand', 'runway']:
                values = (frame.airport.astype('string') + ':' + values).where(values.notna())
            if categories is None:
                mapping[c] = sorted(values.dropna().unique().tolist())
            # Unseen categories become missing for LightGBM; the separate kNN
            # representation retains unseen-versus-missing distinctions.
            out[c] = pd.Categorical(values.where(values.isin(mapping[c])), categories=mapping[c])
        else:
            out[c] = pd.to_numeric(frame[c]).astype('float32')
    assert not set(out.columns) & EXCLUDED
    return out, mapping


def fit_calibrator(logits, y, weights):
    """Monotone Platt scaling on separate dates, with balanced diagnostic priors."""
    x = np.asarray(logits, dtype=float)
    y, w = np.asarray(y, dtype=float), np.asarray(weights, dtype=float)
    w = w/w.sum()
    def loss(theta):
        z = theta[0]*x + theta[1]
        error = expit(z)-y
        objective = np.sum(w*(np.logaddexp(0, z)-y*z)) + 1e-6*theta[0]**2
        gradient = np.array([np.sum(w*error*x) + 2e-6*theta[0], np.sum(w*error)])
        return objective, gradient
    result = minimize(loss, [1., 0.], jac=True, method='L-BFGS-B', bounds=[(0., None), (None, None)])
    assert result.success, result.message
    return {'slope': float(result.x[0]), 'intercept': float(result.x[1])}


def calibrated(logits, fit):
    return expit(fit['slope']*np.asarray(logits) + fit['intercept'])


def classification_metrics(y, p, weights=None):
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    bins = np.minimum((np.clip(p, 0, 1)*5).astype(int), 4)
    ece, reliability = 0., []
    for b in range(5):
        keep = bins == b
        if not keep.any():
            continue
        mean_p, observed = np.average(p[keep], weights=w[keep]), np.average(y[keep], weights=w[keep])
        ece += w[keep].sum()/w.sum()*abs(mean_p-observed)
        reliability.append(dict(bin=b, n=int(keep.sum()), mean_probability=float(mean_p), observed_fraction=float(observed)))
    return dict(auc=float(roc_auc_score(y, p, sample_weight=w)),
                brier=float(np.average((p-y)**2, weights=w)), ece=float(ece), reliability=reliability)


def _numeric(frame, cols):
    a = frame[cols].to_numpy(dtype=float, na_value=np.nan, copy=True)
    for j, c in enumerate(cols):
        if c.startswith('takeoff_minus_'):
            a[:,j] = np.sign(a[:,j])*np.log1p(np.abs(a[:,j]))
    a[~np.isfinite(a)] = np.nan
    return a


def fit_distance(reference):
    """Fit group-specific encodings/scales from source donor rows only."""
    result = {}
    for group, columns in GROUPS.items():
        categorical = [c for c in columns if c in CATEGORICAL]
        numeric = [c for c in columns if c not in CATEGORICAL]
        spec = dict(categorical=categorical, numeric=numeric, categories={})
        for c in categorical:
            spec['categories'][c] = {value: i for i, value in enumerate(sorted(reference[c].astype('string').dropna().unique()))}
        if numeric:
            values = _numeric(reference, numeric)
            median, scale = [], []
            for col in values.T:
                known = col[np.isfinite(col)]
                if len(known):
                    p10, p25, p50, p75, p90 = np.quantile(known, [.1,.25,.5,.75,.9])
                    width = p75-p25 if p75-p25 > 1e-8 else p90-p10
                    median.append(p50); scale.append(width if width > 1e-8 else 1.)
                else:
                    median.append(0.); scale.append(1.)
            spec.update(median=np.asarray(median), scale=np.asarray(scale))
        result[group] = spec
    return result


def encode_distance(frame, specs):
    result = {}
    for group, spec in specs.items():
        cat = np.empty((len(frame), len(spec['categorical'])), dtype=float)
        for i, c in enumerate(spec['categorical']):
            v = frame[c].astype('string')
            # Known category >=0; missing=-1; unseen=-2. Donors never have -2.
            cat[:,i] = v.map(spec['categories'][c]).fillna(-2).where(v.notna(), -1).to_numpy(float)
        if spec['numeric']:
            raw = _numeric(frame, spec['numeric'])
            missing = ~np.isfinite(raw)
            z = (np.where(missing, spec['median'], raw)-spec['median'])/spec['scale']
            # Each numeric variable contributes half a scaled squared difference
            # and half a missing-indicator difference; no fitting on queries.
            num = np.c_[z, missing.astype(float)] / np.sqrt(2*len(spec['numeric']))
        else:
            num = np.empty((len(frame), 0))
        result[group] = (cat, num)
    return result


def distance_blocks(reference, queries, chunk_size=256):
    """Exact mixed distances; nominal categories use mismatch, not code magnitude."""
    n = len(next(iter(queries.values()))[0])
    for start in range(0, n, chunk_size):
        stop = min(n, start+chunk_size)
        sums = {}
        for group, (ref_cat, ref_num) in reference.items():
            cat, num = queries[group]
            d = np.zeros((stop-start, len(ref_cat)))
            parts = 0
            if ref_cat.shape[1]:
                d += cdist(cat[start:stop], ref_cat, metric='hamming'); parts += 1
            if ref_num.shape[1]:
                d += cdist(num[start:stop], ref_num, metric='sqeuclidean'); parts += 1
            sums[group] = d/parts
        raw = sum(sums[group] for group in RAW_GROUPS)/len(RAW_GROUPS)
        rich = sum(sums.values())/len(sums)
        yield start, stop, {'raw': raw, 'rich': rich}


def nearest(reference, query, k=10):
    """Distances and reference indices for k neighbours, in deterministic ID order.

    Reference rows must be sorted by movement ID. Stable distance sorting then
    breaks equal-distance ties by ID. Only each chunk's candidates are sorted.
    """
    assert len(reference) >= k
    assert reference.mvt_id.is_monotonic_increasing, 'Sort donors by movement ID for deterministic ties'
    specs = fit_distance(reference)
    ref, qry = encode_distance(reference, specs), encode_distance(query, specs)
    result = {view: {'distance': np.empty((len(query), k)), 'index': np.empty((len(query), k), dtype=np.int32)}
              for view in ['raw', 'rich']}
    for start, stop, distances in distance_blocks(ref, qry):
        for view, matrix in distances.items():
            # Argpartition can choose arbitrary equal-distance neighbours. Use
            # the kth distance, then include ties in reference ID order.
            kth = np.partition(matrix, k-1, axis=1)[:, k-1]
            chosen = np.empty((len(matrix), k), dtype=np.int32)
            for row in range(len(matrix)):
                candidates = np.flatnonzero(matrix[row] <= kth[row])
                order = np.argsort(matrix[row, candidates], kind='stable')[:k]
                chosen[row] = candidates[order]
            result[view]['index'][start:stop] = chosen
            result[view]['distance'][start:stop] = np.sqrt(np.take_along_axis(matrix, chosen, axis=1))
    return result
