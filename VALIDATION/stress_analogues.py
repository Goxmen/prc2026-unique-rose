"""Bounded observable-input matching; there is no forced nearest neighbour."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import weather

SNOW = ['dis_snow_6h_fraction', 'dis_snow_24h_fraction']
COUNTS = [f'dis_arr_{m}m_count' for m in (60, 180)]
MEANS = [f'dis_arr_{m}m_mean_sec' for m in (60, 180)]
SHARES30 = [f'dis_arr_{m}m_over30_share' for m in (60, 180)]
SHARES60 = [f'dis_arr_{m}m_over60_share' for m in (60, 180)]
RAW_MEANS = [f'stress_arr_{m}m_uncapped_mean_sec' for m in (60, 180)]
SHARES120 = [f'stress_arr_{m}m_over120_share' for m in (60, 180)]
DIAGNOSTICS = RAW_MEANS + SHARES120
MATCH_COLUMNS = SNOW + COUNTS + MEANS + SHARES30 + SHARES60 + DIAGNOSTICS
SEVERITY = SNOW + MEANS + SHARES30 + SHARES60 + DIAGNOSTICS
COHORTS = ['prolonged_snow', 'heavy_arrival_congestion', 'snow_and_congestion']
GROUP = ['airport', 'raw_flight_id_missing', 'gap_band']
ROUNDING_TOLERANCE = 1e-6  # Fixed float32 model-input rounding, not an adaptive caliper.


def gap_band(seconds):
    values = np.asarray(seconds, float)
    out = np.zeros(len(values), dtype='int8')
    known = np.isfinite(values)
    out[known] = np.searchsorted([0, 900, 1800, 3600, 7200, 14400, 43200], values[known], side='right') + 1
    return out


def cohort(frame, name):
    snow = frame.dis_snow_24h_fraction.gt(.25)
    heavy = ((frame.dis_arr_60m_count.ge(5) & frame.dis_arr_60m_mean_sec.ge(1800))
             | (frame.dis_arr_180m_count.ge(10) & frame.dis_arr_180m_mean_sec.ge(1800)))
    return {'prolonged_snow': snow, 'heavy_arrival_congestion': heavy,
            'snow_and_congestion': snow & heavy}[name]


def coordinates(frame):
    """Chebyshev radius one is the intersection of all fixed matching limits."""
    values = frame[MATCH_COLUMNS].to_numpy(float).copy()
    values[:, :2] *= [6., 24.]
    values[:, 2:4] = np.log1p(values[:, 2:4]) / np.log(1.5)
    values[:, 4:6] = np.log(np.maximum(values[:, 4:6], 60)) / np.log(1.25)
    values[:, 6:8] /= .10
    values[:, 8:10] /= .05
    values[:, 10:12] = np.log(np.maximum(values[:, 10:12], 60)) / np.log(1.25)
    values[:, 12:14] /= .025
    return values


def arrival_tail_checks(raw, arrivals):
    """Uncapped checks for matching only; frozen model features stay unchanged."""
    if arrivals.mvt_id.isna().any() or arrivals.mvt_id.duplicated().any():
        raise ValueError('Arrival donors must have unique nonmissing IDs')
    landed = weather._utc_naive(arrivals.MVT_TIME_UTC_mvt)
    taxi = arrivals.taxi_sec.to_numpy(float)
    valid = landed.notna().to_numpy() & np.isfinite(taxi) & (taxi >= 0)
    donors = pd.DataFrame({'airport': arrivals.ADES_mvt.to_numpy()[valid], 'taxi': taxi[valid],
        'completed': landed.loc[valid].to_numpy() + pd.to_timedelta(taxi[valid], unit='s').to_numpy()})
    out = pd.DataFrame(np.nan, index=np.arange(len(raw)), columns=DIAGNOSTICS)
    takeoff = weather._utc_naive(raw.MVT_TIME_UTC_mvt)
    for airport in sorted(raw.ADEP_mvt.dropna().unique()):
        indices = np.flatnonzero(raw.ADEP_mvt.eq(airport).to_numpy())
        times = takeoff.iloc[indices].to_numpy(dtype='datetime64[ns]').astype('int64')
        group = donors.loc[donors.airport.eq(airport)].sort_values('completed', kind='stable')
        completed = group.completed.to_numpy(dtype='datetime64[ns]').astype('int64')
        values = group.taxi.to_numpy(float)
        cumulative = [np.r_[0., np.cumsum(v)] for v in (values, values > 7200)]
        right = np.searchsorted(completed, times, side='left')
        for minutes in (60, 180):
            left = np.searchsorted(completed, times - minutes * 60 * 10**9, side='left')
            n = right - left
            for suffix, total in zip(('uncapped_mean_sec', 'over120_share'), cumulative):
                out.loc[indices, f'stress_arr_{minutes}m_{suffix}'] = np.divide(
                    total[right] - total[left], n, out=np.full(len(n), np.nan), where=n > 0)
    out.index = raw.index
    return out


def match_targets(source, target, name, neighbours=10):
    """Return accepted edges and a reason for every target in the stress group.

    Source and target need metadata and MATCH_COLUMNS only. Departure outcomes
    and predictions are never inspected. Maxima are within airport/cohort;
    the tree also requires the same missing-ID and schedule-gap class.
    """
    for frame in (source, target):
        if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
            raise ValueError('Expected unique nonmissing movement IDs')
    s = source.loc[cohort(source, name)].sort_values('mvt_id').copy().reset_index(drop=True)
    t = target.loc[cohort(target, name)].sort_values('mvt_id').copy().reset_index(drop=True)
    out = t[['mvt_id', 'airport', 'mvt_time']].copy()
    out['reason'] = 'missing_measurements'
    valid_s = np.isfinite(s[MATCH_COLUMNS].to_numpy(float)).all(axis=1)
    valid_t = np.isfinite(t[MATCH_COLUMNS].to_numpy(float)).all(axis=1)
    s = s.loc[valid_s].reset_index(drop=True)
    out.loc[valid_t, 'reason'] = 'no_historical_cohort'
    for airport in t.airport.unique():
        si = s.airport.eq(airport)
        ti = t.airport.eq(airport).to_numpy() & valid_t
        if not si.any() or not ti.any():
            continue
        maximum = s.loc[si, SEVERITY].max().to_numpy()
        beyond = (t.loc[ti, SEVERITY].to_numpy() > maximum + 1e-7).any(axis=1)
        indices = np.flatnonzero(ti)
        out.loc[indices, 'reason'] = 'no_match_within_limits'
        out.loc[indices[beyond], 'reason'] = 'beyond_observed_severity'
    edges = []
    target_groups = t.loc[out.reason.eq('no_match_within_limits')].groupby(GROUP, sort=True).groups
    source_groups = s.groupby(GROUP, sort=True).groups
    for group, indices in target_groups.items():
        if group not in source_groups:
            continue
        historical = s.loc[source_groups[group]]
        current = t.loc[indices]
        tree = cKDTree(coordinates(historical))
        distance, position = tree.query(coordinates(current), k=neighbours, p=np.inf,
                                        distance_upper_bound=1 + ROUNDING_TOLERANCE, workers=1)
        distance, position = np.asarray(distance).reshape(len(current), neighbours), np.asarray(position).reshape(len(current), neighbours)
        accepted = np.isfinite(distance) & (position < len(historical))
        row, column = np.nonzero(accepted)
        if not len(row):
            continue
        source_ids = historical.mvt_id.to_numpy()[position[row, column]]
        target_ids = current.mvt_id.to_numpy()[row]
        edges.append(pd.DataFrame(dict(target_id=target_ids, source_id=source_ids,
                                        distance=distance[row, column], cohort=name)))
        out.loc[current.index[np.unique(row)], 'reason'] = 'matched'
    pairs = pd.concat(edges, ignore_index=True) if edges else pd.DataFrame(columns=['target_id', 'source_id', 'distance', 'cohort'])
    return pairs, out
