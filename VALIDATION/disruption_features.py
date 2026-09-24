"""Past snowfall and taxi-in congestion available before a departure's takeoff.

Callers supply only their permitted arrival months. Departure labels are never read.
"""
import numpy as np
import pandas as pd

import weather

SNOW_FEATURES = [f'dis_snow_{h}h_{s}' for h in (6, 24) for s in ('fraction', 'coverage')]
ARRIVAL_FEATURES = [f'dis_arr_{m}m_{s}' for m in (60, 180)
                    for s in ('count', 'mean_sec', 'over30_share', 'over60_share')]
FEATURES = SNOW_FEATURES + ARRIVAL_FEATURES
SETTINGS = dict(snow_codes=['SN', 'SG'], snow_grid_minutes=30, report_max_age_minutes=90,
                snow_windows_hours=[6, 24], minimum_known_samples=[10, 40],
                arrival_windows_minutes=[60, 180], arrival_mean_cap_sec=7200,
                arrival_available='landing + finite nonnegative taxi_sec < departure takeoff',
                arrival_window='[takeoff - window, takeoff)', features=FEATURES)


def snow_grid(metar):
    reports = metar[['station', 'valid', 'wxcodes']].copy()
    reports['valid'] = weather._utc_naive(reports.valid)
    reports['snow'] = weather._any(weather._codes(reports.wxcodes),
                                   lambda t: 'SN' in t or 'SG' in t).astype(float)
    # Any simultaneous snow report wins; repeated reports cannot change duration.
    reports = reports.groupby(['station', 'valid'], as_index=False).snow.max()
    pieces = []
    for station, group in reports.groupby('station', sort=True):
        group = group.sort_values('valid')
        grid = pd.DataFrame({'sample': pd.date_range(group.valid.min().floor('30min'),
                             group.valid.max().ceil('30min'), freq='30min')})
        grid = pd.merge_asof(grid, group[['valid', 'snow']], left_on='sample',
                             right_on='valid', direction='backward')
        known = grid.valid.notna() & (grid['sample'] - grid.valid).le(weather.MAX_AGE)
        grid.loc[~known, 'snow'] = np.nan
        for hours, minimum in ((6, 10), (24, 40)):
            rolled = grid.snow.rolling(hours * 2, min_periods=minimum)
            grid[f'dis_snow_{hours}h_fraction'] = rolled.mean()
            grid[f'dis_snow_{hours}h_coverage'] = grid.snow.rolling(hours * 2, min_periods=1).count() / (hours * 2)
        grid['station'] = station
        pieces.append(grid[['station', 'sample', *SNOW_FEATURES]])
    return pd.concat(pieces, ignore_index=True)


def snow_features(raw, grid):
    flights = pd.DataFrame({'row': np.arange(len(raw)), 'station': raw.ADEP_mvt.to_numpy(),
                            'takeoff': weather._utc_naive(raw.MVT_TIME_UTC_mvt).to_numpy()})
    if flights.takeoff.isna().any():
        raise ValueError('Missing departure takeoff')
    joined = pd.merge_asof(flights.sort_values('takeoff'), grid.sort_values('sample'),
                           by='station', left_on='takeoff', right_on='sample',
                           direction='backward', tolerance=pd.Timedelta(minutes=30))
    return pd.DataFrame(joined.sort_values('row')[SNOW_FEATURES].to_numpy(),
                        index=raw.index, columns=SNOW_FEATURES).astype('float32')


def arrival_features(raw, arrivals):
    if arrivals.mvt_id.isna().any() or arrivals.mvt_id.duplicated().any():
        raise ValueError('Arrival donors must have unique, present IDs')
    landing = weather._utc_naive(arrivals.MVT_TIME_UTC_mvt)
    taxi = arrivals.taxi_sec.to_numpy(float)
    valid = landing.notna().to_numpy() & np.isfinite(taxi) & (taxi >= 0)
    donors = pd.DataFrame({'airport': arrivals.ADES_mvt.to_numpy()[valid],
                           'taxi': taxi[valid],
                           'completed': landing.loc[valid].to_numpy() + pd.to_timedelta(taxi[valid], unit='s').to_numpy()})
    times = weather._utc_naive(raw.MVT_TIME_UTC_mvt)
    if times.isna().any():
        raise ValueError('Missing departure takeoff')
    out = pd.DataFrame(np.nan, index=np.arange(len(raw)), columns=ARRIVAL_FEATURES)
    for airport in sorted(raw.ADEP_mvt.dropna().unique()):
        rows = np.flatnonzero(raw.ADEP_mvt.eq(airport).to_numpy())
        target = times.iloc[rows].to_numpy(dtype='datetime64[ns]').astype('int64')
        a = donors.loc[donors.airport.eq(airport)].sort_values('completed', kind='stable')
        completed = a.completed.to_numpy(dtype='datetime64[ns]').astype('int64')
        durations = a.taxi.to_numpy(float)
        right = np.searchsorted(completed, target, side='left')  # Strictly completed before takeoff.
        sums = [np.r_[0., np.cumsum(values)] for values in
                (np.minimum(durations, 7200), durations > 1800, durations > 3600)]
        for minutes in (60, 180):
            left = np.searchsorted(completed, target - minutes * 60 * 10**9, side='left')
            n = right - left
            out.loc[rows, f'dis_arr_{minutes}m_count'] = n
            for suffix, cumulative in zip(('mean_sec', 'over30_share', 'over60_share'), sums):
                values = np.divide(cumulative[right] - cumulative[left], n,
                                   out=np.full(len(n), np.nan), where=n > 0)
                out.loc[rows, f'dis_arr_{minutes}m_{suffix}'] = values
    out.index = raw.index
    return out.astype('float32')


def add_features(X, raw, arrivals, grid):
    if not X.index.equals(raw.index) or set(FEATURES) & set(X.columns):
        raise ValueError('Misaligned rows or duplicate disruption features')
    return pd.concat([X, snow_features(raw, grid), arrival_features(raw, arrivals)], axis=1)
