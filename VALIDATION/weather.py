"""Weather inputs from METAR reports (external, public-domain IEM archive).

Each departure gets the airport's latest report at or before its takeoff, if
that report is at most 90 minutes old, plus summaries of the reports in the two
hours before it. Only report fields and the visible takeoff time are used, so
the same inputs exist for 2025 training rows and 2026 ranking rows.

Download the reports first: .venv/bin/python VALIDATION/tools/fetch_metar.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METAR = ROOT / 'data' / 'external' / 'metar' / 'metar.parquet'
MAX_AGE = pd.Timedelta(minutes=90)
WINDOW = '2h'
VISIBILITY_CAP_M = 10_000.0
NO_CEILING_FT = 25_000.0

CURRENT_FEATURES = [
    'wx_temp_c', 'wx_dewpoint_spread_c', 'wx_wind_kt', 'wx_gust_kt', 'wx_wind_dir',
    'wx_visibility_m', 'wx_ceiling_ft', 'wx_thunder', 'wx_snow', 'wx_freezing',
    'wx_fog', 'wx_rain', 'wx_heavy', 'wx_deicing', 'wx_age_min',
]
WINDOW_FEATURES = [
    'wx_2h_min_visibility_m', 'wx_2h_min_ceiling_ft', 'wx_2h_max_gust_kt',
    'wx_2h_thunder', 'wx_2h_snow_or_freezing', 'wx_2h_fog',
]
WEATHER_FEATURES = CURRENT_FEATURES + WINDOW_FEATURES
SETTINGS = dict(source='Iowa Environmental Mesonet ASOS/METAR archive', max_age_min=90,
                window=WINDOW, visibility_cap_m=VISIBILITY_CAP_M, no_ceiling_ft=NO_CEILING_FT,
                features=WEATHER_FEATURES)


def _utc_naive(values):
    stamp = pd.to_datetime(values)
    if getattr(stamp.dt, 'tz', None) is not None:
        stamp = stamp.dt.tz_convert('UTC').dt.tz_localize(None)
    return stamp.astype('datetime64[ns]')


def _codes(wxcodes):
    """Present-weather tokens without intensity signs, e.g. '-SN FZFG' -> ['SN', 'FZFG']."""
    return wxcodes.fillna('').str.replace(r'[+-]', '', regex=True).str.split()


def _any(tokens, test):
    return tokens.map(lambda items: any(test(item) for item in items)).astype('int8')


def report_table(metar):
    """One row per report with physical units and weather flags."""
    frame = metar.drop_duplicates(['station', 'valid', 'metar']).copy()
    out = pd.DataFrame({'station': frame['station'].to_numpy(), 'valid': _utc_naive(frame['valid']).to_numpy()})
    temp = (frame['tmpf'].to_numpy(float) - 32) * 5 / 9
    dew = (frame['dwpf'].to_numpy(float) - 32) * 5 / 9
    out['wx_temp_c'] = temp
    out['wx_dewpoint_spread_c'] = temp - dew
    wind = frame['sknt'].to_numpy(float)
    out['wx_wind_kt'] = wind
    out['wx_gust_kt'] = np.where(np.isfinite(frame['gust'].to_numpy(float)), frame['gust'].to_numpy(float), wind)
    out['wx_wind_dir'] = frame['drct'].to_numpy(float)
    visibility = np.minimum(frame['vsby'].to_numpy(float) * 1609.344, VISIBILITY_CAP_M)
    cavok = frame['metar'].fillna('').str.contains('CAVOK').to_numpy()
    out['wx_visibility_m'] = np.where(np.isnan(visibility) & cavok, VISIBILITY_CAP_M, visibility)
    ceiling = np.full(len(frame), NO_CEILING_FT)
    for layer in range(1, 5):
        cover = frame[f'skyc{layer}'].fillna('').str.strip().to_numpy()
        base = frame[f'skyl{layer}'].to_numpy(float)
        base = np.where((cover == 'VV') & np.isnan(base), 0.0, base)
        closes = np.isin(cover, ['BKN', 'OVC', 'VV']) & np.isfinite(base)
        ceiling = np.where(closes, np.minimum(ceiling, base), ceiling)
    has_sky = frame[[f'skyc{layer}' for layer in range(1, 5)]].notna().any(axis=1).to_numpy() | cavok
    out['wx_ceiling_ft'] = np.where(has_sky, ceiling, np.nan)
    tokens = _codes(frame['wxcodes']).reset_index(drop=True)
    raw_codes = frame['wxcodes'].fillna('').reset_index(drop=True)
    out['wx_thunder'] = _any(tokens, lambda t: 'TS' in t).to_numpy()
    out['wx_snow'] = _any(tokens, lambda t: any(k in t for k in ('SN', 'SG', 'PL', 'GS'))).to_numpy()
    out['wx_freezing'] = _any(tokens, lambda t: 'FZ' in t).to_numpy()
    out['wx_fog'] = _any(tokens, lambda t: t in ('FG', 'FZFG')).to_numpy()
    out['wx_rain'] = _any(tokens, lambda t: 'RA' in t or 'DZ' in t).to_numpy()
    out['wx_heavy'] = raw_codes.str.contains(r'(?:^|\s)\+', regex=True).astype('int8').to_numpy()
    moist = ((out['wx_snow'] + out['wx_freezing'] + out['wx_fog'] + out['wx_rain']) > 0) | (out['wx_dewpoint_spread_c'] <= 1)
    out['wx_deicing'] = ((out['wx_temp_c'] <= 3) & moist).astype('int8')
    out['wx_snow_or_freezing'] = ((out['wx_snow'] + out['wx_freezing']) > 0).astype('int8')
    return out.sort_values(['station', 'valid'], kind='stable').reset_index(drop=True)


def _with_window(reports):
    parts = []
    for _, group in reports.groupby('station', sort=False):
        rolled = group.set_index('valid').rolling(WINDOW, closed='both')
        group = group.copy()
        group['wx_2h_min_visibility_m'] = rolled['wx_visibility_m'].min().to_numpy()
        group['wx_2h_min_ceiling_ft'] = rolled['wx_ceiling_ft'].min().to_numpy()
        group['wx_2h_max_gust_kt'] = rolled['wx_gust_kt'].max().to_numpy()
        group['wx_2h_thunder'] = rolled['wx_thunder'].max().to_numpy()
        group['wx_2h_snow_or_freezing'] = rolled['wx_snow_or_freezing'].max().to_numpy()
        group['wx_2h_fog'] = rolled['wx_fog'].max().to_numpy()
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def load_reports(path=METAR):
    return _with_window(report_table(pd.read_parquet(path)))


def weather_features(raw, reports):
    """Weather inputs aligned to `raw` (needs ADEP_mvt and MVT_TIME_UTC_mvt)."""
    flights = pd.DataFrame({'row': np.arange(len(raw)), 'station': raw['ADEP_mvt'].to_numpy(),
                            'takeoff': _utc_naive(raw['MVT_TIME_UTC_mvt']).to_numpy()})
    known = flights['takeoff'].notna()
    joined = pd.merge_asof(flights.loc[known].sort_values('takeoff'), reports.sort_values('valid'),
                           left_on='takeoff', right_on='valid', by='station', direction='backward')
    age = joined['takeoff'] - joined['valid']
    joined['wx_age_min'] = age.dt.total_seconds() / 60
    stale = ~(age <= MAX_AGE)
    joined.loc[stale, WEATHER_FEATURES] = np.nan
    out = pd.DataFrame(np.nan, index=np.arange(len(raw)), columns=WEATHER_FEATURES)
    out.loc[joined['row'].to_numpy(), WEATHER_FEATURES] = joined[WEATHER_FEATURES].to_numpy(float)
    out.index = raw.index
    return out.astype('float32')


def add_weather(X, raw, reports):
    if not X.index.equals(raw.index):
        raise ValueError('Matrix and raw movement rows must align before adding weather')
    overlap = set(X.columns) & set(WEATHER_FEATURES)
    if overlap:
        raise ValueError(f'Weather inputs already present: {sorted(overlap)}')
    return pd.concat([X, weather_features(raw, reports)], axis=1)
