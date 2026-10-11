"""Public EUROCONTROL daily punctuality, separate from PRU delay totals.

Source: https://www.eurocontrol.int/Economics/DailyPunctuality-Airports.html
The dashboard directly links pun_apt_data.json. It uses airline schedules;
population and timestamp differences prohibit treating it as an exact identity.
Features use public values and supplied departure clocks only, never targets.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import public_delay_features as daily
from records import write_json

ROOT = daily.base.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261011_public_punctuality'
SOURCE = daily.base.SOURCE / 'pun_apt_data.json'
SOURCE_SHA256 = 'c239cd2fc7ae8b274fad56ab100d7bf4401e07bd238e4f7e8d91f4048318e4f1'
AIRPORTS = {'Amsterdam': 'EHAM', 'Barcelona': 'LEBL', 'Frankfurt': 'EDDF',
    'Istanbul': 'LTFM', 'London Heathrow': 'EGLL', 'Madrid Barajas': 'LEMD',
    'Munich': 'EDDM', 'Paris Charles de Gaulle': 'LFPG',
    'Rome Fiumicino': 'LIRF', 'Zurich': 'LSZH'}
DIRECT = {'Departure Punctuality %': 'pun_dep_otp15',
    'Arrival Punctuality %': 'pun_arr_otp15',
    'Avg Departure Schedule Delay': 'pun_dep_mean',
    'Avg Arrival Schedule Delay': 'pun_arr_mean',
    'Operated Schedules %': 'pun_operated'}
FEATURES = list(DIRECT.values()) + ['pun_otp900_resid', 'pun_otp1800_resid',
    'pun_mean900_resid', 'pun_clock_mean_resid']


def load():
    assert daily.base.sha256(SOURCE) == SOURCE_SHA256
    raw = json.loads(SOURCE.read_text())
    assert raw[0] == ['Date', 'Airport', 'Departure Punctuality %',
        'Arrival Punctuality %', 'Avg Departure Schedule Delay',
        'Avg Arrival Schedule Delay', 'Avg Departure - Arrival Schedule Delay',
        'Operated Schedules %']
    assert raw[1] == ['date', 'string'] + ['number'] * 6
    f = pd.DataFrame(raw[2:], columns=raw[0])
    f['airport'] = f.Airport.map(AIRPORTS)
    f = f[f.airport.notna()].rename(columns={'Date': 'day', **DIRECT})
    for name in DIRECT.values():
        f[name] = pd.to_numeric(f[name], errors='raise')
    for name in ['pun_dep_mean', 'pun_arr_mean']:
        f[name] *= 60
    assert not f.duplicated(['airport', 'day']).any()
    assert np.isfinite(f[list(DIRECT.values())]).all().all()
    assert f[['pun_dep_otp15', 'pun_arr_otp15', 'pun_operated']].ge(0).all().all()
    assert f[['pun_dep_otp15', 'pun_arr_otp15', 'pun_operated']].le(1).all().all()
    return f[['airport', 'day'] + list(DIRECT.values())]


def build(rows, source):
    f = rows[['mvt_id', 'airport', 'day', 'gap', 'clock', 'fltrule']].copy()
    eligible = ~f.fltrule.eq('V')
    clock = np.clip(f.clock.fillna(1200).to_numpy(float), 30, 7200)
    f['otp900'] = (f.gap <= 1800).astype(float)
    f['otp1800'] = (f.gap <= 2700).astype(float)
    f['mean900'] = f.gap-900
    f['clock_mean'] = f.gap-clock
    columns = ['otp900', 'otp1800', 'mean900', 'clock_mean']
    a = f.loc[eligible].groupby(['airport', 'day'], as_index=False)[columns].mean()
    g = f[['mvt_id', 'airport', 'day']].merge(a, on=['airport', 'day'],
        how='left', validate='many_to_one').merge(source, on=['airport', 'day'],
        how='left', validate='many_to_one')
    g['pun_otp900_resid'] = g.pun_dep_otp15-g.otp900
    g['pun_otp1800_resid'] = g.pun_dep_otp15-g.otp1800
    g['pun_mean900_resid'] = g.pun_dep_mean-g.mean900
    g['pun_clock_mean_resid'] = g.pun_dep_mean-g.clock_mean
    out = g[['mvt_id'] + FEATURES]
    assert len(out) == len(rows) and out.mvt_id.is_unique
    assert np.isfinite(out[FEATURES]).all().all(), 'Missing source/eligible population'
    return out


def checks():
    rows = pd.DataFrame(dict(mvt_id=[1, 2, 3], airport='LIRF', day='2025-01-01',
        gap=[1000., 5000., 8000.], clock=[700., 800., 900.], fltrule=['I', None, 'V']))
    source = pd.DataFrame(dict(airport=['LIRF'], day=['2025-01-01'],
        pun_dep_otp15=.7, pun_arr_otp15=.8, pun_dep_mean=900., pun_arr_mean=600., pun_operated=.9))
    f = build(rows, source)
    np.testing.assert_allclose(f.pun_otp900_resid, .2)
    np.testing.assert_allclose(f.pun_otp1800_resid, .2)
    np.testing.assert_allclose(f.pun_mean900_resid, -1200.)
    np.testing.assert_allclose(f.pun_clock_mean_resid, -1350.)
    pd.testing.assert_frame_equal(f, build(rows.assign(y=1e12, BLOCK_TIME_UTC_mvt='2099-01-01'), source))
    pd.testing.assert_frame_equal(f, build(rows.iloc[::-1], source).sort_values('mvt_id').reset_index(drop=True))
    return dict(hand_arithmetic=True, explicit_vfr_excluded=True, unknown_included=True,
        target_isolation=True, row_order_invariant=True)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    identity = dict(source=daily.base.artifact(SOURCE), builder=daily.base.artifact(__file__),
        inputs=daily.base.artifact(daily.OUT / 'inputs2025.parquet'), features=FEATURES,
        mapping=AIRPORTS, checks=checks())
    manifest = OUT / 'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        assert all(saved[k] == v for k, v in identity.items())
        assert saved['output'] == daily.base.artifact(OUT / 'features2025.parquet')
        return saved
    source = load()
    rows = pd.read_parquet(daily.OUT / 'inputs2025.parquet')
    f = build(rows, source)
    assert len(f) == 2085047
    f.to_parquet(OUT / 'features2025.parquet', index=False)
    source.to_parquet(OUT / 'source.parquet', index=False)
    saved = dict(identity, rows=len(f), airport_days=int(rows[['airport', 'day']].drop_duplicates().shape[0]),
        coverage_2026=int(source.day.between('2026-01-01', '2026-07-31').sum()),
        output=daily.base.artifact(OUT / 'features2025.parquet'))
    write_json(manifest, saved)
    return saved


if __name__ == '__main__':
    print(prepare(), flush=True)
