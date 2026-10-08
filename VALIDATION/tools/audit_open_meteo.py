"""Independent checks of cached ERA5 sums and the candidate's flight alignment.

Run after training, so audits do not compete with a fit on the desktop.
"""
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline')]
import external_conditions as wx
import hpo_jan_jul as hpo
import tuning_seasonal as ts
from records import sha256, write_json


def main():
    out = ts.RUNS / 'v3_20261008_open_meteo_jan_jul'
    manifest = json.loads((wx.CACHE/'weather_manifest.json').read_text())
    for r in manifest:
        assert sha256(ROOT/r['path']) == r['sha256']
    # Direct source arrays, no call to feature construction or join functions.
    hourly = {}
    for ap in wx.AIRPORTS:
        h = pd.DataFrame(json.loads((wx.CACHE/'weather'/f'{ap}_2025.json').read_text())['hourly'])
        h.index = pd.to_datetime(h.pop('time'), utc=True)
        hourly[ap] = h
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET TimeZone='UTC'")
    raw = con.execute("SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt, MVT_TIME_UTC_mvt, TAXITIME_SEC_mvt AS y "
                      "FROM read_parquet(?) WHERE PHASE_mvt='DEP'", [str(ROOT/'data/training_2025-*.parquet')]).df().set_index('mvt_id')
    con.close()
    results = {}
    for month in ts.SPLITS:
        path = out/'inputs'/f'{month}.pkl'
        info = ts.read(path.with_suffix('.json'))
        assert sha256(path) == info['input_sha256']
        d = pd.read_pickle(path)
        original = pd.read_pickle(ts.split_dir(hpo.SPLITS_FROM,month)/'after_drop.pkl')
        shared = [c for c in d['baseline'] if c in original['X'].columns]
        pd.testing.assert_frame_equal(d['X'][shared], original['X'].loc[d['X'].index, shared])
        truth = raw.loc[d['train_ids']]
        assert np.array_equal(truth.y.to_numpy(), d['y'])
        assert np.array_equal(truth.ADEP_mvt.to_numpy(), d['X'].airport.astype(str).to_numpy())
        assert set(pd.to_datetime(truth.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')) == set(ts.FOLDS[month])
        assert not set(d['train_ids']) & set(d['ids'])
        n = 0
        max_error = 0.
        for matrix, ids in [(d['X'],d['train_ids']), (d['V'],d['ids'])]:
            truth = raw.loc[ids]
            for ap in wx.AIRPORTS:
                positions = np.flatnonzero(truth.ADEP_mvt.to_numpy()==ap)
                chosen = np.random.default_rng(42).choice(positions, min(32,len(positions)), replace=False)
                h = hourly[ap]
                for i in chosen:
                    stamp = pd.Timestamp(truth.iloc[i].MVT_TIME_UTC_mvt)
                    stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
                    end = stamp.floor('h')
                    k = h.index.get_loc(end)
                    expected = {f'om_{v}':float(h.iloc[k][v]) for v in wx.VARIABLES}
                    for v in ['precipitation','snowfall']:
                        for hours in [6,24,48]:
                            expected[f'om_{v}_{hours}h'] = float(sum(h[v].iloc[k-hours+1:k+1]))
                    actual = matrix.iloc[i][wx.WEATHER].to_numpy(float)
                    e = np.array([expected[c] for c in wx.WEATHER])
                    np.testing.assert_allclose(actual,e,rtol=1e-6,atol=1e-5)
                    max_error = max(max_error,float(np.max(np.abs(actual-e))))
                    n += len(wx.WEATHER)
        results[month] = dict(training_rows=len(d['X']), held_rows=len(d['V']),
            original_columns_unchanged=len(shared), independently_checked_weather_values=n, max_error=max_error)
        del d,original,truth
    write_json(out/'audit.json',dict(passed=True,source_files_verified=len(manifest),checks=results))
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
