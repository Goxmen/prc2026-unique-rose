"""Audit v61's full-year data, raw weather alignment and saved model inputs.

Read-only with respect to models, predictions and submissions; writes one report.
No fitting, downloading or uploading. Uses all rows for weather-value checks.
"""
import gc
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import external_conditions as wx
from movement_data import load_phase
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261008_open_meteo_submit'
WORK = OUT / 'work'


def weather_check(raw, matrix):
    """Direct raw-array lookup and cumulative sums, independent of weather_join."""
    max_error, count = 0., 0
    for ap in wx.AIRPORTS:
        frames = []
        for path in sorted((wx.CACHE / 'weather').glob(f'{ap}_*.json')):
            h = pd.DataFrame(json.loads(path.read_text())['hourly'])
            h.index = pd.to_datetime(h.pop('time'), utc=True)
            frames.append(h)
        h = pd.concat(frames).sort_index()
        duplicate = h.index.duplicated(keep=False)
        assert h.loc[duplicate].groupby(level=0).nunique().le(1).all().all()
        h = h.loc[~h.index.duplicated()]
        pos = np.flatnonzero(raw.ADEP_mvt.to_numpy() == ap)
        stamp = pd.DatetimeIndex(pd.to_datetime(raw.iloc[pos].MVT_TIME_UTC_mvt, utc=True)).floor('h')
        k = h.index.get_indexer(stamp)
        assert (k >= 47).all()
        assert ((h.index[k] - h.index[k - 47]) == pd.Timedelta(hours=47)).all()
        expected = {'om_' + c: h[c].to_numpy()[k] for c in wx.VARIABLES}
        for c in ['precipitation', 'snowfall']:
            cumulative = np.r_[0., np.cumsum(h[c].to_numpy(dtype=float))]
            for hours in [6, 24, 48]:
                expected[f'om_{c}_{hours}h'] = cumulative[k + 1] - cumulative[k + 1 - hours]
        for c in wx.WEATHER:
            actual = matrix[c].to_numpy(dtype=float)[pos]
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected[c], rtol=1e-6, atol=1e-5)
            max_error = max(max_error, float(np.max(np.abs(actual - expected[c]), initial=0)))
            count += len(pos)
    return dict(rows=len(raw), values=count, missing=0, max_error=max_error)


def main():
    prepared = json.loads((WORK / 'prepared.json').read_text())
    original = ROOT / 'VALIDATION/runs/v3_20261003_ideas_submit/work/inputs.pkl'
    for p, key in [(WORK / 'inputs.pkl', 'inputs_sha256'), (WORK / 'rome.pkl', 'rome_sha256'),
                   (original, 'original_inputs_sha256')]:
        assert sha256(p) == prepared[key]
    manifest = json.loads((wx.CACHE / 'weather_manifest.json').read_text())
    for item in manifest:
        assert sha256(ROOT / item['path']) == item['sha256']
    d, old = pd.read_pickle(WORK / 'inputs.pkl'), pd.read_pickle(original)
    for part in ['Xn', 'Vn']:
        assert d[part].index.equals(old[part].index)
        for c in old[part]:
            assert d[part][c].equals(old[part][c]), (part, c)
    for key in ['yn', 'tmonth', 'rank_ids']:
        np.testing.assert_array_equal(d[key], old[key])
    report = dict(original_columns_unchanged=len(old['Xn'].columns), source_files_verified=len(manifest))
    del old
    gc.collect()
    con = duckdb.connect()
    con.execute('SET threads=2')
    con.execute("SET TimeZone='UTC'")
    dep = load_phase(con, str(ROOT / 'data/training_2025-*.parquet'), 'DEP')
    phase_counts = con.execute("SELECT PHASE_mvt, count(*) FROM read_parquet(?) GROUP BY 1",
                               [str(ROOT / 'data/training_2025-*.parquet')]).fetchall()
    con.close()
    raw = dep.loc[d['Xn'].index]
    np.testing.assert_array_equal(raw.taxi_sec, d['yn'])
    np.testing.assert_array_equal(raw.ADEP_mvt, d['Xn'].airport.astype(str))
    assert not np.intersect1d(raw.mvt_id, d['rank_ids']).size
    months = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m')
    np.testing.assert_array_equal(months, d['tmonth'])
    assert set(months) == {f'2025-{m:02}' for m in range(1, 13)}
    rome = pd.read_pickle(WORK / 'rome.pkl')
    r = dep.loc[rome['X'].index]
    assert r.ADEP_mvt.eq('LIRF').all()
    from opdi_run import rule_rows
    cfg = d['v46_identity']['config']
    valid = dep.loc[dep.taxi_sec.between(1, cfg['max_taxi_sec'])]
    excluded = rule_rows(valid, cfg['prediction_rule'])
    np.testing.assert_array_equal(np.union1d(d['Xn'].index, rome['X'].index), valid.index[~excluded])
    report['population'] = dict(raw_phase_counts=dict(phase_counts), ordinary_rows=len(raw),
        ordinary_by_month=months.value_counts().sort_index().to_dict(),
        ordinary_by_airport=raw.ADEP_mvt.value_counts().sort_index().to_dict(),
        rome_rows=len(r), rome_positive_rows=int(rome['y'].sum()),
        substituted_rows=d['substituted_rows'], invalid_labels=len(dep)-len(valid),
        rule_exclusions=int(excluded.sum()), every_eligible_row_accounted_for=True)
    rank = pd.concat([d['months'][m]['raw'] for m in sorted(d['months'])], ignore_index=True)
    np.testing.assert_array_equal(rank.mvt_id, d['rank_ids'])
    report['weather'] = {}
    for name, rows, matrix in [('ordinary', raw, d['Xn']), ('ranking', rank, d['Vn']), ('rome', r, rome['X'])]:
        report['weather'][name] = weather_check(rows, matrix)
        print(name, report['weather'][name], flush=True)
    report['weather_ranges'] = {c: dict(train_min=float(d['Xn'][c].min()), train_max=float(d['Xn'][c].max()),
        rank_min=float(d['Vn'][c].min()), rank_max=float(d['Vn'][c].max()),
        rank_above_train_max=int((d['Vn'][c] > d['Xn'][c].max()).sum())) for c in wx.WEATHER}
    import lightgbm as lgb
    report['models'] = {}
    for p in sorted((WORK / 'models').glob('*.txt')):
        model = lgb.Booster(model_file=str(p), params={'num_threads': 2})
        names = model.feature_name()
        expected = list(rome['X']) if p.stem == 'rome' else list(d['Xn'])
        assert names == expected
        imp = dict(zip(names, model.feature_importance('split').tolist()))
        report['models'][p.name] = dict(columns=len(names), iterations=model.current_iteration(),
            weather_split_counts={c: imp[c] for c in wx.WEATHER})
        del model
    from catboost import CatBoostRegressor
    cb = CatBoostRegressor().load_model(str(WORK / 'models/catboost.cbm'))
    assert cb.feature_names_ == d['names'] + wx.WEATHER
    importance = dict(zip(cb.feature_names_, cb.get_feature_importance(thread_count=2).tolist()))
    report['models']['catboost.cbm'] = dict(columns=len(cb.feature_names_), iterations=cb.tree_count_,
        weather_importance={c: importance[c] for c in wx.WEATHER})
    nn = json.loads((WORK / 'parts/nn.json').read_text())
    assert nn['columns'] == list(d['Xn']) and nn['network_members'] == 15
    models = list((WORK / 'models').glob('nn_*.pt'))
    assert len(models) == 15
    report['networks'] = dict(members=15, columns=len(nn['columns']),
        stop_epochs={s: json.loads((WORK / f'parts/nn_group_{s}.json').read_text())['stop_epoch'] for s in [42,142,242]},
        full_year_refit_verified_in_code=True)
    parts = {}
    for part in ['nn', 'lgb', 'airport', 'catboost', 'rome']:
        p = WORK / f'parts/{part}.parquet'
        assert sha256(p) == json.loads(p.with_suffix('.json').read_text())['sha256']
        f = pd.read_parquet(p)
        np.testing.assert_array_equal(f.mvt_id, d['rank_ids'])
        values = f.drop(columns='mvt_id').to_numpy(float)
        assert np.isfinite(values).all()
        parts[part] = values.mean(axis=1)
    base = .71 * parts['nn'] + .29 * np.mean([parts[p] for p in ['lgb','airport','catboost']], axis=0)
    built = pd.concat([pd.read_parquet(OUT / f'submission_v61/ranking_{m}.parquet')
                       for m in sorted(d['months'])], ignore_index=True)
    np.testing.assert_array_equal(built.mvt_id, d['rank_ids'])
    np.testing.assert_allclose(built.blend_base, base, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(built.p, parts['rome'])
    rec = json.loads((OUT / 'submission_v61/submission_v61.json').read_text())
    assert sha256(ROOT / rec['file']) == rec['sha256']
    sub, template = pq.read_table(ROOT / rec['file']), pq.read_table(ROOT / 'data/submitting.parquet')
    assert sub['MVT_ID_mvt'].equals(template['MVT_ID_mvt']) and sub.num_rows == 344841
    vals = sub['TAXITIME_SEC_mvt'].to_numpy()
    assert np.isfinite(vals).all() and (vals >= 0).all()
    report.update(passed=True, prediction_parts_verified=5, final_blend_reproduced=True,
                  submission_rows=sub.num_rows, submission_sha256=rec['sha256'],
                  source_sha256=sha256(Path(__file__)))
    write_json(OUT / 'full_year_audit.json', report)
    print('Full-year audit passed.', flush=True)


if __name__ == '__main__':
    main()
