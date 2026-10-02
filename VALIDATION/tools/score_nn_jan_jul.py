"""Score `nn_jan_jul.py` (MLP beside v46's ordinary LightGBM) at three seeds, saved predictions only.

The blend weight is fixed at 0.3 NN / 0.7 LightGBM before any v46 run: the middle of the
cross-month weight choices (0.25-0.35) of the earlier exploratory MLP run on v45's inputs.
Squared errors are averaged over seeds within each fit, as the Jan/Jul search did; the
interval resamples whole days within each fit. Pass: interval excludes zero and every seed
improves.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
import tuning_seasonal as ts
from records import write_json

SPLITS_FROM = ts.RUNS / 'v3_20261002_adsblol_jan_jul'
OUT = ts.RUNS / 'v3_20261002_nn_v46_jan_jul'
SEEDS, WEIGHT = [42, 142, 242], 0.3


def main():
    preds = pd.read_parquet(OUT / 'predictions' / 'nn_seeds42_142_242.parquet')
    table = adsblol.load_features()
    parts = []
    for month in ts.SPLITS:
        held = pd.read_pickle(ts.split_dir(SPLITS_FROM, month) / 'heldout.pkl')
        meta = ts.read(ts.split_dir(SPLITS_FROM, month) / 'meta.json')
        raw = held['raw']
        p = preds[preds.month == month].set_index('mvt_id').reindex(raw.mvt_id)
        if p.isna().any().any():
            raise ValueError(f'{month}: prediction rows differ from the held-out rows')
        final = lambda base: ts.final_prediction(base, held['p'], held['V'], raw, meta['parameters'], meta['corrections'])
        frame = pd.DataFrame({'mvt_id': raw.mvt_id.to_numpy(), 'month': month, 'y': raw.taxi_sec.to_numpy(float),
                              'airport': raw.ADEP_mvt.astype(str).to_numpy(),
                              'has_id': raw.FLIGHT_ID_mvt.notna().to_numpy(),
                              'day': pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').to_numpy()})
        for s in SEEDS:
            lgb, nn = p[f'lgb_base_{s}'].to_numpy(float), p[f'nn_base_{s}'].to_numpy(float)
            frame[f'lgb_{s}'] = final(lgb)
            frame[f'nn_{s}'] = final(nn)
            frame[f'blend_{s}'] = final(WEIGHT * nn + (1 - WEIGHT) * lgb)
            frame[f'touched_{s}'] = np.abs(frame[f'lgb_{s}'].to_numpy() - lgb) > 1e-6
        parts.append(frame)
    f = pd.concat(parts, ignore_index=True)
    ab = table.loc[f.mvt_id, ['ab_matched', 'ab_seg_sec']]
    f['adsb'] = np.select([ab.ab_matched.fillna(0).to_numpy() == 0, ab.ab_seg_sec.isna().to_numpy()],
                          ['unmatched', 'matched_no_ground'], 'seen_on_ground')
    se = {arm: np.mean([(f.y - f[f'{arm}_{s}']) ** 2 for s in SEEDS], axis=0) for arm in ('lgb', 'nn', 'blend')}
    month, day = f.month.to_numpy(), f.day.to_numpy()
    out = dict(weight=WEIGHT, seeds=SEEDS, level={}, change={}, by_seed={}, by_airport={}, by_adsb={})
    for arm in se:
        out['level'][arm] = {m: float(np.sqrt(se[arm][month == m].mean())) for m in ts.SPLITS}
        out['level'][arm]['average'] = float(np.mean([out['level'][arm][m] for m in ts.SPLITS]))
    for arm in ('nn', 'blend'):
        out['change'][arm] = ts.paired_bootstrap(day, month, se['lgb'], se[arm])
        k = f.has_id.to_numpy()
        out['change'][f'{arm}_id'] = ts.paired_bootstrap(day[k], month[k], se['lgb'][k], se[arm][k])
    for s in SEEDS:
        lv = {arm: np.mean([np.sqrt(((f.y - f[f'{arm}_{s}']) ** 2)[month == m].mean()) for m in ts.SPLITS])
              for arm in ('lgb', 'nn', 'blend')}
        out['by_seed'][s] = {k: float(v) for k, v in lv.items()}
    for (m, a), g in f.groupby(['month', 'airport']):
        i = g.index.to_numpy()
        out['by_airport'].setdefault(m, {})[a] = dict(n=len(i), lgb=float(np.sqrt(se['lgb'][i].mean())),
                                                      blend=float(np.sqrt(se['blend'][i].mean())))
    for (m, a), g in f.groupby(['month', 'adsb']):
        i = g.index.to_numpy()
        out['by_adsb'].setdefault(m, {})[a] = dict(n=len(i), lgb=float(np.sqrt(se['lgb'][i].mean())),
                                                   blend=float(np.sqrt(se['blend'][i].mean())))
    # Bulk versus tail: squared-error change split by the LightGBM error size.
    err = np.sqrt(se['lgb'])
    for name, k in (('lgb_abs_err_lt_1200', err < 1200), ('lgb_abs_err_ge_1200', err >= 1200)):
        out.setdefault('sse_change_share', {})[name] = {
            m: float((se['blend'][k & (month == m)] - se['lgb'][k & (month == m)]).sum() / se['lgb'][month == m].sum())
            for m in ts.SPLITS}
    write_json(OUT / 'score_3seed.json', out)
    print('levels', {a: round(v['average'], 4) for a, v in out['level'].items()})
    print('per month', {a: {m: round(v[m], 2) for m in ts.SPLITS} for a, v in out['level'].items()})
    for k, v in out['change'].items():
        print(k, round(v['change'], 4), v['ci95'], v['decision'])
    print('by seed', {s: {a: round(x, 3) for a, x in v.items()} for s, v in out['by_seed'].items()})
    print('SSE change share (fraction of LightGBM SSE)', out['sse_change_share'])
    for m in ts.SPLITS:
        print(m, 'airport', {a: round(v['blend'] - v['lgb'], 2) for a, v in out['by_airport'][m].items()})
        print(m, 'adsb', {a: (v['n'], round(v['blend'] - v['lgb'], 2)) for a, v in out['by_adsb'][m].items()})


if __name__ == '__main__':
    main()
