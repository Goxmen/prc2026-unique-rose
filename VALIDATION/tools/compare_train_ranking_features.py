"""Compare training-side and ranking-side feature distributions on the richer inputs.

The adversarial diagnostic ran on raw movement and flight-plan columns only. The
weather, runway-queue and stand/day inputs that the model actually uses were never
part of that comparison -- they are absent from the saved prediction files, so
they have to be built. This builds them for both sides and compares them, and it
does not depend on reweighting passing any test.

Information limits are respected per population, as protocol rule 5 requires:

  * a validation month sees its own fold's training window plus that month only;
  * a ranking month sees all of 2025 plus that month only, which is what the
    submitted models were given.

Existing caches are used rather than rebuilt: METAR reports come from
`data/external/metar/metar.parquet` via `VALIDATION/weather.py`. Queue and
stand/day inputs have no cache and are derived from movement timestamps.

Departure-history inputs are deliberately NOT compared. A fold freezes them at its
own cutoff while the submitted model uses all of 2025, so any difference there is
produced by the protocol rather than by the data, and would not mean what a
distribution gap is supposed to mean here.

Importance is held-out permutation importance, never training gain: the
high-cardinality categoricals absorb most of the gain even when a model is at
chance. No taxi-out fits, no uploads, no change to any recorded score.
"""

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'VALIDATION/tools'), str(ROOT / 'pipeline')]
import runway_queue  # noqa: E402
import stand_day  # noqa: E402
import weather  # noqa: E402
import analyze_adversarial_shift as adv  # noqa: E402
from common import connect  # noqa: E402
from modeling import observable  # noqa: E402
from movement_data import load_phase  # noqa: E402
from records import write_json  # noqa: E402

DATA = ROOT / 'data'
RUN = ROOT / 'VALIDATION/runs/v3_20260919_artifact_experiments'
OUT = ROOT / 'VALIDATION/reports/train_ranking_features.json'
TRAIN_GLOB = str(DATA / 'training_*.parquet')
RANKING = str(DATA / 'ranking.parquet')
A_FOLDS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
RANK_MONTHS = ['2026-01', '2026-07']
RICH = (weather.CURRENT_FEATURES + weather.WINDOW_FEATURES +
        runway_queue.PAST_FEATURES + runway_queue.FUTURE_FEATURES +
        stand_day.STAND_FEATURES + stand_day.DAY_FEATURES)
SEED = 42


def month_of(frame):
    return frame.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m')


def build(raw, dep_events, arr_events, reports, label):
    """Weather, queue and stand/day inputs for one population, at its own exposure."""
    t0 = time.time()
    out = weather.weather_features(raw, reports)
    out = out.join(runway_queue.queue_features(
        raw, observable(dep_events), observable(arr_events)))
    out = out.join(stand_day.stand_day_features(raw, dep_events, arr_events))
    out['mvt_id'] = raw.mvt_id.to_numpy()
    print(f'  {label}: {len(out):,} rows, {len(RICH)} inputs, {time.time() - t0:.0f}s', flush=True)
    return out


def describe(frame, columns):
    rows = {}
    for c in columns:
        v = frame[c].to_numpy(dtype='float64', na_value=np.nan)
        ok = np.isfinite(v)
        rows[c] = {
            'missing_pct': float(100 * (~ok).mean()),
            'mean': float(v[ok].mean()) if ok.any() else None,
            'sd': float(v[ok].std()) if ok.any() else None,
            'p10': float(np.quantile(v[ok], 0.10)) if ok.any() else None,
            'p50': float(np.quantile(v[ok], 0.50)) if ok.any() else None,
            'p90': float(np.quantile(v[ok], 0.90)) if ok.any() else None,
        }
    return rows


def standardised(a, b):
    """Standardised mean difference, pooled sd; a robust scale-free distribution gap."""
    if a['mean'] is None or b['mean'] is None:
        return None
    pooled = np.sqrt(((a['sd'] or 0) ** 2 + (b['sd'] or 0) ** 2) / 2)
    return float((b['mean'] - a['mean']) / pooled) if pooled > 0 else None


def main():
    con = connect()
    con.execute('SET threads=6')
    print('loading movements...')
    dep = load_phase(con, TRAIN_GLOB, 'DEP')
    arr = load_phase(con, TRAIN_GLOB, 'ARR')
    rank_dep = load_phase(con, RANKING, 'DEP')
    rank_arr = load_phase(con, RANKING, 'ARR')
    reports = weather.load_reports()
    dep['ym'], arr['ym'] = month_of(dep), month_of(arr)
    rank_dep['ym'], rank_arr['ym'] = month_of(rank_dep), month_of(rank_arr)
    print(f'  2025 {len(dep):,} DEP / {len(arr):,} ARR; '
          f'2026 {len(rank_dep):,} DEP / {len(rank_arr):,} ARR')

    record = json.loads((RUN / 'result.json').read_text())
    print('\nbuilding validation-side inputs, each fold at its own exposure:')
    parts = []
    for fold in A_FOLDS:
        info = record['folds'][fold]
        lo, hi = pd.Timestamp(info['train_start'], tz='UTC'), pd.Timestamp(info['train_end'] + ' 23:59:59', tz='UTC')
        td = dep[(dep.MVT_TIME_UTC_mvt >= lo) & (dep.MVT_TIME_UTC_mvt <= hi)]
        ta = arr[(arr.MVT_TIME_UTC_mvt >= lo) & (arr.MVT_TIME_UTC_mvt <= hi)]
        vd, va = dep[dep.ym.eq(info['val'])], arr[arr.ym.eq(info['val'])]
        built = build(vd, pd.concat([td, vd], ignore_index=True),
                      pd.concat([ta, va], ignore_index=True), reports, f"{fold} ({info['val']})")
        built['population'] = 'validation'
        parts.append(built)

    print('\nbuilding ranking-side inputs, all of 2025 plus each month itself:')
    for m in RANK_MONTHS:
        rd, ra = rank_dep[rank_dep.ym.eq(m)], rank_arr[rank_arr.ym.eq(m)]
        built = build(rd, pd.concat([dep, rd], ignore_index=True),
                      pd.concat([arr, ra], ignore_index=True), reports, m)
        built['population'] = m
        parts.append(built)

    rich = pd.concat(parts, ignore_index=True)
    del parts
    populations = {'validation': rich.population.eq('validation'),
                   '2026-01': rich.population.eq('2026-01'),
                   '2026-07': rich.population.eq('2026-07')}
    stats = {name: describe(rich[mask], RICH) for name, mask in populations.items()}

    print('\nlargest distribution gaps against pooled validation '
          '(standardised mean difference, January and July separately):')
    gaps = []
    for c in RICH:
        gaps.append({'feature': c,
                     'validation_missing_pct': stats['validation'][c]['missing_pct'],
                     'jan_missing_pct': stats['2026-01'][c]['missing_pct'],
                     'jul_missing_pct': stats['2026-07'][c]['missing_pct'],
                     'smd_jan': standardised(stats['validation'][c], stats['2026-01'][c]),
                     'smd_jul': standardised(stats['validation'][c], stats['2026-07'][c])})
    ranked = sorted(gaps, key=lambda g: -max(abs(g['smd_jan'] or 0), abs(g['smd_jul'] or 0)))
    print(f"   {'feature':<28}{'SMD Jan':>9}{'SMD Jul':>9}   missing val/Jan/Jul")
    for g in ranked[:15]:
        print(f"   {g['feature']:<28}{g['smd_jan'] or 0:>9.3f}{g['smd_jul'] or 0:>9.3f}   "
              f"{g['validation_missing_pct']:5.2f}% {g['jan_missing_pct']:5.2f}% {g['jul_missing_pct']:5.2f}%")

    # Does the richer set expose separation the raw columns miss? Held-out AUC, two fits.
    print('\nraw-only versus raw+rich discriminators (held-out AUC, permutation importance):')
    raw = adv.load().set_index('mvt_id')
    joined = rich.set_index('mvt_id').join(raw[adv.FEATURES], how='inner')
    joined['label'] = (~joined.population.eq('validation')).astype(int)
    rng = np.random.default_rng(SEED)
    holdout = rng.random(len(joined)) < 0.30
    discriminators = {}
    for name, columns in [('raw_only', adv.FEATURES), ('raw_plus_rich', adv.FEATURES + RICH)]:
        original, adv.FEATURES_BACKUP = adv.FEATURES, adv.FEATURES
        adv.FEATURES = columns
        model = adv.fit(joined[~holdout], joined.label.to_numpy()[~holdout])
        auc = float(roc_auc_score(joined.label.to_numpy()[holdout],
                                  model.predict(joined[holdout][columns])))
        perm = adv.permutation_importance(
            model, joined[holdout].reset_index(drop=True), joined.label.to_numpy()[holdout], rng)
        adv.FEATURES = original
        discriminators[name] = {'n_features': len(columns), 'holdout_auc': auc,
                                'permutation': perm}
        print(f'   {name:<14} AUC {auc:.4f} over {len(columns)} inputs')
        for f in perm['top_features'][:8]:
            print(f"      {f['feature']:<28} AUC drop {f['auc_drop']:.4f}")

    write_json(OUT, {'rich_features': RICH, 'population_stats': stats,
                     'gaps_ranked': ranked, 'discriminators': discriminators,
                     'note': 'Departure-history inputs excluded: a fold freezes them at its cutoff '
                             'while the submitted model uses all of 2025, so a difference there '
                             'reflects the protocol, not the data.'})
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
