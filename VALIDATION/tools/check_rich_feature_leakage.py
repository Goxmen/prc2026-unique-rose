"""Is the rich-feature separation a real shift, or a calendar fingerprint?

`compare_train_ranking_features.py` reaches AUC 0.9999 once weather and day-level
inputs join the raw columns. Near-perfect separation of two large samples usually
means a column identifies the calendar rather than a difference that matters:
`wx_temp_c` is a thermometer, and the `dd_*` inputs are per-airport-day aggregates
that are close to unique per day.

The test: season-match. July 2025 against July 2026 removes the seasonal contrast
almost entirely. If AUC stays near 1, the model is reading the date, and the rich
features cannot be used for reweighting -- the propensity would be degenerate.
A within-2025 pair (July against December) bounds what an ordinary season change
looks like on the same inputs.

No taxi-out fits, no uploads, no change to any recorded score.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'VALIDATION/tools'), str(ROOT / 'pipeline')]
import weather  # noqa: E402
import analyze_adversarial_shift as adv  # noqa: E402
import compare_train_ranking_features as cmp  # noqa: E402
from common import connect  # noqa: E402
from movement_data import load_phase  # noqa: E402
from records import write_json  # noqa: E402

OUT = ROOT / 'VALIDATION/reports/rich_feature_leakage.json'
SEED = 42
GROUPS = {'wx_only': [c for c in cmp.RICH if c.startswith('wx_')],
          'dd_only': [c for c in cmp.RICH if c.startswith(('dd_', 'dh_'))],
          'queue_only': [c for c in cmp.RICH if c.startswith('q_')],
          'stand_only': [c for c in cmp.RICH if c.startswith('st_')],
          'raw_plus_rich': None}


def discriminate(frame, label, columns, rng):
    holdout = rng.random(len(frame)) < 0.30
    original, adv.FEATURES = adv.FEATURES, columns
    try:
        model = adv.fit(frame[~holdout], label[~holdout])
        auc = float(roc_auc_score(label[holdout], model.predict(frame[holdout][columns])))
    finally:
        adv.FEATURES = original
    return auc


def main():
    con = connect()
    con.execute('SET threads=6')
    dep = load_phase(con, str(ROOT / 'data/training_*.parquet'), 'DEP')
    arr = load_phase(con, str(ROOT / 'data/training_*.parquet'), 'ARR')
    rank_dep = load_phase(con, str(ROOT / 'data/ranking.parquet'), 'DEP')
    rank_arr = load_phase(con, str(ROOT / 'data/ranking.parquet'), 'ARR')
    reports = weather.load_reports()
    for f in (dep, arr, rank_dep, rank_arr):
        f['ym'] = cmp.month_of(f)

    record = json.loads((cmp.RUN / 'result.json').read_text())
    built = {}
    for fold, month in [('A1', '2025-07'), ('A6', '2025-12')]:
        info = record['folds'][fold]
        lo = pd.Timestamp(info['train_start'], tz='UTC')
        hi = pd.Timestamp(info['train_end'] + ' 23:59:59', tz='UTC')
        td = dep[(dep.MVT_TIME_UTC_mvt >= lo) & (dep.MVT_TIME_UTC_mvt <= hi)]
        ta = arr[(arr.MVT_TIME_UTC_mvt >= lo) & (arr.MVT_TIME_UTC_mvt <= hi)]
        vd, va = dep[dep.ym.eq(month)], arr[arr.ym.eq(month)]
        built[month] = cmp.build(vd, pd.concat([td, vd], ignore_index=True),
                                 pd.concat([ta, va], ignore_index=True), reports, month)
    for month in ('2026-07',):
        rd, ra = rank_dep[rank_dep.ym.eq(month)], rank_arr[rank_arr.ym.eq(month)]
        built[month] = cmp.build(rd, pd.concat([dep, rd], ignore_index=True),
                                 pd.concat([arr, ra], ignore_index=True), reports, month)

    raw = adv.load().set_index('mvt_id')
    pairs = {'jul25_vs_jul26': ('2025-07', '2026-07'),   # season matched, different year
             'jul25_vs_dec25': ('2025-07', '2025-12')}   # same year, different season
    out = {}
    for name, (a, b) in pairs.items():
        frame = pd.concat([built[a].assign(label=0), built[b].assign(label=1)], ignore_index=True)
        frame = frame.set_index('mvt_id').join(raw[adv.FEATURES], how='inner')
        label = frame.label.to_numpy()
        out[name] = {'n_negative': int((label == 0).sum()), 'n_positive': int((label == 1).sum())}
        print(f"\n{name}: {out[name]['n_negative']:,} vs {out[name]['n_positive']:,}")
        for group, extra in GROUPS.items():
            columns = adv.FEATURES + (cmp.RICH if extra is None else extra)
            auc = discriminate(frame, label, columns, np.random.default_rng(SEED))
            out[name][group] = auc
            print(f'   raw + {group:<14} ({len(columns):>2} inputs)  AUC {auc:.4f}')
        auc_raw = discriminate(frame, label, adv.FEATURES, np.random.default_rng(SEED))
        out[name]['raw_only'] = auc_raw
        print(f'   raw only         ({len(adv.FEATURES):>2} inputs)  AUC {auc_raw:.4f}')

    verdict = {
        'season_matched_auc_raw_plus_rich': out['jul25_vs_jul26']['raw_plus_rich'],
        'within_year_auc_raw_plus_rich': out['jul25_vs_dec25']['raw_plus_rich'],
        'reading': 'A season-matched AUC near 1 means the rich inputs identify the calendar, '
                   'not a shift that costs predictions. Weights built on them would be degenerate.',
    }
    write_json(OUT, {'pairs': out, 'verdict': verdict, 'groups': {k: (v or cmp.RICH) for k, v in GROUPS.items()}})
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
