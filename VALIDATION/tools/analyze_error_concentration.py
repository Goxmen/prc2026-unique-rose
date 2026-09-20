"""Measure how few flights carry the validation score.

Reads saved out-of-fold predictions only: no fits, no uploads, and no change to
any recorded score. It answers one question -- how many flights does the squared
error actually live in -- so that a distribution-shift diagnostic can be aimed at
the right scale. A mean over a million rows cannot resolve a gap that lives in
ten of them.

Usage: analyze_error_concentration.py [prediction_column ...]
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from reporting import load_losses, score_groups  # noqa: E402
from records import write_json  # noqa: E402

RUN = ROOT / 'VALIDATION/runs/v3_20260919_artifact_experiments'
OUT = ROOT / 'VALIDATION/reports/error_concentration.json'
GROUPS = ['R_A', 'R_B_long']
KS = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
SHARES = [0.25, 0.50, 0.80, 0.90, 0.95, 0.99]
ATTRS = ['mvt_id', 'rule_triggered', 'raw_flight_id_missing', 'late_rows']
# Leaderboard anchors, for framing the CV gap in rows rather than seconds.
RANKING_ROWS = 344841
OFFICIAL = {'pred_combined': 279.3805, 'pred_noid_cap': 280.4658}


def attributes(record):
    """Row indicators that `load_losses` does not carry into its meta frame."""
    frames = [pd.read_parquet(RUN / info['predictions']['42'], columns=ATTRS)
              for info in record['folds'].values()]
    return pd.concat(frames).drop_duplicates('mvt_id').set_index('mvt_id')


def masks(meta, attrs):
    """Overlapping row classes, plus one mutually exclusive ordinary class."""
    a = attrs.reindex(meta.index)
    y = meta.y.to_numpy()
    rule = a.rule_triggered.fillna(False).to_numpy(bool)
    noid = a.raw_flight_id_missing.fillna(False).to_numpy(bool)
    return {
        'rome_rule_row': rule,
        'no_flight_id': noid,
        'no_flight_plan': ~meta.has_flight_plan.to_numpy(bool),
        'label_over_2h': y > 7_200,
        'label_over_6h': y > 21_600,
        'label_over_12h': y > 43_200,
        'ordinary_with_id_under_2h': (~noid) & (y <= 7_200),
    }


def concentration(per_flight):
    """Cumulative squared-error share of the worst k flights."""
    order = np.argsort(per_flight)[::-1]
    ranked = per_flight[order]
    total = float(ranked.sum())
    n = len(ranked)
    cumulative = np.cumsum(ranked)
    top = {}
    for k in KS:
        if k > n:
            break
        remaining = total - float(cumulative[k - 1])
        top[k] = {
            'sse_share_pct': 100 * float(cumulative[k - 1]) / total,
            'row_share_pct': 100 * k / n,
            # RMSE that would remain if those k flights were predicted exactly.
            'rmse_without': float(np.sqrt(remaining / n)),
        }
    rows_for = {f'{int(100 * s)}pct': int(np.searchsorted(cumulative, s * total) + 1)
                for s in SHARES}
    return {'n_flights': n, 'rmse': float(np.sqrt(total / n)), 'sse': total,
            'top_k': top, 'rows_to_reach': rows_for,
            'rows_to_reach_share_pct': {k: 100 * v / n for k, v in rows_for.items()}}


def worst_rows(meta, per_flight, attrs, limit=15):
    a = attrs.reindex(meta.index)
    order = np.argsort(per_flight)[::-1][:limit]
    out = []
    for i in order:
        out.append({
            'mvt_id': float(meta.index[i]),
            'airport': str(meta.airport.iloc[i]),
            'month': meta.mvt_time.iloc[i].strftime('%Y-%m-%d'),
            'label_sec': float(meta.y.iloc[i]),
            'abs_error_sec': float(np.sqrt(per_flight[i])),
            'sse_share_pct': 100 * float(per_flight[i]) / float(per_flight.sum()),
            'rome_rule_row': bool(a.rule_triggered.iloc[i]),
            'no_flight_id': bool(a.raw_flight_id_missing.iloc[i]),
        })
    return out


def analyse(column, record, attrs):
    record = {**record, 'prediction_column': column}
    groups = score_groups(load_losses(record, RUN))
    out = {}
    for name in GROUPS:
        loss = groups[name]
        per_flight = loss.sq.mean(axis=0)          # mean over seeds, as scored
        total = float(per_flight.sum())
        result = concentration(per_flight)
        result['classes'] = {}
        for label, mask in masks(loss.meta, attrs).items():
            n = int(mask.sum())
            sse = float(per_flight[mask].sum())
            result['classes'][label] = {
                'n': n, 'row_share_pct': 100 * n / len(per_flight),
                'sse_share_pct': 100 * sse / total,
                'rmse_in_class': float(np.sqrt(sse / n)) if n else None,
            }
        worst = np.argsort(per_flight)[::-1][:100]
        result['top_100_classes'] = {
            label: int(mask[worst].sum()) for label, mask in masks(loss.meta, attrs).items()}
        result['top_100_airports'] = (
            loss.meta.airport.iloc[worst].value_counts().to_dict())
        result['worst_rows'] = worst_rows(loss.meta, per_flight, attrs)
        out[name] = result
    w = record['ranking_weights']['jan_fraction']
    combined = lambda a, b: float(np.sqrt(w * a ** 2 + (1 - w) * b ** 2))
    out['R_combined'] = {
        'jan_fraction': w,
        'rmse': combined(out['R_A']['rmse'], out['R_B_long']['rmse']),
        'top_k_rmse_without': {
            k: combined(out['R_A']['top_k'][k]['rmse_without'],
                        out['R_B_long']['top_k'][k]['rmse_without'])
            for k in out['R_A']['top_k'] if k in out['R_B_long']['top_k']},
    }
    if column in OFFICIAL:
        lb, cv = OFFICIAL[column], out['R_combined']['rmse']
        excess = (lb ** 2 - cv ** 2) * RANKING_ROWS
        out['gap_to_leaderboard'] = {
            'official_rmse': lb, 'cv_combined': cv, 'excess_sse': excess,
            'one_row_error_sec': float(np.sqrt(excess)),
            'rows_equivalent': {f'{h}h': excess / (h * 3600) ** 2
                                for h in (24, 12, 8, 4, 2, 1)},
        }
    return out


def main():
    columns = sys.argv[1:] or ['pred_combined', 'pred_noid_cap']
    record = json.loads((RUN / 'result.json').read_text())
    attrs = attributes(record)
    report = {'run': RUN.name, 'columns': {c: analyse(c, record, attrs) for c in columns}}
    write_json(OUT, report)
    for column, result in report['columns'].items():
        print(f"\n=== {column} — combined {result['R_combined']['rmse']:.4f} ===")
        for name in GROUPS:
            r = result[name]
            print(f"{name}: {r['n_flights']:,} flights, RMSE {r['rmse']:.4f}")
            for k, v in r['top_k'].items():
                print(f"   top {k:>5}: {v['sse_share_pct']:6.2f}% of SSE "
                      f"({v['row_share_pct']:.5f}% of rows) → RMSE {v['rmse_without']:8.4f}")
            print('   rows to reach:', r['rows_to_reach'])
            for label, c in r['classes'].items():
                print(f"   {label:<26} n={c['n']:>8,} rows={c['row_share_pct']:6.3f}% "
                      f"sse={c['sse_share_pct']:6.2f}% rmse={c['rmse_in_class']}")
            print('   top-100 classes:', r['top_100_classes'])
            print('   top-100 airports:', r['top_100_airports'])
        if 'gap_to_leaderboard' in result:
            g = result['gap_to_leaderboard']
            print(f"   LB {g['official_rmse']} vs CV {g['cv_combined']:.4f}: excess SSE "
                  f"{g['excess_sse']:.4g} = one row wrong by {g['one_row_error_sec']:,.0f} s")
            print('   rows equivalent:', {k: round(v, 1) for k, v in g['rows_equivalent'].items()})
    print(f"\nwrote {OUT}")


if __name__ == '__main__':
    main()
