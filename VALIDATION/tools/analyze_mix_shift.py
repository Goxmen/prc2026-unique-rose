"""Account for the CV-to-leaderboard gap from row-class prevalence alone.

Reads the two saved diagnostics -- `error_concentration.json` (per-class squared
error in the validation folds) and `recording_regime_2026.json` (row-class
prevalence in 2025 and in the ranking months) -- and asks one question: if every
class kept its measured validation error and only the class MIX changed to the
ranking set's, how much of the gap would that explain? The residual is what
prevalence cannot account for.

No fits, no uploads, no change to any recorded score.
"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION')]
from records import write_json  # noqa: E402

REPORTS = ROOT / 'VALIDATION/reports'
OUT = REPORTS / 'mix_shift_accounting.json'
CLASS = 'no_flight_id'
RANKING_ROWS = {'2026-01': 152719, '2026-07': 192122}

# Leaderboard anchors from SUBMISSIONS.md rows 30 and 33. v33 moves 58 ranking
# rows off v30's predictions and changes nothing else; its report attributes
# almost all of the official gain to one of them (ITY340).
V30_LB, V33_LB, V33_ROWS_MOVED = 309.9872, 282.4525, 58
TOTAL_ROWS = 344841


def split(group):
    """Squared error and row counts for the class and for everything else."""
    n, sse = group['n_flights'], group['sse']
    c = group['classes'][CLASS]
    sse_class = sse * c['sse_share_pct'] / 100
    return {
        'n': n, 'rmse': group['rmse'],
        'class_n': c['n'], 'class_share_pct': c['row_share_pct'],
        'class_mse': sse_class / c['n'],
        'rest_n': n - c['n'], 'rest_mse': (sse - sse_class) / (n - c['n']),
    }


def main():
    conc = json.loads((REPORTS / 'error_concentration.json').read_text())['columns']['pred_combined']
    regime = json.loads((REPORTS / 'recording_regime_2026.json').read_text())
    monthly = {(d['ym'], d['phase']): d for d in regime['monthly']}

    # Ranking-set prevalence of the class, over departures only.
    missing = sum(RANKING_ROWS[m] * monthly[(m, 'DEP')]['pct_no_flight_id'] / 100
                  for m in RANKING_ROWS)
    p_rank = missing / TOTAL_ROWS
    w = conc['R_combined']['jan_fraction']

    groups, reweighted = {}, {}
    for name in ('R_A', 'R_B_long'):
        s = split(conc[name])
        s['mse_at_ranking_mix'] = p_rank * s['class_mse'] + (1 - p_rank) * s['rest_mse']
        s['rmse_at_ranking_mix'] = float(np.sqrt(s['mse_at_ranking_mix']))
        s['rest_rmse'] = float(np.sqrt(s['rest_mse']))
        s['class_rmse'] = float(np.sqrt(s['class_mse']))
        groups[name] = s
        reweighted[name] = s['rmse_at_ranking_mix']

    combine = lambda a, b: float(np.sqrt(w * a ** 2 + (1 - w) * b ** 2))
    cv = conc['R_combined']['rmse']
    shifted = combine(reweighted['R_A'], reweighted['R_B_long'])
    lb = conc['gap_to_leaderboard']['official_rmse']

    # If every artifact row were predicted exactly, only ordinary rows would remain.
    ordinary = combine(conc['R_A']['classes']['ordinary_with_id_under_2h']['rmse_in_class'],
                       conc['R_B_long']['classes']['ordinary_with_id_under_2h']['rmse_in_class'])

    # Top-1 concentration: validation versus what the leaderboard demonstrably holds.
    v30_sse = V30_LB ** 2 * TOTAL_ROWS
    v33_gain_sse = (V30_LB ** 2 - V33_LB ** 2) * TOTAL_ROWS

    report = {
        'class': CLASS,
        'prevalence': {
            'ranking_pct': 100 * p_rank,
            'ranking_rows': missing,
            'validation_R_A_pct': groups['R_A']['class_share_pct'],
            'validation_R_B_long_pct': groups['R_B_long']['class_share_pct'],
            'jan_2025_pct': monthly[('2025-01', 'DEP')]['pct_no_flight_id'],
            'jan_2026_pct': monthly[('2026-01', 'DEP')]['pct_no_flight_id'],
            'jul_2025_pct': monthly[('2025-07', 'DEP')]['pct_no_flight_id'],
            'jul_2026_pct': monthly[('2026-07', 'DEP')]['pct_no_flight_id'],
        },
        'groups': groups,
        'accounting': {
            'cv_combined': cv,
            'cv_at_ranking_mix': shifted,
            'official': lb,
            'gap_total': lb - cv,
            'gap_explained_by_mix': shifted - cv,
            'gap_explained_pct': 100 * (shifted - cv) / (lb - cv),
            'gap_residual': lb - shifted,
            'residual_excess_sse': (lb ** 2 - shifted ** 2) * TOTAL_ROWS,
            'residual_one_row_sec': float(np.sqrt((lb ** 2 - shifted ** 2) * TOTAL_ROWS)),
            'residual_rows_at_4h': (lb ** 2 - shifted ** 2) * TOTAL_ROWS / (4 * 3600) ** 2,
        },
        'perfect_artifact_rows': {'cv_combined_if_perfect': ordinary,
                                  'remaining_share_of_cv_pct': 100 * ordinary ** 2 / cv ** 2},
        'leaderboard_concentration': {
            'v30_official': V30_LB, 'v33_official': V33_LB,
            'rows_moved': V33_ROWS_MOVED,
            'gain_sse': v33_gain_sse,
            'moved_rows_share_of_v30_sse_pct': 100 * v33_gain_sse / v30_sse,
            'validation_top_1_share_pct': conc['R_A']['top_k']['1']['sse_share_pct'],
            'validation_top_50_share_pct': conc['R_A']['top_k']['50']['sse_share_pct'],
            'validation_top_100_share_pct': conc['R_A']['top_k']['100']['sse_share_pct'],
        },
    }
    write_json(OUT, report)
    print(json.dumps(report['prevalence'], indent=2))
    print(json.dumps(report['accounting'], indent=2))
    print(json.dumps(report['perfect_artifact_rows'], indent=2))
    print(json.dumps(report['leaderboard_concentration'], indent=2))
    for name, g in groups.items():
        print(f"{name}: class rmse {g['class_rmse']:.1f} ({g['class_share_pct']:.3f}% of rows), "
              f"rest {g['rest_rmse']:.2f}, at ranking mix {g['rmse_at_ranking_mix']:.4f} "
              f"(from {g['rmse']:.4f})")
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
