"""Do unfamiliar stands actually produce larger errors? Tested on labelled flights.

Counting ranking rows whose stand never appears in validation says how many there
are, not what they cost. This measures the cost where the labels are known: for
each fold, flag validation flights whose (airport, stand) -- or (airport, stand,
runway), or (airport, destination) -- never appears in that fold's own training
window, and compare their error against the rest of the same fold.

The hypothetical benefit keeps every scored flight in the denominator, including
rows whose key is missing and which therefore belong to neither slice:

    new_sse  = total_sse - unseen_sse + n_unseen * seen_mse
    new_rmse = sqrt(new_sse / total_n)

Dropping the missing-key rows overstates the benefit; they are few but expensive,
so they are reported as their own slice rather than silently excluded.

Saved out-of-fold predictions and the raw parquets only. No fits, no uploads, no
change to any recorded score. Independently verified in
`tmp/unseen_support_review/` (check.py, results.json).
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import connect  # noqa: E402
from records import write_json  # noqa: E402
from reporting import load_losses  # noqa: E402

DATA = ROOT / 'data'
RUN = ROOT / 'VALIDATION/runs/v3_20260919_artifact_experiments'
OUT = ROOT / 'VALIDATION/reports/unseen_support.json'
A_FOLDS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
KEYS = {'stand': ['airport', 'stand'],
        'stand_runway': ['airport', 'stand', 'runway'],
        'destination': ['airport', 'ades']}
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260920

QUERY = """
SELECT MVT_ID_mvt AS mvt_id, MVT_TIME_UTC_mvt AS t,
       ADEP_mvt AS airport, STAND_mvt AS stand,
       RUNWAY_mvt AS runway, ADES_mvt AS ades
FROM read_parquet($training) WHERE PHASE_mvt = 'DEP'
"""


def key_series(df, cols):
    """One string key per row; a missing component makes the row unkeyable."""
    parts = [df[c].astype('string') for c in cols]
    out = parts[0]
    for q in parts[1:]:
        out = out + '\u001f' + q
    return out


def slice_rmse(sq, mask):
    return float(np.sqrt(sq[mask].mean())) if mask.any() else None


def day_bootstrap(parts):
    """Descriptive interval for RMSE(unseen) - RMSE(seen), resampling days in fold."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    tables = [p.to_numpy() for p in parts]
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        totals = np.zeros(4)
        for arr in tables:
            totals += arr[rng.integers(len(arr), size=len(arr))].sum(axis=0)
        if totals[0] > 0 and totals[2] > 0:
            draws.append(np.sqrt(totals[1] / totals[0]) - np.sqrt(totals[3] / totals[2]))
    return [float(v) for v in np.quantile(draws, [0.025, 0.5, 0.975])]


def main():
    con = connect()
    con.execute('SET threads=6')
    dep = con.execute(QUERY, {'training': [str(p) for p in sorted(DATA.glob('training_*.parquet'))]}).df()
    dep['t'] = pd.to_datetime(dep.t, utc=True)
    dep = dep.set_index('mvt_id')
    print(f'{len(dep):,} departures of 2025 loaded')

    record = json.loads((RUN / 'result.json').read_text())
    record = {**record, 'prediction_column': 'pred_combined'}
    losses = load_losses(record, RUN)

    keyed = {name: key_series(dep, cols) for name, cols in KEYS.items()}
    report, pooled = {}, {name: [] for name in KEYS}

    for fold in A_FOLDS:
        info = record['folds'][fold]
        window = ((dep.t >= pd.Timestamp(info['train_start'], tz='UTC')) &
                  (dep.t <= pd.Timestamp(info['train_end'] + ' 23:59:59', tz='UTC')))
        loss = losses[fold]
        sq = loss.sq.mean(axis=0)                      # mean over seeds, as scored
        day = loss.meta.mvt_time.dt.floor('D')
        row = {'val_month': info['val'], 'train': f"{info['train_start']}..{info['train_end']}",
               'n': len(sq), 'rmse': float(np.sqrt(sq.mean())), 'keys': {}}
        for name in KEYS:
            seen = set(keyed[name][window].dropna().unique())
            k = keyed[name].reindex(loss.meta.index)
            missing = k.isna().to_numpy()
            unseen = (~k.isin(seen)).to_numpy() & ~missing
            known = ~unseen & ~missing
            entry = {
                'n_unseen': int(unseen.sum()), 'pct_unseen': 100 * float(unseen.mean()),
                'rmse_unseen': slice_rmse(sq, unseen), 'rmse_seen': slice_rmse(sq, known),
                'sse_share_unseen_pct': 100 * float(sq[unseen].sum() / sq.sum()),
                # Rows whose key is missing belong to neither slice but are still scored.
                'n_missing': int(missing.sum()),
                'missing_sse': float(sq[missing].sum()),
                'rmse_missing': slice_rmse(sq, missing),
                'sse_share_missing_pct': 100 * float(sq[missing].sum() / sq.sum()),
            }
            entry['ratio'] = (entry['rmse_unseen'] / entry['rmse_seen']
                              if entry['rmse_unseen'] and entry['rmse_seen'] else None)
            entry['cost_sec'] = float(np.sqrt(sq.mean()) - np.sqrt(
                (sq.sum() - sq[unseen].sum() + unseen.sum() * sq[known].mean()) / len(sq)))
            row['keys'][name] = entry
            pooled[name].append((sq, unseen, known, missing, pd.DataFrame(
                {'un': unseen.astype(float), 'us': sq * unseen,
                 'kn': known.astype(float), 'ks': sq * known}).groupby(day.to_numpy()).sum()))
        report[fold] = row
        print(f"\n{fold} (val {info['val']}, train {row['train']}): "
              f"{row['n']:,} flights, RMSE {row['rmse']:.2f}")
        for name, e in row['keys'].items():
            print(f"   {name:<13} unseen {e['n_unseen']:>5,} ({e['pct_unseen']:5.2f}%)  "
                  f"RMSE {0 if e['rmse_unseen'] is None else round(e['rmse_unseen'], 1):>8} "
                  f"vs {e['rmse_seen']:7.1f} seen  ×{e['ratio']:.2f}  cost {e['cost_sec']:+.3f} s  "
                  f"| missing {e['n_missing']:>3} rows, RMSE "
                  f"{0 if e['rmse_missing'] is None else round(e['rmse_missing'], 1)}")

    print('\npooled over A1-A6 (each month scored by one model):')
    summary = {}
    for name, parts in pooled.items():
        sq = np.concatenate([p[0] for p in parts])
        unseen = np.concatenate([p[1] for p in parts])
        known = np.concatenate([p[2] for p in parts])
        missing = np.concatenate([p[3] for p in parts])
        base = float(np.sqrt(sq.mean()))
        # Every scored flight stays in the denominator, missing-key rows included.
        corrected = float(np.sqrt(
            (sq.sum() - sq[unseen].sum() + unseen.sum() * sq[known].mean()) / len(sq)))
        fold_matched = float(np.sqrt(
            (sq.sum() - sum(s[u].sum() - u.sum() * s[k].mean() for s, u, k, _, _ in parts)) / len(sq)))
        summary[name] = {
            'n_total': int(len(sq)),
            'n_unseen': int(unseen.sum()), 'pct_unseen': 100 * float(unseen.mean()),
            'n_missing': int(missing.sum()), 'missing_sse': float(sq[missing].sum()),
            'rmse_missing': slice_rmse(sq, missing),
            'rmse_unseen': float(np.sqrt(sq[unseen].mean())),
            'rmse_seen': float(np.sqrt(sq[known].mean())),
            'ratio': float(np.sqrt(sq[unseen].mean()) / np.sqrt(sq[known].mean())),
            'unseen_minus_seen_rmse': float(np.sqrt(sq[unseen].mean()) - np.sqrt(sq[known].mean())),
            'sse_share_unseen_pct': 100 * float(sq[unseen].sum() / sq.sum()),
            'rmse': base,
            'corrected_rmse_if_unseen_equal_seen': corrected,
            'corrected_cost_sec': base - corrected,
            'fold_matched_cost_sec': base - fold_matched,
            'day_bootstrap_unseen_minus_seen_p025_p50_p975': day_bootstrap([p[4] for p in parts]),
        }
        e = summary[name]
        lo, _, hi = e['day_bootstrap_unseen_minus_seen_p025_p50_p975']
        print(f"   {name:<13} unseen {e['n_unseen']:>6,} ({e['pct_unseen']:5.2f}%)  "
              f"RMSE {e['rmse_unseen']:7.1f} vs {e['rmse_seen']:7.1f}  ×{e['ratio']:.2f}  "
              f"difference {e['unseen_minus_seen_rmse']:+7.1f} s  95% [{lo:+.1f}, {hi:+.1f}]")
        print(f"   {'':<13} hypothetical benefit if they matched the rest: "
              f"{e['corrected_cost_sec']:+.3f} s (fold-matched {e['fold_matched_cost_sec']:+.3f} s); "
              f"missing-key rows {e['n_missing']} at RMSE {e['rmse_missing']:.0f}")

    write_json(OUT, {'run': RUN.name, 'prediction_column': 'pred_combined',
                     'bootstrap': {'reps': BOOTSTRAP_REPS, 'seed': BOOTSTRAP_SEED,
                                   'note': 'days resampled within fold; descriptive, conditional on '
                                           'saved models; does not control airport mix'},
                     'folds': report, 'pooled_A': summary})
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
