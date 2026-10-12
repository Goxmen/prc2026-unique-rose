"""Describe retained CV errors; no fitting, candidate selection or ranking labels."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/reports/current_error_concentration_20261012'
PRED = ROOT / 'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo/scored.parquet'
RAW = ROOT / 'VALIDATION/runs/v3_20261011_daily_delay/rows2025.parquet'


def artifact(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'result.json').exists(), 'Preserve completed diagnostic'
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        con.read_parquet(str(PRED)).create_view('p')
        con.read_parquet(str(RAW)).create_view('r')
        f = con.execute('''SELECT p.*, r.airport, r.gap, r.has_id, r.nm_gap
            FROM p JOIN r USING(mvt_id)
            WHERE p.y IS NOT DISTINCT FROM r.y AND p.month=r.month AND p.day=r.day''').fetchdf()
        assert len(f) == f.mvt_id.nunique() == 344419
        sql = dict(con.execute('SELECT month, sqrt(avg(pow(three_tabm_runway-y,2))) FROM p GROUP BY month').fetchall())
    f['se'] = (f.three_tabm_runway - f.y) ** 2
    f['old_se'] = (f.control - f.y) ** 2
    f['gap_band'] = pd.cut(f.gap, [-np.inf, 3600, 14400, np.inf],
                           labels=['<=1h', '1-4h', '>4h']).astype(str)
    n = f.groupby('month').size()
    mse = f.groupby('month').se.mean()
    baseline = float(np.sqrt(mse).mean())
    for month, value in sql.items():
        assert abs(value - np.sqrt(mse[month])) < 1e-8
    f['balanced_se'] = f.se / f.month.map(n)
    rows = []
    for keys, g in f.groupby(['airport', 'has_id', 'gap_band'], observed=True):
        remaining = mse - g.groupby('month').se.sum().reindex(n.index, fill_value=0) / n
        rows.append(dict(airport=keys[0], has_id=bool(keys[1]), gap_band=keys[2], rows=len(g),
            balanced_sse_share=float(g.balanced_se.sum() / f.balanced_se.sum()),
            rmse=float(np.sqrt(g.se.mean())), old_rmse=float(np.sqrt(g.old_se.mean())),
            mean_residual=float((g.three_tabm_runway-g.y).mean()),
            oracle_cv_gain=float(baseline-np.sqrt(np.maximum(remaining, 0)).mean()),
            label_over_4h=int((g.y > 14400).sum()), days=g.day.nunique()))
    groups = pd.DataFrame(rows).sort_values('balanced_sse_share', ascending=False)
    groups.to_csv(OUT / 'input_groups.csv', index=False)
    airports = f.groupby(['month', 'airport']).agg(rows=('mvt_id', 'size'),
        mse=('se', 'mean'), old_mse=('old_se', 'mean'), balanced_sse=('balanced_se', 'sum'))
    airports['rmse'] = np.sqrt(airports.mse)
    airports['change'] = airports.rmse - np.sqrt(airports.old_mse)
    airports.to_csv(OUT / 'airports.csv')
    worst = f.nlargest(40, 'balanced_se')
    worst.to_csv(OUT / 'largest_errors.csv', index=False)
    concentration = {str(k): float(f.nlargest(k, 'balanced_se').balanced_se.sum() / f.balanced_se.sum())
                     for k in [1, 10, 100, 1000]}
    result = dict(status='descriptive_only', created_at=datetime.now(timezone.utc).isoformat(),
        rows=len(f), cv=baseline, monthly_rmse=sql, top_k_balanced_sse_fraction=concentration,
        top_groups=groups.head(8).to_dict('records'),
        interpretation='Reused held-out 2025 errors; group oracle gains are unattainable diagnostic bounds, not candidate gains. No attribution to hidden 2026 errors.',
        sources=[artifact(p) for p in [Path(__file__), PRED, RAW]],
        no_fit=True, no_candidate=True, no_upload=True)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(cv=baseline, concentration=concentration, top_groups=result['top_groups'][:5])))


if __name__ == '__main__':
    main()
