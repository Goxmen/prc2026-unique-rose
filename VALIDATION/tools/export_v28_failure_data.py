"""Compact aggregates for the interactive v28 failure page, and the page itself.

Reuses `analyze_v28_failures` so the page and VALIDATION/reports/V28_FAILURES.md quote
the same saved predictions. Refits nothing. Writes the aggregates as JSON, then splices
them into `page_template.html` to produce the self-contained `v28_failures.html`.

    .venv/bin/python VALIDATION/tools/export_v28_failure_data.py
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import write_json
from analyze_v28_failures import (CONTROL, MODEL, PAST, SEEDS, Y_NAMES, frame, label,
                                  load_predictions, numbers, rmse, stats, worst_flights)

SEGMENTS = ['Rome rule rows', 'No flight ID (rule did not fire)', 'LIRF, flight ID present',
            'Other airports, flight ID present']
FLAGS = [('wx_thunder', 'Thunder'), ('wx_freezing', 'Freezing'),
         ('wx_2h_thunder', 'Thunder in the last 2h'),
         ('wx_2h_snow_or_freezing', 'Snow or freezing, last 2h'),
         ('wx_heavy', 'Heavy precipitation'), ('wx_snow', 'Snow'), ('wx_fog', 'Fog'),
         ('wx_deicing', 'De-icing conditions')]


def r(value, places=3):
    return None if value is None or (isinstance(value, float) and not np.isfinite(value)) \
        else round(float(value), places)


def curves(d, tot):
    """Cumulative squared-error share and the perfect-prediction headroom, log-sampled."""
    n = len(d)
    order = np.sort(d['se'].to_numpy())[::-1]
    cum = np.cumsum(order)
    ranks = np.unique(np.geomspace(1, n, 260).astype(int))
    rest = np.clip(tot - cum, 0, None)
    return [[int(k), r(cum[k - 1] / tot, 5), r(np.sqrt(rest[k - 1] / n), 2)] for k in ranks]


def survival(values, points=110):
    v = np.sort(np.asarray(values, float))
    v = v[v > 0]
    n = len(v)
    idx = np.unique(np.clip(n - np.geomspace(1, n, points).astype(int), 0, n - 1))
    return [[r(v[i], 1), r((n - i) / n * 100, 6)] for i in idx]


def heatmap(x, y, bins=46, lo=1.7, hi=5.08):
    """Sparse log-log density: [column, row, count], both axes decades from lo to hi."""
    edges = np.linspace(lo, hi, bins + 1)
    ix = np.clip(np.digitize(np.log10(np.clip(x, 1, None)), edges) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(np.log10(np.clip(y, 1, None)), edges) - 1, 0, bins - 1)
    counts = np.bincount(ix * bins + iy, minlength=bins * bins)
    hit = np.nonzero(counts)[0]
    return dict(bins=bins, lo=lo, hi=hi,
                cells=[[int(k // bins), int(k % bins), int(counts[k])] for k in hit])


def build(d, tot):
    n = len(d)
    keys = numbers(d, tot)
    keys['seed_rmse'] = {str(s): rmse(*(lambda p: (p[MODEL], p['y']))(load_predictions(s)))
                         for s in SEEDS}
    clean = d[~d['broken']]
    lirf = d[d['airport'].eq('LIRF')]
    rule = d[d['rule_pred_queue_all']]

    apt = stats(d, 'airport', tot, n).sort_values('sse', ascending=False)
    apt_clean = stats(clean, 'airport', tot, n).reindex(apt.index)
    control = d.groupby('airport', observed=True).apply(
        lambda p: rmse(p[CONTROL], p['y']), include_groups=False).reindex(apt.index)

    ybin = d.groupby(['y_bin', 'broken'], observed=True)['se'].sum().unstack(fill_value=0)
    ybin_n = d['y_bin'].value_counts().reindex(Y_NAMES)
    ybin_rmse = stats(d, 'y_bin', tot, n).reindex(Y_NAMES)
    gain = d.groupby('y_bin', observed=True).apply(lambda p: pd.Series({
        'past': float(((p[PAST] - p['y']) ** 2).sum() - ((p[CONTROL] - p['y']) ** 2).sum()),
        'all': float(((p[MODEL] - p['y']) ** 2).sum() - ((p[CONTROL] - p['y']) ** 2).sum())}),
        include_groups=False).reindex(Y_NAMES) / float(((d[CONTROL] - d['y']) ** 2).sum()) * 100

    band = pd.qcut(d[MODEL], 20, labels=False, duplicates='drop')
    bias = d.groupby(band, observed=True).agg(bias=('err', 'mean'), mid=(MODEL, 'median'),
                                              n=('se', 'size'))

    rome_kind = np.select(
        [(lirf['y'] - lirf['gap_sec']).abs() <= 60, (lirf['y'] - lirf['gap_sec']).abs() <= 600],
        ['Label = schedule gap (±60 s)', 'Label near the gap (±600 s)'],
        default='Label unrelated to the gap')
    rome = stats(lirf.assign(kind=rome_kind), 'kind', tot, n).reindex(
        ['Label = schedule gap (±60 s)', 'Label near the gap (±600 s)',
         'Label unrelated to the gap'])
    pdec = lirf.groupby(pd.qcut(lirf['p_pred_queue_all'], 10, labels=False, duplicates='drop'),
                        observed=True).agg(n=('se', 'size'), sse=('se', 'sum'),
                                           p=('p_pred_queue_all', 'median'))
    broken_lirf = lirf[lirf['broken'] & (lirf['gap_sec'] > 0)]

    hour = stats(d, 'hour', tot, n).sort_index()
    hour_clean = stats(clean, 'hour', tot, n).sort_index()
    day = d.groupby('date', observed=True).agg(sse=('se', 'sum'), n=('se', 'size'),
                                               broken=('broken', 'sum'))
    cong = d[d['q_apt_takeoffs_prev30'].notna()]
    cdec = pd.qcut(cong['q_apt_takeoffs_prev30'], 10, labels=False, duplicates='drop')
    cong_clean = cong[~cong['broken']]

    counts = d.drop_duplicates('mvt_id').groupby(['airport', 'date'], observed=True)['broken'].sum()
    lam, top = counts.mean(), 14
    obs = counts.clip(upper=top).value_counts().reindex(range(top + 1), fill_value=0)
    pmf, term = [], np.exp(-lam)
    for k in range(top + 1):
        pmf.append(term)
        term = term * lam / (k + 1)
    pmf = np.array(pmf)
    pmf[-1] = max(1 - pmf[:-1].sum(), 0)

    folds = [f'A{i}' for i in range(1, 7)] + [f'B{i}' for i in range(1, 11)]
    seed_folds = {str(s): [r(v, 2) for v in load_predictions(s).groupby('fold', observed=True)
                           .apply(lambda p: rmse(p[MODEL], p['y']), include_groups=False)
                           .reindex(folds)] for s in SEEDS}

    worst = worst_flights(d)
    return dict(
        keys={k: (r(v, 6) if isinstance(v, float) else v) for k, v in keys.items()},
        curve=curves(d, tot),
        survival=dict(clean=survival(clean['abs_err']), broken=survival(d.loc[d['broken'], 'abs_err'])),
        segments=[dict(name=i, n=int(row.n), n_share=r(row.n_share, 6), rmse=r(row.rmse, 1),
                       se_share=r(row.se_share, 5), bias=r(row.bias, 1))
                  for i, row in stats(d, 'segment', tot, n).reindex(SEGMENTS).iterrows()],
        scatter=heatmap(d['y'], d[MODEL]),
        airports=[dict(code=i, n=int(row.n), rmse=r(row.rmse, 1),
                       rmse_clean=r(apt_clean.loc[i, 'rmse'], 1), se_share=r(row.se_share, 5),
                       broken=r(row.broken, 5), bias_clean=r(apt_clean.loc[i, 'bias'], 1),
                       queue_gain=r(row.rmse - control[i], 2)) for i, row in apt.iterrows()],
        y_bins=[dict(name=name, n=int(ybin_n[name]), n_share=r(ybin_n[name] / n, 6),
                     rmse=r(ybin_rmse.loc[name, 'rmse'], 1),
                     se_clean=r(ybin.loc[name, False] / tot * 100, 4),
                     se_broken=r(ybin.loc[name, True] / tot * 100, 4),
                     gain_past=r(gain.loc[name, 'past'], 4), gain_all=r(gain.loc[name, 'all'], 4))
                for name in Y_NAMES],
        bias_bands=[[r(row.mid, 0), r(row.bias, 2), int(row.n)] for _, row in bias.iterrows()],
        rome=dict(
            recorded=[dict(name=i, n=int(row.n), rmse=r(row.rmse, 1), se_share=r(row.se_share, 5),
                           row_share=r(row.n / len(lirf), 5)) for i, row in rome.iterrows()],
            p_deciles=[dict(p=r(row.p, 3), n=int(row.n), rmse=r(np.sqrt(row.sse / row.n), 1),
                            se_share=r(row.sse / tot, 5)) for _, row in pdec.iterrows()],
            rule_sorted=dict(
                rule=[r(v, 1) for v in np.sort(rule['abs_err'].to_numpy())[::-1]],
                model=[r(v, 1) for v in np.sort(
                    np.abs(rule['ordinary_pred_queue_all'] - rule['y']).to_numpy())[::-1]]),
            clean_density=heatmap(lirf.loc[~lirf['broken'] & (lirf['gap_sec'] > 0), 'gap_sec'],
                                  lirf.loc[~lirf['broken'] & (lirf['gap_sec'] > 0), 'y'],
                                  bins=44, lo=1.6, hi=5.08),
            broken_points=[[r(g, 0), r(v, 0), int(f)] for g, v, f in zip(
                broken_lirf['gap_sec'], broken_lirf['y'], broken_lirf['rule_pred_queue_all'])]),
        hours=[dict(hour=int(h), rmse=r(hour.loc[h, 'rmse'], 1),
                    rmse_clean=r(hour_clean.loc[h, 'rmse'], 1), n=int(hour.loc[h, 'n']))
               for h in hour.index],
        days=[[str(i.date()), r(row.sse / tot * 100, 4), int(row.n), int(row.broken)]
              for i, row in day.iterrows()],
        congestion=[dict(mid=r(m, 0), rmse=r(np.sqrt(s / c), 1), n=int(c))
                    for m, s, c in zip(
                        cong.groupby(cdec, observed=True)['q_apt_takeoffs_prev30'].median(),
                        cong.groupby(cdec, observed=True)['se'].sum(),
                        cong.groupby(cdec, observed=True)['se'].size())],
        congestion_clean=[r(np.sqrt(v), 1) for v in cong_clean.groupby(
            pd.qcut(cong_clean['q_apt_takeoffs_prev30'], 10, labels=False, duplicates='drop'),
            observed=True)['se'].mean()],
        weather=[dict(name=name, rmse_on=r(np.sqrt(clean.loc[clean[col] == 1, 'se'].mean()), 1),
                      rmse_off=r(np.sqrt(clean.loc[clean[col] == 0, 'se'].mean()), 1),
                      n=int((clean[col] == 1).sum())) for col, name in FLAGS],
        folds=folds, seed_folds=seed_folds,
        bursts=dict(observed=[r(v / len(counts) * 100, 4) for v in obs],
                    poisson=[r(v * 100, 6) for v in pmf], mean=r(lam, 2),
                    dispersion=r(counts.var() / lam, 1), days=int(len(counts))),
        worst=[dict(airport=row.airport, takeoff=f'{row.mvt_time:%Y-%m-%d %H:%M}',
                    y=int(row.y), pred=r(row[MODEL], 0), err=r(row.err, 0),
                    gap_min=r(row.sched_gap_min, 0), rule=bool(row.rule_pred_queue_all),
                    broken=bool(row.broken), type=label(row.AIRCRAFT_TYPE_mvt),
                    stand=str(row.STAND_mvt), folds=int(row.folds))
               for _, row in worst.iterrows()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'VALIDATION/reports/v28_failures/v28_failures_page.json')
    args = parser.parse_args()
    d = frame(42)
    payload = build(d, float(d['se'].sum()))
    write_json(args.output, payload)
    print(args.output, f'{args.output.stat().st_size / 1024:.0f} KB')

    template = args.output.parent / 'page_template.html'
    page = template.read_text().replace('/*__DATA__*/', args.output.read_text().strip())
    if '/*__DATA__*/' in page:
        raise ValueError(f'No data placeholder in {template}')
    target = args.output.parent / 'v28_failures.html'
    target.write_text(page)
    print(target, f'{target.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
