"""Compact aggregates for the interactive v37 failure page, and the page itself.

v37 is row 37 of SUBMISSIONS.md, "Dayshift-hedge" (LB 280.1743). Its saved
predictions are the `pred_dayshift` column of run v3_20260919T112057144323Z, which
keeps only a few columns. Everything else the page needs is joined back, row for row,
from the runs v37 was derived from:

    v28  pred_queue_all   v3_20260917T172934966809Z  (also the congestion input)
    v29  pred             v3_20260918T151436598224Z  (row 30's control is row 29 exactly)
    v30  pred_stand       v3_20260918T151436598224Z  (also ordinary_stand, p_stand, gap_sec)
    v33  pred_window      v3_20260918T181924599658Z
    v34  pred_noid_cap    v3_20260919T104157403241Z
    v37  pred_dayshift    v3_20260919T112057144323Z

Refits nothing. Reuses the v28 page's helpers so both pages measure the same way, writes
the aggregates as JSON, then splices them into `page_template.html` to produce the
self-contained `v37_failures.html`.

    .venv/bin/python VALIDATION/tools/export_v37_failure_data.py
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from records import write_json
from weather import load_reports, weather_features
from analyze_v28_failures import RAW_COLS, SEEDS, Y_EDGES, Y_NAMES, label, rmse, stats
from export_v28_failure_data import FLAGS, curves, heatmap, r, survival

RUNS = ROOT / 'VALIDATION/runs'
V37_RUN = 'v3_20260919T112057144323Z'
STEPS = [('v28', 'v3_20260917T172934966809Z', 'pred_queue_all'),
         ('v29', 'v3_20260918T151436598224Z', 'pred'),
         ('v30', 'v3_20260918T151436598224Z', 'pred_stand'),
         ('v33', 'v3_20260918T181924599658Z', 'pred_window'),
         ('v34', 'v3_20260919T104157403241Z', 'pred_noid_cap'),
         ('v37', V37_RUN, 'pred_dayshift')]
EXTRA = {'v3_20260917T172934966809Z': ['q_apt_takeoffs_prev30'],
         'v3_20260918T151436598224Z': ['ordinary_stand', 'p_stand', 'gap_sec'],
         V37_RUN: ['mvt_id', 'mvt_time', 'airport', 'y', 'fold', 'month', 'raw_flight_id_missing',
                   'rule_triggered']}
MODEL, OLD, RULE, ORDINARY, P = 'v37', 'v28', 'v34', 'ordinary_stand', 'p_stand'
POPS = ['Rome rule rows', 'Rome, no flight ID, rule silent', 'Rome, flight ID present',
        'Other airports, no flight ID', 'Other airports, flight ID present']
FOLDS = [f'A{i}' for i in range(1, 7)] + [f'B{i}' for i in range(1, 11)]


def load_predictions(seed):
    """Pooled out-of-fold rows for one seed, one column per version, checked row for row."""
    parts = []
    for fold in FOLDS:
        name = f'oof_{fold}_seed{seed}.parquet'
        base = pd.read_parquet(RUNS / V37_RUN / name, columns=EXTRA[V37_RUN])
        cols = {}
        for run in dict.fromkeys([run for _, run, _ in STEPS]):
            want = [c for step, rr, c in STEPS if rr == run] + [c for c in EXTRA.get(run, [])
                                                                   if c not in base]
            got = pd.read_parquet(RUNS / run / name, columns=['mvt_id', 'y'] + want)
            if not (np.array_equal(got['mvt_id'], base['mvt_id']) and np.array_equal(got['y'], base['y'])):
                raise ValueError(f'{run}/{name} is not row-aligned with v37')
            cols.update({c: got[c].to_numpy() for c in want})
        frame = base.assign(**cols)
        parts.append(frame.rename(columns={c: step for step, _, c in STEPS}))
    d = pd.concat(parts, ignore_index=True)
    d['mvt_id'] = d['mvt_id'].astype('int64')
    d['population'] = population(d)
    return d


def population(d):
    lirf, broken = d['airport'].eq('LIRF'), d['raw_flight_id_missing'].astype(bool)
    return np.select([d['rule_triggered'].astype(bool), lirf & broken, lirf, broken],
                     POPS[:4], default=POPS[4])


def load_context(ids):
    """Raw movement/flight columns (plus the flight number) and METAR weather."""
    cols = RAW_COLS + ['FLIGHT_mvt', 'FLIGHT_ID_mvt', 'LOBT_flt']
    raw = pd.concat([pd.read_parquet(f, columns=cols)
                     for f in sorted((ROOT / 'data').glob('training_2025-*.parquet'))],
                    ignore_index=True)
    raw['MVT_ID_mvt'] = raw['MVT_ID_mvt'].astype('int64')
    raw = raw[raw['MVT_ID_mvt'].isin(ids)].reset_index(drop=True)
    if len(raw) != len(ids):
        raise ValueError(f'Context join covers {len(raw)} of {len(ids)} scored flights')
    return pd.concat([raw, weather_features(raw, load_reports())], axis=1)


def frame(seed=42):
    d = load_predictions(seed)
    d = d.merge(load_context(pd.Index(d['mvt_id'].unique())), left_on='mvt_id',
                right_on='MVT_ID_mvt', how='left', validate='many_to_one')
    d['err'] = d[MODEL] - d['y']
    d['se'] = d['err'] ** 2
    d['abs_err'] = d['err'].abs()
    d['se_old'] = (d[OLD] - d['y']) ** 2
    d['broken'] = d['raw_flight_id_missing'].astype(bool)
    d['rule'] = d['rule_triggered'].astype(bool)
    d['hour'] = d['mvt_time'].dt.hour
    d['date'] = d['mvt_time'].dt.normalize().dt.tz_localize(None)
    d['y_bin'] = pd.cut(d['y'], Y_EDGES, labels=Y_NAMES, right=False)
    d['sched_gap_min'] = (d['MVT_TIME_UTC_mvt'] - d['SCHED_TIME_UTC_mvt']).dt.total_seconds() / 60
    return d


def numbers(d, tot):
    n = len(d)
    cum = np.cumsum(np.sort(d['se'].to_numpy())[::-1]) / tot
    rule, clean = d[d['rule']], d[~d['broken']]
    miss_type = d['AIRCRAFT_TYPE_mvt'].isna()
    per_day = d.groupby('date')['se'].sum()
    normal = d[d['population'].eq(POPS[4])]
    long_taxis = {str(c): dict(rows=float((normal['y'] >= c).mean()),
                               se_share=float(normal.loc[normal['y'] >= c, 'se'].sum() / tot),
                               bias=float(normal.loc[normal['y'] >= c, 'err'].mean()),
                               median_y=float(normal.loc[normal['y'] >= c, 'y'].median()),
                               median_pred=float(normal.loc[normal['y'] >= c, MODEL].median()))
                  for c in (1800, 2400, 3600)}
    return dict(long_taxis=long_taxis,
        run=V37_RUN, seed=42, rows=n, flights=int(d['mvt_id'].nunique()),
        rmse=rmse(d[MODEL], d['y']), rmse_v28=rmse(d[OLD], d['y']),
        rmse_steps={step: rmse(d[step], d['y']) for step, _, _ in STEPS},
        top_share={str(k): float(cum[k - 1]) for k in (100, 1_000, 10_000, 100_000)},
        top_share_v28={str(k): float(v) for k, v in zip(
            (100, 1_000, 10_000, 100_000), (np.cumsum(np.sort(d['se_old'].to_numpy())[::-1])
                                            / d['se_old'].sum())[[99, 999, 9_999, 99_999]])},
        rmse_if_worst_exact={str(k): float(np.sqrt((tot - cum[k - 1] * tot) / n))
                             for k in (100, 1_000, 10_000)},
        broken_rows=int(d['broken'].sum()), broken_share=float(d['broken'].mean()),
        broken_rmse=rmse(d.loc[d['broken'], MODEL], d.loc[d['broken'], 'y']),
        broken_se_share=float(d.loc[d['broken'], 'se'].sum() / tot),
        broken_se_share_v28=float(d.loc[d['broken'], 'se_old'].sum() / d['se_old'].sum()),
        clean_rmse=rmse(clean[MODEL], clean['y']), clean_rmse_v28=rmse(clean[OLD], clean['y']),
        rule_rows=int(len(rule)), rule_se_share=float(rule['se'].sum() / tot),
        rule_rmse=rmse(rule[MODEL], rule['y']), rule_rmse_v34=rmse(rule[RULE], rule['y']),
        rule_model_only_rmse=rmse(rule[ORDINARY], rule['y']),
        rule_exact_rows=int((rule['abs_err'] <= 60).sum()),
        rule_bad_rows=int((rule['abs_err'] > 3600).sum()),
        rule_bad_se_share=float(rule.loc[rule['abs_err'] > 3600, 'se'].sum() / tot),
        rule_hedged_rows=int((rule[MODEL] != rule[RULE]).sum()),
        rmse_without_rule=float(np.sqrt(
            (tot - rule['se'].sum() + ((rule[ORDINARY] - rule['y']) ** 2).sum()) / n)),
        lirf_se_share=float(d.loc[d['airport'].eq('LIRF'), 'se'].sum() / tot),
        lirf_se_share_v28=float(d.loc[d['airport'].eq('LIRF'), 'se_old'].sum() / d['se_old'].sum()),
        lirf_clean_rmse=rmse(*(lambda p: (p[MODEL], p['y']))(d[d['airport'].eq('LIRF') & ~d['broken']])),
        missing_type_rows=int(miss_type.sum()),
        missing_type_rmse=rmse(d.loc[miss_type, MODEL], d.loc[miss_type, 'y']),
        missing_type_se_share=float(d.loc[miss_type, 'se'].sum() / tot),
        tail_3600_rows=int((d['y'] > 3600).sum()),
        tail_3600_se_share=float(d.loc[d['y'] > 3600, 'se'].sum() / tot),
        tail_precision=float((d.loc[d[MODEL] > 3600, 'y'] > 3600).mean()),
        tail_recall=float((d.loc[d['y'] > 3600, MODEL] > 3600).mean()),
        tail_precision_v28=float((d.loc[d[OLD] > 3600, 'y'] > 3600).mean()),
        tail_recall_v28=float((d.loc[d['y'] > 3600, OLD] > 3600).mean()),
        y_p999=float(np.percentile(d['y'], 99.9)), pred_p999=float(np.percentile(d[MODEL], 99.9)),
        worst_day=str(per_day.idxmax().date()), worst_day_share=float(per_day.max() / tot))


def ladder():
    """Squared error per population for every version, all three seeds pooled."""
    sse, n = {}, {}
    for seed in SEEDS:
        d = load_predictions(seed)
        for step, _, _ in STEPS:
            s = ((d[step] - d['y']) ** 2).groupby(d['population']).sum()
            for pop, v in s.items():
                sse[step, pop] = sse.get((step, pop), 0.0) + float(v)
        for pop, v in d['population'].value_counts().items():
            n[pop] = n.get(pop, 0) + int(v)
    rows = sum(n.values())
    base = sum(sse['v28', pop] for pop in POPS)
    return dict(
        rows=rows,
        populations=[dict(name=pop, n=n[pop]) for pop in POPS],
        steps=[dict(name=step, rmse=r(np.sqrt(sum(sse[step, pop] for pop in POPS) / rows), 3),
                    share=[r(sse[step, pop] / base * 100, 3) for pop in POPS],
                    rmse_pop=[r(np.sqrt(sse[step, pop] / n[pop]), 1) for pop in POPS])
               for step, _, _ in STEPS])


def hedged(d, tot):
    """Rome rule rows the day-shift hedge moved, one line per distinct flight."""
    moved = d[d[MODEL] != d[RULE]]
    kinds = {row['mvt_id']: row['kind'] for rows in json.loads(
        (RUNS / V37_RUN / 'result.json').read_text())['dayshift_hedge']['moved'].values() for row in rows}
    out = []
    for mvt, g in moved.groupby('mvt_id'):
        row = g.iloc[0]
        out.append(dict(
            flight=label(row['FLIGHT_mvt']), takeoff=f"{row['mvt_time']:%Y-%m-%d %H:%M}",
            late_h=r(row['sched_gap_min'] / 60, 1), kind=kinds[mvt], y=int(row['y']),
            before=r(row[RULE], 0), after=r(g[MODEL].mean(), 0),
            after_lo=r(g[MODEL].min(), 0), after_hi=r(g[MODEL].max(), 0), folds=int(len(g)),
            change=r((g['se'].sum() - ((g[RULE] - g['y']) ** 2).sum()) / tot * 100, 3)))
    return sorted(out, key=lambda x: x['change'])


def hedge_rows(d):
    """Every fold row the hedge moved, with both labels it weighs: bet(q) = q * wrong + (1 - q) * planned."""
    moved = d[d[MODEL] != d[RULE]]
    weights = {(fold, row['mvt_id']): row['p'] for key, rows in json.loads(
        (RUNS / V37_RUN / 'result.json').read_text())['dayshift_hedge']['moved'].items()
        for fold in [key.split('/')[0]] if key.endswith('/42') for row in rows}
    out = []
    for row in moved.itertuples():
        p = weights[row.fold, row.mvt_id]
        wrong = (getattr(row, MODEL) - (1 - p) * getattr(row, RULE)) / p
        out.append(dict(flight=label(row.FLIGHT_mvt), fold=row.fold, planned=r(getattr(row, RULE), 0),
                        wrong=r(wrong, 0), p=r(p, 3), y=int(row.y)))
    return out


def queue_day(d, airport='LTFM', day='2025-07-26', start='13:30', end='15:10'):
    """One afternoon of departures at one airport, for the runway timeline."""
    g = d[d['airport'].eq(airport) & d['date'].eq(pd.Timestamp(day))].drop_duplicates('mvt_id')
    t = g['mvt_time'].dt.tz_convert('UTC')
    g = g[(t >= pd.Timestamp(f'{day} {start}', tz='UTC')) & (t <= pd.Timestamp(f'{day} {end}', tz='UTC'))]
    minutes = lambda s: ((s - pd.Timestamp(day, tz='UTC')).dt.total_seconds() / 60).round(2)
    return dict(airport=airport, day=day, rows=[
        dict(flight=label(f), runway=str(rw), type=label(ty), sched=float(sc), off=float(ob), takeoff=float(to),
             y=int(y), guess=r(v, 0))
        for f, rw, ty, sc, ob, to, y, v in zip(
            g['FLIGHT_mvt'], g['RUNWAY_mvt'], g['AIRCRAFT_TYPE_mvt'], minutes(g['SCHED_TIME_UTC_mvt']),
            minutes(g['BLOCK_TIME_UTC_mvt']), minutes(g['mvt_time']), g['y'], g[MODEL])])


def build(d, tot):
    n = len(d)
    keys = numbers(d, tot)
    keys['seed_rmse'] = {str(s): rmse(*(lambda p: (p[MODEL], p['y']))(load_predictions(s)))
                         for s in SEEDS}
    clean = d[~d['broken']]
    lirf = d[d['airport'].eq('LIRF')]
    rule = d[d['rule']]
    old_tot = float(d['se_old'].sum())

    apt = stats(d, 'airport', tot, n).sort_values('sse', ascending=False)
    apt_clean = stats(clean, 'airport', tot, n).reindex(apt.index)
    apt_old = d.groupby('airport')['se_old'].mean().pow(.5).reindex(apt.index)

    ybin = d.groupby(['y_bin', 'broken'], observed=True)[['se', 'se_old']].sum()
    ybin_n = d['y_bin'].value_counts().reindex(Y_NAMES)
    ybin_rmse = stats(d, 'y_bin', tot, n).reindex(Y_NAMES)

    def ycell(name, broken, col):
        return float(ybin[col].get((name, broken), 0.0))

    band = pd.qcut(d[MODEL], 20, labels=False, duplicates='drop')
    bias = d.groupby(band, observed=True).agg(bias=('err', 'mean'), mid=(MODEL, 'median'), n=('se', 'size'))

    rome_kind = np.select(
        [(lirf['y'] - lirf['gap_sec']).abs() <= 60, (lirf['y'] - lirf['gap_sec']).abs() <= 600],
        ['Record = lateness (±1 min): typed in', 'Record near the lateness (±10 min)'],
        default='Record unrelated to the lateness')
    rome = stats(lirf.assign(kind=rome_kind), 'kind', tot, n).reindex(
        ['Record = lateness (±1 min): typed in', 'Record near the lateness (±10 min)',
         'Record unrelated to the lateness'])
    pdec = lirf.groupby(pd.qcut(lirf[P], 10, labels=False, duplicates='drop'), observed=True).agg(
        n=('se', 'size'), sse=('se', 'sum'), p=(P, 'median'))
    broken_lirf = lirf[lirf['broken'] & (lirf['gap_sec'] > 0)]

    hour = stats(d, 'hour', tot, n).sort_index()
    hour_clean = stats(clean, 'hour', tot, n).sort_index()
    day = d.groupby('date', observed=True).agg(sse=('se', 'sum'), n=('se', 'size'), broken=('broken', 'sum'))
    cong = d[d['q_apt_takeoffs_prev30'].notna()]
    cdec = pd.qcut(cong['q_apt_takeoffs_prev30'], 10, labels=False, duplicates='drop')
    cong_clean = cong[~cong['broken']]

    normal = d[~d['broken'] & ~d['airport'].eq('LIRF')]
    month = [dict(month=str(m), n=int(len(g)), rmse=r(np.sqrt(g['se'].mean()), 1),
                  rmse_v28=r(np.sqrt(g['se_old'].mean()), 1),
                  rmse_normal=r(np.sqrt(normal.loc[normal['month'] == m, 'se'].mean()), 1),
                  rmse_lirf_id=r(np.sqrt(g.loc[g['airport'].eq('LIRF') & ~g['broken'], 'se'].mean()), 1),
                  se_share=r(g['se'].sum() / tot, 5))
             for m, g in d.groupby('month', observed=True)]
    july = normal[normal['month'].astype(str).str.endswith('-07')]
    july_days = (july.groupby(['airport', 'date'])['se'].agg(['sum', 'size', 'mean'])
                 .sort_values('sum', ascending=False).head(12))
    july_apt = july.groupby('airport')['se'].agg(['sum', 'size']).sort_values('sum', ascending=False)

    seed_folds = {str(s): [r(v, 2) for v in load_predictions(s).groupby('fold', observed=True)
                           .apply(lambda p: rmse(p[MODEL], p['y']), include_groups=False)
                           .reindex(FOLDS)] for s in SEEDS}

    worst = (d.assign(folds=d.groupby('mvt_id')['fold'].transform('size'))
             .sort_values('se', ascending=False).drop_duplicates('mvt_id').head(60))
    return dict(
        keys={k: (r(v, 6) if isinstance(v, float) else v) for k, v in keys.items()},
        curve=curves(d, tot),
        curve_v28=[[k, s] for k, s, _ in curves(d.assign(se=d['se_old']), old_tot)],
        survival=dict(clean=survival(clean['abs_err']), broken=survival(d.loc[d['broken'], 'abs_err'])),
        segments=[dict(name=i, n=int(row.n), n_share=r(row.n_share, 6), rmse=r(row.rmse, 1),
                       rmse_v28=r(np.sqrt(d.loc[d['population'] == i, 'se_old'].mean()), 1),
                       se_share=r(row.se_share, 5), bias=r(row.bias, 1),
                       se_share_v28=r(d.loc[d['population'] == i, 'se_old'].sum() / old_tot, 5))
                  for i, row in stats(d, 'population', tot, n).reindex(POPS).iterrows()],
        scatter=heatmap(d['y'], d[MODEL]),
        airports=[dict(code=i, n=int(row.n), rmse=r(row.rmse, 1), rmse_v28=r(apt_old[i], 1),
                       rmse_clean=r(apt_clean.loc[i, 'rmse'], 1), se_share=r(row.se_share, 5),
                       broken=r(row.broken, 5), bias_clean=r(apt_clean.loc[i, 'bias'], 1),
                       change=r(row.rmse - apt_old[i], 2)) for i, row in apt.iterrows()],
        y_bins=[dict(name=name, n=int(ybin_n[name]), n_share=r(ybin_n[name] / n, 6),
                     rmse=r(ybin_rmse.loc[name, 'rmse'], 1),
                     se_clean=r(ycell(name, False, 'se') / tot * 100, 4),
                     se_broken=r(ycell(name, True, 'se') / tot * 100, 4),
                     gain_clean=r((ycell(name, False, 'se') - ycell(name, False, 'se_old')) / old_tot * 100, 4),
                     gain_broken=r((ycell(name, True, 'se') - ycell(name, True, 'se_old')) / old_tot * 100, 4))
                for name in Y_NAMES],
        bias_bands=[[r(row.mid, 0), r(row.bias, 2), int(row.n)] for _, row in bias.iterrows()],
        rome=dict(
            recorded=[dict(name=i, n=int(row.n), rmse=r(row.rmse, 1), se_share=r(row.se_share, 5),
                           row_share=r(row.n / len(lirf), 5)) for i, row in rome.iterrows()],
            p_deciles=[dict(p=r(row.p, 3), n=int(row.n), rmse=r(np.sqrt(row.sse / row.n), 1),
                            se_share=r(row.sse / tot, 5)) for _, row in pdec.iterrows()],
            rule_sorted=dict(
                v37=[r(v, 1) for v in np.sort(rule['abs_err'].to_numpy())[::-1]],
                v34=[r(v, 1) for v in np.sort(np.abs(rule[RULE] - rule['y']).to_numpy())[::-1]],
                model=[r(v, 1) for v in np.sort(np.abs(rule[ORDINARY] - rule['y']).to_numpy())[::-1]]),
            clean_density=heatmap(lirf.loc[~lirf['broken'] & (lirf['gap_sec'] > 0), 'gap_sec'],
                                  lirf.loc[~lirf['broken'] & (lirf['gap_sec'] > 0), 'y'],
                                  bins=44, lo=1.6, hi=5.08),
            broken_points=[[r(g, 0), r(v, 0), int(f)] for g, v, f in zip(
                broken_lirf['gap_sec'], broken_lirf['y'], broken_lirf['rule'])]),
        hedged=hedged(d, tot),
        hedge_rows=hedge_rows(d),
        queue=queue_day(d),
        hours=[dict(hour=int(h), rmse=r(hour.loc[h, 'rmse'], 1),
                    rmse_clean=r(hour_clean.loc[h, 'rmse'], 1), n=int(hour.loc[h, 'n']))
               for h in hour.index],
        days=[[str(i.date()), r(row.sse / tot * 100, 4), int(row.n), int(row.broken)]
              for i, row in day.iterrows()],
        months=month,
        airport_months=[dict(code=a, month=str(m), n=int(len(g)), rmse=r(np.sqrt(g['se'].mean()), 1),
                             rmse_v28=r(np.sqrt(g['se_old'].mean()), 1))
                        for (a, m), g in clean.groupby(['airport', 'month'], observed=True)],
        airport_days={a: [r(v / tot * 100, 4) for v in g.groupby('date')['se'].sum().reindex(day.index, fill_value=0)]
                      for a, g in d.groupby('airport', observed=True)},
        july=dict(
            normal_rmse=r(np.sqrt(july['se'].mean()), 1),
            other_rmse=r(np.sqrt(normal.loc[~normal.index.isin(july.index), 'se'].mean()), 1),
            airports=[dict(code=a, share=r(row['sum'] / july['se'].sum(), 4), n=int(row['size']))
                      for a, row in july_apt.iterrows()],
            days=[dict(airport=a, date=str(t.date()), rmse=r(np.sqrt(row['mean']), 0), n=int(row['size']),
                       share=r(row['sum'] / july['se'].sum(), 4)) for (a, t), row in july_days.iterrows()]),
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
        folds=FOLDS, seed_folds=seed_folds,
        ladder=ladder(),
        worst=[dict(airport=row.airport, flight=label(row.FLIGHT_mvt), takeoff=f'{row.mvt_time:%Y-%m-%d %H:%M}',
                    y=int(row.y), pred=r(row[MODEL], 0), pred_v28=r(row[OLD], 0), err=r(row.err, 0),
                    gap_min=r(row.sched_gap_min, 0), rule=bool(row.rule), broken=bool(row.broken),
                    type=label(row.AIRCRAFT_TYPE_mvt), stand=str(row.STAND_mvt), folds=int(row.folds),
                    share=r(row.se / tot * 100, 3), population=POPS.index(row.population))
               for _, row in worst.iterrows()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'VALIDATION/reports/v37_failures/v37_failures_page.json')
    args = parser.parse_args()
    d = frame(42)
    payload = build(d, float(d['se'].sum()))
    write_json(args.output, payload)
    print(args.output, f'{args.output.stat().st_size / 1024:.0f} KB')

    template = args.output.parent / 'page_template.html'
    if not template.exists():
        return
    page = template.read_text().replace('/*__DATA__*/', args.output.read_text().strip())
    if '/*__DATA__*/' in page:
        raise ValueError(f'No data placeholder in {template}')
    target = args.output.parent / 'v37_failures.html'
    target.write_text(page)
    print(target, f'{target.stat().st_size / 1024:.0f} KB')


if __name__ == '__main__':
    main()
