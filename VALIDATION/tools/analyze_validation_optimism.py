"""Diagnose validation score levels from saved errors and observable populations.

No training, submissions, row deletion or benchmark changes. The two-row removal
is an influence diagnostic only. It never changes the reported experiment score.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import connect
from records import sha256, write_json
from reporting import COVERAGE, IDENTITY, Loss, boot_scores, bootstrap_calendar, same_flights

REPORTS = ROOT / 'VALIDATION/reports'
SEASONAL = ROOT / 'VALIDATION/runs/seasonal_jan_jul'
MONTHS = ['2025-01', '2025-07']


def observations():
    con = connect()  # Explicit UTC, including month-boundary movements.
    try:
        con.execute('SET threads=2')
        return con.sql("""
            WITH visible AS (
                SELECT 'training' AS src, strftime(MVT_TIME_UTC_mvt, '%Y-%m') AS ym,
                       ADEP_mvt AS airport, AOBT_3_flt IS NULL AS missing_aobt,
                       FLIGHT_ID_mvt IS NULL AS missing_id,
                       epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt) AS schedule_gap
                FROM read_parquet('data/training_2025-*.parquet') WHERE PHASE_mvt='DEP'
                UNION ALL
                SELECT 'ranking', strftime(MVT_TIME_UTC_mvt, '%Y-%m'), ADEP_mvt,
                       AOBT_3_flt IS NULL, FLIGHT_ID_mvt IS NULL,
                       epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt)
                FROM read_parquet('data/ranking.parquet') WHERE PHASE_mvt='DEP'
            )
            SELECT src, ym, airport, count(*) AS n,
                   count(*) FILTER (WHERE missing_aobt) AS missing_aobt,
                   count(*) FILTER (WHERE missing_id) AS missing_id,
                   count(*) FILTER (WHERE airport='LIRF' AND missing_id AND schedule_gap>=14400) AS rule4,
                   count(*) FILTER (WHERE airport='LIRF' AND missing_id AND schedule_gap>=21600) AS rule6
            FROM visible GROUP BY src, ym, airport ORDER BY src, ym, airport
        """).df()
    finally:
        con.close()


def baseline_losses(progress, strategy, hashes):
    losses, predictions = {}, {}
    for month in MONTHS:
        parts = []
        for seed in (42, 142, 242):
            entries = [p for p in progress['predictions'].values()
                       if (p['strategy'], p['parent'], p['month'], p['seed'])
                       == (strategy, 'baseline', month, seed)]
            if len(entries) != 1:
                raise ValueError('Incomplete seasonal baseline predictions')
            item = entries[0]
            path = SEASONAL / item['path']
            digest = sha256(path)
            if digest != item['artifacts'][item['path']]:
                raise ValueError('Saved prediction hash changed')
            hashes[str(path.relative_to(ROOT))] = digest
            frame = pd.read_parquet(path).set_index('mvt_id').sort_index()
            frame['mvt_time'] = pd.to_datetime(frame.mvt_time, utc=True)
            if parts:
                same_flights(parts[0], frame)
            parts.append(frame)
        meta = parts[0][IDENTITY + COVERAGE].copy()
        pred = np.stack([p['Q0-4h'].to_numpy() for p in parts])
        squares = (pred - meta.y.to_numpy()[None, :])**2
        losses[month] = Loss(meta, squares, (42, 142, 242))
        predictions[month] = pred
    return losses, predictions


def influence(losses, predictions, ids, weight):
    jan, jul = [losses[m] for m in MONTHS]
    removed = jan.meta.index.isin(ids)
    if removed.sum() != 2:
        raise ValueError('Expected the same two January flights in both strategies')
    total_jan = jan.sq.mean(axis=0).sum()
    removed_sse = jan.sq[:, removed].mean(axis=0).sum()
    original_mse = weight * jan.sq.mean() + (1-weight) * jul.sq.mean()
    remaining_jan_mse = (total_jan-removed_sse) / (len(jan.meta)-2)
    rows = jan.meta.loc[removed, ['mvt_time', 'airport', 'y', 'has_flight_plan']].copy()
    rows['mean_prediction'] = predictions['2025-01'][:, removed].mean(axis=0)
    rows['mean_squared_error'] = jan.sq[:, removed].mean(axis=0)
    rows = rows.reset_index()
    rows['mvt_time'] = rows.mvt_time.astype(str)
    return dict(rows=rows.to_dict('records'),
                january_sse_share=removed_sse/total_jan,
                combined_mse_share=weight*removed_sse/len(jan.meta)/original_mse,
                actual_combined=float(np.sqrt(original_mse)),
                diagnostic_combined_without_two=float(np.sqrt(weight*remaining_jan_mse+(1-weight)*jul.sq.mean())))


def main():
    manifest = REPORTS / 'leaderboard_selected_20260916.json'
    seasonal_path = SEASONAL / 'result.json'
    batch, seasonal = json.loads(manifest.read_text()), json.loads(seasonal_path.read_text())
    if seasonal['status'] != 'complete' or seasonal['signature']['config']['sample'] != 1:
        raise ValueError('Requires the completed full seasonal experiment')
    progress = json.loads((SEASONAL / 'progress.json').read_text())
    weight = seasonal['signature']['ranking_weights']['jan_fraction']
    hashes = {str(p.relative_to(ROOT)):sha256(p) for p in [manifest,seasonal_path,Path(__file__)]}
    old, comparisons = {}, {}
    for item in batch['submissions']:
        code = item['codename']
        path = ROOT / 'VALIDATION/runs' / item['validation_run'] / 'result.json'
        old[code] = json.loads(path.read_text())
        hashes[str(path.relative_to(ROOT))] = sha256(path)
        cv = item['cv_standard']
        comparisons[code] = dict(original_cv=cv, official_lb=item['score'], seasonal={})
        for strategy in ('10m', '11m'):
            scores = seasonal['report']['strategies'][strategy]['modes']['standard']['levels'][code]['scores']
            j = weight*(scores['2025-01']**2-cv['R_A']**2)
            u = (1-weight)*(scores['2025-07']**2-cv['R_B_long']**2)
            original_july = old[code]['report']['folds']['A1']['rmse']
            comparisons[code]['seasonal'][strategy] = dict(
                scores=scores, january_component_mse_increase=j, july_component_mse_increase=u,
                july_share_of_total_mse_increase=u/(j+u), original_july=original_july,
                same_july_rmse_change=scores['2025-07']-original_july,
                hybrid_same_january_original_july=float(np.sqrt(weight*scores['2025-01']**2+(1-weight)*original_july**2)))
            np.testing.assert_allclose(j+u, scores['combined']**2-cv['R_combined']**2, atol=1e-8, rtol=0)
    fold = json.loads((REPORTS / 'fold_alignment.json').read_text())
    result = dict(reference_date=batch['scores_checked_at'], comparisons=comparisons,
                  original_month_weights=fold['headline_month_weights'], focus_candidate='Q0-4h',
                  january_influence={}, conditional_absolute_intervals={}, populations=[],
                  no_fits=True, no_scoring_population_change=True, source_hashes=hashes)
    # Select the two largest January errors in 11m, with deterministic ID ties.
    losses, preds = baseline_losses(progress,'11m',hashes)
    ranked = pd.Series(losses['2025-01'].sq.mean(axis=0),index=losses['2025-01'].meta.index).sort_values(
        ascending=False,kind='stable')
    ids = ranked.index[:2].tolist()
    for strategy in ('11m','10m'):
        if strategy == '10m':
            losses,preds = baseline_losses(progress,strategy,hashes)
        result['january_influence'][strategy] = influence(losses,preds,ids,weight)
        saved = comparisons['Q0-4h']['seasonal'][strategy]['scores']['combined']
        np.testing.assert_allclose(result['january_influence'][strategy]['actual_combined'],saved,atol=1e-9,rtol=0)
        calendar,draw_weights = bootstrap_calendar(losses,2000,0)
        draws = {m:boot_scores(losses[m],calendar,draw_weights) for m in MONTHS}
        draws['combined'] = np.sqrt(weight*draws['2025-01']**2+(1-weight)*draws['2025-07']**2)
        result['conditional_absolute_intervals'][strategy] = {
            m:np.quantile(values[np.isfinite(values)],[.025,.975]).tolist() for m,values in draws.items()}
    for name in [f'A{i}' for i in range(1,7)]:
        row = old['Q0-4h']['report']['folds'][name]
        result['populations'].append(dict(name=name,month=old['Q0-4h']['folds'][name]['val'],
                                         rmse=row['rmse'],ordinary_rmse=row['ordinary']['rmse'],
                                         tail_n=row['tail']['n'],tail_sse_pct=row['tail']['pct_sse']))
    for month,row in seasonal['report']['strategies']['11m']['slices']['Q0-4h'].items():
        result['populations'].append(dict(name='11m',month=month,rmse=row['rmse'],
                                         ordinary_rmse=row['ordinary']['rmse'],tail_n=row['tail']['n'],
                                         tail_sse_pct=row['tail']['pct_sse']))
    visible = observations()
    total = visible.groupby(['src','ym'],sort=True)[['n','missing_aobt','missing_id','rule4','rule6']].sum()
    total['missing_aobt_pct'] = 100*total.missing_aobt/total.n
    if total.loc[('ranking','2026-01'),'n'] != seasonal['signature']['ranking_weights']['jan_rows']:
        raise ValueError('Ranking month counts do not match the UTC scoring population')
    if total.loc[('ranking','2026-07'),'n'] != seasonal['signature']['ranking_weights']['jul_rows']:
        raise ValueError('Ranking July counts do not match the UTC scoring population')
    def missing_fraction(src, month):
        row = total.loc[(src,month)]
        return float(row.missing_aobt/row.n)
    result['effective_missing_aobt_fraction'] = {
        'original_combined':sum(w*missing_fraction('training',m) for m,w in result['original_month_weights'].items()),
        'seasonal':weight*missing_fraction('training','2025-01')+(1-weight)*missing_fraction('training','2025-07'),
        'ranking':weight*missing_fraction('ranking','2026-01')+(1-weight)*missing_fraction('ranking','2026-07'),
    }
    # Sensitivity, not a corrected score: it assumes unverified stability of
    # conditional errors inside the present/missing-clock strata across years.
    result['missing_clock_composition_sensitivity'] = {}
    for strategy in ('10m','11m'):
        mse = []
        for old_month,new_month in zip(MONTHS,['2026-01','2026-07']):
            bins = seasonal['report']['strategies'][strategy]['slices']['Q0-4h'][old_month]['all_airports_aobt3']
            fraction = missing_fraction('ranking',new_month)
            mse.append(fraction*bins['missing']['rmse']**2+(1-fraction)*bins['present']['rmse']**2)
        result['missing_clock_composition_sensitivity'][strategy] = {
            'january':float(np.sqrt(mse[0])), 'july':float(np.sqrt(mse[1])),
            'combined':float(np.sqrt(weight*mse[0]+(1-weight)*mse[1]))}
    result['observable_month_counts'] = total.reset_index().to_dict('records')
    result['observable_airport_counts'] = visible.to_dict('records')
    write_json(REPORTS/'validation_optimism.json',result)
    visible.to_csv(REPORTS/'validation_optimism_observable_counts.csv',index=False)
    print('Candidate: original combined / 10m / 11m / LB')
    for code,row in comparisons.items():
        print(code,*(round(x,4) for x in [row['original_cv']['R_combined'],
                  row['seasonal']['10m']['scores']['combined'],row['seasonal']['11m']['scores']['combined'],row['official_lb']]))
    for strategy,item in result['january_influence'].items():
        print(strategy,'two January rows:',item)
        print('Conditional absolute intervals:',result['conditional_absolute_intervals'][strategy])
    print(total.round(3).to_string())


if __name__ == '__main__':
    main()
