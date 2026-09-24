"""Step 3: relate saved train/ranking similarities to held-out prediction cost.

Uses local files only. No models are fitted and validation weights are untouched.
The compact seasonal inputs are made by export_shift_cost_inputs.py on the desktop.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/reports/train_ranking_step3'
CACHE = ROOT / 'tmp/train_ranking_step3'
INPUTS = ROOT / 'tmp/train_ranking_step3_inputs'
sys.path.insert(0, str(ROOT / 'VALIDATION'))
from reporting import group_loss, load_losses


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def check_identity(frame, reference):
    assert frame.index.is_unique and frame.index.notna().all()
    assert frame.index.isin(reference.index).all(), 'unknown movement ID'
    ref = reference.loc[frame.index]
    assert frame.airport.astype(str).equals(ref.airport.astype(str)), 'airport mismatch'
    dates = pd.to_datetime(frame.mvt_time, utc=True).dt.tz_localize(None)
    np.testing.assert_array_equal(dates.to_numpy(), ref.mvt_time.to_numpy())
    np.testing.assert_array_equal(frame.y.to_numpy(), ref.y.to_numpy())
    assert np.isfinite(frame[['y', 'mean_sq']].to_numpy()).all()
    assert frame.mean_sq.ge(0).all()


def unseen_stands(query, train):
    """Airport-scoped novelty; missing stands never count as unseen."""
    known = pd.MultiIndex.from_frame(train.loc[train.stand.notna(), ['airport', 'stand']])
    keys = pd.MultiIndex.from_frame(query[['airport', 'stand']])
    return query.stand.notna().to_numpy() & ~keys.isin(known)


def masks(frame):
    noid = frame.no_flight_id.eq(1)
    late = frame.takeoff_minus_sched_min.gt(240)
    result = {'all': np.ones(len(frame), bool), 'missing_flight_id': noid,
              'schedule_gap_over_4h': late, 'missing_id_or_long_gap': noid | late,
              'missing_id_and_long_gap': noid & late,
              'snow': frame.wx_snow.eq(1),
              'snow_without_clock_flags': frame.wx_snow.eq(1) & ~(noid | late),
              'missing_stand': frame.stand.isna()}
    for c in ['stand_unseen_in_fold_training', 'new_stand_full_2025']:
        if c in frame:
            result[c] = frame[c].fillna(False)
    for view in ['raw', 'rich']:
        if f'{view}_far5' not in frame or frame[f'{view}_far5'].isna().all():
            continue
        far = frame[f'{view}_far5'].fillna(False)
        result[f'{view}_far5'] = far
        result[f'{view}_far5_without_clock_flags'] = far & ~(noid | late)
        p = frame[f'{view}_propensity']
        for a, b in zip([0, .25, .5, .75], [.25, .5, .75, 1]):
            result[f'{view}_propensity_{a:g}_{b:g}'] = p.ge(a) & (p.lt(b) if b < 1 else p.le(b))
    return {k: np.asarray(v, dtype=bool) for k, v in result.items()}


def cost_stats(frame, mask, value='mean_sq'):
    values = frame[value].to_numpy(float)
    selected = values[mask]
    n = len(selected)
    total, subtotal = values.sum(), selected.sum()
    days = pd.to_datetime(frame.loc[mask, 'date']).dt.strftime('%Y-%m-%d')
    by_day = pd.Series(selected, index=days).groupby(level=0).sum()
    return dict(n=n, population_n=len(frame), row_pct=100*n/len(frame),
                rmse=float(np.sqrt(selected.mean())) if n else None,
                complement_rmse=float(np.sqrt(values[~mask].mean())) if (~mask).any() else None,
                sse=float(subtotal), sse_pct=float(100*subtotal/total) if total else None,
                top1_sse_pct=float(100*selected.max()/subtotal) if subtotal else None,
                top10_sse_pct=float(100*np.sort(selected)[-10:].sum()/subtotal) if subtotal else None,
                top_day_sse_pct=float(100*by_day.max()/subtotal) if subtotal else None,
                n_days=len(by_day))


def slice_rows(frame, **keys):
    rows = []
    for (month, airport), group in frame.groupby(['month', 'airport'], observed=True):
        for name, mask in masks(group).items():
            rows.append(dict(**keys, month=month, airport=airport, group=name,
                             **cost_stats(group, mask)))
    return rows


def neighbour_risk(query, donors, view, k):
    ids = query[[f'{view}_neighbour{i}' for i in range(1, k+1)]].to_numpy()
    loc = donors.index.get_indexer(ids.ravel())
    assert (loc >= 0).all(), 'missing donor OOF loss; do not silently drop neighbours'
    picked = donors.iloc[loc]
    np.testing.assert_array_equal(picked.airport.to_numpy(), np.repeat(query.airport.to_numpy(), k))
    donor_month = pd.to_datetime(picked.date).dt.month.to_numpy()
    np.testing.assert_array_equal(donor_month, np.repeat(pd.to_datetime(query.date).dt.month.to_numpy(), k))
    # All source neighbours must come from other step-2 date blocks, separated
    # from the query by at least one day. Ranking dates naturally differ by year.
    source = query.label.eq(0).to_numpy()
    if source.any():
        delta = np.abs((picked.date.to_numpy().reshape(-1, k)[source]
                        - query.loc[source, 'date'].to_numpy()[:, None]) / np.timedelta64(1, 'D'))
        assert (delta > 1).all()
        assert not (ids[source] == query.index.to_numpy()[source, None]).any()
        assert (picked['fold'].to_numpy().reshape(-1,k)[source]
                != query.loc[source,'fold'].to_numpy()[:,None]).all()
    return picked.mean_sq.to_numpy().reshape(-1, k).mean(axis=1)


def proxy_analysis(sim, loss, model):
    """Evaluate the historical error proxy before describing ranking groups."""
    months = set(pd.to_datetime(loss.date).dt.month)
    query = sim[pd.to_datetime(sim.date).dt.month.isin(months)].copy()
    donors = query[query.label.eq(0)].copy()
    assert set(donors.index) == set(loss.index), 'OOF donor population differs'
    donors['mean_sq'] = loss.loc[donors.index, 'mean_sq']
    summaries, checks, blocks = [], [], []
    for view in ['raw', 'rich']:
        for k in [1, 5, 10]:
            print(f'  {model} {view} k={k}', flush=True)
            risk = neighbour_risk(query, donors, view, k)
            query['proxy_sq'] = risk
            for (case, airport), group in query.groupby(['case', 'airport'], observed=True):
                source = group[group.label.eq(0)].copy()
                actual = donors.loc[source.index, 'mean_sq'].to_numpy()
                # Exactly ceil(20%) rows, deterministic ID tie break, no labels
                # used to define which rows enter the highest-risk group.
                order = np.lexsort((source.index.to_numpy(), -source.proxy_sq.to_numpy()))
                n_top = int(np.ceil(.2 * len(source)))
                high = np.zeros(len(source), bool); high[order[:n_top]] = True
                checks.append(dict(model=model, case=case, airport=airport, view=view, k=k,
                    n=len(source), top_n=n_top,
                    actual_rmse=float(np.sqrt(actual.mean())),
                    proxy_rmse=float(np.sqrt(source.proxy_sq.mean())),
                    high20_actual_sse_pct=float(100*actual[high].sum()/actual.sum()),
                    high20_actual_rmse=float(np.sqrt(actual[high].mean())),
                    other80_actual_rmse=float(np.sqrt(actual[~high].mean()))))
                for fold, part in source.groupby('fold'):
                    blocks.append(dict(model=model, case=case, airport=airport, view=view, k=k,
                        date_block=int(fold), n=len(part),
                        actual_rmse=float(np.sqrt(donors.loc[part.index, 'mean_sq'].mean())),
                        proxy_rmse=float(np.sqrt(part.proxy_sq.mean()))))
                for label, part in group.groupby('label'):
                    for name, mask in masks(part).items():
                        # Empty source novelty is recorded explicitly, not matched
                        # to non-existent 2025 errors for new 2026 stands.
                        stats = cost_stats(part, mask, value='proxy_sq')
                        selected = part.loc[mask]
                        ids = selected[[f'{view}_neighbour{i}' for i in range(1,k+1)]].to_numpy().ravel()
                        unique, counts = np.unique(ids, return_counts=True)
                        contribution = (donors.loc[unique,'mean_sq'].to_numpy()*counts/k)
                        summaries.append(dict(model=model, case=case, airport=airport,
                            domain='ranking' if label else 'source', view=view, k=k, group=name,
                            n=stats['n'], row_pct=stats['row_pct'],
                            neighbour_proxy_rmse=stats['rmse'], proxy_sse_pct=stats['sse_pct'],
                            unique_donors=len(unique),
                            top10_donor_proxy_sse_pct=float(100*np.sort(contribution)[-10:].sum()/contribution.sum()) if contribution.sum() else None))
            if k == 5:
                query[f'{view}_neighbour_proxy_sq5'] = risk
    columns = ['date', 'airport', 'label', 'raw_neighbour_proxy_sq5', 'rich_neighbour_proxy_sq5']
    query[columns].to_parquet(CACHE/f'{model}_neighbour_risk.parquet', compression='zstd')
    return summaries, checks, blocks


def main():
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    design = read(OUT/'design.json')
    delivery = read(ROOT/'VALIDATION/reports/train_ranking_step2/delivery.json')
    path = ROOT/delivery['row_cache']
    assert sha(path) == delivery['row_cache_sha256']
    sim = pd.read_parquet(path).set_index('mvt_id')
    assert sim.index.is_unique
    audit = dict(design_sha256=sha(OUT/'design.json'), source_sha256=sha(Path(__file__)),
                 similarity_sha256=sha(path), prediction_sha256={}, reproduced_scores={},
                 coverage=[], limits=design['limits'], new_taxi_out_fits=0, uploads=0)
    cols = ['mvt_id','date','airport','stand','no_flight_id','takeoff_minus_sched_min','wx_snow']
    features = []
    for month, digest in delivery['source_cache_sha256'].items():
        p = ROOT/'tmp/train_ranking_step1'/f'{month}.parquet'
        assert sha(p) == digest
        f = pd.read_parquet(p, columns=cols); f['month'] = month
        features.append(f)
    features = pd.concat(features).set_index('mvt_id')
    assert features.index.is_unique
    paths = sorted((ROOT/'data').glob('training_*.parquet'))
    with duckdb.connect() as con:
        # DuckDB inherits the laptop timezone; casting TIMESTAMPTZ otherwise
        # moves late-night UTC rows to the following local calendar date.
        con.execute("SET TimeZone='UTC'")
        con.execute('SET threads=6')
        raw = con.execute("""SELECT MVT_ID_mvt AS mvt_id, ADEP_mvt AS airport,
            CAST(MVT_TIME_UTC_mvt AS TIMESTAMP) AS mvt_time, TAXITIME_SEC_mvt AS y
            FROM read_parquet(?) WHERE PHASE_mvt='DEP'""", [[str(p) for p in paths]]).df().set_index('mvt_id')
    assert raw.index.is_unique and set(raw.index) == set(features[features.month.str.startswith('2025')].index)
    features['y'] = raw.y
    for view in ['raw', 'rich']:
        for suffix in ['far5', 'propensity']:
            features[f'{view}_{suffix}'] = sim[f'{view}_{suffix}']
    # Check the raw reference independently against the feature cache.
    np.testing.assert_array_equal(raw.airport.astype(str), features.loc[raw.index,'airport'].astype(str))
    np.testing.assert_array_equal(raw.mvt_time.dt.floor('D'), features.loc[raw.index,'date'])
    actual_rows, all_proxy, all_checks, all_blocks = [], [], [], []

    print('Verify and score original v38 predictions', flush=True)
    run = ROOT/'VALIDATION/runs/v3_20260919_artifact_experiments'
    record = read(run/'result.json')
    recorded = next(x for x in read(ROOT/'VALIDATION/reports/weighted_cv_leaderboard.json')['submissions'] if x['version']==38)
    assert sha(run/'result.json') == recorded['record_sha256']
    for item in record['data_identity']:
        p = ROOT/'data'/item['file']
        if p.exists():
            assert sha(p) == item['sha256']
    names = ['A1','A2','A3','A4','A5','A6','B3','B4','B7']
    record['folds'] = {k:record['folds'][k] for k in names}
    record['prediction_column'] = 'pred_combined'
    for info in record['folds'].values():
        for p in info['predictions'].values():
            digest = sha(run/p)
            assert digest == recorded['prediction_sha256'][p]
            audit['prediction_sha256'][str((run/p).relative_to(ROOT))] = digest
    losses = load_losses(record, run)
    for name, folds in [('R_A', names[:6]), ('R_B_long', names[6:])]:
        loss = group_loss(losses, folds)
        score = float(np.sqrt(loss.sq.mean()))
        np.testing.assert_allclose(score, recorded['expected_scores'][name], rtol=0, atol=1e-8)
        audit['reproduced_scores']['v38_'+name] = score
    a_frames = []
    for fold, loss in losses.items():
        frame = loss.meta.copy(); frame['mean_sq'] = loss.sq.mean(axis=0)
        check_identity(frame, raw)
        info = record['folds'][fold]
        f = features.loc[frame.index].copy(); f['mean_sq'] = frame.mean_sq
        train = features[features.date.between(info['train_start'],info['train_end']) & features.y.between(1,200000)]
        assert not train.index.isin(f.index).any()
        f['stand_unseen_in_fold_training'] = unseen_stands(f,train)
        actual_rows += slice_rows(f, model='v38', protocol='next_month' if fold.startswith('A') else 'long_gap', taxi_fold=fold)
        if fold.startswith('A'):
            a_frames.append(f)
        audit['coverage'].append(dict(model='v38',taxi_fold=fold,month=info['val'],n=len(f)))
    all_a = pd.concat(a_frames); assert all_a.index.is_unique
    all_a.to_parquet(CACHE/'v38_A_losses.parquet',compression='zstd')
    july = all_a[all_a.month.eq('2025-07')]
    results = proxy_analysis(sim,july,'v38')
    all_proxy += results[0]; all_checks += results[1]; all_blocks += results[2]

    print('Verify and score cached seasonal v15 holdouts', flush=True)
    manifest = read(INPUTS/'manifest.json')
    seasonal_path = ROOT/'VALIDATION/runs/seasonal_jan_jul/result.json'
    assert sha(seasonal_path) == manifest['result_sha256']
    seasonal = read(seasonal_path)
    audit['seasonal_export'] = manifest
    for strategy in ['11m', '10m']:
        parts = []
        for month in ['2025-01','2025-07']:
            name = f'v15_{strategy}_{month}.parquet'; info = manifest['files'][name]
            p = INPUTS/name; assert sha(p) == info['sha256']
            f = pd.read_parquet(p).set_index('mvt_id'); check_identity(f,raw)
            assert len(f) == info['rows']
            score = float(np.sqrt(f.mean_sq.mean()))
            expected = seasonal['report']['strategies'][strategy]['modes']['standard']['levels']['Q0-4h']['scores'][month]
            np.testing.assert_allclose(score,expected,rtol=0,atol=1e-8)
            audit['reproduced_scores'][name] = score
            out = features.loc[f.index].copy(); out['mean_sq'] = f.mean_sq
            train = features[features.month.isin(info['job']['train']) & features.y.between(1,200000)]
            assert not train.index.isin(out.index).any()
            assert set(out.index) == set(sim[sim.label.eq(0) & sim.date.dt.strftime('%Y-%m').eq(month)].index)
            out['stand_unseen_in_fold_training'] = unseen_stands(out,train)
            parts.append(out)
            actual_rows += slice_rows(out,model='v15',protocol=strategy,taxi_fold=info['job']['id'])
            audit['coverage'].append(dict(model='v15',protocol=strategy,month=month,n=len(out)))
        full = pd.concat(parts); full.to_parquet(CACHE/f'v15_{strategy}_losses.parquet',compression='zstd')
        results = proxy_analysis(sim,full,'v15_'+strategy)
        all_proxy += results[0]; all_checks += results[1]; all_blocks += results[2]
    pd.DataFrame(actual_rows).to_csv(OUT/'actual_cost.csv',index=False)
    pd.DataFrame(all_proxy).to_csv(OUT/'neighbour_proxies.csv',index=False)
    pd.DataFrame(all_checks).to_csv(OUT/'proxy_validation.csv',index=False)
    pd.DataFrame(all_blocks).to_csv(OUT/'proxy_validation_blocks.csv',index=False)
    audit['missing_current_model'] = 'v40 OOF files absent locally and on desktop; v38 is the main anchor, not the current recipe.'
    audit['january_limit'] = 'January actual errors and neighbour proxies use the older v15 retrospective holdout; no v38/v40 January OOF exists.'
    audit['outputs'] = {p.name:sha(p) for p in OUT.glob('*.csv')}
    audit['cache'] = {str(p.relative_to(ROOT)):sha(p) for p in CACHE.glob('*.parquet')}
    write(OUT/'audit.json',audit)
    print('Complete. No fits, uploads or weighting changes.',flush=True)


if __name__ == '__main__':
    main()
