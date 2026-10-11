"""Test the public daily total on estimated off-block dates, using retained priors.

Source identity diagnostics use labelled training rows only. Inference assigns
dates from takeoff minus the frozen v70 prediction, never the departure label.
This is a fixed one-pass approximation: uncertain date assignments are not
marginalized, and posterior changes may cross the assigned date boundary.
"""
import argparse
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_delay_posterior as posterior
import daily_delay_ifr as ifr
from records import write_json

base = posterior.base
OUT = base.ROOT / 'VALIDATION/runs/v3_20261011_daily_offblock_alignment'
DESIGN = dict(parent='serialized original v70; no new public-input models',
    control='same retained daily posterior, explicit VFR excluded',
    change='group public totals by takeoff minus incoming prediction, once',
    noise='true off-block dates in allowed training months; exclude calendar edge days',
    population='explicit VFR excluded; unknown included',
    edge='retain matched control on first/last/missing held off-block dates',
    settings=posterior.DESIGN, no_parameter_or_weight_search=True,
    limitation='hard prior date assignment, without marginalizing boundary uncertainty',
    selection='motivated by source-identity diagnostics across reused 2025 data')


def read(path):
    return json.loads(path.read_text())


def dates(epoch, taxi):
    epoch, taxi = np.asarray(epoch, float), np.asarray(taxi, float)
    assert epoch.shape == taxi.shape and np.isfinite(epoch).all() and np.isfinite(taxi).all()
    return pd.to_datetime(epoch-taxi, unit='s', utc=True).strftime('%Y-%m-%d').to_numpy()


def interior(day, month):
    d = pd.DatetimeIndex(pd.to_datetime(day, utc=True))
    return (d.strftime('%Y-%m') == month) & (d.day > 1) & (d.day < d.days_in_month)


def inference_rows(rows, prediction):
    # The whitelist is deliberate: y and recorded block timestamps cannot
    # influence grouping, aggregate likelihoods or posterior inference.
    f = rows[['mvt_id', 'airport', 'takeoff_epoch', 'gap', 'has_id', 'nm_gap', 'fltrule']].copy()
    f['day'] = dates(f.takeoff_epoch, prediction)
    return f


def checks():
    epoch = [pd.Timestamp(s).timestamp() for s in ['2025-07-02T00:10:00Z', '2025-07-02T00:20:00Z']]
    f = pd.DataFrame(dict(mvt_id=[1., 2.], airport='LIRF', takeoff_epoch=epoch,
        gap=[5000., 1000.], has_id=False, nm_gap=np.nan, fltrule=['I', None]))
    p = np.array([1200., 600.])
    actual = inference_rows(f, p)
    assert actual.day.tolist() == ['2025-07-01', '2025-07-02']
    pd.testing.assert_frame_equal(actual, inference_rows(f.assign(y=1e12, true_day='1900-01-01'), p))
    perm = [1, 0]
    pd.testing.assert_frame_equal(actual.iloc[perm], inference_rows(f.iloc[perm], p[perm]))
    np.testing.assert_array_equal(interior(['2025-06-30', '2025-07-01', '2025-07-02',
        '2025-07-30', '2025-07-31', '2025-08-01'], '2025-07'), [False, False, True, True, False, False])
    # Exactly midnight is assigned to the new UTC day; a one-second earlier
    # off-block is assigned to the previous day.
    midnight = pd.Timestamp('2025-07-02T00:00:00Z').timestamp()
    assert dates([midnight+600, midnight+600], [600, 601]).tolist() == ['2025-07-02', '2025-07-01']
    return dict(utc_midnight=True, target_isolation=True, row_order=True, edge_guard=True)


def aggregate_diagnostic(rows, source, name, day):
    f = rows.assign(day=np.asarray(day))
    g = base.aggregate_truth(f, source)
    g['assignment'] = name
    summaries = []
    for airport, q in g.groupby('airport'):
        q = q.dropna(subset=['eps', 'dn']).sort_values('day')
        summaries.append(dict(assignment=name, airport=airport, days=len(q),
            equal_count_days=int(q.dn.eq(0).sum()), median_abs_seconds=float(q.eps.abs().median()),
            p95_abs_seconds=float(q.eps.abs().quantile(.95)), lag1_correlation=float(q.eps.autocorr())))
    return g, summaries


def audit():
    assert not (OUT / 'source_audit.json').exists(), 'Preserve completed audit'
    check_result = checks()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((base.ROOT / 'data').glob('training_2025-*.parquet'))
    cached = base.ROOT / 'VALIDATION/runs/v3_20261011_daily_ifr/rows2025.parquet'
    rows = pd.read_parquet(cached)
    with duckdb.connect(config={'threads': 1}) as con:
        con.execute("SET TimeZone='UTC'")
        clocks = con.execute('''select MVT_ID_mvt mvt_id, epoch(MVT_TIME_UTC_mvt) takeoff_epoch,
            strftime(BLOCK_TIME_UTC_mvt, '%Y-%m-%d') true_day,
            epoch(MVT_TIME_UTC_mvt-BLOCK_TIME_UTC_mvt)-TAXITIME_SEC_mvt target_difference
            from read_parquet(?) where PHASE_mvt='DEP' ''', [list(map(str, paths))]).fetchdf()
    assert clocks.mvt_id.is_unique and len(clocks) == len(rows)
    np.testing.assert_array_equal(clocks.target_difference, np.zeros(len(clocks)))
    rows = rows.merge(clocks.drop(columns='target_difference'), on='mvt_id', validate='one_to_one')
    np.testing.assert_array_equal(dates(rows.takeoff_epoch, rows.y), rows.true_day)
    path = OUT / 'rows2025.parquet'
    rows.to_parquet(path, index=False)
    eligible = rows[~rows.fltrule.eq('V')]
    source = base.public(2025)
    frames, summary = [], []
    for name, day in [('takeoff', eligible.day), ('recorded_offblock', eligible.true_day)]:
        g, s = aggregate_diagnostic(eligible, source, name, day)
        frames.append(g)
        summary.extend(s)
    g = pd.concat(frames, ignore_index=True)
    g.to_parquet(OUT / 'daily_source_identity.parquet', index=False)
    pd.DataFrame(summary).to_csv(OUT / 'source_identity.csv', index=False)
    write_json(OUT / 'source_audit.json', dict(status='pass', rows=len(rows),
        all_recorded_blocks_equal_takeoff_minus_target=True, checks=check_result,
        source=base.artifact(__file__), inputs=[base.artifact(p) for p in paths],
        cached_rows=base.artifact(cached), prepared_rows=base.artifact(path),
        public=base.artifact(base.SOURCE / 'all_pre_departure_delays_2025.csv'),
        aggregates=base.artifact(OUT / 'daily_source_identity.parquet'), statistics=summary,
        interpretation='Descriptive source alignment only; recorded dates cannot be used at ranking inference'))
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


def guard():
    audit_record = read(OUT / 'source_audit.json')
    assert audit_record['status'] == 'pass'
    assert audit_record['source'] == base.artifact(__file__)
    assert audit_record['prepared_rows'] == base.artifact(OUT / 'rows2025.parquet')
    value = dict(design=DESIGN, source=base.artifact(__file__), checks=checks(),
        helpers=[base.artifact(m.__file__) for m in [posterior, base, ifr]],
        source_audit=base.artifact(OUT / 'source_audit.json'),
        prior=base.artifact(posterior.OUT / 'v70/predictions.parquet'),
        prior_result=base.artifact(posterior.OUT / 'v70/result.json'))
    path = OUT / 'identity.json'
    if path.exists():
        assert read(path) == value
    else:
        write_json(path, value)


def screen():
    assert not (OUT / 'result.json').exists(), 'Preserve completed screen'
    guard()
    rows = pd.read_parquet(OUT / 'rows2025.parquet')
    prior = pd.read_parquet(posterior.OUT / 'v70/predictions.parquet')
    aligned = rows.set_index('mvt_id').loc[prior.mvt_id].reset_index()
    np.testing.assert_array_equal(aligned.y, prior.y)
    source = base.public(2025)
    control = prior.parent.to_numpy(float).copy()
    candidate = control.copy()
    logs, params, diagnostics = [], {}, []
    assignments = np.empty(len(prior), dtype=object)
    changed_scope = np.zeros(len(prior), bool)
    for month, allowed in base.FOLDS.items():
        train = rows[rows.month.isin(allowed)]
        assert month not in set(train.month)
        prm_control = ifr.fit(train, source)
        canonical = train[~train.fltrule.eq('V')].assign(day=lambda x: x.true_day)
        # Exclude source-day boundaries around each allowed calendar month.
        safe = np.zeros(len(canonical), bool)
        for m in allowed:
            safe |= interior(canonical.day, m)
        canonical = canonical.loc[safe].copy()
        prm_new = posterior.fit(canonical, source)
        params[month] = dict(control=prm_control, offblock=prm_new, allowed=allowed,
            canonical_training_rows=len(canonical))
        ix = np.flatnonzero(prior.month.eq(month))
        held = aligned.iloc[ix].reset_index(drop=True)
        incoming = prior.parent.iloc[ix].to_numpy(float)
        control[ix], oldlog = ifr.apply(held, incoming, source, prm_control)
        new_rows = inference_rows(held, incoming)
        assignments[ix] = new_rows.day
        eligible = ~new_rows.fltrule.eq('V')
        use = eligible & interior(new_rows.day, month)
        j = np.flatnonzero(use)
        candidate[ix] = control[ix]
        candidate[ix[j]], newlog = posterior.apply(new_rows.iloc[j].reset_index(drop=True),
            incoming[j], source, prm_new)
        changed_scope[ix[j]] = True
        oldlog['variant'], newlog['variant'] = 'takeoff_IFR', 'estimated_offblock'
        oldlog['month'], newlog['month'] = month, month
        logs.extend([oldlog, newlog])
        # Held labels enter diagnostics/scoring only, after inference above.
        diagnostic_rows = held[eligible].copy()
        _, stat = aggregate_diagnostic(diagnostic_rows, source, 'estimated_offblock_'+month,
            new_rows.loc[eligible, 'day'])
        diagnostics.extend(stat)
    f = prior.rename(columns={'pred': 'old_all_departures'}).copy()
    f['control'] = control
    f['pred'] = candidate
    f['assigned_day'] = assignments
    f['new_scope'] = changed_scope
    assert np.isfinite(candidate).all() and (candidate >= 0).all()
    np.testing.assert_array_equal(candidate[~changed_scope], control[~changed_scope])
    f.to_parquet(OUT / 'scored.parquet', index=False)
    pd.concat(logs, ignore_index=True).to_parquet(OUT / 'days.parquet', index=False)
    import tuning_seasonal as ts
    columns = ['parent', 'old_all_departures', 'control', 'pred']
    scores = {m: {c: float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in columns} for m, g in f.groupby('month')}
    result = dict(design=DESIGN, scores=scores,
        cv={c: float(np.mean([v[c] for v in scores.values()])) for c in columns},
        paired={c: ts.paired_bootstrap(f.day, f.month, (f[c]-f.y)**2, (f.pred-f.y)**2)
            for c in ['parent', 'old_all_departures', 'control']},
        parameters=params, estimated_date_source_identity=diagnostics,
        prior_assigned_date_wrong=int((assignments != aligned.true_day.to_numpy()).sum()),
        posterior_crosses_assigned_date=int(((dates(aligned.takeoff_epoch, candidate) != assignments) & changed_scope).sum()),
        new_scope_rows=int(changed_scope.sum()), fallback_rows=int((~changed_scope).sum()),
        identity=base.artifact(OUT / 'identity.json'), predictions=base.artifact(OUT / 'scored.parquet'),
        interpretation='Fixed one-pass source alignment screen; no production selected')
    write_json(OUT / 'result.json', result)
    print(json.dumps({k: result[k] for k in ['cv', 'paired', 'prior_assigned_date_wrong',
        'posterior_crosses_assigned_date']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'screen', 'guard'])
    args = parser.parse_args()
    with threadpool_limits(1):
        globals()[args.stage]()
