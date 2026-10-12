"""Label-free identifiability and coverage pilot for a daily peer encoder.

This is not a taxi validation, model selection, or aggregate-loss experiment.
The retained local context used at most 16 peers within 30 minutes; this pilot
measures additional paired information strictly outside that window but inside
the public departure-total UTC day. No held labels or predictions are read.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import flight_cohort_context as prior
import public_delay_features as public

OUT = ROOT / 'VALIDATION/reports/cohort_learning_20261012'
BANK = prior.OUT / 'train'
DESIGN = dict(
    task='input-only full-day donor coverage and non-identifiability of retained summaries',
    cohort='airport and UTC takeoff day, the existing public departure-delay population',
    donors='explicit VFR excluded; unknown flight rule retained, as in selected public features',
    paired='valid original clock and at least one gated OPDI/ADSB own physical clock',
    new_donors='strictly more than 30 minutes away inside the same airport UTC day',
    aliases='far donors automatically fail the earlier <=120-second input-alias criteria',
    population='all 2025 cached departure inputs; ordinary routing does not remove context donors',
    scope='daily groups cannot mix disjoint allowed UTC months; exact original fold membership retained',
    comparison='closed 16-neighbor local encoder and retained 12 public scalar features',
    interpretation='coverage/identifiability only; no fitted model, predictive gain, or CV claim',
    no_labels=True, no_predictions=True, no_fit=True, no_upload=True)


def art(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest(), bytes=path.stat().st_size)


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')


def collision():
    # All observed own clocks/gaps, source totals, and marginal sensor values
    # are fixed. Only two distant donors' physical clock assignments swap.
    rows = pd.DataFrame(dict(mvt_id=[1, 2, 3], airport='LIRF', day='2025-01-10',
        gap=[6000., 4000., 8000.], clock=[1000., 1000., 1000.], fltrule='I'))
    source = pd.DataFrame(dict(airport=['LIRF'], day=['2025-01-10'], n_public=[3.], total=[12000.]))
    a, b = rows.copy(), rows.copy()
    a['physical'] = [1000., 300., 1700.]
    b['physical'] = [1000., 1700., 300.]
    expected, actual = public.build(a, source), public.build(b, source)
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    np.testing.assert_array_equal(np.sort(a.physical), np.sort(b.physical))
    # Query at noon, donors at 06:00/18:00: neither enters its old local context.
    contribution = lambda x: float(np.mean((x.gap.iloc[1:] - 900) *
        (x.physical.iloc[1:] - x.clock.iloc[1:])))
    assert contribution(a) != contribution(b)
    return dict(retained_12_features_identical=True, physical_marginals_identical=True,
        query_local_context_identical=True, query_clock_and_physical_identical=True,
        joint_donor_moment_a=contribution(a), joint_donor_moment_b=contribution(b),
        proves='additional input information exists; does not establish predictive relevance')


def main():
    started = time.monotonic()
    assert not (OUT / 'pilot_result.json').exists(), 'Preserve completed pilot'
    manifest = json.loads((BANK / 'manifest.json').read_text())
    assert manifest['no_labels_loaded'] and manifest['source'] == prior.art(prior.__file__)
    files = [Path(__file__), Path(prior.__file__), Path(public.__file__), BANK/'manifest.json',
        BANK/'meta.npy', BANK/'values.npy', public.OUT/'inputs2025.parquet']
    identity = dict(design=DESIGN, sources=[art(p) for p in files], before_loading_values=True)
    path = OUT / 'pilot_identity.json'
    if path.exists():
        assert json.loads(path.read_text()) == identity
    else:
        write(path, identity)
    for name in ['meta.npy', 'values.npy']:
        expected = next(x for x in manifest['outputs'] if x['path'].endswith('/' + name))
        assert art(BANK/name) == expected
    meta = np.load(BANK/'meta.npy', mmap_mode='r')
    values = np.load(BANK/'values.npy', mmap_mode='r')
    raw = pd.read_parquet(public.OUT/'inputs2025.parquet', columns=['mvt_id', 'airport', 'day', 'fltrule'])
    raw = raw.set_index('mvt_id').loc[meta[:, 0]].reset_index()
    assert len(raw) == raw.mvt_id.nunique() == 2085047
    np.testing.assert_array_equal(raw.mvt_id, meta[:, 0])
    np.testing.assert_array_equal(pd.to_datetime(meta[:, 1], unit='ms', utc=True).strftime('%Y-%m-%d'), raw.day)
    donor = ~raw.fltrule.eq('V').to_numpy()
    paired = (values[:, 4] > 0) & (values[:, 5:8] > 0).any(1) & donor
    own_absent = ~(values[:, 5:8] > 0).any(1)
    records, group_records = [], []
    groups = raw.groupby(['airport', 'day'], sort=True).indices
    far = np.zeros(len(raw), dtype=np.int32)
    available = np.zeros(len(raw), dtype=np.int32)
    for (airport, day), ix in groups.items():
        ix = np.asarray(ix)
        order = ix[np.argsort(meta[ix, 1], kind='stable')]
        time_ms = meta[order, 1]
        support = paired[order]
        prefix = np.r_[0, np.cumsum(support)]
        left = np.searchsorted(time_ms, time_ms - prior.WINDOW_MS, side='left')
        right = np.searchsorted(time_ms, time_ms + prior.WINDOW_MS, side='right')
        far[order] = prefix[-1] - (prefix[right] - prefix[left])
        available[order] = prefix[-1] - support
        group_records.append(dict(airport=airport, day=day, rows=len(ix),
            donor_rows=int(donor[ix].sum()), paired_rows=int(paired[ix].sum())))
    for month in sorted(raw.day.str[:7].unique()):
        use = raw.day.str.startswith(month).to_numpy()
        for name, mask in [('all', use), ('without_own_physical', use & own_absent)]:
            records.append(dict(month=month, subset=name, rows=int(mask.sum()),
                any_paired_same_day=float((available[mask] > 0).mean()),
                any_paired_outside_30min=float((far[mask] > 0).mean()),
                median_paired_outside_30min=float(np.median(far[mask])),
                p90_paired_outside_30min=float(np.quantile(far[mask], .9))))
    # Independent SQL full-population/calendar/group checks, using only the
    # declared input columns. Fixed evenly spread query positions check the
    # temporal prefix calculation against direct donor predicates.
    count_frame = raw[['mvt_id', 'airport', 'day']].assign(paired=paired, donor=donor,
        time_ms=meta[:, 1], far=far)
    g = pd.DataFrame(group_records)
    with duckdb.connect(config={'threads': 1}) as con:
        con.register('f', count_frame)
        con.register('g', g)
        bad = con.execute('''WITH q AS (SELECT airport,day,count(*) AS rows,
            sum(donor::INTEGER) AS donor_rows,sum(paired::INTEGER) AS paired_rows FROM f GROUP BY airport,day)
            SELECT count(*) FROM q FULL JOIN g USING(airport,day)
            WHERE q.rows IS DISTINCT FROM g.rows OR q.donor_rows IS DISTINCT FROM g.donor_rows
            OR q.paired_rows IS DISTINCT FROM g.paired_rows''').fetchone()[0]
        assert bad == 0
        con.register('query', count_frame.iloc[np.linspace(0, len(raw)-1, 512, dtype=int)])
        checks = con.execute('''SELECT q.mvt_id,q.far,count(d.mvt_id) AS direct FROM query q
            LEFT JOIN f d ON d.airport=q.airport AND d.day=q.day AND d.paired
            AND abs(d.time_ms-q.time_ms)>1800000 GROUP BY q.mvt_id,q.far''').df()
        np.testing.assert_array_equal(checks.far, checks.direct)
    # Complete UTC-day pooling respects both non-contiguous training scopes.
    scope = {}
    for month, allowed in prior.FOLDS.items():
        train = raw.day.str[:7].isin(allowed)
        held = raw.day.str.startswith(month)
        assert not (train & held).any()
        assert set(raw.loc[train, 'day'].str[:7]) == set(allowed)
        scope[month] = dict(allowed=allowed, train_source_rows=int(train.sum()), held_rows=int(held.sum()),
            reference_days_overlap=False, held_labels_loaded=False)
    g.to_parquet(OUT/'pilot_cohorts.parquet', index=False)
    result = dict(design=DESIGN, identity=art(path), collision=collision(), summary=records,
        cohorts=len(g), rows=len(raw), scope=scope, seconds=time.monotonic()-started,
        memory='memory-mapped audited scalar bank; no model matrices/raw track tensors',
        audit=dict(status='pass', independent_sql_group_rows=len(raw), independent_time_queries=len(checks),
            all_ids_and_UTC_dates_exact=True, frozen_bank_hashes_exact=True),
        batch_feasibility=dict(median_source_rows_per_day=float(g.rows.median()),
            p95_source_rows_per_day=float(g.rows.quantile(.95)), max_source_rows_per_day=int(g.rows.max())),
        cohort_artifact=art(OUT/'pilot_cohorts.parquet'),
        no_model_fit=True, no_scores=True, no_candidate=True)
    write(OUT/'pilot_result.json', result)
    print(json.dumps(dict(rows=result['rows'], cohorts=result['cohorts'], seconds=result['seconds'],
        jan_jul=[r for r in records if r['month'] in ['2025-01', '2025-07']],
        batch_feasibility=result['batch_feasibility'], audit=result['audit']), indent=2))


if __name__ == '__main__':
    main()
