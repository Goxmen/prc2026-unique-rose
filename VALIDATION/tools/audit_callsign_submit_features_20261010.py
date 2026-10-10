"""Independent bounded production callsign-clock audit; no departure labels read.

Reconstruct raw intervals and the scoped previous-arrival comparison for a
stratified sample. Reuse the sibling's completed all-row arithmetic audit.
Old/final ranking views deliberately remain separate throughout this audit.
"""
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/derived/callsign_submit_20261010'
TRAIN = ROOT / 'data/derived/aircraft_rotation_20261010/identity.parquet'
REPORT = ROOT / 'tmp/research_20261010/signal_scout/callsign_submit_feature_audit.json'
TABLE_AUDIT = ROOT / 'tmp/research_20261010/error_mechanisms/callsign_production_features_audit.json'
COLS = ['cc_call_start_sec', 'cc_squawk_start_sec', 'cc_previous_call_end_sec',
        'cc_before_own_arrival', 'cc_ambiguous', 'cc_censored']
MONTHS = [f'2025-{i:02d}' for i in range(1, 13)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def equal(a, b):
    return (pd.isna(a) and pd.isna(b)) or np.isclose(a, b, atol=.002, rtol=0)


def scalar(r, raw, history, arrivals):
    """Pandas scalar reference, independent of production SQL/day_clocks."""
    expected = {c: np.nan for c in ['call_start_ms', 'squawk_start_ms',
        'previous_call_end_ms', 'active_interval_count', *COLS]}
    aircraft = raw[raw.hex.eq(r.hex)] if pd.notna(r.hex) else raw.iloc[:0]
    cur, other = aircraft[aircraft.cs.eq(r.cs)], aircraft[~aircraft.cs.eq(r.cs)]
    active = cur[(cur.first_ms - 180000 <= r.to_ms) & (cur.last_ms + 180000 >= r.to_ms)]
    bridged, aid, age = False, None, np.nan
    if len(active):
        envelopes = cur.groupby('day', sort=False).agg(first=('first_ms', 'min'), last=('last_ms', 'max'))
        eligible = envelopes[(envelopes['first'] - 180000 <= r.to_ms) & (envelopes['last'] + 180000 >= r.to_ms)]
        day = sorted(eligible.index, key=lambda d: (d != r.day, abs((pd.Timestamp(d) - pd.Timestamp(r.day)).days), d))[0]

        def earliest(first, intervals):
            initial = first
            while True:
                prior = [a for a, b in intervals if a < first and 0 <= first - b <= 600000]
                if not prior:
                    return first, first != initial
                first = min(prior)

        start, bridged = earliest(envelopes.loc[day, 'first'], list(envelopes[['first', 'last']].itertuples(index=False, name=None)))
        squawk = active.loc[active.first_ms.idxmax()]
        sq = cur[cur.squawk.eq(squawk.squawk)]
        sqstart, _ = earliest(squawk.first_ms, list(sq[['first_ms', 'last_ms']].itertuples(index=False, name=None)))
        previous = other[(other.last_ms >= r.to_ms - 14400000) & (other.last_ms < r.to_ms)]
        end = previous.last_ms.max() if len(previous) else np.nan
        earlier = history[history.hex.eq(r.hex) & history.cs.eq(r.cs)
                          & history.t.dt.strftime('%Y-%m-%d').eq(r.day) & history.t.lt(r.t)]
        prior_t = earlier.t.max().value // 1000000 if len(earlier) else np.nan
        ambiguous = bool(((other.first_ms < r.to_ms) & (other.last_ms > start)).any()
                         or active.squawk.nunique() > 1
                         or (pd.notna(prior_t) and start <= prior_t < r.to_ms))
        expected.update(call_start_ms=start, squawk_start_ms=sqstart,
            previous_call_end_ms=end, active_interval_count=len(active),
            cc_ambiguous=float(ambiguous), cc_censored=float(start % 1800000 <= 20000 or start > r.to_ms))
        for name, key, lower in [(COLS[0], 'call_start_ms', -180),
                                 (COLS[1], 'squawk_start_ms', -180),
                                 (COLS[2], 'previous_call_end_ms', 0)]:
            delta = (r.to_ms - expected[key]) / 1000
            expected[name] = delta if lower <= delta <= 14400 else np.nan
    # Latest strictly preceding landing is chosen before checking in-block or
    # airport validity, matching the frozen conservative reference semantics.
    past = arrivals[arrivals.hex.eq(r.hex) & arrivals.t.lt(r.t)
                    & arrivals.inblock.notna() & arrivals.month.isin(MONTHS + [r.month])]
    if len(past):
        last = past.sort_values('t').iloc[-1]
        aid = int(last.mvt_id)
        gap = (r.t - last.inblock).total_seconds()
        if last.airport == r.airport and 0 < gap <= 259200:
            age = np.float32(gap)
    if pd.notna(expected['call_start_ms']) and pd.notna(age):
        expected['cc_before_own_arrival'] = float((r.to_ms - expected['call_start_ms']) / 1000 > age)
    return expected, dict(bridged=bool(bridged), active_intervals=len(active),
                         arrival_id=aid, valid_previous_arrival=bool(pd.notna(age)))


def main():
    started = time.monotonic()
    design = read(OUT / 'design.json')
    table_audit = read(TABLE_AUDIT)
    assert design['settings']['feature_names'] == COLS
    assert sha(ROOT / 'VALIDATION/callsign_submit_features.py') == design['source_sha256']
    for path, digest in design['helpers'].items():
        assert sha(ROOT / path) == digest
    assert sha(TRAIN) == design['original_identity_sha256']
    con = duckdb.connect(config={'threads': 2, 'memory_limit': '2GB'})
    con.execute("SET TimeZone='UTC'")
    con.read_parquet(str(TRAIN)).create_view('train')
    for view in ['old', 'final']:
        con.read_parquet(str(OUT / view / 'identity.parquet')).create_view(view + '_i')
        con.read_parquet(str(OUT / view / 'shared.parquet')).create_view(view + '_s')
    pair_ids = con.execute('''SELECT o.mvt_id FROM old_i o JOIN final_i f USING(mvt_id)
        WHERE o.phase='DEP' AND o.t<>f.t AND o.hex=f.hex
        ORDER BY hash(o.mvt_id+717) LIMIT 8''').df()
    assert len(pair_ids) == 8
    con.register('paired_ids', pair_ids)
    cache, checked_sources, samples, views = {}, {}, [], []
    source_expected = {}
    for view in ['old', 'final']:
        for rec in read(OUT / view / 'shared.json')['sources']:
            if '/callsigns/' in rec['path']:
                if rec['path'] in source_expected:
                    assert source_expected[rec['path']] == rec['sha256']
                source_expected[rec['path']] = rec['sha256']

    def source(day):
        if day not in cache:
            p = ROOT / 'data/external/adsblol_heatmap/callsigns' / f'{day}.parquet'
            if p.exists():
                key = str(p.relative_to(ROOT))
                assert sha(p) == source_expected[key]
                checked_sources[key] = source_expected[key]
                f = pd.read_parquet(p)
                f = f[f.records >= 3].copy()
                f['cs'] = f.callsign.fillna('').str.replace(' ', '', regex=False).str.upper().str.strip()
                f = f[f.cs.ne('')]
                f['day'] = day
            else:
                f = pd.DataFrame(columns=['hex', 'cs', 'first_ms', 'last_ms', 'squawk', 'day'])
            cache[day] = f
        return cache[day]

    paired = {}
    for view, count, months in [('old', 344841, ['2026-01', '2026-07']),
                               ('final', 670790, ['2026-01', '2026-02', '2026-06', '2026-07'])]:
        manifests = [read(OUT / f'{view}.json'), read(OUT / view / 'identity.json'), read(OUT / view / 'shared.json')]
        for meta in manifests:
            assert meta['design_sha256'] == sha(OUT / 'design.json')
            assert meta['ranking_view'] == view
            assert sha(ROOT / meta['artifact']['path']) == meta['artifact']['sha256']
        meta, identity, shared = manifests
        assert meta['rows'] == shared['rows'] == identity['departure_rows'] == count
        assert meta['per_query_month_reference_months'] == {m: MONTHS + [m] for m in months}
        assert identity['ranking_months'] == months
        assert identity['sources'][0]['path'] == f'data/{"ranking" if view == "old" else "final_ranking"}.parquet'
        assert sha(ROOT / identity['sources'][0]['path']) == identity['sources'][0]['sha256']
        expected_sources = {str((OUT / view / (n + '.parquet')).relative_to(ROOT)) for n in ['identity', 'shared']} | {str(TRAIN.relative_to(ROOT))}
        assert {r['path'] for r in meta['sources']} == expected_sources
        con.read_parquet(str(OUT / f'{view}.parquet')).create_view('f', replace=True)
        con.execute(f'CREATE OR REPLACE VIEW i AS SELECT * FROM {view}_i')
        con.execute(f'CREATE OR REPLACE VIEW s AS SELECT * FROM {view}_s')
        audited = table_audit['views'][view]
        assert audited['all_checks_pass'] and audited['rows'] == count
        assert audited['sha256'] == meta['artifact']['sha256']
        assert audited['nonmissing'] == meta['nonmissing']
        selected = con.execute('''WITH chosen AS (
          SELECT mvt_id FROM paired_ids
          UNION (SELECT mvt_id FROM s ORDER BY hash(mvt_id+817) LIMIT 6)
          UNION (SELECT mvt_id FROM s WHERE cc_ambiguous=1 ORDER BY hash(mvt_id+817) LIMIT 4)
          UNION (SELECT mvt_id FROM s WHERE cc_censored=1 ORDER BY hash(mvt_id+817) LIMIT 4)
          UNION (SELECT mvt_id FROM s WHERE floor(call_start_ms/86400000)<floor(to_ms/86400000)
                 AND cc_call_start_sec IS NOT NULL ORDER BY hash(mvt_id+817) LIMIT 6)
          UNION (SELECT s.mvt_id FROM s JOIN i USING(mvt_id) WHERE i.hex IS NULL ORDER BY hash(s.mvt_id+817) LIMIT 4)
          UNION (SELECT s.mvt_id FROM s JOIN i USING(mvt_id) WHERE i.hex IS NOT NULL AND s.call_start_ms IS NULL ORDER BY hash(s.mvt_id+817) LIMIT 4)
          UNION (SELECT s.mvt_id FROM s JOIN i USING(mvt_id) QUALIFY row_number() OVER(PARTITION BY i.month ORDER BY hash(s.mvt_id+919))<=2)
        ) SELECT s.*,i.hex,i.cs,i.t,i.airport,i.month,f.cc_before_own_arrival
          FROM chosen JOIN s USING(mvt_id) JOIN i USING(mvt_id) JOIN f USING(mvt_id) ORDER BY i.t''').df()
        selected['day'] = selected.t.dt.strftime('%Y-%m-%d')
        # Read only observable departure fields to verify the independently
        # jittered source clock and callsign fallback used by each view.
        con.register('sample_ids', selected[['mvt_id']])
        raw_dep = con.execute('''SELECT MVT_ID_mvt mvt_id,MVT_TIME_UTC_mvt t,
            CALLSIGN_flt,FLIGHT_mvt,ADEP_mvt airport FROM read_parquet(?)
            WHERE PHASE_mvt='DEP' AND MVT_ID_mvt IN(SELECT mvt_id FROM sample_ids)''',
            [str(ROOT / identity['sources'][0]['path'])]).df().set_index('mvt_id').reindex(selected.mvt_id)
        cs = raw_dep.CALLSIGN_flt.fillna('').str.strip()
        cs = cs.where(cs.ne(''), raw_dep.FLIGHT_mvt.fillna('')).str.replace(' ', '', regex=False).str.upper()
        np.testing.assert_array_equal(cs.to_numpy(), selected.cs.to_numpy())
        np.testing.assert_array_equal(raw_dep.t.to_numpy(), selected.t.to_numpy())
        np.testing.assert_array_equal(raw_dep.airport.to_numpy(), selected.airport.to_numpy())
        con.register('picked_hex', selected[['hex']].dropna().drop_duplicates())
        history = con.execute("SELECT hex,cs,t FROM i WHERE phase='DEP' AND hex IN(SELECT hex FROM picked_hex)").df()
        arrivals = con.execute("""SELECT mvt_id,hex,airport,t,inblock,month FROM train WHERE phase='ARR' AND hex IN(SELECT hex FROM picked_hex)
            UNION ALL SELECT mvt_id,hex,airport,t,inblock,month FROM i WHERE phase='ARR' AND hex IN(SELECT hex FROM picked_hex)""").df()
        arrival_sample_ids = []
        for r in selected.itertuples(index=False):
            raw = pd.concat([source((pd.Timestamp(r.day) + pd.Timedelta(days=k)).strftime('%Y-%m-%d')) for k in [-2, -1, 0, 1]], ignore_index=True)
            expected, details = scalar(r, raw, history, arrivals)
            for name, value in expected.items():
                assert equal(getattr(r, name), value), (view, r.mvt_id, name, getattr(r, name), value)
            row = dict(view=view, mvt_id=int(r.mvt_id), day=r.day,
                missing_identity=bool(pd.isna(r.hex)), **details)
            samples.append(row)
            if details['arrival_id'] is not None:
                arrival_sample_ids.append(details['arrival_id'])
            if r.mvt_id in set(pair_ids.mvt_id):
                paired[(view, r.mvt_id)] = dict(to_ms=int(r.to_ms),
                    call_start_ms=float(r.call_start_ms), call_age=float(r.cc_call_start_sec))
        # Supplied arrival taxi-in labels are permitted; no departure label or
        # departure block column is selected anywhere in this auditor.
        con.register('arrival_sample_ids', pd.DataFrame({'mvt_id': arrival_sample_ids}))
        raw_arr = con.execute('''SELECT MVT_ID_mvt mvt_id,MVT_TIME_UTC_mvt t,
            TAXITIME_SEC_mvt taxiin,ADES_mvt airport FROM read_parquet(?)
            WHERE PHASE_mvt='ARR' AND MVT_ID_mvt IN(SELECT mvt_id FROM arrival_sample_ids)''',
            [str(ROOT / identity['sources'][0]['path'])]).df()
        cached_arr = arrivals.set_index('mvt_id').reindex(raw_arr.mvt_id)
        np.testing.assert_array_equal(raw_arr.t.to_numpy(), cached_arr.t.to_numpy())
        np.testing.assert_array_equal(raw_arr.airport.to_numpy(), cached_arr.airport.to_numpy())
        inblock = raw_arr.t + pd.to_timedelta(raw_arr.taxiin.where(raw_arr.taxiin >= 0), unit='s')
        np.testing.assert_array_equal(inblock.to_numpy(), cached_arr.inblock.to_numpy())
        coverage = con.execute('''SELECT i.month,count(*) AS row_count,count(i.hex) identified,
            count(f.cc_call_start_sec) call_start,count(f.cc_before_own_arrival) arrival_comparison
            FROM f JOIN i USING(mvt_id) GROUP BY i.month ORDER BY i.month''').df().to_dict('records')
        views.append(dict(view=view, rows=count, identity_sha256=identity['artifact']['sha256'],
            shared_sha256=shared['artifact']['sha256'], features_sha256=meta['artifact']['sha256'],
            nonmissing=meta['nonmissing'], coverage=coverage, sampled_rows=len(selected),
            raw_departure_input_rows=len(raw_dep), raw_arrival_input_rows=len(raw_arr)))
    jitter = []
    for mid in pair_ids.mvt_id:
        old, final = paired[('old', mid)], paired[('final', mid)]
        delta = (final['to_ms'] - old['to_ms']) / 1000
        assert 0 < abs(delta) <= 12
        same_envelope = equal(old['call_start_ms'], final['call_start_ms'])
        if same_envelope and np.isfinite(old['call_age']) and np.isfinite(final['call_age']):
            assert equal(final['call_age'] - old['call_age'], delta)
        jitter.append(dict(mvt_id=int(mid), takeoff_jitter_sec=delta, same_envelope=bool(same_envelope)))
    assert sum(r['bridged'] for r in samples) >= 2
    assert any(r['missing_identity'] for r in samples)
    report = dict(status='passed', auditor_sha256=sha(Path(__file__)),
        design_sha256=sha(OUT / 'design.json'), views=views,
        reused_table_audit=dict(path=str(TABLE_AUDIT.relative_to(ROOT)), sha256=sha(TABLE_AUDIT)),
        raw_sample_rows=len(samples), raw_sample_values=len(samples) * 10,
        bridge_examples=sum(r['bridged'] for r in samples),
        raw_source_files=len(checked_sources), checked_raw_sources=checked_sources,
        old_final_jitter_pairs=jitter, samples=samples,
        scope='Independent bounded raw-clock and scoped-arrival arithmetic, monthly coverage and separate jittered views; reused hashed sibling all-row table audit. No departure labels read.',
        elapsed_sec=time.monotonic() - started)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['samples', 'checked_raw_sources']}, indent=2))


if __name__ == '__main__':
    main()
