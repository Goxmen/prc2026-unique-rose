"""Bounded falsifier for shared flight-plan clock error across both endpoints.

Coverage/matching is input-only. Target diagnostics use twelve dates inside both
outer training folds; no January/July departure labels are loaded. No fit,
candidate, new gate selection, external download or frozen-source modification.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
from records import sha256, write_json

OUT = ROOT / 'tmp/research_20261012/error_mechanisms/paired_endpoint'
PHASE_MANIFEST = ROOT / 'data/derived/source_clock_phase_20261012/manifest.json'
MONTHS = ['2025-03', '2025-09', '2025-10', '2025-11']
DATES = [f'{month}-{day:02d}' for month in MONTHS for day in [5, 15, 25]]
AIRPORTS = ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
DESIGN = dict(
    mechanism='A common NM trajectory timestamp error should appear at both departure AOBT3 and destination ARVT3.',
    match='Unique original FLIGHT_ID per phase, equal movement route and callsign; both plan routes agree with movement route; known aircraft types agree; destination landing 10min..20h after takeoff.',
    endpoint_feature='same-flight destination original landing minus original ARVT3, seconds',
    source_views='all2025, old2026, final2026 isolated; all matching uses only own view inputs',
    pilot_dates=DATES, pilot_reference_months=MONTHS,
    target='departure BLOCK minus genuine AOBT3, training diagnostics only',
    diagnostics='fixed absolute landing discrepancy >60/>600, Pearson after clipping both residuals to +/-3600, direct additive-error comparison and route-demeaned association',
    temporal='each of four training months and March+September versus October+November',
    no_held_departure_labels=True, no_matching_with_departure_block=True,
    no_model_fit=True, no_candidate=True, no_new_threshold_selection=True)


def make_inputs(con, files):
    con.execute('''CREATE OR REPLACE TEMP TABLE source AS SELECT
        MVT_ID_mvt::BIGINT mvt_id, FLIGHT_ID_mvt::BIGINT flight_id,
        PHASE_mvt phase, ADEP_mvt origin, ADES_mvt destination,
        ADEP_flt plan_origin, ADES_flt plan_destination,
        CALLSIGN_flt callsign, AIRCRAFT_TYPE_mvt aircraft,
        epoch(MVT_TIME_UTC_mvt) event_sec, epoch(AOBT_3_flt) aobt_sec,
        epoch(ARVT_3_flt) arvt_sec,
        strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month,
        strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS day
        FROM read_parquet(?)''', [[str(p) for p in files]])
    con.execute('''CREATE OR REPLACE TEMP VIEW dep AS SELECT *,
        count(*) OVER(PARTITION BY flight_id) AS n_id
        FROM source WHERE phase='DEP' ''')
    con.execute('''CREATE OR REPLACE TEMP VIEW arr AS SELECT *,
        count(*) OVER(PARTITION BY flight_id) AS n_id
        FROM source WHERE phase='ARR' ''')
    con.execute('''CREATE OR REPLACE TEMP TABLE pairs AS SELECT
        d.mvt_id,d.flight_id,d.origin,d.destination,d.month,d.day,
        d.event_sec takeoff_sec,d.aobt_sec,d.arvt_sec dep_arvt_sec,
        a.mvt_id donor_id,a.month donor_month,a.event_sec landing_sec,
        a.aobt_sec donor_aobt_sec,a.arvt_sec donor_arvt_sec,
        a.event_sec-a.arvt_sec landing_error_sec
        FROM dep d JOIN arr a ON d.flight_id=a.flight_id
        WHERE d.flight_id IS NOT NULL AND d.n_id=1 AND a.n_id=1
          AND d.origin=a.origin AND d.destination=a.destination
          AND d.origin=d.plan_origin AND d.destination=d.plan_destination
          AND a.origin=a.plan_origin AND a.destination=a.plan_destination
          AND d.callsign=a.callsign AND d.callsign IS NOT NULL
          AND (d.aircraft IS NULL OR a.aircraft IS NULL OR d.aircraft=a.aircraft)
          AND a.event_sec-d.event_sec BETWEEN 600 AND 72000''')


def coverage():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'coverage.json').exists(), 'Preserve completed research'
    write_json(OUT / 'design.json', DESIGN)
    upstream = json.loads(PHASE_MANIFEST.read_text())
    report = dict(design=DESIGN, source_sha256=sha256(Path(__file__)),
        source_manifest=dict(path=str(PHASE_MANIFEST.relative_to(ROOT)), sha256=sha256(PHASE_MANIFEST)),
        raw_sources=upstream['raw_sources'],
        source_validation='Reuses completed all-row phase source audit; no concurrent full-file hashing.', views={})
    started = time.monotonic()
    with duckdb.connect(config={'threads':1, 'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.register('airports', pd.DataFrame({'airport':AIRPORTS}))
        for view, records in upstream['raw_sources'].items():
            files = [ROOT / r['path'] for r in records]
            for p, r in zip(files, records):
                assert p.stat().st_size == r['bytes']
            make_inputs(con, files)
            rows = con.execute('''SELECT d.origin,d.month,count(*) AS departures,
                count(*) FILTER(WHERE d.flight_id IS NULL) AS missing_id,
                count(*) FILTER(WHERE d.destination IN(SELECT airport FROM airports)) AS within_ten,
                count(p.mvt_id) AS pairs,count(p.landing_error_sec) AS finite_landing_error,
                count(*) FILTER(WHERE p.aobt_sec IS NOT NULL AND p.landing_error_sec IS NOT NULL) AS both_clocks,
                count(*) FILTER(WHERE p.aobt_sec IS DISTINCT FROM p.donor_aobt_sec) AS paired_aobt_disagreements,
                count(*) FILTER(WHERE p.dep_arvt_sec IS DISTINCT FROM p.donor_arvt_sec) AS paired_arvt_disagreements,
                count(*) FILTER(WHERE abs(p.landing_error_sec)>60) AS landing_over60,
                count(*) FILTER(WHERE abs(p.landing_error_sec)>600) AS landing_over600,
                median(p.landing_error_sec) AS landing_median,
                quantile_cont(p.landing_error_sec,.25) AS landing_q25,
                quantile_cont(p.landing_error_sec,.75) AS landing_q75
                FROM dep d LEFT JOIN pairs p USING(mvt_id)
                GROUP BY d.origin,d.month ORDER BY d.origin,d.month''').df()
            rows.to_parquet(OUT / f'{view}_coverage.parquet', index=False)
            totals = {c:int(rows[c].sum()) for c in rows.columns if c not in
                ['origin','month','landing_median','landing_q25','landing_q75']}
            report['views'][view] = dict(totals=totals, summary_sha256=sha256(OUT / f'{view}_coverage.parquet'))
            if view == 'training':
                con.register('pilot_dates', pd.DataFrame({'day':DATES}))
                con.register('pilot_months', pd.DataFrame({'month':MONTHS}))
                pilot = con.execute('''SELECT * FROM pairs WHERE day IN(SELECT day FROM pilot_dates)
                    AND donor_month IN(SELECT month FROM pilot_months) ORDER BY mvt_id''').df()
                assert pilot.mvt_id.is_unique and pilot.donor_id.is_unique
                pilot.to_parquet(OUT / 'training_input_pairs.parquet', index=False)
                report['training_pilot_input_rows'] = len(pilot)
            report['elapsed_seconds'] = time.monotonic() - started
            write_json(OUT / 'coverage.json', report)
            print(view, json.dumps(totals), flush=True)


def describe(frame):
    valid = frame.dropna(subset=['block_error_sec','landing_error_sec']).copy()
    if not len(valid): return dict(rows=0)
    b = valid.block_error_sec.to_numpy(float)
    z = valid.landing_error_sec.to_numpy(float)
    bc, zc = np.clip(b,-3600,3600), np.clip(z,-3600,3600)
    corr = lambda x,y: float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
    valid['bc'], valid['zc'] = bc, zc
    demeaned = valid[['bc','zc']] - valid.groupby(['origin','destination'],observed=True)[['bc','zc']].transform('mean')
    return dict(rows=len(valid), landing_over60=int((abs(z)>60).sum()),landing_over600=int((abs(z)>600).sum()),
        departure_clock_rmse=float(np.sqrt(np.mean(b*b))),
        same_flight_shift_rmse=float(np.sqrt(np.mean((b-z)**2))),
        departure_clock_mae=float(np.mean(abs(b))),same_flight_shift_mae=float(np.mean(abs(b-z))),
        clipped_correlation=corr(bc,zc),route_demeaned_clipped_correlation=corr(demeaned.bc,demeaned.zc),
        clock_median=float(np.median(b)),landing_median=float(np.median(z)),
        common_shift_within60=int((abs(b-z)<=60).sum()),block_clock_within60=int((abs(b)<=60).sum()))


def pilot():
    started=time.monotonic()
    record=json.loads((OUT/'coverage.json').read_text())
    assert record['source_sha256']==sha256(Path(__file__)) and record['design']==DESIGN
    assert not (OUT/'pilot.json').exists()
    q=pd.read_parquet(OUT/'training_input_pairs.parquet')
    files=[ROOT/r['path'] for r in record['raw_sources']['training']
        if any(Path(r['path']).name.startswith('training_'+month+'-') for month in MONTHS)]
    assert len(files)==4
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.register('pilot_dates',pd.DataFrame({'day':DATES}))
        labels=con.execute('''SELECT MVT_ID_mvt::BIGINT mvt_id,
            epoch(BLOCK_TIME_UTC_mvt)-epoch(AOBT_3_flt) block_error_sec
            FROM read_parquet(?) WHERE PHASE_mvt='DEP'
            AND strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') IN(SELECT day FROM pilot_dates)''',
            [[str(p) for p in files]]).df()
    assert labels.mvt_id.is_unique and set(q.day)<=set(DATES)
    f=q.merge(labels,on='mvt_id',validate='one_to_one')
    assert len(f)==len(q)
    report=dict(design=DESIGN,coverage_sha256=sha256(OUT/'coverage.json'),
        matching_inputs_sha256=sha256(OUT/'training_input_pairs.parquet'),
        source_sha256=sha256(Path(__file__)),all=describe(f),
        by_month={str(k):describe(g) for k,g in f.groupby('month')},
        by_airport={str(k):describe(g) for k,g in f.groupby('origin')},
        by_landing_regime={name:describe(f.loc[mask]) for name,mask in {
            'absolute_over60':f.landing_error_sec.abs()>60,
            'absolute_over600':f.landing_error_sec.abs()>600}.items()},
        halves={name:describe(f[f.month.isin(months)]) for name,months in
            [('March_September',MONTHS[:2]),('October_November',MONTHS[2:])]},
        departure_label_dates=sorted(f.day.unique()),elapsed_seconds=time.monotonic()-started)
    f.to_parquet(OUT/'training_pilot.parquet',index=False)
    report['pilot_sha256']=sha256(OUT/'training_pilot.parquet')
    write_json(OUT/'pilot.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['coverage','pilot'])
    args=parser.parse_args(); coverage() if args.stage=='coverage' else pilot()
