"""Frozen, label-free falsifier of undocumented numeric NM flight-ID ordering.

No assertion is made that FLIGHT_ID_mvt is an IFPLID, TACT ID, or timestamp.
This is a source-only chronology/identity check, never a predictive feature.
The pilot is deliberately separate from the previously failed MVT_ID study.
"""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
from records import sha256, write_json

OUT = ROOT / 'tmp/research_20261012/signal_scout/flight_id_order'
PHASE_MANIFEST = ROOT / 'data/derived/source_clock_phase_20261012/manifest.json'
PREVIOUS = ROOT / 'VALIDATION/reports/research_network_20261010/id_clock_order.json'
MONTHS = ['2025-01', '2025-07', '2026-01', '2026-07']
DAYS = [5, 15, 25]
DATES = [f'{month}-{day:02d}' for month in MONTHS for day in DAYS]
COLUMNS = ['MVT_ID_mvt', 'FLIGHT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt',
           'ADES_mvt', 'MVT_TIME_UTC_mvt', 'IOBT_flt', 'EOBT_1_flt']
DESIGN = dict(
    hypothesis='Undocumented numeric flight-ID relative order may retain plan-generation chronology; not a timestamp or IFPLID claim.',
    months=MONTHS, query_dates=DATES, source_columns=COLUMNS,
    views='2025 January/July original monthly training inputs; accepted old January/July2026 ranking only; no final view.',
    clocks=['original IOBT_flt', 'original EOBT_1_flt'], timezone='UTC',
    targets='No BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt projection, loading, association, or model fit.',
    reference='Within each movement UTC month, unique positive integral IDs <=2**53 with exactly one DEP, <=one ARR, consistent route, and <=one known value per plan clock.',
    missing='Null/invalid IDs, phase duplicates, cross-endpoint null mismatches, conflicting clocks/routes reported separately. Missing clocks remain missing; no schedule fill.',
    rank='ID percentile among eligible departures in its month; clock rank only among rows with that clock; average tie ranks for Spearman. Raw absolute ID never enters a model.',
    endpoint='Within-month same-ID DEP/ARR only; clock equality and null disagreement are descriptive checks, not matching proof; cross-month endpoints are not available.',
    chronology='All-month Spearman plus fixed-date and fixed-date x origin Spearman, >=100 valid observations per supported group. Report both clocks independently.',
    format_checks='Decimal digit lengths and first-three-decimal-digit frequency by month/origin are diagnostics only. Report month-relative ID rank quartiles by origin and clock-rank quintile.',
    falsification=dict(
        meaning='Necessary evidence for simple monotonic chronology only; passing does not establish creation time, stale-plan detection, incremental target information, or CV gain.',
        eligible_known_id_share_min=.80,
        min_both_known_endpoint_pairs_each_month=100,
        max_endpoint_known_clock_conflict_share=.01,
        min_supported_airport_day_groups_per_month=10,
        min_abs_median_signed_airport_day_spearman_each_month=.20,
        min_paired_year_same_season_airport_day_sign_agreement=.80,
        both_clocks_required=True,
        missing_support='Fail feasibility; no threshold repair or alternative encoding search.'),
    resources=dict(threads=1, duckdb_memory='256MB', external_timeout_seconds=60,
                   expected_seconds='5–20, unmeasured', incremental_workspace_budget_mib=512),
    previous_movement_id_evidence=str(PREVIOUS.relative_to(ROOT)),
    public_semantic_caveat=[
        'https://www.nm.eurocontrol.int/HELP/Filing.html',
        'https://www.nm.eurocontrol.int/HELP/Operational_Log.html'],
    no_feature_build=True, no_fit=True, no_held_score=True, no_download=True)


def source_records():
    upstream = json.loads(PHASE_MANIFEST.read_text())
    records = [r for r in upstream['raw_sources']['training']
               if any(Path(r['path']).name.startswith('training_' + m + '-') for m in MONTHS[:2])]
    records += upstream['raw_sources']['old']
    assert len(records) == 3
    return records


def freeze():
    """Metadata-only stage: fixes cases/source identities before chronology reads."""
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'design.json').exists(), 'Preserve previous freeze'
    records = source_records()
    metadata = []
    for record in records:
        path = ROOT / record['path']
        assert path.stat().st_size == record['bytes']
        parquet = pq.ParquetFile(path)
        selected = 0
        for j in range(parquet.metadata.num_row_groups):
            rowgroup = parquet.metadata.row_group(j)
            for k in range(rowgroup.num_columns):
                column = rowgroup.column(k)
                if column.path_in_schema in COLUMNS:
                    selected += column.total_compressed_size
        metadata.append(dict(**record, source_rows=parquet.metadata.num_rows,
                             selected_column_compressed_bytes=selected,
                             mtime_ns=path.stat().st_mtime_ns))
    write_json(OUT / 'design.json', dict(design=DESIGN, source_sha256=sha256(__file__),
        phase_manifest_sha256=sha256(PHASE_MANIFEST), previous_sha256=sha256(PREVIOUS),
        source_records=metadata, hash_scope='Inherited independently audited phase raw-source hashes; pilot checks size/mtime, no redundant concurrent full-source hash scan.'))
    print(json.dumps(dict(metadata=metadata, design=DESIGN), indent=2), flush=True)


def make_source(con, path, months):
    con.execute('''CREATE OR REPLACE TEMP TABLE raw AS SELECT
        MVT_ID_mvt::BIGINT mvt_id, FLIGHT_ID_mvt raw_id,
        CASE WHEN isfinite(FLIGHT_ID_mvt) AND FLIGHT_ID_mvt>0
              AND FLIGHT_ID_mvt<=9007199254740992
              AND FLIGHT_ID_mvt=floor(FLIGHT_ID_mvt)
             THEN FLIGHT_ID_mvt::BIGINT END flight_id,
        PHASE_mvt phase,ADEP_mvt origin,ADES_mvt destination,
        epoch(MVT_TIME_UTC_mvt) event_sec,
        strftime(MVT_TIME_UTC_mvt,'%Y-%m') month_key,
        strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') day_key,
        epoch(IOBT_flt) iobt,epoch(EOBT_1_flt) eobt
        FROM read_parquet(?)
        WHERE strftime(MVT_TIME_UTC_mvt,'%Y-%m') IN
              (SELECT unnest(?))''', [str(path), months])
    con.execute('''CREATE OR REPLACE TEMP TABLE ids AS SELECT month_key,flight_id,
        count(*) FILTER(WHERE phase='DEP') n_dep,
        count(*) FILTER(WHERE phase='ARR') n_arr,
        count(*) FILTER(WHERE phase NOT IN('DEP','ARR') OR phase IS NULL) n_other,
        count(DISTINCT struct_pack(a:=origin,b:=destination)) n_routes,
        count(DISTINCT iobt) iobt_nvalues,count(iobt) iobt_known,
        count(DISTINCT eobt) eobt_nvalues,count(eobt) eobt_known,
        count(*) n_rows,
        min(iobt) FILTER(WHERE phase='DEP') iobt,
        min(eobt) FILTER(WHERE phase='DEP') eobt,
        min(origin) origin,min(destination) destination,
        min(day_key) FILTER(WHERE phase='DEP') dep_day,
        min(mvt_id) FILTER(WHERE phase='DEP') dep_id
        FROM raw WHERE flight_id IS NOT NULL GROUP BY month_key,flight_id''')
    con.execute('''CREATE OR REPLACE TEMP TABLE eligible AS SELECT *,
        (rank() OVER(PARTITION BY month_key ORDER BY flight_id)-.5)
        /count(*) OVER(PARTITION BY month_key) id_percentile
        FROM ids WHERE n_dep=1 AND n_arr<=1 AND n_other=0
        AND n_routes=1 AND iobt_nvalues<=1 AND eobt_nvalues<=1''')


def correlation(frame, clock):
    group = frame.dropna(subset=[clock])
    n = len(group)
    if n < 100 or group[clock].nunique() < 2 or group.flight_id.nunique() < 2:
        return dict(n=n, rho=None)
    return dict(n=n, rho=float(group.flight_id.rank().corr(group[clock].rank())))


def describe(con):
    source = con.execute('''SELECT month_key,count(*) AS "rows",count(*) FILTER(WHERE phase='DEP') departures,
        count(*) FILTER(WHERE raw_id IS NULL) null_ids,
        count(*) FILTER(WHERE raw_id IS NOT NULL AND flight_id IS NULL) invalid_ids,
        count(*) FILTER(WHERE phase='DEP' AND flight_id IS NOT NULL) known_id_departures,
        count(*) FILTER(WHERE iobt IS NULL) missing_iobt,
        count(*) FILTER(WHERE eobt IS NULL) missing_eobt,
        count(*)-count(DISTINCT mvt_id) duplicate_movement_ids
        FROM raw GROUP BY month_key ORDER BY month_key''').df().to_dict('records')
    identity = con.execute('''SELECT month_key,count(*) distinct_ids,
        count(*) FILTER(WHERE n_dep>1) duplicate_departure_ids,
        count(*) FILTER(WHERE n_arr>1) duplicate_arrival_ids,
        count(*) FILTER(WHERE n_routes>1) route_conflicts,
        count(*) FILTER(WHERE iobt_nvalues>1) iobt_conflicts,
        count(*) FILTER(WHERE eobt_nvalues>1) eobt_conflicts,
        count(*) FILTER(WHERE n_dep=1 AND n_arr=1) unique_endpoint_pairs,
        count(*) FILTER(WHERE n_dep=1 AND n_arr=1 AND iobt_known=1) endpoint_iobt_null_disagreement,
        count(*) FILTER(WHERE n_dep=1 AND n_arr=1 AND eobt_known=1) endpoint_eobt_null_disagreement,
        count(*) FILTER(WHERE n_dep=1 AND n_arr=1 AND iobt_known=2) endpoint_iobt_both_known,
        count(*) FILTER(WHERE n_dep=1 AND n_arr=1 AND eobt_known=2) endpoint_eobt_both_known,
        count(*) FILTER(WHERE n_dep=1 AND n_arr=1 AND iobt_nvalues=2) endpoint_iobt_known_conflicts,
        count(*) FILTER(WHERE n_dep=1 AND n_arr=1 AND eobt_nvalues=2) endpoint_eobt_known_conflicts
        FROM ids GROUP BY month_key ORDER BY month_key''').df().to_dict('records')
    frame = con.execute('SELECT month_key,flight_id,origin,dep_day,iobt,eobt,id_percentile FROM eligible').df()
    monthly, groups, format_rows, rank_strata = [], [], [], []
    for month, part in frame.groupby('month_key', observed=True):
        monthly.append(dict(month=month, eligible_departures=len(part),
                            clocks={c:correlation(part,c) for c in ['iobt','eobt']}))
        for (day, airport), group in part[part.dep_day.isin(DATES)].groupby(['dep_day','origin'],observed=True):
            groups.append(dict(month=month,day=day,airport=airport,
                               clocks={c:correlation(group,c) for c in ['iobt','eobt']}))
        for day, group in part[part.dep_day.isin(DATES)].groupby('dep_day',observed=True):
            groups.append(dict(month=month,day=day,airport='ALL',
                               clocks={c:correlation(group,c) for c in ['iobt','eobt']}))
        p = part.copy()
        as_text = p.flight_id.astype('int64').astype(str)
        p['digits'], p['prefix3'] = as_text.str.len(), as_text.str[:3]
        formats=p.groupby(['origin','digits','prefix3'],observed=True).size().reset_index(name='n')
        format_rows.extend(dict(month=month,**r) for r in formats.to_dict('records'))
        for clock in ['iobt','eobt']:
            p['clock_quintile']=np.floor(p[clock].rank(pct=True)*5).clip(upper=4)
            for (airport,quintile), group in p.dropna(subset=[clock]).groupby(['origin','clock_quintile'],observed=True):
                rank_strata.append(dict(month=month,airport=airport,clock=clock,clock_quintile=int(quintile),
                    n=len(group),id_percentile_quantiles=group.id_percentile.quantile([.25,.5,.75]).tolist()))
    return dict(source=source,identity=identity,monthly=monthly,chronology_groups=groups,
                format_groups=format_rows,clock_conditioned_id_ranks=rank_strata)


def decision(reports):
    sources={r['month_key']:r for report in reports for r in report['source']}
    identities={r['month_key']:r for report in reports for r in report['identity']}
    months={r['month']:r for report in reports for r in report['monthly']}
    groups=[g for report in reports for g in report['chronology_groups'] if g['airport']!='ALL']
    eligible={m:months.get(m,dict(eligible_departures=0))['eligible_departures']/max(1,sources[m]['known_id_departures']) for m in MONTHS}
    result=dict(eligible_share=eligible,clocks={})
    for clock in ['iobt','eobt']:
        signed={m:[g['clocks'][clock]['rho'] for g in groups if g['month']==m and g['clocks'][clock]['rho'] is not None] for m in MONTHS}
        summaries={m:dict(supported_groups=len(v),median_signed_rho=float(np.median(v)) if v else None) for m,v in signed.items()}
        bykey={(g['month'],g['day'][-2:],g['airport']):g['clocks'][clock]['rho'] for g in groups}
        comparisons=[]
        for season in ['01','07']:
            for day in DAYS:
                airports={k[2] for k in bykey}
                for airport in sorted(airports):
                    a,b=[bykey.get((year+'-'+season,f'{day:02d}',airport)) for year in ['2025','2026']]
                    if a is not None and b is not None:comparisons.append(bool(np.sign(a)==np.sign(b)))
        agreement=float(np.mean(comparisons)) if comparisons else None
        conflicts={m:identities[m]['endpoint_'+clock+'_known_conflicts']/max(1,identities[m]['endpoint_'+clock+'_both_known']) for m in MONTHS}
        endpoint_counts={m:identities[m]['endpoint_'+clock+'_both_known'] for m in MONTHS}
        passed=(all(x>=.8 for x in eligible.values()) and all(x<=.01 for x in conflicts.values())
                and all(x>=100 for x in endpoint_counts.values())
                and all(v['supported_groups']>=10 and abs(v['median_signed_rho'])>=.2 for v in summaries.values())
                and agreement is not None and agreement>=.8)
        result['clocks'][clock]=dict(months=summaries,paired_sign_comparisons=len(comparisons),
            sign_agreement=agreement,endpoint_conflict_share=conflicts,endpoint_both_known_counts=endpoint_counts,
            passes_necessary_evidence=passed)
    result['retain_monotonic_chronology_hypothesis']=all(v['passes_necessary_evidence'] for v in result['clocks'].values())
    result['interpretation']=DESIGN['falsification']['meaning']
    return result


def pilot():
    frozen=json.loads((OUT/'design.json').read_text())
    assert frozen['design']==DESIGN and frozen['source_sha256']==sha256(__file__)
    assert frozen['phase_manifest_sha256']==sha256(PHASE_MANIFEST)
    assert frozen['previous_sha256']==sha256(PREVIOUS)
    assert not (OUT/'result.json').exists(), 'No retries or overwrite'
    started=time.monotonic(); reports=[]
    initial_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        con.execute("SET preserve_insertion_order=false")
        for record in frozen['source_records']:
            path=ROOT/record['path'];stat=path.stat()
            assert stat.st_size==record['bytes'] and stat.st_mtime_ns==record['mtime_ns']
            months=MONTHS[2:] if path.name=='ranking.parquet' else [m for m in MONTHS[:2] if m in path.name]
            assert months
            make_source(con,path,months)
            report=describe(con);report['source_path']=record['path'];reports.append(report)
            write_json(OUT/'partial.json',dict(design_sha256=sha256(OUT/'design.json'),reports=reports,
                elapsed_seconds=time.monotonic()-started))
            print(path.name,round(time.monotonic()-started,3),'seconds',flush=True)
    peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    incremental_mib=max(0,peak_kib-initial_peak_kib)/1024
    write_json(OUT/'result.json',dict(design_sha256=sha256(OUT/'design.json'),source_sha256=sha256(__file__),
        reports=reports,decision=decision(reports),elapsed_seconds=time.monotonic()-started,
        initial_peak_rss_kib=initial_peak_kib,peak_rss_kib=peak_kib,
        incremental_peak_rss_mib=incremental_mib,
        incremental_workspace_budget_pass=incremental_mib<=512,
        no_targets_read=True,no_model_or_feature_constructed=True))
    assert incremental_mib<=512, 'Preserve failed resource result; no retry'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','pilot'])
    globals()[parser.parse_args().stage]()
