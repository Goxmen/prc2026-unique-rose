"""Audited label-free correction inputs for the separately jittered final set."""
import argparse
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import daily_delay_reconcile as daily
import monthly_taxi_reconcile as monthly
import recovered_plan_exclusion_fixed as exclusion
from records import write_json,sha256

ROOT=daily.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_correction_inputs'
MANIFEST=OUT/'identity.json'
art=daily.artifact

def read(p):
    import json
    return json.loads(Path(p).read_text())

def independent_matches(path):
    """Enumerate all admissible raw pairs in pandas, not the producer's SQL."""
    columns=['MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt',
        'AIRCRAFT_TYPE_mvt','MVT_TIME_UTC_mvt','LOBT_flt','ADEP_flt','ADES_flt']
    raw=pd.read_parquet(path,columns=columns,filters=[('ADEP_mvt','in',['LIRF','LTFM'])])
    keys=['FLIGHT_mvt','ADEP_mvt','ADES_mvt']
    raw=raw.dropna(subset=keys)
    dep=raw[raw.PHASE_mvt.eq('DEP')&raw.FLIGHT_ID_mvt.isna()&raw.FLIGHT_mvt.ne('')]
    arr=raw[raw.PHASE_mvt.eq('ARR')&raw.FLIGHT_ID_mvt.notna()&raw.LOBT_flt.notna()]
    pairs=dep.merge(arr,on=keys,suffixes=('_d','_a'))
    elapsed=(pairs.MVT_TIME_UTC_mvt_a-pairs.MVT_TIME_UTC_mvt_d).dt.total_seconds()
    pairs=pairs[elapsed.gt(600)&elapsed.lt(72000)].copy()
    counts=pairs.groupby('MVT_ID_mvt_d').MVT_ID_mvt_a.transform('size')
    aircraft=pairs.AIRCRAFT_TYPE_mvt_d.isna()|pairs.AIRCRAFT_TYPE_mvt_a.isna()|pairs.AIRCRAFT_TYPE_mvt_d.eq(pairs.AIRCRAFT_TYPE_mvt_a)
    route=pairs.ADEP_flt_a.eq(pairs.ADEP_mvt)&pairs.ADES_flt_a.eq(pairs.ADES_mvt)
    expected=pd.DataFrame(dict(mvt_id=pairs.MVT_ID_mvt_d,airport=pairs.ADEP_mvt,takeoff=pairs.MVT_TIME_UTC_mvt_d,
        donor_mvt_id=pairs.MVT_ID_mvt_a,donor_plan_id=pairs.FLIGHT_ID_mvt_a,landing=pairs.MVT_TIME_UTC_mvt_a,
        recovered_lobt=pairs.LOBT_flt_a,candidates=counts,
        recovered_gap=(pairs.MVT_TIME_UTC_mvt_d-pairs.LOBT_flt_a).dt.total_seconds(),
        donor_month=pairs.MVT_TIME_UTC_mvt_a.dt.strftime('%Y-%m'),aircraft_ok=aircraft,route_ok=route,
        accepted=counts.eq(1)&aircraft.fillna(False)&route.fillna(False)))
    assert not expected.duplicated(['mvt_id','donor_mvt_id']).any()
    return expected.sort_values(['mvt_id','donor_mvt_id']).reset_index(drop=True)

def prepare():
    assert not MANIFEST.exists()
    paths=[Path(__file__),Path(daily.__file__),Path(monthly.__file__),Path(exclusion.__file__),
        ROOT/'data/ranking.parquet',ROOT/'data/final_ranking.parquet',
        ROOT/'data/submitting.parquet',ROOT/'data/final_submitting.parquet',
        daily.OUT/'ranking_inputs.parquet',exclusion.OUT/'ranking_matches.parquet']
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(MANIFEST,dict(sources=[art(p) for p in paths],views=['old','final'],
        no_label_reads=True,no_calibration_fit=True,no_prediction=True))
    for view in ['old','final']:
        raw=ROOT/'data'/('ranking.parquet' if view=='old' else 'final_ranking.parquet')
        template=ROOT/'data'/('submitting.parquet' if view=='old' else 'final_submitting.parquet')
        ids=pd.read_parquet(template,columns=['MVT_ID_mvt']).MVT_ID_mvt
        folder=OUT/view;folder.mkdir(exist_ok=True)
        rows=daily.load_rows([raw],labelled=False)
        months=monthly.load([raw],labelled=False)
        clocks=pd.read_parquet(raw,columns=['MVT_ID_mvt','MVT_TIME_UTC_mvt','FLIGHT_RULE_mvt'],
            filters=[('PHASE_mvt','==','DEP')]).rename(columns=dict(MVT_ID_mvt='mvt_id',
            MVT_TIME_UTC_mvt='takeoff',FLIGHT_RULE_mvt='fltrule'))
        assert all(len(f)==f.mvt_id.nunique()==len(ids) for f in [rows,months,clocks])
        rows=rows.set_index('mvt_id').loc[ids].reset_index()
        months=months.set_index('mvt_id').loc[ids].reset_index()
        clocks=clocks.set_index('mvt_id').loc[ids].reset_index()
        matches=exclusion.recover([raw])
        assert 'y' not in rows and 'y' not in months
        if view=='old':
            prior=pd.read_parquet(daily.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
            pd.testing.assert_frame_equal(rows,prior,check_exact=True)
            pd.testing.assert_frame_equal(matches,pd.read_parquet(exclusion.OUT/'ranking_matches.parquet'),check_exact=True)
        for name,frame in [('daily',rows),('monthly',months),('clocks',clocks),('matches',matches)]:
            frame.to_parquet(folder/f'{name}.parquet',index=False)
        write_json(folder/'prepared.json',dict(identity=art(MANIFEST),rows=len(ids),months=rows.month.value_counts().sort_index().to_dict(),
            matched_candidates=len(matches),accepted=int(matches.accepted.sum()),
            outputs=[art(folder/f'{name}.parquet') for name in ['daily','monthly','clocks','matches']],
            old_replay_exact=view=='old',no_departure_labels_read=True))
    print(dict(prepared=True,final_rows=len(ids),final_pairs=len(matches),final_accepted=int(matches.accepted.sum())),flush=True)

def audit():
    identity=read(MANIFEST);assert not (OUT/'audit.json').exists()
    for a in identity['sources']:assert sha256(ROOT/a['path'])==a['sha256']
    checks=[]
    with duckdb.connect(config={'threads':1,'memory_limit':'768MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for view,n in [('old',344841),('final',670790)]:
            folder=OUT/view;receipt=read(folder/'prepared.json');assert receipt['identity']==art(MANIFEST)
            for a in receipt['outputs']:assert sha256(ROOT/a['path'])==a['sha256']
            raw=ROOT/'data'/('ranking.parquet' if view=='old' else 'final_ranking.parquet')
            con.read_parquet(str(raw)).create_view('raw',replace=True)
            for name in ['daily','monthly','clocks','matches']:
                con.read_parquet(str(folder/f'{name}.parquet')).create_view(name,replace=True)
            zone='CASE r.ADEP_mvt '+' '.join(f"WHEN '{a}' THEN '{z}'" for a,z in monthly.ZONES.items())+' END'
            rows,unique,bad=con.execute(f'''SELECT count(*),count(distinct r.MVT_ID_mvt),sum(
                (d.airport IS DISTINCT FROM r.ADEP_mvt)::INT+
                (d.day IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m-%d'))::INT+
                (d.month IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m'))::INT+
                (d.gap IS DISTINCT FROM epoch(r.MVT_TIME_UTC_mvt-r.SCHED_TIME_UTC_mvt))::INT+
                (d.has_id IS DISTINCT FROM (r.FLIGHT_ID_mvt IS NOT NULL))::INT+
                (d.nm_gap IS DISTINCT FROM epoch(r.MVT_TIME_UTC_mvt-r.LOBT_flt))::INT+
                (m.airport IS DISTINCT FROM r.ADEP_mvt)::INT+
                (m.utc_month IS DISTINCT FROM d.month)::INT+
                (m.local_month IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt AT TIME ZONE ({zone}),'%Y-%m'))::INT+
                (m.eligible IS DISTINCT FROM (r.STAND_mvt IS NOT NULL AND r.RUNWAY_mvt IS NOT NULL))::INT+
                (c.takeoff IS DISTINCT FROM r.MVT_TIME_UTC_mvt)::INT+
                (c.fltrule IS DISTINCT FROM r.FLIGHT_RULE_mvt)::INT)
                FROM raw r JOIN daily d ON r.MVT_ID_mvt=d.mvt_id
                JOIN monthly m USING(mvt_id) JOIN clocks c USING(mvt_id) WHERE r.PHASE_mvt='DEP' ''').fetchone()
            assert rows==unique==n and bad==0,(view,rows,bad)
            expected=independent_matches(raw)
            actual=pd.read_parquet(folder/'matches.parquet').sort_values(['mvt_id','donor_mvt_id']).reset_index(drop=True)
            # SQL NULL booleans and pandas' false comparisons have the same
            # acceptance meaning; normalize only these two witness flags.
            for name in ['aircraft_ok','route_ok']:
                actual[name]=actual[name].fillna(False)
                expected[name]=expected[name].fillna(False)
            pd.testing.assert_frame_equal(actual,expected[actual.columns],check_dtype=False,check_exact=True)
            # Verify actual matched donors against their raw public records,
            # separately from the recovery window query and supplied receipt.
            wrong=con.execute('''SELECT count(*) FROM matches q
                JOIN raw d ON q.mvt_id=d.MVT_ID_mvt JOIN raw a ON q.donor_mvt_id=a.MVT_ID_mvt
                WHERE d.PHASE_mvt!='DEP' OR a.PHASE_mvt!='ARR' OR d.FLIGHT_ID_mvt IS NOT NULL
                OR a.FLIGHT_ID_mvt IS NULL OR a.LOBT_flt IS NULL
                OR d.FLIGHT_mvt IS DISTINCT FROM a.FLIGHT_mvt OR d.FLIGHT_mvt IS NULL OR d.FLIGHT_mvt=''
                OR d.ADEP_mvt IS DISTINCT FROM a.ADEP_mvt OR d.ADES_mvt IS DISTINCT FROM a.ADES_mvt
                OR d.ADEP_mvt NOT IN ('LIRF','LTFM')
                OR epoch(a.MVT_TIME_UTC_mvt-d.MVT_TIME_UTC_mvt)<=600
                OR epoch(a.MVT_TIME_UTC_mvt-d.MVT_TIME_UTC_mvt)>=72000
                OR q.recovered_gap IS DISTINCT FROM epoch(d.MVT_TIME_UTC_mvt-a.LOBT_flt)
                OR q.donor_month IS DISTINCT FROM strftime(a.MVT_TIME_UTC_mvt,'%Y-%m')
                OR q.accepted IS DISTINCT FROM (q.candidates=1 AND
                    (d.AIRCRAFT_TYPE_mvt IS NULL OR a.AIRCRAFT_TYPE_mvt IS NULL OR d.AIRCRAFT_TYPE_mvt=a.AIRCRAFT_TYPE_mvt)
                    AND coalesce(a.ADEP_flt=d.ADEP_mvt AND a.ADES_flt=d.ADES_mvt,false))''').fetchone()[0]
            assert wrong==0
            count_errors=con.execute('''SELECT count(*) FROM matches q JOIN
                (SELECT mvt_id,count(*) n FROM matches GROUP BY mvt_id) c USING(mvt_id)
                WHERE q.candidates!=c.n''').fetchone()[0]
            assert count_errors==0
            checks.append(dict(view=view,rows=n,independent_raw_values=n*12,matched_pairs=receipt['matched_candidates'],
                all_donor_fields_and_acceptance_exact=True,complete_raw_pair_enumeration_independent=True,
                nullable_witness_flags_compared_by_acceptance_semantics=True))
    write_json(OUT/'audit.json',dict(status='pass',identity=art(MANIFEST),checks=checks,
        original_daily_and_match_replay_exact=True,no_predictions_or_labels=True))
    print(dict(audit='pass',raw_values=sum(x['independent_raw_values'] for x in checks)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','audit'])
    with threadpool_limits(1):globals()[p.parse_args().stage]()
