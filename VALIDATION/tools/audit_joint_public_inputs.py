"""Independently check every compact joint-source input against raw caches."""
from pathlib import Path
import sys

import duckdb
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import joint_public_moments_cv as runner


def audit():
    runner.guard();out=runner.OUT/'input_audit.json';assert not out.exists()
    paths={'i':runner.OUT/'inputs.parquet',
        'r':ROOT/'VALIDATION/runs/v3_20261011_daily_offblock_alignment/rows2025.parquet',
        'a':ROOT/'VALIDATION/runs/v3_20261011_daily_offblock_alignment/scored.parquet',
        't':ROOT/'tmp/research_20261011/daily_taxi_source/training_rows.parquet',
        'q':ROOT/'VALIDATION/runs/v3_20261011_public_daily_taxi_quality/scored.parquet',
        's':runner.OUT/'taxi_source.parquet',
        'p':ROOT/'data/external/eurocontrol_daily_taxi_20261011/daily_runway.parquet',
        'd':runner.OUT/'delay_source.parquet'}
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,path in paths.items():con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct i.mvt_id),sum(case when
            r.airport!='LIRF' or i.month is distinct from r.month or i.gap is distinct from r.gap
            or i.fltrule is distinct from r.fltrule or i.prior is distinct from a.parent
            or i.delay_scope is distinct from a.new_scope or i.assigned_day is distinct from a.assigned_day
            or i.assigned_day!=strftime(to_timestamp(r.takeoff_epoch-i.prior),'%Y-%m-%d')
            or i.eligible!=(t.stand is not null and t.runway is not null)
            or i.local_day is distinct from t.local_takeoff or i.runway is distinct from t.runway
            or i.quality is distinct from q.quality_eligible then 1 else 0 end)
            from i join r using(mvt_id) join a using(mvt_id) join t using(mvt_id) join q using(mvt_id)''').fetchone()
        assert n==u==26528 and errors==0
        missing=con.execute('''select count(*) from r where airport='LIRF' and month in ('2025-01','2025-07')
            and mvt_id not in (select mvt_id from i)''').fetchone()[0];assert missing==0
        cells,bad,maxdiff=con.execute('''select count(*),sum(case when s.source_n!=p.n then 1 else 0 end),
            max(abs(s.source_mean-60*p.total_min/p.n)) from s join p
            on p.airport='LIRF' and s.local_day=p.day and s.runway=p.runway''').fetchone()
        assert cells==con.execute('select count(*) from s').fetchone()[0] and bad==0 and maxdiff<1e-8
        path=ROOT/'data/external/eurocontrol_performance_20261011/all_pre_departure_delays_2025.csv'
        con.read_csv(str(path)).create_view('csv')
        days,bad=con.execute('''select count(*),sum(case when d.n_public!=csv.FLT_DEP_IFR_2
            or abs(d.total-60*csv.DLY_ALL_PRE_2)>1e-8 then 1 else 0 end)
            from d join csv on csv.APT_ICAO='LIRF' and cast(csv.FLT_DATE as varchar)=d.day''').fetchone()
        assert days==con.execute('select count(*) from d').fetchone()[0] and bad==0
    prm=runner.read(runner.OUT/'parameters.json')
    old=runner.read(ROOT/'VALIDATION/runs/v3_20261011_taxi_count_sum_support/parameters.json')
    for m,p in prm.items():
        assert p['support']==old[m]['held_parameters']
        assert p['taxi']=={k:v for k,v in old[m].items() if k!='held_parameters'}
    runner.write(out,dict(status='pass',rows=n,taxi_cells=cells,delay_days=days,
        identity=runner.art(runner.OUT/'identity.json'),auditor=runner.art(__file__),
        all_input_fields_ids_dates_scope_public_sources_exact=True,
        source_parameters_reused_exact=True,inference_labels_absent=True))
    print(dict(status='pass',rows=n,taxi_cells=cells,delay_days=days),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):audit()
