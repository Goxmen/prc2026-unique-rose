"""Independent raw inputs, native training calendars, source totals and scores."""
import argparse
from datetime import datetime,timedelta,timezone
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import joint_public_airports_cv as run
import taxi_count_sum_jan_jul as taxi
import daily_heavy_tail as heavy


def inputs():
    identity=run.guard();out=run.OUT/'input_audit.json';assert not out.exists()
    for source in identity['sources']:run.verify(source)
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        paths={'i':run.OUT/'inputs.parquet','r':heavy.aligned.OUT/'rows2025.parquet',
            'a':heavy.aligned.OUT/'scored.parquet','t':taxi.daily.CACHE,
            'q':taxi.quality.OUT/'scored.parquet','s':run.OUT/'taxi_source.parquet',
            'p':taxi.daily.SOURCE/'daily_runway.parquet','d':run.OUT/'delay_source.parquet','cal':run.OUT/'calibration.parquet'}
        for name,path in paths.items():con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct i.mvt_id),sum(case when
            i.airport is distinct from r.airport or i.airport='LIRF' or i.month is distinct from r.month
            or i.gap is distinct from r.gap or i.fltrule is distinct from r.fltrule or i.prior is distinct from a.parent
            or i.delay_scope is distinct from a.new_scope or i.assigned_day is distinct from a.assigned_day
            or i.assigned_day!=strftime(to_timestamp(r.takeoff_epoch-i.prior),'%Y-%m-%d')
            or i.eligible!=(t.stand is not null and t.runway is not null) or i.local_day is distinct from t.local_takeoff
            or i.runway is distinct from t.runway or i.quality is distinct from q.quality_eligible then 1 else 0 end)
            from i join r using(mvt_id) join a using(mvt_id) join t using(mvt_id) join q using(mvt_id)''').fetchone()
        assert n==u==317891 and bad==0
        missing=con.execute('''select count(*) from r where airport!='LIRF' and month in ('2025-01','2025-07')
            and mvt_id not in(select mvt_id from i)''').fetchone()[0];assert missing==0
        nc,bad,diff=con.execute('''select count(*),sum(cast(s.source_n is distinct from p.n as int)),
            max(abs(s.source_mean-60*p.total_min/p.n)) from s join p on s.airport=p.airport
            and s.local_day=p.day and s.runway=p.runway''').fetchone()
        assert nc==con.execute('select count(*) from s').fetchone()[0] and bad==0 and diff<1e-8
        source=run.model.base.prior.base.SOURCE/'all_pre_departure_delays_2025.csv';con.read_csv(str(source)).create_view('csv')
        nd,bad=con.execute('''select count(*),sum(case when d.n_public!=csv.FLT_DEP_IFR_2
            or abs(d.total-60*csv.DLY_ALL_PRE_2)>1e-8 then 1 else 0 end)
            from d join csv on d.airport=csv.APT_ICAO and d.day=cast(csv.FLT_DATE as varchar)''').fetchone()
        assert nd==con.execute('select count(*) from d').fetchone()[0] and bad==0
        cells=0
        for month,allowed in taxi.daily.base.FOLDS.items():
            agg=con.execute('''select cal.airport,cal.local_day,cal.runway,cal.n,cal.valid,cal.total,
                count(*) sql_n,count(*) filter(where t.y between 0 and 7200) sql_valid,
                sum(case when t.y between 0 and 7200 then t.y else 0 end) sql_total
                from cal join t on cal.airport=t.airport and cal.local_day=t.local_takeoff and cal.runway=t.runway
                where cal.fold=? and strftime(t.takeoff AT TIME ZONE 'UTC','%Y-%m') in (select unnest(?))
                and t.stand is not null and t.runway is not null group by all''',[month,allowed]).df()
            assert len(agg)==con.execute('select count(*) from cal where fold=?',[month]).fetchone()[0]
            for a,b in [('n','sql_n'),('valid','sql_valid'),('total','sql_total')]:np.testing.assert_array_equal(agg[a],agg[b])
            for row in agg.itertuples():
                zone=ZoneInfo(taxi.daily.ZONES[row.airport]);day=datetime.fromisoformat(row.local_day)
                start=day.replace(tzinfo=zone).astimezone(timezone.utc)
                end=(day+timedelta(days=1)).replace(tzinfo=zone).astimezone(timezone.utc)-timedelta(microseconds=1)
                assert start.strftime('%Y-%m') in allowed and end.strftime('%Y-%m') in allowed
            cells+=len(agg)
    parameters=run.read(run.OUT/'parameters.json');old=run.read(heavy.aligned.OUT/'result.json')['parameters']
    support=run.read(taxi.TAU)['parameters']
    for month,airports in parameters.items():
        for airport,p in airports.items():
            assert p['delay']==old[month]['offblock'][airport] and p['support']==support[month][airport]
            assert p['taxi']['allowed']==taxi.daily.base.FOLDS[month]
    assert not {'y','true_day','BLOCK_TIME_UTC_mvt'}.intersection(pd.read_parquet(run.OUT/'inputs.parquet').columns)
    run.write(out,dict(status='pass',identity=run.art(run.OUT/'identity.json'),source=run.art(__file__),rows=n,
        taxi_source_cells=nc,delay_source_days=nd,calibration_cells=cells,all_input_fields_ids_and_raw_source_values_exact=True,
        native_airport_training_calendars_independently_checked=True,old_delay_and_support_parameters_exact=True,
        inference_labels_and_true_dates_absent=True))
    print(dict(status='pass',rows=n,calibration_cells=cells),flush=True)


def scores():
    run.guard();result=run.read(run.OUT/'result.json');run.verify(result['scored']);assert not (run.OUT/'audit.json').exists()
    assert run.read(run.OUT/'input_audit.json')['status']=='pass'
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('f',run.OUT/'scored.parquet'),('p',run.OUT/'predictions.parquet'),('i',run.OUT/'inputs.parquet'),
                         ('old',run.parent.OUT/'scored.parquet'),('raw',taxi.daily.base.OUT/'rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct f.mvt_id),sum(case when
            f.y is distinct from raw.y or f.day is distinct from raw.day or f.month is distinct from raw.month
            or f.airport is distinct from raw.airport or f.control is distinct from old.candidate
            or f.increment!=coalesce(p.increment,0) or f.touched!=coalesce(p.touched,false)
            or ((not f.touched or f.airport='LIRF') and f.candidate!=f.control) then 1 else 0 end)
            from f join raw using(mvt_id) join old using(mvt_id) left join p using(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        bad=con.execute('''select count(*) from p join i using(mvt_id) where p.prior!=i.prior
            or p.increment!=p.joint-p.dep_only or (p.touched and (i.airport='LIRF' or not i.eligible or not i.quality
            or not i.delay_scope or i.fltrule='V' or i.gap<=3600 or i.gap>=50400))''').fetchone()[0]
        assert bad==0
        for name in ['control','candidate']:
            values=con.execute(f'select month,sqrt(avg(pow({name}-y,2))) from f group by month').fetchall()
            for month,value in values:np.testing.assert_allclose(value,result['scores'][month][name],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for _,v in values]),result['cv'][name],atol=1e-8,rtol=0)
    f=pd.read_parquet(run.OUT/'scored.parquet');ids=f.mvt_id.to_numpy()
    raw=pd.read_parquet(taxi.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches=pd.read_parquet(taxi.daily.constraints.exclusion.OUT/'matches2025.parquet')
    np.testing.assert_array_equal(taxi.daily.constrain(raw,ids,f.month,f.control.to_numpy()+.5*f.increment.to_numpy(),matches),f.candidate)
    run.write(run.OUT/'audit.json',dict(status='pass',identity=run.art(run.OUT/'identity.json'),result=run.art(run.OUT/'result.json'),
        source=run.art(__file__),rows=n,all_raw_truth_parent_scope_increment_and_sql_scores=True,final_constraints_replayed=True,
        message_audit_pending=True))
    print(dict(status='pass',rows=n),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['inputs','scores']);args=p.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
