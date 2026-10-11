"""Independent SQL replay of all12 public-aggregate inputs on every original-ranking row."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import duckdb,numpy as np,pandas as pd
import public_delay_features as feature
from records import write_json

def main():
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        con.execute('''create temp table rows as select MVT_ID_mvt id,ADEP_mvt airport,
            strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS day,
            epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt) d,
            least(7200,greatest(30,coalesce(epoch(MVT_TIME_UTC_mvt-AOBT_3_flt),1200))) clock,
            coalesce(FLIGHT_RULE_mvt!='V',true) eligible
            from read_parquet('data/ranking.parquet') where PHASE_mvt='DEP' ''')
        con.execute('''create temp table totals as select airport,day,sum(eligible::int) n,
            sum((eligible and d>3600)::int) nl,
            sum(case when eligible and d>3600 then d else 0 end) dl,
            sum(case when eligible then greatest(0,d-900) else 0 end) c900,
            sum(case when eligible then greatest(0,d-1800) else 0 end) c1800,
            sum(case when eligible then greatest(0,d-clock) else 0 end) cc,
            sum(case when eligible and d>3600 then greatest(0,d-900) else 0 end) cl
            from rows group by airport,day''')
        con.execute('''create temp table src as select APT_ICAO airport,strftime(FLT_DATE,'%Y-%m-%d') AS day,
            FLT_DEP_IFR_2 np,DLY_ALL_PRE_2*60 total from read_csv_auto(?)''',[str(feature.base.SOURCE/'all_pre_departure_delays_2026.csv')])
        q=con.execute('''select r.id mvt_id,
            s.np-g.n pub_dep_count_gap,s.total/greatest(s.np,1) pub_dep_mean,
            g.nl pub_dep_long_count,g.dl pub_dep_long_gap_sum,
            (s.total-g.c900)/greatest(g.n,1) pub_dep_resid900_mean,
            (s.total-g.c1800)/greatest(g.n,1) pub_dep_resid1800_mean,
            (s.total-g.cc)/greatest(g.n,1) pub_dep_resid_clock_mean,
            s.total-g.c900+case when r.eligible then greatest(r.d-900,0) else 0 end pub_dep_own900_left,
            s.total-g.c1800+case when r.eligible then greatest(r.d-1800,0) else 0 end pub_dep_own1800_left,
            (s.total-g.c900+case when r.eligible then greatest(r.d-900,0) else 0 end)/greatest(case when r.eligible then greatest(r.d-900,0) else 0 end,600) pub_dep_own900_fraction,
            (s.total-g.c1800+case when r.eligible then greatest(r.d-1800,0) else 0 end)/greatest(case when r.eligible then greatest(r.d-1800,0) else 0 end,600) pub_dep_own1800_fraction,
            (s.total-(g.c900-g.cl))/greatest(g.cl,600) pub_dep_long900_fraction
            from rows r join totals g using(airport,day) join src s using(airport,day) order by r.id''').fetchdf()
    actual=pd.read_parquet(ROOT/'VALIDATION/runs/v3_20261011_rome_public_delay_submit/ranking_features.parquet').sort_values('mvt_id').reset_index(drop=True)
    np.testing.assert_array_equal(q.mvt_id,actual.mvt_id)
    np.testing.assert_allclose(q[feature.FEATURES],actual[feature.FEATURES],rtol=1e-12,atol=1e-8,equal_nan=True)
    result=dict(status='pass',rows=len(q),values=len(q)*len(feature.FEATURES),columns=feature.FEATURES,
        max_abs_error=float(np.nanmax(np.abs(q[feature.FEATURES].to_numpy()-actual[feature.FEATURES].to_numpy()))),
        source=feature.base.artifact(__file__),feature_manifest=feature.base.artifact(feature.OUT/'manifest.json'))
    result.update(raw_ranking=feature.base.artifact(ROOT/'data/ranking.parquet'),public=feature.base.artifact(feature.base.SOURCE/'all_pre_departure_delays_2026.csv'))
    write_json(ROOT/'VALIDATION/runs/v3_20261011_rome_public_delay_submit/ranking_feature_audit.json',result);print(result,flush=True)
if __name__=='__main__':main()
