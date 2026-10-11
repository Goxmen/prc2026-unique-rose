"""Independent inputs, calibration, scalar messages, bounds and CV review."""
import math
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logsumexp
from scipy.stats import multivariate_t
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import joint_public_moments_submit as run


def message_audit():
    """Independent row masks and scalar sums, reusing tested moment primitives."""
    model=run.cv.model
    rows=pd.read_parquet(run.OUT/'inputs.parquet').reset_index(drop=True)
    saved=pd.read_parquet(run.OUT/'inference.parquet').set_index('mvt_id').loc[rows.mvt_id]
    prm=run.read(run.OUT/'parameters.json')
    ts=pd.read_parquet(run.OUT/'taxi_source.parquet').set_index(['local_day','runway'])
    ds=pd.read_parquet(run.OUT/'delay_source.parquet').set_index('day')
    summaries=[]
    for month,global_ix in rows.groupby('month').indices.items():
        r=rows.iloc[global_ix].reset_index(drop=True);p=saved.iloc[global_ix].reset_index(drop=True)
        v,c,prob=model.normal_moments(r.prior.to_numpy(),r.gap.to_numpy());supports={}
        for i in np.flatnonzero(r.gap.gt(3600)):
            values,weights,_=model.prior.support(float(r.gap.iloc[i]),float(r.prior.iloc[i]),False,np.nan,'LIRF',prm['support'])
            w=np.column_stack([(values>=0)&(values<=7200),np.where((values>=0)&(values<=7200),values,0),np.maximum(r.gap.iloc[i]-values,0)])
            v[i]=weights@w;c[i]=(w-v[i]).T@(weights[:,None]*(w-v[i]));supports[i]=(values,weights,w)
        active=np.flatnonzero(p.touched)
        chosen=np.unique(np.r_[active[np.linspace(0,len(active)-1,min(64,len(active)),dtype=int)],
            np.flatnonzero(p.touched&r.local_day.ne(r.assigned_day))[:16]])
        worst=0.
        for i in chosen:
            a=r.iloc[i]
            tm=(r.eligible&r.quality&r.local_day.eq(a.local_day)&r.runway.eq(a.runway)).to_numpy()
            dm=(r.delay_scope&~r.fltrule.eq('V')&r.assigned_day.eq(a.assigned_day)).to_numpy()
            union=np.flatnonzero(tm|dm);assert tm[i] and dm[i]
            mean=np.zeros(3);cov=np.zeros((3,3))
            for k in range(3):
                mean[k]=math.fsum(v[j,k] for j in union if j!=i and (tm[j] if k<2 else dm[j]))
                for l in range(3):
                    cov[k,l]=math.fsum(c[j,k,l] for j in union if j!=i
                        and (tm[j] if k<2 else dm[j]) and (tm[j] if l<2 else dm[j]))
            n=int(tm.sum());pub=ts.loc[(a.local_day,a.runway)];delay=ds.loc[a.assigned_day]
            bias=np.asarray(prm['taxi']['coefficients'])[0]+n/100*np.asarray(prm['taxi']['coefficients'])[1]
            observed=np.array([pub.source_n+bias[0],pub.source_n*pub.source_mean+bias[1],delay.total-prm['delay']['bias']])
            cov[:2,:2]+=n*np.asarray(prm['taxi']['covariance_per_row'])+np.diag([1.,3600.])
            dn=delay.n_public-int(dm.sum())
            cov[2,2]+=3*(prm['delay']['sigma']**2+(prm['delay']['kappa']*dn)**2)
            count=math.fsum(v[j,0] for j in np.flatnonzero(tm))
            load=math.fsum(prob[j] for j in np.flatnonzero(dm) if r.gap.iloc[j]<=3600)
            loading=np.array([0.,count,-load]);cov+=225*np.outer(loading,loading)
            values,weights,contrib=supports[i]
            actual=[]
            for dims in [[2],[0,1,2]]:
                delta=observed[dims]-mean[dims]-contrib[:,dims]
                # Counts and sums have very different units. Standardize before
                # SciPy's eigen decomposition; the omitted Jacobian is constant
                # across this flight's support and cancels in normalization.
                block=.6*cov[np.ix_(dims,dims)]
                scales=np.sqrt(np.diag(block))
                ll=multivariate_t.logpdf(delta/scales,
                    shape=block/np.outer(scales,scales),df=5)
                lw=np.log(np.maximum(weights,1e-300))+ll;post=np.exp(lw-logsumexp(lw))
                actual.append(float(values@post))
            np.testing.assert_allclose(actual,[p.dep_only.iloc[i],p.joint.iloc[i]],atol=2e-6,rtol=0)
            worst=max(worst,float(np.max(abs(np.asarray(actual)-[p.dep_only.iloc[i],p.joint.iloc[i]]))))
        summaries.append(dict(month=month,sampled_rows=len(chosen),max_mean_error=worst))
    return summaries


def audit():
    identity=run.guard();assert not (run.OUT/'audit.json').exists(),'Preserve completed review'
    result=run.read(run.OUT/'result.json')
    for a in result['artifacts']:run.verify(a)
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        paths={'i':run.OUT/'inputs.parquet','v':run.OUT/'predictions.parquet','p':run.OUT/'inference.parquet',
            'parent':ROOT/identity['parent_file'],'prior':ROOT/'submissions/unique-rose_v70.parquet',
            'rank':ROOT/'data/ranking.parquet','q':run.parent.OUT/'predictions.parquet',
            'cal':run.OUT/'taxi_calibration.parquet','tr':run.taxi.daily.CACHE,
            'dc':run.OUT/'delay_calibration_rows.parquet','raw':run.alignment.OUT/'rows2025.parquet',
            'cv':run.OUT/'cv.parquet','oldcv':run.cv.OUT/'scored.parquet',
            'truth':run.taxi.daily.base.OUT/'rows2025.parquet'}
        for name,path in paths.items():con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''select count(*),count(distinct i.mvt_id),sum(case when
            rank.PHASE_mvt!='DEP' or rank.ADEP_mvt!='LIRF' or i.prior!=prior.TAXITIME_SEC_mvt
            or i.gap!=epoch(rank.MVT_TIME_UTC_mvt-rank.SCHED_TIME_UTC_mvt)
            or i.fltrule is distinct from rank.FLIGHT_RULE_mvt
            or i.local_day!=strftime(rank.MVT_TIME_UTC_mvt at time zone 'Europe/Rome','%Y-%m-%d')
            or i.assigned_day!=strftime(to_timestamp(epoch(rank.MVT_TIME_UTC_mvt)-i.prior),'%Y-%m-%d')
            or i.runway is distinct from rank.RUNWAY_mvt or i.eligible!=(rank.STAND_mvt is not null and rank.RUNWAY_mvt is not null)
            or i.quality is distinct from q.quality_eligible then 1 else 0 end)
            from i join rank on i.mvt_id=rank.MVT_ID_mvt join prior on i.mvt_id=prior.MVT_ID_mvt join q using(mvt_id)''').fetchone()
        assert n==u==26899 and bad==0
        missing=con.execute("select count(*) from rank where PHASE_mvt='DEP' and ADEP_mvt='LIRF' and MVT_ID_mvt not in (select mvt_id from i)").fetchone()[0]
        assert missing==0
        count,unique,bad=con.execute('''select count(*),count(distinct v.mvt_id),sum(case when
            v.parent!=parent.TAXITIME_SEC_mvt or v.increment!=coalesce(p.increment,0)
            or v.touched!=coalesce(p.touched,false) or (not v.touched and v.pred!=v.parent)
            then 1 else 0 end) from v join parent on v.mvt_id=parent.MVT_ID_mvt left join p using(mvt_id)''').fetchone()
        assert count==unique==344841 and bad==0
        cells=con.execute('''select cal.local_day,cal.runway,cal.n,cal.valid,cal.total,count(*) sql_n,
            count(*) filter(where tr.y between 0 and 7200) sql_valid,
            sum(case when tr.y between 0 and 7200 then tr.y else 0 end) sql_total
            from cal join tr on cal.airport=tr.airport and cal.local_day=tr.local_takeoff and cal.runway=tr.runway
            where tr.stand is not null and tr.runway is not null group by all''').df()
        assert len(cells)==600
        for a,b in [('n','sql_n'),('valid','sql_valid'),('total','sql_total')]:np.testing.assert_array_equal(cells[a],cells[b])
        size,unique,bad=con.execute('''select count(*),count(distinct dc.mvt_id),sum(case when dc.y!=raw.y
            or dc.gap!=raw.gap or dc.day!=raw.true_day or dc.fltrule is distinct from raw.fltrule
            or dc.fltrule='V' or raw.airport!='LIRF' or substr(dc.day,1,4)!='2025'
            or extract(day from cast(dc.day as date))<=1 or cast(dc.day as date)=last_day(cast(dc.day as date))
            then 1 else 0 end) from dc join raw using(mvt_id)''').fetchone()
        assert size==unique==150093 and bad==0
        ncv,ucv,bad=con.execute('''select count(*),count(distinct cv.mvt_id),sum(case when cv.y!=truth.y
            or cv.month!=truth.month or cv.day!=truth.day or cv.pred!=oldcv.candidate or cv.control!=oldcv.control
            then 1 else 0 end) from cv join truth using(mvt_id) join oldcv using(mvt_id)''').fetchone()
        assert ncv==ucv==344419 and bad==0
        scores=dict(con.execute('select month,sqrt(avg(pow(pred-y,2))) from cv group by month').fetchall())
        for m,s in scores.items():np.testing.assert_allclose(s,result['cv_months'][m],atol=1e-8,rtol=0)
        np.testing.assert_allclose(np.mean(list(scores.values())),result['jan_jul_cv'],atol=1e-8,rtol=0)
    # Independent constraint arithmetic, including nearest accepted exclusion edge.
    f=pd.read_parquet(run.OUT/'predictions.parquet');ids=f.mvt_id.to_numpy()
    raw=pd.read_parquet(run.taxi.daily.base.OUT/'ranking_inputs.parquet').set_index('mvt_id').loc[ids]
    matches=pd.read_parquet(run.taxi.daily.constraints.exclusion.OUT/'ranking_matches.parquet')
    values=f.parent.to_numpy()+.5*f.increment.to_numpy();known=raw.has_id.to_numpy(bool)&raw.nm_gap.notna().to_numpy()
    gaps=raw.nm_gap.to_numpy();values[known]=np.maximum(0,np.minimum(np.maximum(values[known],gaps[known]-3606),gaps[known]+3606))
    values=np.maximum(values,0);positions={v:i for i,v in enumerate(ids)}
    for row in matches[matches.accepted].itertuples():
        i=positions[row.mvt_id];lo,hi=row.recovered_gap-3606,row.recovered_gap+3606
        if lo<values[i]<hi:values[i]=min([v for v in [lo,hi] if v>=0],key=lambda x:abs(x-values[i]))
    np.testing.assert_array_equal(np.rint(np.maximum(values,0)),f.pred)
    template=pq.read_table(ROOT/'data/submitting.parquet');table=pq.read_table(run.OUT/'candidate.parquet')
    assert template.schema==table.schema and table.num_rows==344841
    for name in template.column_names:
        if name!='TAXITIME_SEC_mvt':assert table[name].equals(template[name])
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),f.pred)
    assert np.isfinite(f.pred).all() and f.pred.ge(0).all()
    checks=message_audit()
    run.write(run.OUT/'audit.json',dict(status='pass',identity=run.art(run.MANIFEST),result=run.art(run.OUT/'result.json'),
        source=run.art(__file__),rows=count,rome_rows=n,cv_rows=ncv,
        independent_sql_raw_inputs_calibration_parent_increment_and_cv=True,
        independent_scalar_message_checks=checks,independent_final_bounds_and_serialization=True,
        all_template_non_target_columns_exact=True,reviewer='root; retained subagents unavailable on account quota'))
    print(dict(status='pass',rows=count,rome_rows=n,cv=result['jan_jul_cv'],messages=checks),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):audit()
