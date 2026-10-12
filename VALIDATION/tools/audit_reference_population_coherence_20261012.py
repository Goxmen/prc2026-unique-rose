"""Independent source-cell, deletion, calendar and score audit of fixed pilot."""
from pathlib import Path
import hashlib
import json
import sys

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VALIDATION/runs/v3_20261012_reference_population_coherence'
INVERSE = ROOT/'VALIDATION/runs/v3_20261011_public_reference_inverse_features'


def read(path): return json.loads(path.read_text())


def main():
    result = read(OUT/'result.json')
    identity = read(OUT/'identity.json')
    for a in identity['sources']+result['artifacts']:
        assert hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest() == a['sha256']
    history = read(OUT/'prelabel_failure/identity.json')
    assert history['design'] == identity['design']
    assert hashlib.sha256((OUT/'prelabel_failure/source.py').read_bytes()).hexdigest() == history['sources'][0]['sha256']
    assert 'Parser Error' in (OUT/'prelabel_failure/failure.log').read_text()
    assert history['sources'][1:] == identity['sources'][1:]
    rows = pd.read_parquet(OUT/'source_evaluation.parquet')
    months = identity['folds']
    errors = []
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        for name, path in [('r',OUT/'source_evaluation.parquet'),
                           ('t',ROOT/'tmp/research_20261011/daily_taxi_source/training_rows.parquet'),
                           ('p',ROOT/'data/external/eurocontrol_daily_taxi_20261011/daily_runway.parquet'),
                           ('ref',OUT/'reference2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        for fold, allowed in months.items():
            expected = con.execute('''select t.airport,t.local_takeoff local_day,t.runway,
                avg(t.y) truth,count(*) n from t where strftime(t.takeoff,'%Y-%m') in (select unnest(?))
                and t.stand is not null and t.runway is not null and t.y between 0 and 7200 group by all''',[allowed]).df()
            got = rows[rows.fold.eq(fold)]
            join = got.merge(expected,on=['airport','local_day','runway'],validate='one_to_one',suffixes=('','_sql'))
            assert len(join)==len(got)
            np.testing.assert_allclose(join.truth,join.truth_sql,atol=1e-10,rtol=0)
            np.testing.assert_array_equal(join.n,join.n_sql)
            assert set(got.month).issubset(allowed) and fold not in set(got.month)
        bad = con.execute('''select count(*) from r join p on r.airport=p.airport and r.runway=p.runway and r.local_day=p.day
            join ref on r.airport=ref.airport and r.runway=ref.runway and r.local_day=ref.day
            where r.source_n!=p.n or abs(r.source_mean-60*p.total_min/p.n)>1e-10
            or r.reference_error!=ref.reference_error or r.n<20 or r.source_n<20
            or r.source_n/r.n not between .5 and 1.2''').fetchone()[0]
        assert bad==0
        for view in ['2025','ranking']:
            ref=pd.read_parquet(OUT/f'reference{view}.parquet')
            counts=con.execute('''select airport,runway,day,stand,count(*) n from read_parquet(?)
                where stand is not null and runway is not null
                and substr(utc_month,1,4)=substr(month,1,4)
                and ((cast(substr(utc_month,6,2) as integer)-1)%6)=((cast(substr(month,6,2) as integer)-1)%6)
                group by all''',[str(INVERSE/f'inputs{view}.parquet')]).df()
            for key,g in ref.groupby(['airport','runway','pair']):
                g=g.sort_values('day').reset_index(drop=True)
                c=counts[counts.airport.eq(key[0])&counts.runway.eq(key[1])&counts.day.isin(g.day)]
                A=c.pivot(index='day',columns='stand',values='n').fillna(0).reindex(g.day).fillna(0)
                A=A.to_numpy(float)/g.metadata_n.to_numpy()[:,None]
                y=g.reference.to_numpy(); m=g.month.to_numpy()
                # Explicit deletion, independently centering only retained days.
                at=len(g)//2; keep=np.arange(len(g))!=at
                X=A[keep].copy(); response=y[keep].copy(); levels={}
                for period in sorted(set(m)):
                    ix=m[keep]==period
                    xmean=X[ix].mean(0);ymean=response[ix].mean()
                    X[ix]-=xmean;response[ix]-=ymean
                    levels[period]=(xmean,ymean)
                alpha=np.linalg.solve(X@X.T+(10./300.)**2*np.eye(len(X)),response)
                b=X.T@alpha;xm,ym=levels[m[at]]
                prediction=(A[at]-xm)@b+ym
                error=abs(float(prediction-y[at]-g.reference_error.iloc[at]))
                assert error<1e-7
                errors.append(error)
    # Coefficients are measured only on disjoint inner calibration months;
    # reconstruct every saved error without invoking the producing fitter.
    airports=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
    # Producer inherits insertion order, which is bound by its archived source.
    sys.path.insert(0,str(ROOT/'VALIDATION'))
    import monthly_taxi_reconcile
    airports=list(monthly_taxi_reconcile.ZONES)
    for record in read(OUT/'coefficients.json'):
        fold,held=record['fold'],record['held_source_month']
        assert held not in record['fitting_months'] and fold not in record['fitting_months']
        assert set(record['fitting_months']).issubset(months[fold])
        e=rows[rows.fold.eq(fold)&rows.month.eq(held)]
        one=np.column_stack([e.airport.eq(a).to_numpy(float) for a in airports])
        base=np.column_stack([one,one*(1-e.source_n.to_numpy()/e.n.to_numpy())[:,None]/.1])
        cand=np.column_stack([base,one*e.reference_error.to_numpy()[:,None]/10.])
        for name,X in [('baseline',base),('conditional',cand)]:
            expected=(e.truth-e.source_mean).to_numpy()-X@np.array(record[name])
            np.testing.assert_allclose(expected,e[name+'_error'],atol=1e-10,rtol=0)
    for fold,s in result['windows'].items():
        e=rows[rows.fold.eq(fold)]
        for name in ['baseline','conditional']:
            value=float(np.sqrt(np.mean(e[name+'_error']**2)))
            assert abs(value-s[name+'_rmse'])<1e-10
        # Independent day/calendar-block resampling arithmetic.
        group=e.groupby('month');names=sorted(group.groups)
        sums=np.array([[len(group.get_group(m)),(group.get_group(m).baseline_error**2).sum(),
            (group.get_group(m).conditional_error**2).sum()] for m in names])
        index=np.random.default_rng(721).integers(0,len(names),(10000,len(names)))
        resampled=sums[index].sum(1)
        delta=np.sqrt(resampled[:,2]/resampled[:,0])-np.sqrt(resampled[:,1]/resampled[:,0])
        np.testing.assert_allclose(np.quantile(delta,[.025,.975]),s['paired_month_ci95'],atol=1e-10,rtol=0)
        assert s['gate']==(s['conditional_rmse']<=.95*s['baseline_rmse'] and s['paired_month_ci95'][1]<0)
    audit=dict(status='pass',source_rows=len(rows),reference_direct_deletions=len(errors),
        max_reference_deletion_error=max(errors),all_source_hashes=True,raw_SQL_source_truth=True,
        full_prediction_arithmetic=True,outer_and_inner_training_membership=True,
        independent_month_bootstrap=True,gate=False,
        no_taxi_cv_or_candidate=True,prelabel_operational_failure_preserved=True)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    with threadpool_limits(1):main()
