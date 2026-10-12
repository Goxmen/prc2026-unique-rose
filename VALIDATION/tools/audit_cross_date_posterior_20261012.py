"""Separate direct-conditioning and SciPy-density audit of the closed screen."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import duckdb
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.special import logsumexp
from scipy.stats import multivariate_t
from threadpoolctl import threadpool_limits

import joint_public_cross_date_cv as run


def main():
    run.guard()
    result=run.read(run.OUT/'result.json')
    assert not result['standalone_gate_pass'] and result['combination'] is None
    prediction=pd.read_parquet(run.OUT/'predictions.parquet').set_index('mvt_id')
    inputs=pd.read_parquet(run.BASE/'inputs.parquet',columns=run.model.INPUTS)
    taxi=pd.read_parquet(run.BASE/'taxi_source.parquet')
    delay=pd.read_parquet(run.BASE/'delay_source.parquet')
    parameters=run.read(run.BASE/'parameters.json')
    selected=run.read(run.pilot.OUT/'identity.json')['selected']
    discrepancies=[];cohorts=0
    for month,prm in parameters.items():
        rows=inputs[inputs.month.eq(month)].reset_index(drop=True)
        bank=run.model.Messages(rows,taxi,delay,prm['taxi'],prm['delay'],prm['support'])
        for key in [k for k in selected if k['month']==month]:
            own=(key['local_day'],key['runway'])
            g=bank.cohort(own,key['assigned_day']);cohorts+=1
            run.pilot.independent_group(bank,g,prm['taxi'],prm['delay'])
            hit=(rows.local_day.eq(own[0])&rows.runway.eq(own[1])&rows.assigned_day.eq(key['assigned_day']))
            indices=np.flatnonzero(hit.to_numpy()&prediction.loc[rows.mvt_id,'touched'].to_numpy())
            for i in indices:
                blocks=[];residuals=[]
                for start,stop in [(3,g['old_size']),(g['old_size'],len(g['mean']))]:
                    if start==stop:continue
                    cov=g['cov'][start:stop,start:stop]
                    residual=g['observed'][start:stop]-g['mean'][start:stop]
                    q=float(residual@np.linalg.solve(cov,residual))
                    weight=min(1.,(5+len(residual))/(3+q))
                    blocks.append(cov/weight);residuals.append(residual)
                mean=g['mean'][:3]-bank.v[i]
                cov=g['cov'][:3,:3]-bank.c[i]
                if blocks:
                    cross=g['cov'][:3,3:]
                    extra=block_diag(*blocks)
                    mean=mean+cross@np.linalg.solve(extra,np.concatenate(residuals))
                    cov=cov-cross@np.linalg.solve(extra,cross.T)
                y,w,v=bank.supports[i]
                # Standardize before SciPy's eigenvalue tolerance is applied.
                sd=np.sqrt(np.diag(cov))
                residual=(g['observed'][:3]-mean-v)/sd
                logw=np.log(np.maximum(w,1e-300))+multivariate_t.logpdf(
                    residual,shape=.6*cov/np.outer(sd,sd),df=5)
                posterior=float(y@np.exp(logw-logsumexp(logw)))
                saved=float(prediction.loc[rows.mvt_id.iloc[i],'cross_joint'])
                np.testing.assert_allclose(posterior,saved,atol=1e-7,rtol=0)
                discrepancies.append(abs(posterior-saved))
    with duckdb.connect(config={'threads':1}) as con:
        con.read_parquet(str(run.OUT/'scored.parquet')).create_view('s')
        con.read_parquet(str(run.P79/'cv.parquet')).create_view('p')
        n,u,bad=con.execute('''SELECT count(*),count(DISTINCT s.mvt_id),sum(CASE WHEN
            s.y IS DISTINCT FROM p.y OR s.month IS DISTINCT FROM p.month OR s.day IS DISTINCT FROM p.day
            OR s.v79 IS DISTINCT FROM p.pred
            OR s.same79 IS DISTINCT FROM round_even(greatest(0,s.v79+s.same78-s.v78),0)
            OR s.cross79 IS DISTINCT FROM round_even(greatest(0,s.v79+s.cross78-s.v78),0)
            THEN 1 ELSE 0 END) FROM s JOIN p USING(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for col in result['cv']:
            scores=dict(con.execute(f'SELECT month,sqrt(avg(pow({col}-y,2))) FROM s GROUP BY month').fetchall())
            for month,value in scores.items():assert abs(value-result['scores'][month][col])<1e-8
            assert abs(np.mean(list(scores.values()))-result['cv'][col])<1e-8
    value=dict(status='pass',source=run.art(__file__),result=run.art(run.OUT/'result.json'),
        all_row_audit=run.art(run.OUT/'audit.json'),rows=n,cohorts=cohorts,posterior_checks=len(discrepancies),
        maximum_posterior_error=max(discrepancies),independent_direct_block_conditioning=True,
        independent_scipy_t_density=True,independent_SQL_scores_and_transport=True,
        scope='Support construction reused; independent covariance assembly and posterior arithmetic; not independent validation')
    run.write(run.OUT/'independent_audit.json',value)
    run.write(run.OUT/'closed.json',dict(status='closed_inconclusive',audit=run.art(run.OUT/'independent_audit.json'),
        change_v78=result['paired']['78'],change_v79=result['paired']['79'],
        january_worse=True,no_combination=True,no_production=True,no_upload=True))
    print(dict(status='pass',rows=n,cohorts=cohorts,posteriors=len(discrepancies),maximum_error=max(discrepancies)))


if __name__=='__main__':
    with threadpool_limits(1):main()
