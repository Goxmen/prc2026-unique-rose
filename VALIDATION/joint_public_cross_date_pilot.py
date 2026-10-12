"""Frozen input-only covariance pilot; no target posterior or validation score."""
import argparse
from pathlib import Path
import time
import resource

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import joint_public_cross_date as model
import joint_public_other_runway_pilot as original

ROOT=original.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_joint_public_cross_date_pilot'
PARENT=original.PARENT
DESIGN=dict(settings=model.SETTINGS,
    selected='exact original twelve other-runway pilot cohort keys, chosen before that earlier experiment',
    scope='all previously touched targets within these twelve cohorts',
    budget=dict(seconds=60,incremental_workspace_mib=256,threads=1),
    checks='original3D covariance, positive definiteness, added-date independence, exact target exclusion, independent explicit-flight reconstruction, simultaneous inflated block solve',
    no_target_posterior=True,no_labels=True,no_cv=True,no_parameter_search=True,
    next='Only consider fixed inference if numerical checks pass and added-source variance reduction is nonzero; no outcome-selected subgroup')
read,art,write,verify=original.read,original.art,original.write,original.verify


def freeze():
    assert not (OUT/'identity.json').exists()
    old=read(original.OUT/'design.json')
    census=ROOT/'VALIDATION/runs/v3_20261012_cross_date_taxi_coverage'
    assert read(census/'audit.json')['status']=='pass'
    sources=[Path(__file__),Path(model.__file__),Path(model.same.__file__),Path(model.base.__file__),
        original.OUT/'design.json',census/'result.json',census/'audit.json',
        ROOT/'VALIDATION/reports/cross_date_taxi_20261012/design.json',
        *[PARENT/name for name in ['inputs.parquet','predictions.parquet','taxi_source.parquet','delay_source.parquet','parameters.json']]]
    write(OUT/'identity.json',dict(design=DESIGN,selected=old['selected'],
        sources=[art(p) for p in sources],checks=model.checks()))


def independent_group(messages,group,tp,dp):
    """Reconstruct every component by explicitly walking flight memberships."""
    rows=messages.rows;size=len(group['mean']);peer=np.zeros((size,size));mean=np.zeros(size)
    sets=[(set(messages.taxi[key]['ix'].tolist()),sl) for key,sl in group['blocks']]
    dep=set(group['di'].tolist());used=set(dep)
    for positions,_ in sets:used.update(positions)
    for i in sorted(used):
        projection=np.zeros((size,3))
        for positions,sl in sets:
            if i in positions:projection[sl,:2]=np.eye(2)
        if i in dep:projection[2,2]=1
        mean+=projection@messages.v[i]
        peer+=projection@messages.c[i]@projection.T
    np.testing.assert_allclose(mean,group['mean'],atol=1e-8,rtol=1e-13)
    np.testing.assert_allclose(peer,group['peer'],atol=1e-5,rtol=1e-13)
    independent=np.zeros_like(peer)
    for key,sl in group['blocks']:
        count=len(messages.taxi[key]['ix'])
        independent[sl,sl]=count*np.asarray(tp['covariance_per_row'])+np.diag([1.,3600.])
    source=messages.delay.loc[group['assigned_day']]
    independent[2,2]=3*(dp['sigma']**2+(dp['kappa']*(source.n_public-len(dep)))**2)
    # Explicit date-specific loading list, retaining the original old message.
    oldload=np.zeros(size)
    oldload[2]=-sum(float(messages.prob[i]) for i in sorted(dep) if messages.gap[i]<=3600)
    newload={}
    for key,sl in group['blocks']:
        if key[0]==group['own'][0]:oldload[sl.stop-1]=messages.taxi[key]['n']
        else:newload.setdefault(key[0],np.zeros(size))[sl.stop-1]=messages.taxi[key]['n']
    independent+=225*np.outer(oldload,oldload)
    for loading in newload.values():independent+=225*np.outer(loading,loading)
    reconstructed=peer+independent
    np.testing.assert_allclose(reconstructed,group['cov'],atol=1e-5,rtol=1e-13)
    return dict(explicit_flight_memberships=len(used),maximum_covariance_error=float(abs(reconstructed-group['cov']).max()))


def pilot():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for record in identity['sources']:verify(record)
    assert not (OUT/'result.json').exists()
    started=time.monotonic();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    def budget():
        assert time.monotonic()-started<60,'Fixed60second pilot budget'
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss<256*1024,'Fixed256MiB additional memory budget'
    inputs=pd.read_parquet(PARENT/'inputs.parquet',columns=model.INPUTS)
    touched=pd.read_parquet(PARENT/'predictions.parquet',columns=['mvt_id','touched']).set_index('mvt_id')
    taxi=pd.read_parquet(PARENT/'taxi_source.parquet');delay=pd.read_parquet(PARENT/'delay_source.parquet')
    params=read(PARENT/'parameters.json');records=[]
    for month,prm in params.items():
        rows=inputs[inputs.month.eq(month)].reset_index(drop=True)
        bank=model.Messages(rows,taxi,delay,prm['taxi'],prm['delay'],prm['support'])
        for key in [k for k in identity['selected'] if k['month']==month]:
            budget();own=(key['local_day'],key['runway']);group=bank.cohort(own,key['assigned_day'])
            independent=independent_group(bank,group,prm['taxi'],prm['delay'])
            mask=rows.local_day.eq(own[0])&rows.runway.eq(own[1])&rows.assigned_day.eq(key['assigned_day'])
            ix=np.flatnonzero(mask.to_numpy()&touched.loc[rows.mvt_id,'touched'].to_numpy(bool))
            assert len(ix)>0
            targets=[]
            for i in ix:
                budget();r=bank.condition(i)
                contraction=r['oldcov']-r['newcov'];fraction=float(contraction[2,2]/r['oldcov'][2,2])
                assert -1e-10<=fraction<1
                for name in ['rawcov','oldcov','newcov']:
                    sd=np.sqrt(np.diag(r[name]));np.linalg.cholesky(r[name]/np.outer(sd,sd))
                targets.append(dict(mvt_id=int(rows.mvt_id.iloc[i]),additional_delay_variance_reduction=fraction,
                    new_weight=r['new_diagnostic']['weight'],new_q=r['new_diagnostic']['q'],
                    direct_mean_max_error=r['direct_mean_max_error'],direct_cov_max_error=r['direct_cov_max_error']))
            records.append(dict(**key,new_cells=group['new_cells'],old_cells=group['old_cells'],targets=targets,independent=independent))
    budget()
    contractions=[v['additional_delay_variance_reduction'] for r in records for v in r['targets']]
    value=dict(status='pass',identity=art(OUT/'identity.json'),records=records,cohorts=len(records),
        targets=len(contractions),targets_with_information=sum(v>1e-12 for v in contractions),
        contraction_quantiles=np.quantile(contractions,[0,.5,.9,1]).tolist(),
        seconds=time.monotonic()-started,incremental_workspace_mib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss)/1024,
        no_target_posterior=True,no_labels=True,no_cv=True,scope='Model-information diagnostic, no prediction gain estimate')
    write(OUT/'result.json',value)
    print({k:value[k] for k in ['status','cohorts','targets','targets_with_information','contraction_quantiles','seconds']},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','pilot'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
