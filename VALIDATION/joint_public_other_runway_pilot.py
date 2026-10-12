"""Fixed source-only opportunity test for additional runway observations.

No target posterior, departure labels, CV, fitting, transfer weight or gate.
Only Gaussian Schur moment diagnostics; the prospective 3D t5 is not run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import joint_public_moments as base

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'VALIDATION/runs/v3_20261011_joint_public_moments'
OUT = ROOT / 'tmp/research_20261012/error_mechanisms/other_runway'
MONTHS = ['2025-01', '2025-07']
KEYS = ['local_day', 'runway', 'assigned_day']
INPUTS = ['mvt_id', 'month', 'gap', 'prior', 'eligible', 'quality', 'runway',
          'local_day', 'assigned_day', 'fltrule', 'delay_scope']
DESIGN = dict(
    selection='six evenly spaced integer positions in sorted original touched cohort keys per month; all if fewer',
    extra='all supported other runway cells on same local_day intersecting assigned-day delay cohort; full taxi membership',
    targets='every original touched target within each selected cohort',
    control='unchanged 3D own count/sum plus assigned-day delay; target moments removed exactly once',
    conditioning='Gaussian Schur moments only; no enlarged Student distribution or shared radial scale',
    common='one225*loading*loading.T across all cells, preserving exact original3D block',
    noise='original per-cell taxi block and one delay variance; cross-runway source-error correlation remains unknown',
    prior='unchanged original normal/discrete support and fixed fold parameters; generate only selected union',
    outputs='coverage, covariance contraction, mean-shift/extra-residual leverage, shared-flight/common15 components',
    budget=dict(threads=1,seconds=30,incremental_workspace_mib=256),
    no_target_posterior=True,no_departure_labels=True,no_cv=True,no_fit=True,
    no_new_gate=True,no_weight=True,no_threshold_search=True,
)


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def verify(record):
    assert art(ROOT / record['path']) == {k:record[k] for k in ['path','sha256']}


def write(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def freeze():
    assert not (OUT/'design.json').exists(), 'Preserve fixed cohort selection'
    parent=read(PARENT/'identity.json')
    for item in [parent['model'],*parent['helpers'],*parent['inputs']]:verify(item)
    assert parent['model']==art(base.__file__)
    inference=read(PARENT/'inference.json')
    assert inference['identity']==art(PARENT/'identity.json')
    for item in inference['artifacts']:verify(item)
    # Select keys using old metadata only, before loading extra source cells.
    cells=pd.read_parquet(PARENT/'cells.parquet',columns=['month',*KEYS])
    assert not cells.duplicated(['month',*KEYS]).any()
    selected=[]
    for month in MONTHS:
        group=cells[cells.month.eq(month)].sort_values(KEYS,kind='stable').reset_index(drop=True)
        ix=np.linspace(0,len(group)-1,min(6,len(group)),dtype=int)
        selected.extend(group.iloc[ix].to_dict('records'))
    sources=[art(p) for p in [Path(__file__),PARENT/'identity.json',PARENT/'inference.json',
        PARENT/'inputs.parquet',PARENT/'taxi_source.parquet',PARENT/'delay_source.parquet',
        PARENT/'parameters.json',PARENT/'cells.parquet',PARENT/'predictions.parquet']]
    sources += [parent['model'],*parent['helpers']]
    write(OUT/'design.json',dict(design=DESIGN,selected=selected,sources=sources))
    print(dict(frozen=True,cohorts=len(selected),selected=selected),flush=True)


def condition(mean, covariance, cross, extra_covariance, innovation):
    """Schur computation via Cholesky, without eigenvalue/variance repair."""
    if cross.shape[1]==0 or not np.any(cross):
        return mean.copy(),covariance.copy(),np.zeros_like(covariance),np.zeros_like(mean),0.
    sd=np.sqrt(np.diag(extra_covariance))
    lower=np.linalg.cholesky(extra_covariance/np.outer(sd,sd))
    z=np.linalg.solve(lower,cross.T/sd[:,None])
    residual=np.linalg.solve(lower,innovation/sd)
    contraction=z.T@z
    shift=z.T@residual
    schur=covariance-contraction
    s=np.sqrt(np.diag(schur))
    np.linalg.cholesky(schur/np.outer(s,s))
    return mean+shift,schur,contraction,shift,float(residual@residual)


def checks():
    m=np.array([1.,2.,3.]);a=np.array([[4.,.2,.3],[.2,5.,.4],[.3,.4,6.]])
    for b,c,r in [(np.zeros((3,0)),np.zeros((0,0)),np.zeros(0)),
                   (np.zeros((3,2)),np.eye(2),np.array([1e12,-1e12]))]:
        actual=condition(m,a,b,c,r)
        np.testing.assert_array_equal(actual[0],m);np.testing.assert_array_equal(actual[1],a)
    b=np.array([[.1,.2],[.3,-.1],[.2,.4]]);c=np.array([[2.,.2],[.2,3.]])
    residual=np.array([2.,-1.]);actual=condition(m,a,b,c,residual)
    np.testing.assert_allclose(actual[0],m+b@np.linalg.solve(c,residual),atol=1e-14,rtol=0)
    np.testing.assert_allclose(actual[1],a-b@np.linalg.solve(c,b.T),atol=1e-14,rtol=0)
    shuffled=condition(m,a,b[:,::-1],c[::-1,::-1],residual[::-1])
    np.testing.assert_allclose(shuffled[0],actual[0],atol=1e-14,rtol=0)
    np.testing.assert_allclose(shuffled[1],actual[1],atol=1e-14,rtol=0)
    return dict(no_extra_exact=True,disconnected_exact=True,direct_solve_parity=True,permutation=True)


def pilot():
    identity=read(OUT/'design.json');assert identity['design']==DESIGN
    for item in identity['sources']:verify(item)
    assert not (OUT/'result.json').exists(),'Preserve fixed pilot'
    test=checks();started=time.monotonic();rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    def budget():
        if time.monotonic()-started>=30:raise TimeoutError('Fixed30second diagnostic budget exhausted')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss>256*1024:
            raise MemoryError('Fixed256MiB incremental workspace exceeded')
    records=[];failure=None
    try:
        inputs=pd.read_parquet(PARENT/'inputs.parquet',columns=INPUTS)
        old=pd.read_parquet(PARENT/'predictions.parquet',columns=['mvt_id','touched']).set_index('mvt_id')
        assert inputs.mvt_id.is_unique and old.index.is_unique
        taxi_source=pd.read_parquet(PARENT/'taxi_source.parquet').set_index(['local_day','runway'])
        delay_source=pd.read_parquet(PARENT/'delay_source.parquet').set_index('day')
        parameters=read(PARENT/'parameters.json')
        for month in MONTHS:
            selected=[q for q in identity['selected'] if q['month']==month]
            rows=inputs[inputs.month.eq(month)].reset_index(drop=True)
            p=rows.prior.to_numpy(float);gap=rows.gap.to_numpy(float)
            taxi_ok=(rows.eligible&rows.quality).to_numpy(bool)
            delay_ok=(~rows.fltrule.eq('V')&rows.delay_scope).to_numpy(bool)
            needed=(taxi_ok&rows.local_day.isin([q['local_day'] for q in selected]).to_numpy()) | (
                delay_ok&rows.assigned_day.isin([q['assigned_day'] for q in selected]).to_numpy())
            vectors,covs,prob=base.normal_moments(p,gap)
            prm=parameters[month]
            for count,i in enumerate(np.flatnonzero(needed&(gap>3600))):
                if count%64==0:budget()
                y,w,_=base.prior.support(gap[i],p[i],False,np.nan,'LIRF',prm['support'])
                v=base.contributions(y,gap[i]);mean=w@v;center=v-mean
                vectors[i]=mean;covs[i]=(center.T*w)@center
            positions=np.flatnonzero(taxi_ok)
            groups={key:positions[ix] for key,ix in rows.loc[taxi_ok].groupby(['local_day','runway']).indices.items()}
            taxi={}
            for key,ix in groups.items():
                if key[0] not in {q['local_day'] for q in selected} or key not in taxi_source.index:continue
                src=taxi_source.loc[key];mean=vectors[ix,:2].sum(0);n=mean[0]
                if not(len(ix)>=20 and src.source_n>=20 and n>=20 and .5<=src.source_n/n<=1.2):continue
                bias=np.array([1.,len(ix)/100.])@np.asarray(prm['taxi']['coefficients'])
                taxi[key]=dict(ix=ix,mean=mean,n=n,cov=covs[ix,:2,:2].sum(0),
                    observed=np.array([src.source_n,src.source_n*src.source_mean])+bias,
                    noise=len(ix)*np.asarray(prm['taxi']['covariance_per_row'])+np.diag([1.,3600.]))
            for key in selected:
                budget();own=(key['local_day'],key['runway']);dkey=key['assigned_day']
                ti=taxi[own]['ix'];di=np.flatnonzero(delay_ok&rows.assigned_day.eq(dkey).to_numpy())
                ds=delay_source.loc[dkey];assert len(di)>=20 and np.isfinite(ds.total+ds.n_public)
                extras=sorted(k for k,t in taxi.items() if k[0]==own[0] and k!=own and np.intersect1d(t['ix'],di).size)
                blocks=[(own,slice(0,2)),*[(k,slice(3+2*j,5+2*j)) for j,k in enumerate(extras)]]
                dim=3+2*len(extras);mean=np.zeros(dim);observed=np.zeros(dim)
                peer=np.zeros((dim,dim));measurement=np.zeros_like(peer);loading=np.zeros(dim)
                mean[2]=vectors[di,2].sum();observed[2]=ds.total-prm['delay']['bias']
                peer[2,2]=covs[di,2,2].sum()
                dn=float(ds.n_public-len(di))
                measurement[2,2]=3*(prm['delay']['sigma']**2+(prm['delay']['kappa']*dn)**2)
                loading[2]=-float(np.sum(prob[di]*(gap[di]<=3600)))
                for group,sl in blocks:
                    t=taxi[group];mean[sl]=t['mean'];observed[sl]=t['observed'];peer[sl,sl]=t['cov']
                    measurement[sl,sl]=t['noise'];loading[sl.stop-1]=t['n']
                    shared=np.intersect1d(t['ix'],di)
                    peer[sl,2]=covs[shared,:2,2].sum(0);peer[2,sl]=peer[sl,2]
                common=225*np.outer(loading,loading);full=peer+(measurement+common)
                sd=np.sqrt(np.diag(full));correlation=full/np.outer(sd,sd)
                np.linalg.cholesky(correlation)
                # Literal original three-coordinate construction, before adding cells.
                shared=np.intersect1d(ti,di);old_peer=np.zeros((3,3));old_noise=np.zeros((3,3))
                old_peer[:2,:2]=taxi[own]['cov'];old_peer[2,2]=covs[di,2,2].sum()
                old_peer[:2,2]=old_peer[2,:2]=covs[shared,:2,2].sum(0)
                old_noise[:2,:2]=taxi[own]['noise'];old_noise[2,2]=measurement[2,2]
                old_loading=np.array([0.,taxi[own]['n'],loading[2]])
                old_noise+=225*np.outer(old_loading,old_loading)
                np.testing.assert_array_equal(full[:3,:3],old_peer+old_noise)
                np.testing.assert_array_equal(mean[:3],np.r_[taxi[own]['mean'],vectors[di,2].sum()])
                selected_targets=np.flatnonzero(rows.local_day.eq(own[0]).to_numpy()&rows.runway.eq(own[1]).to_numpy()
                    &rows.assigned_day.eq(dkey).to_numpy()&old.loc[rows.mvt_id,'touched'].to_numpy(bool))
                assert len(selected_targets)>0
                extra_members=np.unique(np.concatenate([taxi[k]['ix'] for k in extras])) if extras else np.array([],int)
                observed_peers=np.intersect1d(extra_members,di)
                target_records=[]
                for i in selected_targets:
                    budget();assert i in ti and i in di and i not in extra_members
                    leave=full.copy();leave[:3,:3]-=covs[i]
                    other=mean.copy();other[:3]-=vectors[i]
                    np.testing.assert_array_equal(leave[:3,:3],old_peer+old_noise-covs[i])
                    np.testing.assert_array_equal(other[:3],mean[:3]-vectors[i])
                    a=leave[:3,:3];cross=leave[:3,3:];b=leave[3:,3:];innovation=observed[3:]-other[3:]
                    conditional,schur,reduction,shift,extra_mahal=condition(other[:3],a,cross,b,innovation)
                    la=np.linalg.cholesky(a/np.outer(np.sqrt(np.diag(a)),np.sqrt(np.diag(a))))
                    sa=np.sqrt(np.diag(a))
                    z=np.linalg.solve(la,reduction/np.outer(sa,sa));standard=np.linalg.solve(la,z.T).T
                    eig=np.linalg.eigvalsh(standard)
                    assert eig.min()>-1e-9 and eig.max()<1+1e-9
                    null=condition(other[:3],a,np.zeros((3,0)),np.zeros((0,0)),np.zeros(0))
                    np.testing.assert_array_equal(null[0],other[:3]);np.testing.assert_array_equal(null[1],a)
                    disconnected=condition(other[:3],a,np.zeros_like(cross),b,innovation)
                    np.testing.assert_array_equal(disconnected[0],other[:3]);np.testing.assert_array_equal(disconnected[1],a)
                    den=covs[di,2,2].sum()-covs[i,2,2]
                    scale=np.outer(sa,np.sqrt(np.diag(b))) if len(b) else np.zeros((3,0))
                    target_records.append(dict(mvt_id=int(rows.mvt_id.iloc[i]),
                        delay_peer_variance_covered=float(covs[observed_peers,2,2].sum()/den) if den>0 else None,
                        delay_variance_reduction=float(reduction[2,2]/a[2,2]),
                        standardized_reduction_eigenvalues=eig.tolist(),
                        standardized_shift_norm=float(np.linalg.norm(np.linalg.solve(la,shift/sa))),
                        shift_count=float(shift[0]),shift_taxi_sum=float(shift[1]),shift_delay_sum=float(shift[2]),
                        extra_mahalanobis=extra_mahal,extra_dimension=len(b),
                        shared_flight_cross_norm=float(np.linalg.norm(peer[:3,3:]/scale)) if len(b) else 0.,
                        common15_cross_norm=float(np.linalg.norm(common[:3,3:]/scale)) if len(b) else 0.,
                        schur_correlation_min_eigenvalue=float(np.linalg.eigvalsh(schur/np.outer(np.sqrt(np.diag(schur)),np.sqrt(np.diag(schur)))).min())))
                records.append(dict(**key,extra_runways=[k[1] for k in extras],extra_cells=len(extras),
                    delay_rows=len(di),target_taxi_rows=len(ti),extra_taxi_rows=len(extra_members),
                    extra_shared_delay_rows=len(observed_peers),targets=target_records,
                    full_correlation_condition=float(np.linalg.cond(correlation)),
                    old_3d_block_exact=True,target_removed_once=True,no_extra_and_disconnected_exact=True))
                print(dict(**key,extra_cells=len(extras),targets=len(target_records)),flush=True)
    except (TimeoutError,MemoryError,np.linalg.LinAlgError,AssertionError) as error:
        failure=f'{type(error).__name__}: {error}'
    result=dict(status='complete' if failure is None else 'fixed_pilot_failed',design=art(OUT/'design.json'),
        source=art(__file__),checks=test,selected=identity['selected'],records=records,failure=failure,
        seconds=time.monotonic()-started,incremental_workspace_mib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss)/1024,
        no_target_posterior=True,no_departure_labels=True,no_cv=True,
        interpretation='Structural information/leverage only; no expected RMSE gain. Independent source-noise blocks may overstate precision.')
    write(OUT/'result.json',result);print(dict(status=result['status'],seconds=result['seconds'],failure=failure),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','checks','pilot'])
    with threadpool_limits(1):
        value=globals()[parser.parse_args().stage]()
        if value is not None:print(value,flush=True)
