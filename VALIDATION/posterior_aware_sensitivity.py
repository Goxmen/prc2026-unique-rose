"""Input-only feasibility of a fixed, target-only posterior support tilt.

This is not a learned model, CV screen, source-noise fit or new posterior solver.
Each target's original peer message and likelihood remain frozen even while its
own prior weights are perturbed. Source-moment Jacobians are a separate algebra
readout; they do not represent recomputation of the posterior peer messages.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

import joint_public_moments as original
from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'VALIDATION/runs/v3_20261011_joint_public_moments'
OUT = ROOT/'VALIDATION/runs/v3_20261012_posterior_aware_sensitivity'
MANIFEST = OUT/'identity.json'
PROPOSAL = ROOT/'VALIDATION/reports/cohort_learning_20261012/posterior_aware_feasibility_design.json'
INPUT_COLUMNS = ['mvt_id','month','gap','prior','eligible','quality','runway',
                 'local_day','assigned_day','fltrule','delay_scope']
PRED_COLUMNS = ['mvt_id','prior','dep_only','joint','increment','touched','month']
MONTHS = ['2025-01', '2025-07']
DESIGN = dict(status='feasibility only; no fit, outcome score or selection',
    likelihood='original fixed joint t5 with original frozen per-target peer messages',
    tilt='weights*exp(eta*(support-prior)/max(gap-tau,60)); eta=x@theta',
    features=['1','log(gap/3600)','logit(clip((prior-tau)/(gap-tau),1e-6,1-1e-6))','product'],
    derivative='Var_post(y)/scale*x, only target weights differentiated',
    source_jacobian='separate derivative of source means if supported target tilts changed together; not posterior mapping',
    peer_policy='fixed original admissible predictions; no own or peer departure labels',
    no_global_joint_reoptimization=True, no_noise_fit=True, no_training=True,
    source_rank='unique count/sum/delay coordinates, fixed units1/3600/3600; local full-source covariance whitening is descriptive only',
    duplicate_warning='source coordinates may recur in different pairs; stacked whitening is not an independent likelihood or Fisher information',
    nuisance='three constant source-type bias directions only, not every airline/month or population bias',
    finite_difference_step=1e-5, finite_difference_atol=2e-5, finite_difference_rtol=2e-5,
    zero_replay_atol=1e-8, relative_rank_threshold=1e-10,
    runtime_budget_seconds=60, threads=1,
    future_training='admissible inner cross-fitted peer packets required; selected caches currently insufficient')


def art(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))


def read(path): return json.loads(Path(path).read_text())


def verify(record):
    path=ROOT/record['path']; assert art(path)==record, path
    return path


def historical_inputs():
    identity=read(PARENT/'identity.json')
    assert identity['labels_absent_from_inference_packet']
    records=[identity['source'],identity['model'],*identity['helpers'],*identity['inputs']]
    for record in records: verify(record)
    assert identity['model']==art(original.__file__)
    assert art(original.prior.__file__) in identity['helpers']
    audit=read(PARENT/'input_audit.json')
    assert audit['status']=='pass' and audit['identity']==art(PARENT/'identity.json')
    inference=read(PARENT/'inference.json')
    assert inference['identity']==art(PARENT/'identity.json')
    assert art(PARENT/'predictions.parquet') in inference['artifacts']
    for record in inference['artifacts']: verify(record)
    return records


def prepare():
    assert not MANIFEST.exists(), 'Preserve frozen design'
    records=historical_inputs()
    files={ROOT/x['path'] for x in records}
    files.update([Path(__file__),PROPOSAL,PARENT/'identity.json',PARENT/'input_audit.json',
                  PARENT/'inference.json',PARENT/'predictions.parquet'])
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(MANIFEST,dict(design=DESIGN,files=[art(p) for p in sorted(files)],
                            sources_authenticated_to_original=True,checks=checks()))
    print(json.dumps(dict(identity=art(MANIFEST),files=len(files))))


def guard():
    frozen=read(MANIFEST); assert frozen['design']==DESIGN
    for record in frozen['files']: verify(record)
    historical_inputs()
    return frozen


def tilted(values,weights,loglike,statistic,eta):
    logw=np.log(np.maximum(weights,1e-300))+loglike+eta*statistic
    post=np.exp(logw-logsumexp(logw)); mean=float(post@values)
    var=float(post@((values-mean)**2))
    derivative=float(post@((values-mean)*(statistic-float(post@statistic))))
    return mean,var,derivative


def checks():
    values=np.array([300.,1200.,4800.]);weights=np.array([.3,.5,.2]);prior=float(weights@values)
    statistic=(values-prior)/3600.; loglike=np.array([-1.7,-.3,-2.1])
    mean,var,derivative=tilted(values,weights,loglike,statistic,0.)
    np.testing.assert_allclose(derivative,var/3600.,atol=1e-12,rtol=1e-12)
    epsilon=DESIGN['finite_difference_step']
    fd=(tilted(values,weights,loglike,statistic,epsilon)[0]-tilted(values,weights,loglike,statistic,-epsilon)[0])/(2*epsilon)
    np.testing.assert_allclose(fd,derivative,atol=DESIGN['finite_difference_atol'],rtol=DESIGN['finite_difference_rtol'])
    pmean,pvar,pderiv=tilted(values,weights,np.zeros(3),statistic,0.)
    np.testing.assert_allclose([pmean,pvar,pderiv],[prior,weights@((values-prior)**2),weights@((values-prior)**2)/3600.],rtol=1e-12,atol=1e-9)
    order=np.array([2,0,1]);permuted=tilted(values[order],weights[order],loglike[order],statistic[order],.2)
    np.testing.assert_allclose(permuted,tilted(values,weights,loglike,statistic,.2),atol=1e-9,rtol=1e-12)
    singleton=tilted(np.array([2.]),np.array([1.]),np.array([0.]),np.array([0.]),.3)
    assert singleton==(2.,0.,0.)
    return dict(target_only_derivative=True,constant_likelihood_parity=True,
                support_permutation=True,singleton_zero_gradient=True,finite_difference_abs=float(abs(fd-derivative)))


def matrix_summary(matrix,reference_threshold=None):
    matrix=np.asarray(matrix,float)
    assert matrix.ndim==2 and matrix.shape[1]==4 and np.isfinite(matrix).all()
    singular=np.linalg.svd(matrix,compute_uv=False) if len(matrix) else np.zeros(4)
    threshold=((singular[0]*DESIGN['relative_rank_threshold']) if len(singular) else 0.) if reference_threshold is None else reference_threshold
    return dict(rows=len(matrix),columns=4,singular_values=singular.tolist(),
                rank=int(np.sum(singular>threshold)),threshold=float(threshold))


def moments_and_groups(rows,source_taxi,source_delay,parameters):
    """Original arithmetic/order, retained explicitly for all-row old replay."""
    p=rows.prior.to_numpy(float);d=rows.gap.to_numpy(float)
    vectors,covs,prob=original.normal_moments(p,d);supports={}
    for i in np.flatnonzero(d>3600):
        values,weights,_=original.prior.support(d[i],p[i],False,np.nan,'LIRF',parameters['support'])
        value=original.contributions(values,d[i]);mean=weights@value;center=value-mean
        vectors[i],covs[i]=mean,(center.T*weights)@center
        supports[i]=(values,weights,value)
    taxi_ok=rows.eligible.to_numpy(bool)&rows.quality.to_numpy(bool)
    dep_ok=~rows.fltrule.eq('V').to_numpy()&rows.delay_scope.to_numpy(bool)
    taxi_positions=np.flatnonzero(taxi_ok);dep_positions=np.flatnonzero(dep_ok)
    tg={key:taxi_positions[ix] for key,ix in rows.loc[taxi_ok].groupby(['local_day','runway']).indices.items()}
    dg={key:dep_positions[ix] for key,ix in rows.loc[dep_ok].groupby('assigned_day').indices.items()}
    taxi_source=source_taxi.set_index(['local_day','runway']);dep_source=source_delay.set_index('day')
    taxi_info={};dep_info={};taxi_parameters=parameters['taxi'];delay_parameters=parameters['delay']
    for key,ix in tg.items():
        if key not in taxi_source.index: continue
        src=taxi_source.loc[key];mean=vectors[ix,:2].sum(0);n=mean[0]
        if not (len(ix)>=20 and src.source_n>=20 and n>=20 and .5<=src.source_n/n<=1.2):continue
        bias=np.array([1.,len(ix)/100.])@np.asarray(taxi_parameters['coefficients'])
        noise=len(ix)*np.asarray(taxi_parameters['covariance_per_row'])+np.diag([1.,60.**2])
        taxi_info[key]=dict(ix=ix,mean=mean,cov=covs[ix,:2,:2].sum(0),noise=noise,
            observed=np.array([src.source_n,src.source_n*src.source_mean])+bias,n=n)
    for key,ix in dg.items():
        if key not in dep_source.index:continue
        src=dep_source.loc[key]
        if len(ix)<20 or not np.isfinite(src.total+src.n_public):continue
        dn=float(src.n_public-len(ix));noise=3*(delay_parameters['sigma']**2+(delay_parameters['kappa']*dn)**2)
        loading=float(np.sum(prob[ix]*(d[ix]<=3600)))
        dep_info[key]=dict(ix=ix,mean=float(vectors[ix,2].sum()),cov=float(covs[ix,2,2].sum()),
            noise=noise,observed=float(src.total-delay_parameters['bias']),loading=loading)
    tau=parameters['support']['tau']
    active=taxi_ok&dep_ok&(d>3600)&(d<50400)&(p>tau)&(p<d)
    pairs={};targets=[]
    for i in np.flatnonzero(active):
        tkey=(rows.local_day.iloc[i],rows.runway.iloc[i]);dkey=rows.assigned_day.iloc[i]
        if tkey not in taxi_info or dkey not in dep_info:continue
        key=(tkey,dkey)
        if key not in pairs:
            t,a=taxi_info[tkey],dep_info[dkey];shared=np.intersect1d(t['ix'],a['ix'])
            covariance=np.zeros((3,3));covariance[:2,:2]=t['cov'];covariance[2,2]=a['cov']
            covariance[:2,2]=covariance[2,:2]=covs[shared,:2,2].sum(0)
            noise=np.zeros((3,3));noise[:2,:2]=t['noise'];noise[2,2]=a['noise']
            loading=np.array([0.,t['n'],-a['loading']]);noise+=15.**2*np.outer(loading,loading)
            pairs[key]=dict(mean=np.r_[t['mean'],a['mean']],covariance=covariance+noise,
                           noise=noise,observed=np.r_[t['observed'],a['observed']])
        targets.append((i,key))
    return p,d,vectors,covs,supports,taxi_info,dep_info,pairs,targets


def month_gate(month,rows,source_taxi,source_delay,parameters,retained):
    p,d,vectors,covs,supports,taxi_info,dep_info,pairs,targets=moments_and_groups(rows,source_taxi,source_delay,parameters)
    result=p.copy();touched=np.zeros(len(rows),bool);records=[]
    source_derivatives=np.zeros((len(rows),3,4));tau=parameters['support']['tau']
    for i,key in targets:
        group=pairs[key];values,weights,value=supports[i]
        covariance=group['covariance']-covs[i];other=group['mean']-vectors[i]
        # Compute ONCE. Every finite difference below retains this exact array.
        loglike=original.loglike(group['observed'][None,:]-other[None,:]-value,covariance)
        scale=max(d[i]-tau,60.);statistic=(values-p[i])/scale
        mean,var,derivative=tilted(values,weights,loglike,statistic,0.)
        np.testing.assert_allclose(derivative,var/scale,rtol=1e-10,atol=1e-8)
        copy=np.clip((p[i]-tau)/(d[i]-tau),1e-6,1-1e-6)
        a=np.log(d[i]/3600.);b=np.log(copy/(1-copy));x=np.array([1.,a,b,a*b])
        gradient=derivative*x;epsilon=DESIGN['finite_difference_step']
        fd=np.array([(tilted(values,weights,loglike,statistic,epsilon*v)[0]-tilted(values,weights,loglike,statistic,-epsilon*v)[0])/(2*epsilon) for v in x])
        np.testing.assert_allclose(fd,gradient,rtol=DESIGN['finite_difference_rtol'],atol=DESIGN['finite_difference_atol'])
        prior_mean=float(weights@values);prior_var=float(weights@((values-prior_mean)**2))
        centered=value-weights@value;covct=(centered.T*weights)@(statistic-weights@statistic)
        source_derivatives[i]=covct[:,None]*x[None,:]
        result[i]=mean;touched[i]=True
        records.append(dict(mvt_id=int(rows.mvt_id.iloc[i]),month=month,local_day=key[0][0],runway=key[0][1],assigned_day=key[1],
            gap=float(d[i]),prior=float(p[i]),posterior=mean,prior_variance=prior_var,posterior_variance=var,
            variance_ratio=var/prior_var if prior_var else 0.,scale=scale,scalar_derivative=derivative,
            finite_difference_abs=float(np.max(abs(fd-gradient))),
            **{f'x{k}':float(x[k]) for k in range(4)},**{f'gradient{k}':float(gradient[k]) for k in range(4)}))
    np.testing.assert_array_equal(retained.mvt_id,rows.mvt_id)
    np.testing.assert_array_equal(retained.prior,p)
    np.testing.assert_array_equal(retained.touched,touched)
    np.testing.assert_allclose(retained.joint,result,rtol=0,atol=DESIGN['zero_replay_atol'])
    frame=pd.DataFrame(records);x=frame[[f'x{k}' for k in range(4)]].to_numpy()
    centered=x-frame.groupby(['local_day','runway','assigned_day'])[[f'x{k}' for k in range(4)]].transform('mean').to_numpy()
    gradient=frame[[f'gradient{k}' for k in range(4)]].to_numpy()
    centered_gradient=centered*frame.scalar_derivative.to_numpy()[:,None]
    output_summary=matrix_summary(gradient)
    contrast_summary=matrix_summary(centered_gradient,reference_threshold=output_summary['threshold'])
    contrast_summary['threshold_uses_original_matrix']=True
    source_records=[];taxi_j={};delay_j={}
    used_t=sorted({key[0] for key in pairs});used_d=sorted({key[1] for key in pairs})
    for key in used_t:
        taxi_j[key]=source_derivatives[taxi_info[key]['ix'],:2,:].sum(0)
        for k,name in enumerate(['count','taxi_sum']):
            source_records.append(dict(month=month,kind=name,day=key[0],runway=key[1],
                **{f'j{c}':float(taxi_j[key][k,c]) for c in range(4)}))
    for key in used_d:
        delay_j[key]=source_derivatives[dep_info[key]['ix'],2,:].sum(0)
        source_records.append(dict(month=month,kind='delay_sum',day=key,runway='',**{f'j{c}':float(delay_j[key][c]) for c in range(4)}))
    source_frame=pd.DataFrame(source_records)
    assert not source_frame.duplicated(['month','kind','day','runway']).any()
    units=np.where(source_frame.kind.eq('count'),1.,3600.)
    unique_matrix=source_frame[[f'j{k}' for k in range(4)]].to_numpy()/units[:,None]
    whites=[];bias=[]
    for key,group in pairs.items():
        # Marginal source uncertainty includes every prior support, not only
        # measurement noise. Freeze this covariance at the original priors.
        jacobian=np.vstack([taxi_j[key[0]],delay_j[key[1]]]);chol=np.linalg.cholesky(group['covariance'])
        whites.append(np.linalg.solve(chol,jacobian));bias.append(np.linalg.solve(chol,np.eye(3)))
    white=np.concatenate(whites);b=np.concatenate(bias);q,_=np.linalg.qr(b,mode='reduced')
    residual=white-q@(q.T@white)
    white_summary=matrix_summary(white)
    residual_summary=matrix_summary(residual,reference_threshold=white_summary['threshold'])
    residual_summary['frobenius_norm_ratio']=float(np.linalg.norm(residual)/np.linalg.norm(white)) if np.linalg.norm(white) else 0.
    residual_summary['threshold_uses_original_matrix']=True
    quantiles=[0.,.1,.5,.9,1.]
    summary=dict(input_rows=len(rows),supported_targets=len(frame),local_days=int(frame.local_day.nunique()),
        assigned_days=int(frame.assigned_day.nunique()),source_pairs=len(pairs),unique_source_coordinates=len(source_frame),
        exact_eligibility_replay=True,zero_tilt_max_difference=float(np.max(abs(retained.joint.to_numpy()-result))),
        maximum_finite_difference_abs=float(frame.finite_difference_abs.max()),
        nonzero_target_derivatives=int(np.count_nonzero(frame.scalar_derivative)),
        quantile_levels=quantiles,posterior_prior_variance_ratio=np.quantile(frame.variance_ratio,quantiles).tolist(),
        posterior_movement_abs=np.quantile(abs(frame.posterior-frame.prior),quantiles).tolist(),
        output_jacobian=output_summary,within_source_pair_contrasts=contrast_summary,
        unique_source_mean_jacobian_fixed_units=matrix_summary(unique_matrix),
        local_pair_whitened_descriptive=white_summary,
        local_pair_whitened_after_three_constant_source_biases=residual_summary,
        source_pair_rows_are_not_independent_observations=True,
        no_predictive_gain_or_global_identifiability_claim=True)
    return frame,source_frame,summary


def run():
    started=time.monotonic();guard()
    assert not (OUT/'result.json').exists(), 'Preserve completed diagnostic'
    assert not (OUT/'targets.parquet').exists(), 'Inspect partial outputs before retry'
    assert pq.ParquetFile(PARENT/'inputs.parquet').schema_arrow.names==INPUT_COLUMNS
    assert pq.ParquetFile(PARENT/'predictions.parquet').schema_arrow.names==PRED_COLUMNS
    rows=pd.read_parquet(PARENT/'inputs.parquet',columns=INPUT_COLUMNS)
    pred=pd.read_parquet(PARENT/'predictions.parquet',columns=PRED_COLUMNS)
    taxi=pd.read_parquet(PARENT/'taxi_source.parquet',columns=['local_day','runway','source_n','source_mean'])
    delay=pd.read_parquet(PARENT/'delay_source.parquet',columns=['day','n_public','total'])
    parameters=read(PARENT/'parameters.json');assert sorted(rows.month.unique())==MONTHS
    frames=[];sources=[];summaries={}
    with threadpool_limits(limits=1):
        for month in MONTHS:
            frame=rows.loc[rows.month.eq(month)].reset_index(drop=True)
            retained=pred.set_index('mvt_id').loc[frame.mvt_id].reset_index()
            assert retained.month.eq(month).all()
            out,source,summary=month_gate(month,frame,taxi,delay,parameters[month],retained)
            frames.append(out);sources.append(source);summaries[month]=summary
            assert time.monotonic()-started<DESIGN['runtime_budget_seconds'], 'Time budget exceeded; preserve failure, no automatic expansion'
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'targets.parquet',index=False)
    pd.concat(sources,ignore_index=True).to_parquet(OUT/'source_jacobians.parquet',index=False)
    write_json(OUT/'result.json',dict(status='feasibility_only',identity=art(MANIFEST),design=DESIGN,
        months=summaries,seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        labels_loaded=False,models_fitted=False,cv_scored=False,original_likelihood_and_peer_messages_frozen=True,
        outcome='Algebraic sensitivity only; no candidate or fit selected; honest training peer packets still required.',
        outputs=[art(OUT/name) for name in ['targets.parquet','source_jacobians.parquet']]))
    print(json.dumps(dict(result=art(OUT/'result.json'),months=summaries,seconds=time.monotonic()-started)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['checks','prepare','run']);args=parser.parse_args()
    if args.stage=='checks':print(json.dumps(checks()))
    else:globals()[args.stage]()
