"""One bounded residual-information certificate; no candidate or optimization.

Reads only already scored original-ranking prediction columns. No CV packet,
taxi truth, base-model inference, hyperparameter search or upload is accessed.
The saved SVD defines one fixed numerical rank, never an outcome-selected rank.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'VALIDATION/runs/v3_20261013_v83_residual_novelty'
OLD=ROOT/'VALIDATION/runs/v3_20261013_portable_feedback_geometry'
STUDENT=ROOT/'VALIDATION/runs/v3_20261013_v80_teacher_distillation'
DESIGN=ROOT/'VALIDATION/reports/v83_residual_novelty_design_20261013.json'
AUDITOR=ROOT/'VALIDATION/tools/audit_v83_residual_novelty_20261013.py'
MANIFEST=OUT/'identity.json'
SCORE_HALF_WIDTH=.00005
NUMERIC_FACTOR=128.
MINIMUM_HEADROOM=.5


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())
def verify(record):
    path=ROOT/record['path'];assert art(path)=={k:record[k] for k in ['path','sha256']}
    return path
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def calculate(P,versions,p83,scores):
    n,k=P.shape;assert k==len(versions) and n>k
    assert np.isfinite(P).all() and np.isfinite(p83).all()
    reference=versions.index(64);best=versions.index(80)
    active=[j for j in range(k) if j!=reference]
    A=(P[:,active]-P[:,reference,None])/np.sqrt(n)
    z=(p83-P[:,reference])/np.sqrt(n)
    U,s,Vt=svd(A,full_matrices=False,check_finite=False,lapack_driver='gesdd')
    eps=np.finfo(np.float64).eps;rank_tolerance=eps*max(n,len(active))*s[0]
    keep=s>rank_tolerance;assert keep.any()
    coordinates=U.T@z
    alpha=Vt[keep].T@(coordinates[keep]/s[keep]);del U
    projected=A@alpha;r=z-projected;q=float(r@r)
    colnorm=np.einsum('ij,ij->j',A,A);z2=float(z@z)
    correction=float(A[:,active.index(best)]@r)
    coefficient=np.r_[-.5*alpha,.5*(alpha.sum()-1),.5]
    score_versions=[versions[j] for j in active]+[64,83]
    center=np.array([scores[str(v)] for v in score_versions],float)
    constant=float(-.5*(z2-alpha@colnorm)+correction)
    h=float(constant+coefficient@(center*center))
    low=(center-SCORE_HALF_WIDTH)**2;high=(center+SCORE_HALF_WIDTH)**2
    bound=NUMERIC_FACTOR*eps*max(n,len(active))
    reconstruction_scale=s[0]*(1+np.linalg.norm(z)+s[0]*np.linalg.norm(alpha))
    numeric_h=bound*(1+z2+abs(alpha)@colnorm+abs(correction)+abs(coefficient)@(high)+reconstruction_scale)
    lo=float(np.nextafter(constant+np.where(coefficient>=0,coefficient*low,coefficient*high).sum()-numeric_h,-np.inf))
    hi=float(np.nextafter(constant+np.where(coefficient>=0,coefficient*high,coefficient*low).sum()+numeric_h,np.inf))
    numeric_q=float(bound*(1+np.linalg.norm(z)+s[0]*np.linalg.norm(alpha))**2)
    orth=float(np.max(abs(A.T@r),initial=0))
    orth_bound=float(bound*s[0]*(1+np.linalg.norm(z)+s[0]*np.linalg.norm(alpha)))
    rank_gap=bool(s[keep].min()>2*rank_tolerance and (not (~keep).any() or s[~keep].max()<rank_tolerance/2))
    precision=bool(q>100*numeric_q and orth<=orth_bound and rank_gap)
    excludes_zero=bool(lo>0 or hi<0);minabs=min(abs(lo),abs(hi)) if excludes_zero else 0.
    maximum_rmse=scores['80']+SCORE_HALF_WIDTH
    reduction=float(minabs**2/(q+numeric_q)) if q+numeric_q>0 else 0.
    consistent=bool(reduction<=maximum_rmse**2)
    headroom=float(maximum_rmse-np.sqrt(max(0.,maximum_rmse**2-reduction))) if consistent else 0.
    upper_headroom=None
    if q>numeric_q:
        optimistic_reduction=max(abs(lo),abs(hi))**2/(q-numeric_q)
        minimum_rmse=max(0.,scores['80']-SCORE_HALF_WIDTH)
        upper_headroom=float(min(maximum_rmse,np.sqrt(optimistic_reduction)) if optimistic_reduction>minimum_rmse**2
            else minimum_rmse-np.sqrt(minimum_rmse**2-optimistic_reduction))
    gate=bool(precision and excludes_zero and consistent and headroom>=MINIMUM_HEADROOM)
    decision=('material_numerical_residual_information' if gate else
              'insufficient_numerical_precision_or_consistency' if not precision or not consistent else
              'no_half_second_oracle_headroom_within_fixed_intervals' if upper_headroom is not None and upper_headroom<MINIMUM_HEADROOM else
              'insufficient_rounding_robust_evidence')
    report=dict(rows=n,columns=k,reference_version=64,baseline_version=80,active_versions=[versions[j] for j in active],
        rank=int(keep.sum()),rank_tolerance=float(rank_tolerance),rank_gap_pass=rank_gap,
        discarded_singular_max=float(s[~keep].max(initial=0)),
        discarded_left_coordinate_energy=float(np.sum(coordinates[~keep]**2)),
        original_box_discarded_direction_RMS_bound=float(4*np.sqrt(len(active))*s[~keep].max(initial=0)),
        retained_condition=float(s[0]/s[keep][-1]),residual_MSE=q,residual_RMS=float(np.sqrt(q)),
        normalized_projection_reconstruction_max=float(np.max(abs(z-projected-r),initial=0)),
        orthogonality_max=orth,orthogonality_bound=orth_bound,q_numeric_envelope=numeric_q,
        h80_center=h,h80_interval=[lo,hi],h_numeric_envelope=float(numeric_h),h80_nonorthogonality_correction=correction,
        cross_moment_constant=constant,score_versions=score_versions,score_coefficients=coefficient.tolist(),
        original_score_half_width=SCORE_HALF_WIDTH,minimum_abs_cross_moment=minabs,
        conservative_oracle_MSE_reduction=reduction,conservative_oracle_RMSE_headroom=headroom,
        optimistic_oracle_RMSE_headroom=upper_headroom,decision=decision,
        gate=dict(precision=precision,excludes_zero=excludes_zero,score_moment_consistent=consistent,
                  minimum_half_second=headroom>=MINIMUM_HEADROOM,passed=gate),
        interpretation='Potential outside the fixed numerical affine tangent on original ranking, not certified exact mathematical rank. Discarded left coordinates can include arbitrary numerical null directions. No candidate, box/ridge-feasible gain, future serialization benefit or validated transfer is established.')
    arrays=dict(singular_values=s,right_vectors=Vt,left_coordinates=coordinates,alpha=alpha,projected_delta=projected*np.sqrt(n),residual=r*np.sqrt(n))
    return report,arrays


def freeze():
    assert not MANIFEST.exists()
    review=read(OUT/'source_review.json');fixture=read(OUT/'synthetic_preflight.json')
    for key,path in [('source',Path(__file__)),('design',DESIGN),('auditor',AUDITOR),('fixture',OUT/'synthetic_preflight.json')]:
        assert review[key]==art(path)
    assert review['status']==fixture['status']=='pass' and fixture['source']==art(Path(__file__))
    old=read(OLD/'identity.json');result=read(OLD/'result.json');audit=read(OLD/'audit.json');closed=read(OLD/'closed.json')
    assert len(old['versions'])==54 and audit['status']=='pass' and closed['gate_pass'] is False
    assert result['identity']==audit['identity']==art(OLD/'identity.json')
    assert audit['result']==art(OLD/'result.json')
    basis=art(OLD/'ranking_basis.parquet');assert basis in result['outputs']
    scored=read(OLD/'solve_started.json');assert scored['identity']==result['identity']
    live=read(verify(scored['live_snapshot']))['rows']
    for version in old['versions']:
        rows=[r for r in live if r.get('filename')==f'unique-rose_v{version}.parquet' and r.get('teamName')=='unique-rose']
        assert len(rows)==1 and rows[0]['usedPairs']==344841 and rows[0]['score']==scored['scores'][str(version)]
    for version,record in scored['scored_manifests'].items():
        past=read(verify(record));assert past['state']=='scored' and past['official_result']==scored['official'][version]
    prod=read(STUDENT/'production/result.json');pa=read(STUDENT/'production/audit.json')
    assert pa['status']=='pass' and pa['result']==art(STUDENT/'production/result.json') and pa['identity']==prod['identity']
    official=read(STUDENT/'submission_v83/official_result.json');submitted=read(STUDENT/'submission_v83/manifest.json')
    assert official['status']=='Succeeded' and official['used_pairs']==344841 and official['file']=='unique-rose_v83.parquet'
    assert submitted['state']=='scored' and submitted['score']==official['score']
    assert submitted['official_result']==official and prod['model']==art(STUDENT/'production/model.json')
    candidate=prod['views']['old']['candidate'];assert pa['views']['old']['candidate']==candidate
    assert submitted['sha256']==submitted['remote_sha256']==candidate['sha256']
    scores=dict(scored['scores']);scores['83']=official['score']
    assert min(scores.values())==scores['80'] and set(scores)=={str(v) for v in old['versions']+[83]}
    paths={Path(__file__),DESIGN,AUDITOR,OUT/'source_review.json',OUT/'synthetic_preflight.json',
           OLD/'identity.json',OLD/'result.json',OLD/'audit.json',OLD/'closed.json',OLD/'solve_started.json',
           STUDENT/'identity.json',STUDENT/'production/result.json',STUDENT/'production/audit.json',
           STUDENT/'production/model.json',STUDENT/'submission_v83/manifest.json',STUDENT/'submission_v83/official_result.json',
           verify(basis),verify(candidate),verify(prod['views']['old']['prediction'])}
    paths.add(verify(scored['live_snapshot']))
    paths.update(verify(record) for record in scored['scored_manifests'].values())
    write(MANIFEST,dict(source=art(Path(__file__)),design=art(DESIGN),auditor=art(AUDITOR),review=art(OUT/'source_review.json'),
        fixture=art(OUT/'synthetic_preflight.json'),versions=old['versions'],scores=scores,basis=basis,candidate=candidate,
        student_prediction=prod['views']['old']['prediction'],student_model=art(STUDENT/'production/model.json'),
        sources=[art(path) for path in sorted(paths)],no_arrays_decoded=True))
    print(json.dumps(dict(identity=art(MANIFEST),no_arrays_decoded=True)))


def run():
    start=time.monotonic();m=read(MANIFEST)
    assert m['source']==art(Path(__file__)) and m['auditor']==art(AUDITOR) and m['design']==art(DESIGN)
    for a in m['sources']:verify(a)
    allocation=read(OUT/'allocation.json')
    assert allocation['authority']=='/root' and allocation['identity']==art(MANIFEST)
    assert allocation['state']=='released_for_single_residual_information_certificate'
    write(OUT/'started.json',dict(identity=art(MANIFEST),allocation=art(OUT/'allocation.json'),pid=os.getpid()))
    versions=m['versions'];assert len(versions)==54
    f=pd.read_parquet(verify(m['basis']),columns=['mvt_id']+[f'v{v}' for v in versions])
    assert len(f)==f.mvt_id.nunique()==344841 and f.mvt_id.notna().all()
    c=pd.read_parquet(verify(m['candidate']),columns=['MVT_ID_mvt','TAXITIME_SEC_mvt'])
    assert len(c)==c.MVT_ID_mvt.nunique()==len(f) and set(c.MVT_ID_mvt)==set(f.mvt_id)
    p83=c.set_index('MVT_ID_mvt').loc[f.mvt_id].TAXITIME_SEC_mvt.to_numpy(float)
    P=f[[f'v{v}' for v in versions]].to_numpy(float)
    for value in [P,p83]:assert np.isfinite(value).all() and (value>=0).all() and np.array_equal(value,np.rint(value))
    retained=pd.read_parquet(verify(m['student_prediction']),columns=['mvt_id','v82','raw','submitted'])
    assert len(retained)==retained.mvt_id.nunique()==len(f) and set(retained.mvt_id)==set(f.mvt_id)
    retained=retained.set_index('mvt_id').loc[f.mvt_id]
    np.testing.assert_array_equal(p83,retained.submitted);np.testing.assert_array_equal(P[:,versions.index(82)],retained.v82)
    model=read(verify(m['student_model']));a=np.zeros(len(versions))
    for v,beta in zip(model['columns'],model['beta']):a[versions.index(v)]=beta
    a[versions.index(82)]=1-sum(model['beta'])
    assert abs(a.sum()-1)<1e-12 and a.min()>=-3 and a.max()<=4
    np.testing.assert_allclose(P@a,retained.raw,atol=1e-7,rtol=0)
    report,arrays=calculate(P,versions,p83,m['scores'])
    arrays.update(mvt_id=f.mvt_id.to_numpy(),raw_student_weights=a)
    np.savez(OUT/'decomposition.npz',**arrays)
    report.update(identity=art(MANIFEST),allocation=art(OUT/'allocation.json'),decomposition=art(OUT/'decomposition.npz'),
        original_affine_student_feasible=True,raw_student_replay_max=float(np.max(abs(P@a-retained.raw))),
        no_candidate=True,no_optimization=True,no_CV_or_taxi_labels=True,no_upload=True,
        final_portability='Original54 contains missing final families. No final counterpart is presumed or silently approximated; even a passed information gate is not production authorization.',
        seconds=time.monotonic()-start)
    write(OUT/'result.json',report)
    print(json.dumps({k:report[k] for k in ['rank','residual_RMS','h80_interval','conservative_oracle_RMSE_headroom','gate','seconds']}))


def preflight():
    assert not MANIFEST.exists();rng=np.random.default_rng(20261013)
    P=rng.normal(size=(127,4))*20+100;P[:,2]=.3*P[:,0]+.7*P[:,1]
    versions=[64,1,80,82];p83=.2*P[:,1]+.8*P[:,3]+rng.normal(size=127)*3
    truth=rng.normal(size=127)*10+100
    scores={str(v):float(np.sqrt(np.mean((P[:,j]-truth)**2))) for j,v in enumerate(versions)}
    scores['83']=float(np.sqrt(np.mean((p83-truth)**2)))
    report,a=calculate(P,versions,p83,scores)
    direct=float(np.mean((P[:,2]-truth)*a['residual']))
    np.testing.assert_allclose(direct,report['h80_center'],atol=1e-10,rtol=0)
    assert report['h80_interval'][0]<=direct<=report['h80_interval'][1]
    np.testing.assert_allclose(p83-P[:,0],a['projected_delta']+a['residual'],atol=1e-12,rtol=0)
    coeff=np.asarray(report['score_coefficients']);base=np.array([scores[str(v)] for v in report['score_versions']])
    for _ in range(100):
        perturbed=base+rng.uniform(-SCORE_HALF_WIDTH,SCORE_HALF_WIDTH,len(base))
        h=report['cross_moment_constant']+coeff@(perturbed*perturbed)
        assert report['h80_interval'][0]<=h<=report['h80_interval'][1]
    zero,_=calculate(P,versions,.2*P[:,1]+.8*P[:,3],dict(scores,**{'83':float(np.sqrt(np.mean((.2*P[:,1]+.8*P[:,3]-truth)**2)))}))
    assert not zero['gate']['precision'] and not zero['gate']['passed']
    write(OUT/'synthetic_preflight.json',dict(status='pass',source=art(Path(__file__)),design=art(DESIGN),synthetic_only=True,
        h80_direct_max_error=abs(direct-report['h80_center']),rank=report['rank'],score_interval_perturbations=100,
        affine_duplicate_precision_gate_fails=True,no_real_arrays=True))
    print(json.dumps(dict(status='pass',fixture=art(OUT/'synthetic_preflight.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['preflight','freeze','run'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
