"""Diagnose one frozen FINAL support failure, without posterior inference/labels."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import daily_delay_posterior as original
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
FOLDER=OUT/'final/support_failure_diagnosis'


def read(p):return json.loads(Path(p).read_text())
def art(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())


def main():
    assert not FOLDER.exists();FOLDER.mkdir()
    identity=read(OUT/'identity.json');spec=identity['specs']['final']
    raw=pd.read_parquet(ROOT/spec['raw']['path'],filters=[('mvt_id','==',213141563.)])
    base=pd.read_parquet(ROOT/spec['base']['path'],columns=['mvt_id','submitted'],filters=[('mvt_id','==',213141563.)])
    assert len(raw)==len(base)==1 and raw.mvt_id.iloc[0]==base.mvt_id.iloc[0]
    row=raw.iloc[0];p=float(base.submitted.iloc[0]);d=float(row.gap);nm_gap=float(row.nm_gap)
    params=read(ROOT/spec['daily_parameters']['path'])['parameters']['LIRF']
    captured={};tilt=original.tilt
    def observe(values,weights,target):
        captured.update(values=values.copy(),weights=weights.copy(),target=float(target))
        return tilt(values,weights,target)
    original.tilt=observe
    try:
        original.support(d,p,bool(row.has_id),nm_gap,'LIRF',params)
    except AssertionError as error:
        failure=str(error)
    else:raise AssertionError('Fixed original row must reproduce the recorded support failure')
    finally:original.tilt=tilt
    values=captured['values'];weights=captured['weights'];target=captured['target']
    scale=max(np.ptp(values),1);z=(values-target)/scale;lw=np.log(np.maximum(weights,1e-300))
    def mean_z(parameter):
        w=np.exp(lw+parameter*z-logsumexp(lw+parameter*z));return float(w@z)
    old_weights=tilt(values,weights,target);old_mean=float(old_weights@values)
    lower,upper=-128.,128.;bounds=[dict(lower=lower,upper=upper,lower_mean_z=mean_z(lower),upper_mean_z=mean_z(upper))]
    for iteration in range(32):
        if mean_z(lower)<=0<=mean_z(upper):break
        if mean_z(lower)>0:lower*=2
        if mean_z(upper)<0:upper*=2
        bounds.append(dict(lower=lower,upper=upper,lower_mean_z=mean_z(lower),upper_mean_z=mean_z(upper)))
    else:raise AssertionError('Mathematical root could not be bracketed')
    a,b=lower,upper
    for _ in range(55):
        mid=(a+b)/2;w=np.exp(lw+mid*z-logsumexp(lw+mid*z))
        if w@z>0:b=mid
        else:a=mid
    fixed_mean=float(w@values)
    independent_root=brentq(mean_z,lower,upper,xtol=1e-11,rtol=1e-14)
    independent_weights=np.exp(lw+independent_root*z-logsumexp(lw+independent_root*z))
    np.testing.assert_allclose(fixed_mean,target,atol=1e-8,rtol=0)
    np.testing.assert_allclose(w,independent_weights,atol=1e-12,rtol=0)
    np.testing.assert_allclose(independent_weights@values,target,atol=1e-8,rtol=0)
    output=dict(source=art(__file__),identity=art(OUT/'identity.json'),original_tilt_source=art(Path(original.__file__)),
        raw=spec['raw'],base=spec['base'],parameters=spec['daily_parameters'],failure_log=art(OUT/'final/completion/01.log'),
        row=dict(mvt_id=float(row.mvt_id),month=row.month,day=row.day,airport=row.airport,gap=d,has_id=bool(row.has_id),nm_gap=nm_gap,prior=target),
        support=dict(values=values.tolist(),base_weights=weights.tolist(),minimum=float(values.min()),maximum=float(values.max()),
                     original_weights=old_weights.tolist(),original_mean=old_mean,original_error=old_mean-target),
        bracket_witness=bounds,fixed=dict(root_parameter=mid,mean=fixed_mean,weights=w.tolist(),
                                       independent_brent_root=independent_root,independent_mean=float(independent_weights@values)),
        explanation='Original fixed +/-128 interval does not bracket zero; 55 iterations cannot correct missing root. The target is inside the feasible support hull.',
        proposed_recovery='Call unchanged original tilt first. Return its exact weights whenever its existing mean assertion passes. Only on that same assertion failure, require failed root bracketing and expand endpoints until monotone root is enclosed; retain55 bisections and the unchanged mean assertion.',
        unchanged_prior_success_proof='Every completed CV/OLD and37FINAL checkpoint required the exact original support mean assertion to pass. A fallback activated only on that same assertion failure is unreachable on those completed calls.',
        no_labels=True,no_posterior_inference=True,no_file_source_mutation=True,original_failure=failure)
    (FOLDER/'result.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(status='unbracketed_original_root_confirmed',result=art(FOLDER/'result.json'),bounds=bounds,
                         old_mean=old_mean,fixed_mean=fixed_mean,independent_root=independent_root)))


if __name__=='__main__':main()
