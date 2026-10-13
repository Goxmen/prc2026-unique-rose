"""Tiny synthetic-only checks of the independent residual-certificate auditor."""
import ast
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import v83_residual_novelty_20261013 as producer
spec=importlib.util.spec_from_file_location('independent_residual_audit',ROOT/'VALIDATION/tools/audit_v83_residual_novelty_20261013.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def main():
    with threadpool_limits(1):
        tree=ast.parse(Path(audit.__file__).read_text())
        function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='audit')
        start=next(i for i,x in enumerate(function.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='n' for t in x.targets))
        end=next(i for i,x in enumerate(function.body) if isinstance(x,ast.Assert) and isinstance(x.test,ast.Call) and isinstance(x.test.func,ast.Name) and x.test.func.id=='all')
        code=compile(ast.fix_missing_locations(ast.Module(body=function.body[start:end],type_ignores=[])),audit.__file__,'exec')
        rng=np.random.default_rng(13);versions=[64,80]+list(range(1,53))
        P=rng.integers(1000,2000,size=(127,54)).astype(float)
        P[:,2]=P[:,3]+P[:,4]-P[:,0]  # Deliberate dependent direction.
        p83=np.rint(.3*P[:,5]+.7*P[:,6]+rng.normal(0,4,len(P)))
        truth=rng.integers(800,1700,size=len(P)).astype(float)
        scores={str(v):float(np.sqrt(np.mean((P[:,j]-truth)**2))) for j,v in enumerate(versions)}
        scores['83']=float(np.sqrt(np.mean((p83-truth)**2)))
        result,saved=producer.calculate(P,versions,p83,scores)
        def check(arrays,report):
            scope=dict(vars(audit));scope.update(P=P,p83=p83,versions=versions,saved=arrays,result=report,m={'scores':scores},ids=np.arange(len(P)))
            exec(code,scope)
            return scope
        scope=check(saved,result)
        assert abs(float(scope['wide_h'])-np.mean((P[:,1]-truth)*saved['residual']))<1e-8
        corrupt=copy.deepcopy(saved);corrupt['alpha'][0]+=.01
        try:check(corrupt,result)
        except AssertionError:pass
        else:raise AssertionError('Corrupt projection accepted')
        corrupt_result=copy.deepcopy(result);corrupt_result['h80_interval'][0]-=1
        try:check(saved,corrupt_result)
        except AssertionError:pass
        else:raise AssertionError('Corrupt score interval accepted')
        coeff=[-2.,1.5,.5];centers=[210.,230.,250.];constant=100.
        lo,hi=audit.decimal_interval(coeff,centers,constant)
        corners=[constant+sum(c*(s+d)**2 for c,s,d in zip(coeff,centers,ds)) for ds in itertools.product([-.00005,.00005],repeat=3)]
        np.testing.assert_allclose([lo,hi],[min(corners),max(corners)],atol=1e-10,rtol=0)
        for path,columns in [('unused.parquet',['y']),('x/cv/packet.parquet',['mvt_id'])]:
            try:audit.projected(path,columns)
            except AssertionError:pass
            else:raise AssertionError('Truth/CV read firewall failed')
        audit.write(producer.OUT/'independent_synthetic_preflight.json',dict(status='pass',source=audit.art(__file__),auditor=audit.art(audit.__file__),producer=producer.art(producer.__file__),rows=len(P),columns=54,
            saved_factor_and_actual_auditor_math_block=True,corrupt_projection_rejected=True,corrupt_interval_rejected=True,
            exact_Decimal_corner_extrema=True,truth_and_CV_projection_rejected=True,
            longdouble_q_difference=abs(float(scope['wide_q'])-scope['q']),longdouble_h_difference=abs(float(scope['wide_h'])-scope['h']),
            no_real_arrays=True,no_candidate=True))
        print(json.dumps({'status':'pass','fixture':audit.art(producer.OUT/'independent_synthetic_preflight.json')}))


if __name__=='__main__':main()
