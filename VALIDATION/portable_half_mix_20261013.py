"""One fixed half-and-half portable mixture screen using audited cached CV only.

Parents are already rounded. Re-round their equal average, with no constraints,
weight search, fits or posterior inference. A failed fixed gate closes this mix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import tuning_seasonal as seasonal

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'VALIDATION/runs'
OUT=RUNS/'v3_20261013_portable_half_mix'
CHAIN=RUNS/'v3_20261013_portable_public_source'
DIRECT=RUNS/'v3_20261013_portable_direct_cv'
DESIGN=ROOT/'VALIDATION/reports/portable_half_mix_design_20261013.json'
AUDITOR=ROOT/'VALIDATION/tools/audit_portable_half_mix_20261013.py'


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())
def verify(record):
    path=ROOT/record['path'];assert art(path)==record,path;return path
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def runtime():return dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)


def freeze():
    assert not (OUT/'identity.json').exists()
    design=read(DESIGN);assert design['weights']==[.5,.5] and design['no_weight_search'] and design['no_new_constraints']
    parent=read(CHAIN/'cv/result.json');checked=read(CHAIN/'cv/audit.json')
    assert checked['status']=='pass' and checked['result']==art(CHAIN/'cv/result.json')
    assert checked['identity']==parent['identity']==art(CHAIN/'identity.json')
    assert checked['prediction']==parent['prediction'] and checked['metrics']['score']==art(CHAIN/'score.json')
    direct=read(DIRECT/'result.json');audit=read(DIRECT/'audit.json')
    assert audit['status']=='pass' and audit['result']==art(DIRECT/'result.json')
    assert audit['identity']==direct['identity']==art(DIRECT/'identity.json')
    paths={Path(__file__),AUDITOR,DESIGN,Path(seasonal.__file__),CHAIN/'identity.json',CHAIN/'cv/result.json',
           CHAIN/'cv/audit.json',CHAIN/'score.json',verify(parent['prediction']),DIRECT/'identity.json',
           DIRECT/'result.json',DIRECT/'audit.json',verify(direct['prediction'])}
    for month in ['2025-01','2025-07']:
        raw=list((ROOT/'data').glob(f'training_{month}-01_*.parquet'))
        assert len(raw)==1;paths.add(raw[0])
    for record in design['sources']:paths.add(verify(record))
    OUT.mkdir(parents=True,exist_ok=True)
    value=dict(design=design,design_source=art(DESIGN),sources=[art(path) for path in sorted(paths)],
               source=art(__file__),auditor=art(AUDITOR),chain=parent['prediction'],direct=direct['prediction'],
               runtime=runtime(),weights=[.5,.5],rows=344419,no_fit=True,no_upload=True)
    write(OUT/'identity.json',value)
    print(json.dumps(dict(status='frozen',identity=art(OUT/'identity.json'))))


def screen():
    identity=read(OUT/'identity.json');assert identity['source']==art(__file__) and identity['runtime']==runtime()
    for record in identity['sources']:verify(record)
    allocation=read(OUT/'allocation.json');assert allocation['authority']=='/root'
    assert allocation['identity']==art(OUT/'identity.json') and allocation['state']=='released_for_fixed_half_mix_cv'
    assert not (OUT/'result.json').exists() and not (OUT/'predictions.parquet').exists()
    parent=pd.read_parquet(verify(identity['chain']));direct=pd.read_parquet(verify(identity['direct']))
    assert parent.mvt_id.is_unique and direct.mvt_id.is_unique and len(parent)==len(direct)==344419
    direct=direct.set_index('mvt_id').loc[parent.mvt_id].reset_index()
    for key in ['month','day','y']:np.testing.assert_array_equal(parent[key],direct[key])
    np.testing.assert_array_equal(parent.direct_reserve,np.rint(np.maximum(direct.tabm_direct,0)).astype(np.int64))
    frame=parent[['mvt_id','month','day','airport','y']].copy()
    frame['chain']=parent.submitted.to_numpy(dtype=np.int64)
    frame['direct']=parent.direct_reserve.to_numpy(dtype=np.int64)
    frame['v80']=parent.v80.to_numpy(dtype=np.int64)
    frame['unrounded']=.5*frame.chain.to_numpy(float)+.5*frame.direct.to_numpy(float)
    frame['mix']=np.rint(np.maximum(frame.unrounded,0)).astype(np.int64)
    assert np.isfinite(frame.select_dtypes(include='number')).all().all()
    frame.to_parquet(OUT/'predictions.parquet',index=False)
    pd.testing.assert_frame_equal(frame,pd.read_parquet(OUT/'predictions.parquet'),check_exact=True)
    columns=['chain','direct','v80','mix']
    months={month:{key:float(np.sqrt(np.mean((part[key]-part.y)**2))) for key in columns}
            for month,part in frame.groupby('month')}
    cv={key:float(np.mean([value[key] for value in months.values()])) for key in columns}
    previous=read(CHAIN/'score.json')['cv']
    for key,old in [('chain','submitted'),('direct','direct_reserve'),('v80','v80')]:
        assert abs(cv[key]-previous[old])<1e-10,(key,cv[key],previous[old])
    paired={key:seasonal.paired_bootstrap(frame.day,frame.month,(frame[key]-frame.y)**2,
             (frame.mix-frame.y)**2) for key in columns[:-1]}
    gate_months=all(value['mix']<value['v80'] for value in months.values())
    gate_paired=all(value['ci95'][1]<0 for value in paired.values())
    groups=[('all','all',np.arange(len(frame)))]
    for month,ix in frame.groupby('month',sort=True).indices.items():groups.append((month,'all',ix))
    for airport,ix in frame.groupby('airport',sort=True).indices.items():groups.append(('all',airport,ix))
    for (month,airport),ix in frame.groupby(['month','airport'],sort=True).indices.items():groups.append((month,airport,ix))
    disagreement=[]
    for month,airport,ix in groups:
        part=frame.iloc[ix];delta=part.chain.to_numpy(float)-part.direct.to_numpy(float)
        disagreement.append(dict(month=month,airport=airport,rows=len(ix),changed=int(np.count_nonzero(delta)),
            rms=float(np.sqrt(np.mean(delta**2))),mean_delta=float(delta.mean()),max_abs=float(abs(delta).max()),
            chain_rmse=float(np.sqrt(np.mean((part.chain-part.y)**2))),
            direct_rmse=float(np.sqrt(np.mean((part.direct-part.y)**2)))))
    write(OUT/'disagreement.json',dict(source=art(__file__),identity=art(OUT/'identity.json'),
          prediction=art(OUT/'predictions.parquet'),groups=disagreement,other_views_not_inferred=True,no_tuning=True))
    result=dict(identity=art(OUT/'identity.json'),allocation=art(OUT/'allocation.json'),rows=len(frame),
                prediction=art(OUT/'predictions.parquet'),disagreement=art(OUT/'disagreement.json'),
                months=months,cv=cv,paired=paired,gate=dict(both_months_vs_v80=gate_months,
                intervals_below_zero_vs_all_three=gate_paired,passed=gate_months and gate_paired),
                reused_development=True,no_weight_search=True,independent_audit_pending=True,
                no_version=True,no_upload=True)
    write(OUT/'result.json',result)
    print(json.dumps({key:result[key] for key in ['months','cv','paired','gate']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','screen'])
    args=parser.parse_args()
    with threadpool_limits(1):globals()[args.stage]()
