"""Independent SQL half-mixture, error identity and paired-day bootstrap audit.

Only saved predictions and raw validation targets are read. No producer/model/
posterior function, optimization, weight search or inference is invoked.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import time
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_half_mix'
CHAIN=ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
DIRECT=ROOT/'VALIDATION/runs/v3_20261013_portable_direct_cv'
DESIGN=ROOT/'VALIDATION/reports/portable_half_mix_design_20261013.json'


def read(p):return json.loads(Path(p).read_text())
def art(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())
def verify(a):
    path=ROOT/a['path'];assert art(path)==a,path;return path

def close(a,b,atol=1e-8):np.testing.assert_allclose(a,b,atol=atol,rtol=0,equal_nan=True)


def audit():
    started=time.monotonic();assert not (OUT/'audit.json').exists()
    identity,result=read(OUT/'identity.json'),read(OUT/'result.json')
    assert identity['auditor']==art(Path(__file__)) and result['identity']==art(OUT/'identity.json')
    assert identity['weights']==[.5,.5] and identity['design']==read(DESIGN) and identity['design_source']==art(DESIGN)
    for a in identity['sources']:verify(a)
    allocation=read(verify(result['allocation']))
    assert allocation['authority']=='/root' and allocation['identity']==result['identity'] and allocation['state']=='released_for_fixed_half_mix_cv'
    for folder,sub in [(CHAIN,'cv/'),(DIRECT,'')]:
        p,a=read(folder/(sub+'result.json')),read(folder/(sub+'audit.json'))
        assert a['status']=='pass' and a['result']==art(folder/(sub+'result.json'))
        assert a['identity']==p['identity']==art(folder/'identity.json')
        assert identity['chain' if folder==CHAIN else 'direct']==p['prediction']
    parent_score=read(CHAIN/'score.json')
    assert read(CHAIN/'cv/audit.json')['metrics']['score']==art(CHAIN/'score.json')
    assert result['rows']==identity['rows']==344419 and result['reused_development'] and result['no_weight_search']
    f=pd.read_parquet(verify(result['prediction']));assert len(f)==f.mvt_id.nunique()==344419
    assert np.isfinite(f.select_dtypes(include='number')).all().all()
    raw=[]
    for month in ['2025-01','2025-07']:
        paths=[verify(a) for a in identity['sources'] if Path(a['path']).name.startswith('training_'+month+'-01_')]
        assert len(paths)==1;raw.extend(map(str,paths))
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.execute("SET TimeZone='UTC'");con.register('f',f)
        con.read_parquet(str(verify(identity['chain']))).create_view('p')
        con.read_parquet(str(verify(identity['direct']))).create_view('d')
        con.read_parquet(raw).create_view('r')
        n,u,bad=con.execute('''SELECT count(*),count(distinct f.mvt_id),count(*) FILTER(WHERE
            f.chain IS DISTINCT FROM p.submitted OR f.direct IS DISTINCT FROM p.direct_reserve OR f.v80 IS DISTINCT FROM p.v80
            OR f.direct IS DISTINCT FROM round_even(greatest(CAST(d.tabm_direct AS DOUBLE),0.),0)
            OR f.y IS DISTINCT FROM p.y OR f.y IS DISTINCT FROM d.y OR f.y IS DISTINCT FROM r.TAXITIME_SEC_mvt
            OR f.month IS DISTINCT FROM p.month OR f.month IS DISTINCT FROM d.month
            OR f.day IS DISTINCT FROM p.day OR f.day IS DISTINCT FROM d.day
            OR f.month IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m')
            OR f.day IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m-%d')
            OR f.airport IS DISTINCT FROM p.airport OR f.airport IS DISTINCT FROM r.ADEP_mvt
            OR r.PHASE_mvt!='DEP' OR f.chain<0 OR f.direct<0 OR f.v80<0
            OR f.unrounded IS DISTINCT FROM .5*CAST(f.chain AS DOUBLE)+.5*CAST(f.direct AS DOUBLE)
            OR f.mix IS DISTINCT FROM round_even(greatest(f.unrounded,0.),0))
            FROM f JOIN p USING(mvt_id) JOIN d USING(mvt_id) JOIN r ON f.mvt_id=r.MVT_ID_mvt''').fetchone()
        assert n==u==344419 and bad==0,(n,u,bad)
        con.execute('''CREATE TEMP VIEW errors AS SELECT *,
            pow(CAST(chain AS DOUBLE)-y,2) AS chain_se,pow(CAST(direct AS DOUBLE)-y,2) AS direct_se,
            pow(CAST(v80 AS DOUBLE)-y,2) AS v80_se,pow(CAST(mix AS DOUBLE)-y,2) AS mix_se,
            pow(unrounded-y,2) AS raw_se,pow(CAST(chain AS DOUBLE)-CAST(direct AS DOUBLE),2) AS disagreement_se,
            CAST(mix AS DOUBLE)-unrounded AS epsilon,row_number() OVER() AS ordinal FROM f''')
        # Independent pointwise identity exposes complementarity separately
        # from the final rounding effect. Check relative numerical scale too.
        point=con.execute('''SELECT max(abs(raw_se-(.5*chain_se+.5*direct_se-.25*disagreement_se))),
            max(abs((mix_se-raw_se)-(2*(unrounded-y)*epsilon+epsilon*epsilon))),max(abs(epsilon)),
            max(chain_se+direct_se+raw_se+disagreement_se) FROM errors''').fetchone()
        assert point[0]<=max(1e-8,point[3]*1e-12) and point[1]<=max(1e-8,point[3]*1e-12) and point[2]<=.5
        metrics={};identities=[]
        for month in ['2025-01','2025-07']:
            values=con.execute('''SELECT sqrt(avg(chain_se)),sqrt(avg(direct_se)),sqrt(avg(v80_se)),sqrt(avg(mix_se)),
                avg(raw_se),avg(.5*chain_se+.5*direct_se-.25*disagreement_se),avg(disagreement_se),
                avg(mix_se-raw_se),avg(2*(unrounded-y)*epsilon+epsilon*epsilon),count(*) FILTER(WHERE epsilon!=0)
                FROM errors WHERE month=?''',[month]).fetchone()
            metrics[month]=dict(zip(['chain','direct','v80','mix'],values[:4]))
            for k,v in metrics[month].items():close(v,result['months'][month][k])
            close(values[4],values[5],1e-7);close(values[7],values[8],1e-7)
            identities.append(dict(month=month,unrounded_MSE=values[4],parent_average_MSE_minus_quarter_disagreement=values[5],
                parent_disagreement_MSE=values[6],rounding_MSE_change=values[7],independent_rounding_expansion=values[8],rounded_half_ties=values[9]))
        for k in ['chain','direct','v80','mix']:close(np.mean([v[k] for v in metrics.values()]),result['cv'][k])
        for new,old in [('chain','submitted'),('direct','direct_reserve'),('v80','v80')]:close(result['cv'][new],parent_score['cv'][old])
        day=con.execute('''SELECT month,day,count(*) AS n,sum(chain_se) AS chain,sum(direct_se) AS direct,
            sum(v80_se) AS v80,sum(mix_se) AS mix,min(ordinal) AS first_position FROM errors
            GROUP BY month,day ORDER BY month,first_position''').fetchdf()
        disagreement=read(verify(result['disagreement']))
        assert disagreement['identity']==result['identity'] and disagreement['prediction']==result['prediction']
        expected_keys={('all','all')}|{(m,'all') for m in metrics}|{('all',a) for a in f.airport.unique()}|set(zip(f.month,f.airport))
        assert {(x['month'],x['airport']) for x in disagreement['groups']}==expected_keys and len(disagreement['groups'])==len(expected_keys)
        for row in disagreement['groups']:
            actual=con.execute('''SELECT count(*),count(*) FILTER(WHERE chain!=direct),sqrt(avg(disagreement_se)),
                avg(CAST(chain AS DOUBLE)-CAST(direct AS DOUBLE)),max(abs(CAST(chain AS DOUBLE)-CAST(direct AS DOUBLE))),
                sqrt(avg(chain_se)),sqrt(avg(direct_se)) FROM errors
                WHERE (?='all' OR month=?) AND (?='all' OR airport=?)''',[row['month'],row['month'],row['airport'],row['airport']]).fetchone()
            for key,val in zip(['rows','changed','rms','mean_delta','max_abs','chain_rmse','direct_rmse'],actual):close(val,row[key])
    paired={}
    for control in ['chain','direct','v80']:
        rng=np.random.default_rng(0);draws=np.zeros(2000)
        for month in sorted(metrics):
            g=day[day.month.eq(month)];counts=g.n.to_numpy();a=g[control].to_numpy();b=g.mix.to_numpy()
            indices=rng.integers(0,len(g),size=(2000,len(g)));den=counts[indices].sum(1)
            draws+=(np.sqrt(b[indices].sum(1)/den)-np.sqrt(a[indices].sum(1)/den))/2
        change=float(np.mean([v['mix']-v[control] for v in metrics.values()]));ci=np.percentile(draws,[2.5,97.5])
        close(change,result['paired'][control]['change']);close(ci,result['paired'][control]['ci95'])
        decision='improvement' if ci[1]<0 else 'regression' if ci[0]>0 else 'inconclusive'
        assert decision==result['paired'][control]['decision'];paired[control]=dict(change=change,ci95=ci.tolist(),decision=decision)
    both=all(v['mix']<v['v80'] for v in metrics.values());intervals=all(v['ci95'][1]<0 for v in paired.values())
    assert result['gate']==dict(both_months_vs_v80=both,intervals_below_zero_vs_all_three=intervals,passed=both and intervals)
    value=dict(status='pass',source=art(Path(__file__)),identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),
        prediction=result['prediction'],rows=len(f),all_raw_ID_truth_calendar_parent_SQL_serialization_exact=True,
        error_identity=identities,pointwise_max_identity_errors=list(point[:2]),paired=paired,gate=result['gate'],
        no_extra_constraints=True,reused_development=True,independent_confirmation=False,no_models_or_inference=True,
        no_weight_scan=True,no_upload=True,seconds=time.monotonic()-started)
    (OUT/'audit.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='pass',audit=art(OUT/'audit.json'),cv=result['cv'],paired=paired,gate=result['gate'],seconds=value['seconds'])))

if __name__=='__main__':
    with threadpool_limits(1):audit()
