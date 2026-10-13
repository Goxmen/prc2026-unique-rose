"""Independent training-only residual dependence audit; no producer import.

Authenticates six honest R20 exports before labels, reconstructs input membership
and original binary outcomes from raw allowed-month files, independently checks
SQL pair moments/exact permutation expectation, 199 seeded null permutations and
2,000 day bootstrap draws. No model inference, posterior, taxi score or upload.
"""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name] = '1'
import hashlib
import json
from pathlib import Path
import re
import time

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VALIDATION/runs/v3_20261013_correlated_modes_pilot'
PRODUCER = ROOT/'VALIDATION/correlated_modes_pilot_20261013.py'
PLAN = ROOT/'VALIDATION/reports/cohort_learning_20261013/correlated_modes_pilot_design.json'
MONTHS = ['2025-01','2025-07']
ALLOWED = {'2025-01':[f'2025-{m:02}' for m in range(3,12)],
           '2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
STRATA = ['airline','month','p_bin','gap_band','clock_band']
DROP = ['op_taxi_30d','op_taxi_30d_n','rwy_taxi_30d','stand_taxi_90d','aphour_taxi_30d']

PAIR_SQL = '''WITH s AS (
    SELECT stratum,count(*) AS n,sum(u) AS total,sum(u*u) AS squares FROM f GROUP BY 1),
    c AS (SELECT airline,day,month,stratum,count(*) AS n FROM f GROUP BY 1,2,3,4),
    x AS (SELECT c.*,c.n*s.total/s.n AS load,
       CASE WHEN s.n>1 THEN c.n*(c.n-1)*(s.total*s.total-s.squares)/(s.n*(s.n-1)) ELSE 0. END AS same
       FROM c JOIN s USING(stratum)),
    e AS (SELECT airline,day,month,sum(same)+pow(sum(load),2)-sum(load*load) AS expected
       FROM x GROUP BY 1,2,3),
    o AS (SELECT airline,day,month,pow(sum(u),2)-sum(u*u) AS observed_cross,sum(v) AS denominator
       FROM f GROUP BY 1,2,3)
    SELECT o.day,o.month,sum(o.observed_cross) AS observed_cross,
       sum(e.expected) AS null_mean,sum(o.denominator) AS denominator
       FROM o JOIN e ON o.airline=e.airline AND o.day=e.day AND o.month=e.month
       GROUP BY 1,2 ORDER BY 1'''


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())
def verify(record):
    path=ROOT/record['path'];assert art(path)==record,path
    return path
def ids_hash(ids):return hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest()
def close(actual,expected,atol=1e-5,rtol=1e-12):
    np.testing.assert_allclose(actual,expected,rtol=rtol,atol=atol,equal_nan=True)
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def provenance():
    m=read(OUT/'identity.json');result=read(OUT/'result.json')
    assert m['source']==art(PRODUCER) and m['auditor']==art(__file__) and m['design']==art(PLAN)
    assert m['no_numeric_read_before_freeze'] and m['no_new_model_fit'] and m['no_held_outer_labels']
    assert m['review']==art(OUT/'source_review.json')
    review=read(verify(m['review']))
    assert review['status']=='pass' and review['source']==m['source'] and review['auditor']==m['auditor'] and review['design']==m['design']
    pinned={r['path']:r for r in m['sources']}
    for record in pinned.values():verify(record)
    for path in [PRODUCER,Path(__file__),PLAN,ROOT/'VALIDATION/rome_hybrid.py',ROOT/'VALIDATION/rome_signals.py']:
        assert pinned[str(path.relative_to(ROOT))]==art(path)
    assert set(m['folds'])==set(MONTHS)
    plan=read(verify(m['design']))
    for record in plan['metadata_sources']:
        verify(record);assert pinned[record['path']]==record
    verify(m['peer_identity']);assert pinned[m['peer_identity']['path']]==m['peer_identity']
    receipts={}
    for outer in MONTHS:
        spec=m['folds'][outer]
        assert spec['months']==ALLOWED[outer] and set(spec['labels'])==set(ALLOWED[outer])
        assert spec['records']==plan['folds'][outer] and len(spec['records'])==3
        seen=[];receipts[outer]=[]
        for k,record in enumerate(spec['records']):
            assert Path(record['receipt']['path']).name==f'{outer}_g{k}.json'
            actual=read(verify(record['receipt']))
            assert actual['identity']==m['peer_identity']
            for key in ['prediction','model','fitting_months','predicted_months','held_rows','held_ids_sha256']:
                assert actual[key]==record[key]
            assert actual['rounds']==1700 and actual['params']['seed']==42 and actual['all_held_reload_exact']
            assert len(actual['columns'])==63 and not set(DROP)&set(actual['columns'])
            assert outer not in actual['fitting_months']+actual['predicted_months']
            assert not set(actual['fitting_months'])&set(actual['predicted_months'])
            assert sorted(actual['fitting_months']+actual['predicted_months'])==ALLOWED[outer]
            assert actual['predicted_months']==ALLOWED[outer][k::3]
            for key in ['prediction','model']:verify(actual[key])
            seen+=actual['predicted_months'];receipts[outer].append(actual)
        assert sorted(seen)==ALLOWED[outer]
    union=sorted(set(ALLOWED[MONTHS[0]]+ALLOWED[MONTHS[1]]))
    assert sorted(m['raw'])==union
    for month,record in m['raw'].items():
        path=Path(record['path'])
        assert path.parent==Path('data') and re.fullmatch(r'training_\d{4}-\d{2}-01_\d{4}-\d{2}-01.parquet',path.name)
        assert path.name[9:16]==month and pinned[record['path']]==record
    for outer in MONTHS:
        for month,record in m['folds'][outer]['labels'].items():
            assert month!=outer and Path(record['path']).name==f'labels_{month}.parquet'
            assert pinned[record['path']]==record
    assert result['identity']==art(OUT/'identity.json') and result['allocation']==art(OUT/'allocation.json')
    allocation=read(verify(result['allocation']))
    assert allocation['authority']=='/root' and allocation['identity']==result['identity']
    assert allocation['state']=='released_for_fixed_correlated_mode_training_pilot'
    assert read(OUT/'started.json')==dict(identity=result['identity'],allocation=result['allocation'])
    assert result['no_fit'] and result['no_held_outer_labels'] and result['no_candidate'] and result['no_upload']
    for record in result['outputs']:verify(record)
    return m,result,receipts


def reconstruct(outer,m,receipts,con):
    vectors=[]
    for receipt in receipts:
        f=pd.read_parquet(verify(receipt['prediction']))
        assert list(f)==['mvt_id','p'] and len(f)==receipt['held_rows']
        assert f.mvt_id.is_unique and f.mvt_id.notna().all() and ids_hash(f.mvt_id)==receipt['held_ids_sha256']
        assert np.isfinite(f.p).all() and f.p.between(0,1).all()
        f['oof_months']=[tuple(receipt['predicted_months'])]*len(f)
        vectors.append(f)
    probabilities=pd.concat(vectors,ignore_index=True)
    assert probabilities.mvt_id.is_unique
    # Raw input-only fields first. Only this outer's allowed months are read.
    paths=[str(verify(m['raw'][month])) for month in ALLOWED[outer]]
    con.from_parquet(paths).create_view('raw_fold',replace=True)
    raw=con.execute('''SELECT MVT_ID_mvt AS mvt_id,FLIGHT_mvt AS flight,
        ADEP_mvt AS airport,strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month,
        strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS day,
        epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt) AS gap,
        epoch(MVT_TIME_UTC_mvt-AOBT_3_flt) AS mvt_minus_aobt3_sec
        FROM raw_fold WHERE PHASE_mvt='DEP' AND ADEP_mvt='LIRF' ''').fetchdf()
    assert raw.mvt_id.is_unique
    frame=probabilities.merge(raw,on='mvt_id',how='left',validate='one_to_one',sort=False)
    assert len(frame)==len(probabilities) and frame.month.notna().all()
    assert set(frame.month)==set(ALLOWED[outer]) and outer not in set(frame.month)
    assert all(month in allowed for month,allowed in zip(frame.month,frame.oof_months))
    # Independent scalar form of the frozen original ICAO/IATA parser.
    frame['airline']=[None if not isinstance(s,str) or len(s)<3 else s[:3] if s[:3].isalpha() else s[:2] for s in frame.flight]
    coverage=dict(oof_rows=len(frame),oof_ids_sha256=ids_hash(frame.mvt_id))
    keep=np.isfinite(frame.gap)&frame.gap.gt(3600)&frame.gap.lt(50400)&frame.airline.notna()&frame.airline.ne('')
    frame=frame.loc[keep].copy();coverage['input_eligible_rows']=len(frame)
    con.register('candidate_inputs',frame)
    frame=con.execute('''WITH bins AS (SELECT *,CAST(least(floor(10*p),9) AS BIGINT) AS p_bin,
        CASE WHEN gap<=14400 THEN 0 ELSE 1 END AS gap_band,
        CASE WHEN NOT coalesce(isfinite(gap-mvt_minus_aobt3_sec),false) THEN 0
             WHEN abs(gap-mvt_minus_aobt3_sec)<=60 THEN 1 ELSE 2 END AS clock_band
        FROM candidate_inputs), c AS (SELECT *,count(distinct day) OVER
        (PARTITION BY airline,month,p_bin,gap_band,clock_band) AS days FROM bins),
        kept AS (SELECT * FROM c WHERE days>=4)
        SELECT mvt_id,p,month,day,gap,airline,p_bin,gap_band,clock_band,
        dense_rank() OVER(ORDER BY airline,month,p_bin,gap_band,clock_band)-1 AS stratum,
        dense_rank() OVER(ORDER BY airline,day)-1 AS "group"
        FROM kept ORDER BY mvt_id''').fetchdf()
    # Membership and full OOF provenance now authenticated; allowed labels only.
    label_parts=[pd.read_parquet(verify(m['folds'][outer]['labels'][month])) for month in ALLOWED[outer]]
    labels=pd.concat(label_parts,ignore_index=True)
    assert labels.mvt_id.is_unique
    labels=labels.set_index('mvt_id').loc[frame.mvt_id].reset_index()
    assert np.isfinite(labels.y).all()
    con.register('kept',frame);con.register('cached_labels',labels)
    raw_labels=con.execute('''SELECT k.mvt_id,CAST(r.TAXITIME_SEC_mvt AS DOUBLE) AS y
        FROM kept k JOIN raw_fold r ON k.mvt_id=r.MVT_ID_mvt ORDER BY k.mvt_id''').fetchdf()
    np.testing.assert_array_equal(raw_labels.mvt_id,frame.mvt_id)
    np.testing.assert_array_equal(raw_labels.y,labels.y)
    con.register('raw_labels',raw_labels)
    frame=con.execute('''WITH z AS (SELECT k.*,CAST(abs(k.gap-y)<=60 AS TINYINT) AS z
        FROM kept k JOIN raw_labels USING(mvt_id))
        SELECT *,gap*(z-p) AS u,gap*gap*p*(1-p) AS v FROM z ORDER BY mvt_id''').fetchdf()
    sizes=frame.groupby('group').size()
    coverage.update(retained_rows=len(frame),retained_strata=frame.stratum.nunique(),
        groups=len(sizes),groups_with_pairs=int(sizes.ge(2).sum()),days=frame.day.nunique(),
        dropped_sparse_stratum_rows=coverage['input_eligible_rows']-len(frame))
    saved=pd.read_parquet(OUT/outer/'retained_rows.parquet')
    assert list(saved)==list(frame)
    for column in frame:
        if column in ['u','v']:close(saved[column],frame[column],atol=1e-8,rtol=1e-14)
        else:pd.testing.assert_series_equal(saved[column],frame[column],check_dtype=False,check_names=False,check_exact=True)
    return frame,coverage


def statistics(frame,con):
    con.register('f',frame)
    days=con.execute(PAIR_SQL).fetchdf()
    days['numerator']=days.observed_cross-days.null_mean
    day_index=pd.Index(days.day)
    day_groups=frame.groupby(['airline','day'],sort=True).indices
    group_indices=list(day_groups.values())
    group_days=np.array([day_index.get_loc(key[1]) for key in day_groups])
    # Match IEEE addition order for exact permutation >= ties. The underlying
    # moments and analytic null have already been independently derived in SQL.
    def ordered_days(values):
        out=np.zeros(len(days))
        for ix,day in zip(group_indices,group_days):
            x=values[ix]
            out[day]+=np.add.accumulate(x)[-1]**2-np.add.accumulate(x*x)[-1]
        return out
    random=np.random.Generator(np.random.PCG64(20261013))
    strata=list(frame.groupby('stratum',sort=True).indices.values())
    values=frame.u.to_numpy(float);denominator=float(frame.v.sum())
    observed_ordered=ordered_days(values)
    close(observed_ordered,days.observed_cross)
    null=np.zeros((199,len(days)))
    for iteration in range(199):
        permuted=values.copy()
        for ix in strata:permuted[ix]=values[random.permutation(ix)]
        null[iteration]=ordered_days(permuted)
    days['mc_null_mean']=null.mean(0)
    permutation=pd.DataFrame(dict(iteration=np.arange(199),null_cross=null.sum(1),
        excess_variance_ratio=(observed_ordered.sum()-null.sum(1))/denominator))
    bootstrap_rng=np.random.Generator(np.random.PCG64(20261014))
    numerator=np.zeros(2000);divisor=np.zeros(2000)
    for month in sorted(days.month.unique()):
        group=days[days.month.eq(month)]
        take=bootstrap_rng.integers(0,len(group),size=(2000,len(group)))
        numerator+=group.numerator.to_numpy()[take].sum(1)
        divisor+=group.denominator.to_numpy()[take].sum(1)
    assert (divisor>0).all()
    draws=numerator/divisor
    estimate=float(days.numerator.sum()/denominator)
    pvalue=float((1+np.count_nonzero(permutation.null_cross.to_numpy()>=observed_ordered.sum()))/200)
    ci=np.percentile(draws,[2.5,97.5])
    return days,permutation,draws,dict(denominator=denominator,
        observed_cross=float(days.observed_cross.sum()),null_mean_cross=float(days.null_mean.sum()),
        mc_null_mean_cross=float(null.sum(1).mean()),
        excess_variance_ratio=estimate,ci95=ci.tolist(),permutation_p=pvalue,
        gate=bool(estimate>=.10 and ci[0]>0 and pvalue<=.05))


def audit():
    start=time.monotonic();assert not (OUT/'independent_audit.json').exists()
    manifest,result,receipts=provenance();reports={}
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for outer in MONTHS:
            report=read(OUT/outer/'result.json')
            assert report==result['results'][outer] and report['identity']==result['identity'] and report['outer']==outer
            assert report['inputs']==manifest['folds'][outer]
            for record in report['outputs']:verify(record)
            frame,coverage=reconstruct(outer,manifest,receipts[outer],con)
            assert report['coverage']==coverage
            denominator=float(frame.v.sum())
            supported=coverage['retained_rows']>=1000 and coverage['groups_with_pairs']>=100 and coverage['days']>=50 and denominator>0
            if not supported:
                assert report['status']=='inconclusive_insufficient_support' and not report['gate']
                close(denominator,report['denominator'])
                assert len(report['outputs'])==1
                for name in ['days.parquet','permutations.parquet','bootstrap.npy']:assert not (OUT/outer/name).exists()
                reports[outer]=dict(status='pass_insufficient_support',coverage=coverage,gate=False)
                continue
            days,permutations,draws,metrics=statistics(frame,con)
            expected_days=pd.read_parquet(OUT/outer/'days.parquet')
            assert set(days.day)==set(expected_days.day) and days.day.is_unique and expected_days.day.is_unique
            expected_days=expected_days.set_index('day').loc[days.day].reset_index()
            np.testing.assert_array_equal(days.month,expected_days.month)
            for column in ['observed_cross','null_mean','denominator','numerator','mc_null_mean']:
                close(days[column],expected_days[column])
            saved_permutations=pd.read_parquet(OUT/outer/'permutations.parquet')
            assert list(saved_permutations)==list(permutations) and len(saved_permutations)==199
            np.testing.assert_array_equal(saved_permutations.iteration,permutations.iteration)
            close(saved_permutations.null_cross,permutations.null_cross)
            close(saved_permutations.excess_variance_ratio,permutations.excess_variance_ratio,atol=1e-11)
            close(np.load(OUT/outer/'bootstrap.npy',allow_pickle=False),draws,atol=1e-11)
            for key in ['denominator','observed_cross','null_mean_cross','mc_null_mean_cross','excess_variance_ratio','ci95','permutation_p']:
                close(metrics[key],report[key],atol=1e-9 if key in ['ci95','excess_variance_ratio','permutation_p'] else 1e-5)
            assert metrics['gate']==report['gate']
            assert report['status']==('material_dependence' if metrics['gate'] else 'closed_fixed_gate_failed')
            reports[outer]=dict(status='pass',coverage=coverage,**metrics,
                full_raw_ID_calendar_label_and_probability_checks=True,
                exact_stratified_null_and_observed_pair_SQL=True,independent_199_permutations_and_2000_bootstrap=True)
    gate=all(report['gate'] for report in reports.values());assert gate==result['gate']
    value=dict(status='pass',source=art(__file__),identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),
        results=reports,gate=gate,no_model_inference=True,no_held_outer_labels=True,no_taxi_score=True,no_upload=True,
        limitation='Training-only residual dependence with coarse nuisance controls. Day bootstrap conditions on empirical stratum marginals; neither omitted daily bias nor nuisance uncertainty is eliminated. Two outer training windows overlap.',
        seconds=time.monotonic()-start)
    write(OUT/'independent_audit.json',value)
    print(json.dumps(dict(status='pass',gate=gate,audit=art(OUT/'independent_audit.json'))))


if __name__=='__main__':
    with threadpool_limits(1):audit()
