"""Independent saved-row audit of the conditional portable consistency screen.

Checks original prefix, seed pool, raw truth/calendar, daily source cells,
serialized constraints and independent paired-day scores. Native historical
direct corrections are replayed through pinned original functions, not newly
derived. No new producer is imported, no models or posterior are evaluated.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import gc
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import audit_portable_public_source_20261013 as historical

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT/'VALIDATION/runs/v3_20261013_nn_public_tabm_track_consistency'
OUT = TRAIN/'portable_comparison'
PRIMARY = TRAIN/'comparison'
PORTABLE = ROOT/'VALIDATION/runs/v3_20261013_portable_public_source'
DIRECT = ROOT/'VALIDATION/runs/v3_20261013_portable_direct_cv'
PRODUCER = ROOT/'VALIDATION/track_consistency_portable_cv.py'
PLAN = ROOT/'VALIDATION/reports/cohort_learning_20261013/target_track_consistency_production_design.json'
ARMS = ['supervised_control', 'training_consistency', 'target_consistency']
ALL = ['control', *ARMS]
REFERENCES = ['v82', 'supervised_control', 'training_consistency']
MONTHS = ['2025-01', '2025-07']
COUNT = 344419
FIELDS = ['N_raw','N','quality_eligible','invalid_share','daily_shift','applied_shift','Q','J','submitted']
read, art, verify, close = historical.read, historical.art, historical.verify, historical.close


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def aligned(record, ids, full=False):
    frame = pd.read_parquet(verify(record))
    assert frame.mvt_id.is_unique and frame.mvt_id.notna().all()
    if full:
        assert len(frame)==len(ids) and set(frame.mvt_id)==set(ids)
    return frame.set_index('mvt_id').loc[ids].reset_index()


def provenance():
    identity = read(OUT/'identity.json')
    assert identity['source']==art(PRODUCER) and identity['auditor']==art(__file__)
    assert identity['plan']==art(PLAN)
    assert identity['before_any_new_score'] and identity['before_new_prediction_decode']
    design = identity['design']
    assert design['arms']==ARMS and design['months']==MONTHS and design['rows']==COUNT
    assert design['selected_arm']=='target_consistency' and design['no_control_promotion']
    assert design['no_fit'] and design['no_calibration'] and design['no_blend_search']
    pinned = {record['path']:record for record in identity['sources']}
    for record in pinned.values():
        verify(record)
    for path in [Path(__file__), Path(historical.__file__), PRODUCER, PLAN,
                 ROOT/'VALIDATION/tuning_seasonal.py']:
        assert pinned[str(path.relative_to(ROOT))]==art(path)
    review = read(verify(identity['source_review']))
    assert review['status']=='pass' and review['source']==identity['source']
    assert review['auditor']==identity['auditor'] and review['design']==identity['plan']
    primary, primary_audit = read(PRIMARY/'result.json'), read(PRIMARY/'independent_audit.json')
    assert primary_audit['status']=='pass' and primary_audit['result']==art(PRIMARY/'result.json')
    assert primary_audit['identity']==primary['identity']==art(PRIMARY/'identity.json')
    assert primary_audit['gate']==primary['gate']
    fit = read(TRAIN/'audit.json')
    assert fit['status']=='pass' and fit['identity']==art(TRAIN/'identity.json')
    assert primary_audit['fit_audit']==art(TRAIN/'audit.json')
    expected = []
    receipts = {}
    for month in MONTHS:
        receipts[month] = {}
        for arm in ARMS:
            path = TRAIN/arm/'networks'/f'{month}_s42.json'
            record = read(path)
            assert record['identity']==fit['identity'] and record['month']==month and record['arm']==arm
            assert record['all_held_reloaded_exact'] and record['epochs']==2
            for key in ['model','prediction','preprocessing','checkpoint','teacher']:
                verify(record[key])
            expected.append(art(path))
            receipts[month][arm] = record
    assert len(fit['members'])==6
    assert sorted(expected,key=lambda x:x['path'])==sorted(fit['members'],key=lambda x:x['path'])
    assert primary_audit['members']==expected
    allocation = read(OUT/'allocation.json')
    assert allocation['authority']=='/root' and allocation['identity']==art(OUT/'identity.json')
    assert allocation['state']=='released_for_fixed_portable_consistency_cv'
    links = dict(identity=art(OUT/'identity.json'), allocation=art(OUT/'allocation.json'),
                 fit_audit=art(TRAIN/'audit.json'), members=expected,
                 primary_result=art(PRIMARY/'result.json'), primary_audit=art(PRIMARY/'independent_audit.json'))
    if not primary['gate']['passed']:
        closed = read(OUT/'closed.json')
        for key,value in links.items():
            assert closed[key]==value
        assert closed['status']=='closed_primary_gate_failed' and closed['gate'] is False
        assert closed['new_prediction_vectors_decoded'] is False and closed['no_alternative_arm']
        assert not (OUT/'result.json').exists() and not (OUT/'scored.parquet').exists() and not (OUT/'started.json').exists()
        return identity, None, receipts, links
    result = read(OUT/'result.json')
    for key,value in links.items():
        assert result[key]==value
    assert result['status']=='complete' and result['no_new_posterior_evaluation']
    assert result['exact_v82_control_all_stages'] and result['unchanged_prefix']
    assert result['no_control_promotion'] and result['no_upload']
    assert read(OUT/'started.json')==links
    return identity, result, receipts, links


def direct_replay(frame, packet, receipts):
    # Original historical postprocessor only. Never call the new producer.
    sys.path.insert(0, str(ROOT/'VALIDATION'))
    import tuning_seasonal as original
    identity = read(DIRECT/'identity.json')
    previous_result, previous_audit = read(DIRECT/'result.json'), read(DIRECT/'audit.json')
    assert previous_audit['status']=='pass' and previous_audit['result']==art(DIRECT/'result.json')
    assert previous_result['identity']==previous_audit['identity']==art(DIRECT/'identity.json')
    verify(previous_result['prediction'])
    reports = []
    for month in MONTHS:
        spec = identity['specs'][month]
        held = pd.read_pickle(verify(spec['held']))
        metadata = read(verify(spec['meta']))
        ids = held['raw'].mvt_id.to_numpy()
        saved = frame.set_index('mvt_id').loc[ids]
        old = packet.set_index('mvt_id').loc[ids]
        np.testing.assert_array_equal(saved.y, held['raw'].taxi_sec)
        members = [aligned(record['prediction'],ids,True).base.to_numpy(float) for record in spec['tabm']]
        for arm in ALL:
            first = members[0] if arm=='control' else aligned(receipts[month][arm]['prediction'],ids,True).base.to_numpy(float)
            assert np.isfinite(first).all() and (first>=0).all()
            pool = (first + members[1] + members[2])/3.
            np.testing.assert_array_equal(pool, saved[arm+'_pool'])
            if arm=='control':
                np.testing.assert_array_equal(pool, old.tabm_pool)
            ordinary = .355*(old.calls.to_numpy(float)+pool)+.29*old.tree.to_numpy(float)
            expected = original.final_prediction(ordinary,old.probability.to_numpy(float),held['V'],held['raw'],metadata['parameters'],metadata['corrections'])
            np.testing.assert_array_equal(expected, saved[arm+'_direct'])
            if arm=='control':
                np.testing.assert_array_equal(expected, old.tabm_direct)
        reports.append(dict(month=month, rows=len(ids), four_seed_pools_exact=True,
                            all_four_original_native_corrections_replayed_exact=True))
        del held, saved, old, members
        gc.collect()
    return reports


def daily_sql(month, params):
    coefficient = params['coefficients']
    assert len(coefficient)==20 and params['allowed']==historical.FOLDS[month]
    assert set(params['noise'])==set(historical.AIRPORTS)
    intercept = historical.airport_case(coefficient[:10])
    slope = historical.airport_case(coefficient[10:])
    noise = historical.airport_case([params['noise'][airport] for airport in historical.AIRPORTS])
    return f'''CREATE OR REPLACE TEMP VIEW dc AS WITH g AS (
        SELECT i.airport,i.local_day,i.runway,count(*) AS n,avg(CAST(f.N AS DOUBLE)) AS mean
        FROM f JOIN i USING(mvt_id) JOIN m USING(mvt_id)
        WHERE f.month='{month}' AND m.eligible AND f.N BETWEEN 0 AND 7200 GROUP BY 1,2,3),
        c AS (SELECT g.*,ds.source_n,ds.source_mean,
            {intercept}+{slope}*(1.-CAST(ds.source_n AS DOUBLE)/g.n)/.1 AS composition,
            {noise} AS noise,40000./g.n+225. AS prior_var
            FROM g LEFT JOIN ds USING(airport,local_day,runway)),
        v AS (SELECT *,coalesce(n>=20 AND source_n>=20 AND CAST(source_n AS DOUBLE)/n BETWEEN .5 AND 1.2
            AND isfinite(source_mean),false) AS supported,
            prior_var/(prior_var+noise*noise) AS gain,source_mean+composition-mean AS raw_gap FROM c)
        SELECT *,CASE WHEN supported THEN gain*least(120.,greatest(-120.,raw_gap)) ELSE 0. END AS shift FROM v'''


def suffix_checks(con, frame, daily_cells, params):
    reports = {}
    for arm in ALL:
        f = frame.copy()
        for field in FIELDS:
            f[field] = f[arm+'_'+field]
        f['new_direct'] = f[arm+'_direct']
        con.register('f', f)
        assert con.execute('''SELECT count(*) FROM f WHERE N_raw IS DISTINCT FROM
            CAST(M AS DOUBLE)+new_direct-direct_control OR applied_shift IS DISTINCT FROM
            CASE WHEN quality_eligible THEN daily_shift ELSE 0. END''').fetchone()[0]==0
        for name, expression in [('N','f.N_raw'), ('Q','CAST(f.N AS DOUBLE)+f.applied_shift'),
                                  ('J','CAST(f.Q AS DOUBLE)+.5*f.joint_increment'),
                                  ('submitted','CAST(f.J AS DOUBLE)+.5*f.robust_delta')]:
            constrained = historical.constraint(expression)
            bad = con.execute(f'''SELECT count(*) FROM f JOIN d USING(mvt_id) LEFT JOIN w USING(mvt_id)
                WHERE f.{name} IS DISTINCT FROM round_even(greatest(({constrained}),0.),0)''').fetchone()[0]
            assert bad==0,(arm,name,bad)
        assert con.execute('SELECT count(*) FROM f WHERE NOT robust_active AND (robust_delta!=0 OR submitted!=J)').fetchone()[0]==0
        assert con.execute('''SELECT count(*) FROM f JOIN i USING(mvt_id)
            LEFT JOIN ms ON f.airport=ms.airport AND substr(i.local_day,1,7)=ms.local_month
            WHERE f.invalid_share IS DISTINCT FROM ms.invalid
            OR f.quality_eligible IS DISTINCT FROM coalesce(ms.invalid<.05,false)''').fetchone()[0]==0
        cells = []
        errors = {}
        for month in MONTHS:
            con.execute(daily_sql(month,params[month]))
            cells.append(con.execute(f"SELECT *, '{month}' AS fold FROM dc").fetchdf())
            expected = con.execute(f'''SELECT f.mvt_id,CASE WHEN m.eligible AND f.N BETWEEN 0 AND 7200
                THEN coalesce(dc.shift,0.) ELSE 0. END AS shift
                FROM f JOIN i USING(mvt_id) JOIN m USING(mvt_id)
                LEFT JOIN dc ON i.airport=dc.airport AND i.local_day=dc.local_day AND i.runway=dc.runway
                WHERE f.month='{month}' ''').fetchdf().set_index('mvt_id')
            subset = f[f.month.eq(month)]
            error = float(np.max(abs(expected.loc[subset.mvt_id,'shift'].to_numpy()-subset.daily_shift.to_numpy())))
            assert error<=1e-8,(arm,month,error)
            errors[month] = error
        count = historical.compare_cells(daily_cells[daily_cells.arm.eq(arm)],pd.concat(cells),
            ['fold','airport','local_day','runway'],
            ['n','mean','source_n','source_mean','composition','noise','prior_var','supported','gain','raw_gap','shift'])
        reports[arm] = dict(daily_cells=count, maximum_daily_shift_errors=errors,
                           all_serialized_constraints_and_source_quality_SQL_exact=True)
    return reports


def metrics(con, frame, result):
    con.register('f', frame.assign(_audit_position=np.arange(len(frame))))
    columns = ['v82', *ARMS]
    rows = con.execute('SELECT month,'+','.join(f'sqrt(avg(pow(CAST({col} AS DOUBLE)-y,2)))' for col in columns)+' FROM f GROUP BY month ORDER BY month').fetchall()
    scores = {row[0]:dict(zip(columns,row[1:])) for row in rows}
    cv = {col:float(np.mean([scores[month][col] for month in MONTHS])) for col in columns}
    terms = ','.join(f'sum(pow(CAST({col} AS DOUBLE)-y,2)) AS {col}' for col in columns)
    day = con.execute(f'SELECT month,day,count(*) AS n,{terms},min(_audit_position) AS pos FROM f GROUP BY month,day ORDER BY month,pos').fetchdf()
    paired = {}
    for reference in REFERENCES:
        random = np.random.Generator(np.random.PCG64(0))
        draws = np.zeros(2000)
        for month in MONTHS:
            group = day[day.month.eq(month)]
            take = random.integers(0,len(group),size=(2000,len(group)))
            denominator = group.n.to_numpy()[take].sum(axis=1)
            a = group[reference].to_numpy()[take].sum(axis=1)
            b = group.target_consistency.to_numpy()[take].sum(axis=1)
            draws += (np.sqrt(b/denominator)-np.sqrt(a/denominator))/2
        interval = np.percentile(draws,[2.5,97.5])
        change = float(np.mean([scores[month]['target_consistency']-scores[month][reference] for month in MONTHS]))
        decision = 'improvement' if interval[1]<0 else 'regression' if interval[0]>0 else 'inconclusive'
        paired[reference] = dict(change=change,ci95=interval.tolist(),decision=decision)
        close(change,result['paired'][reference]['change'])
        close(interval,result['paired'][reference]['ci95'])
        assert decision==result['paired'][reference]['decision']
    for month in MONTHS:
        for col in columns:
            close(scores[month][col],result['scores'][month][col])
    for col in columns:
        close(cv[col],result['cv'][col])
    both = all(scores[month]['target_consistency']<scores[month]['v82'] for month in MONTHS)
    mean = cv['target_consistency']<cv['v82']
    upper = paired['v82']['ci95'][1]<0
    gate = dict(primary=True,mean_improves=mean,both_months=both,interval_below_zero=upper,passed=mean and both and upper)
    assert gate==result['gate']
    return dict(scores=scores,cv=cv,paired=paired,gate=gate)


def audit():
    started = time.monotonic()
    assert not (OUT/'independent_audit.json').exists()
    identity,result,receipts,links = provenance()  # All six states audited before any table decode.
    if result is None:
        value = dict(status='pass_closed_primary_gate_failed',identity=art(OUT/'identity.json'),
                     closed=art(OUT/'closed.json'),source=art(__file__),no_new_prediction_decode=True,**{
                         k:v for k,v in links.items() if k!='identity'})
        write(OUT/'independent_audit.json',value)
        print(json.dumps(dict(status=value['status'],no_new_prediction_decode=True)))
        return
    parent = read(PORTABLE/'identity.json')
    spec = parent['specs']['cv']
    baseline_result, baseline_audit = read(verify(identity['baseline_result'])), read(verify(identity['baseline_audit']))
    assert baseline_audit['status']=='pass' and baseline_audit['result']==identity['baseline_result']
    assert baseline_audit['identity']==baseline_result['identity']==art(PORTABLE/'identity.json')
    frame = pd.read_parquet(verify(result['scored']))
    ids = frame.mvt_id.to_numpy()
    assert len(ids)==len(set(ids))==COUNT and frame.mvt_id.notna().all()
    baseline = aligned(baseline_result['prediction'],ids,True).rename(columns={'submitted':'v82'})
    pd.testing.assert_frame_equal(frame[baseline.columns],baseline,check_exact=True)
    for field in FIELDS:
        np.testing.assert_array_equal(frame['control_'+field],baseline['v82' if field=='submitted' else field])
    for arm in ARMS:
        np.testing.assert_array_equal(frame[arm],frame[arm+'_submitted'])
    invalid = [col for col in frame if col.endswith('invalid_share')]
    assert np.isfinite(frame.drop(columns=invalid).select_dtypes(include='number')).all().all()
    packet = aligned(spec['direct'],ids,True)
    direct = direct_replay(frame,packet,receipts)
    d,m,i = [aligned(spec[key],ids) for key in ['raw','monthly','daily_inputs']]
    i = i.rename(columns={'local_takeoff':'local_day'})
    matches = pd.read_parquet(verify(spec['matches']))
    matches = matches[matches.accepted].merge(frame[['mvt_id','month']],on='mvt_id',validate='one_to_one')
    matches = matches[[row.donor_month in historical.FOLDS[row.month]+[row.month] for row in matches.itertuples()]]
    daily_cells = pd.read_parquet(verify(result['daily_cells']))
    assert set(daily_cells.arm)==set(ALL)
    params = read(verify(spec['quality_parameters']))
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,value in [('f',frame),('d',d),('m',m),('i',i),('w',matches)]:
            con.register(name,value)
        raw_rows = historical.raw_check(con,parent,spec,'cv',frame,d,m,i)
        source = next(record for record in parent['sources'] if Path(record['path']).name=='taxi_out_additional_time_2025.csv')
        con.register('monthly_csv',pd.read_csv(verify(source)))
        con.execute('''CREATE TEMP VIEW ms AS SELECT APT_ICAO AS airport,
            CAST(YEAR AS VARCHAR)||'-'||lpad(CAST(MONTH_NUM AS VARCHAR),2,'0') AS local_month,
            (TF-VALID_FL)/TF AS invalid FROM monthly_csv''')
        source = next(record for record in parent['sources'] if Path(record['path']).name=='daily_runway.parquet')
        con.read_parquet(str(verify(source))).create_view('daily_source')
        assert con.execute("SELECT bool_and(phase='DEP') FROM daily_source").fetchone()[0]
        con.execute('''CREATE TEMP VIEW ds AS SELECT airport,day AS local_day,runway,n AS source_n,
            60.*total_min/n AS source_mean FROM daily_source''')
        for name,keys in [('ms','airport,local_month'),('ds','airport,local_day,runway')]:
            assert con.execute(f'SELECT count(*)=count(distinct ({keys})) FROM {name}').fetchone()[0]
        suffix = suffix_checks(con,frame,daily_cells,params)
        summary = metrics(con,frame,result)
    baseline_cv = float(np.mean([baseline_audit['metrics']['months'][month]['submitted'] for month in MONTHS]))
    close(summary['cv']['v82'],baseline_cv,1e-8)
    value = dict(status='pass',identity=art(OUT/'identity.json'),source=art(__file__),result=art(OUT/'result.json'),
                 rows=COUNT,raw_rows=raw_rows,original_prefix_and_zero_control_all_columns_exact=True,
                 direct_replay=direct,suffix=suffix,**summary,primary_audit=links['primary_audit'],
                 no_model_or_posterior_inference=True,no_calibration=True,no_upload=True,
                 native_direct_scope='Pinned original final_prediction reused; seed pool independently reconstructed. Native correction not independently rederived.',
                 daily_scope='Independent SQL from original frozen source calibration and raw cells.',
                 reused_transductive_development=True,independent_confirmation=False,seconds=time.monotonic()-started)
    write(OUT/'independent_audit.json',value)
    print(json.dumps(dict(status='pass',cv=summary['cv'],gate=summary['gate'])))


if __name__=='__main__':
    with threadpool_limits(1):
        audit()
