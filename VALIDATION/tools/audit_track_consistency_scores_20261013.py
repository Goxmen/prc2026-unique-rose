"""Independent saved-result audit of the fixed target track-consistency screen.

No model or producer module is imported, no model inference is repeated, and no
training targets are opened until all six immutable fit receipts exist. Audits
raw Jan/Jul departure membership/truth/calendar, exact selected pool controls,
all three serialized v80 component transports, SQL RMSE and separately generated
seed0/2000 paired-day bootstraps. The inherited native correction algorithm is
authenticated by frozen sources and exact original controls, not reimplemented.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import hashlib
import json
import math
from pathlib import Path
import time
import duckdb
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'VALIDATION/runs'
TRAIN=RUNS/'v3_20261013_nn_public_tabm_track_consistency'
OUT=TRAIN/'comparison'
ORIGINAL=RUNS/'v3_20261012_nn_public_tabm_pool3_stack'
CURRENT=RUNS/'v3_20261012_tabm_pool3_other_runway_combo'
CURRENT_SUBMIT=RUNS/'v3_20261012_tabm_pool3_other_runway_combo_submit'
OLD_POOL_AUDIT=ROOT/'VALIDATION/reports/public_tabm_20261012/pool3_score_audit.json'
PROTOCOL=ROOT/'VALIDATION/reports/cohort_learning_20261013/target_track_consistency_protocol.json'
SCORER=ROOT/'VALIDATION/nn_public_tabm_track_consistency_score.py'
ARMS=['supervised_control','training_consistency','target_consistency']
REFERENCES=['v80','supervised_control','training_consistency']
MONTHS=['2025-01','2025-07']
RAW=[ROOT/'data/training_2025-01-01_2025-02-01.parquet',ROOT/'data/training_2025-07-01_2025-08-01.parquet']


def read(path):return json.loads(Path(path).read_text())
def art(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8<<20),b''):h.update(part)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())
def verify(record):
    path=ROOT/record['path'];assert art(path)=={k:record[k] for k in ['path','sha256']},record['path']
    if 'bytes' in record:assert path.stat().st_size==record['bytes']
    return path
def close(a,b,atol=1e-8):np.testing.assert_allclose(a,b,atol=atol,rtol=0)


def transport_sql(arm):
    assert arm in ARMS
    return f'''SELECT count(*),count(distinct f.mvt_id),count(*) FILTER(WHERE
        f.mvt_id IS NULL OR o.mvt_id IS NULL OR c.mvt_id IS NULL OR d.mvt_id IS NULL OR r.mvt_id IS NULL
        OR f.y IS DISTINCT FROM r.y OR f.month IS DISTINCT FROM r.month OR f.day IS DISTINCT FROM r.day
        OR f.y IS DISTINCT FROM o.y OR f.month IS DISTINCT FROM o.month OR f.day IS DISTINCT FROM o.day
        OR f.y IS DISTINCT FROM c.y OR f.month IS DISTINCT FROM c.month OR f.day IS DISTINCT FROM c.day
        OR f.y IS DISTINCT FROM d.y OR f.month IS DISTINCT FROM d.month OR f.day IS DISTINCT FROM d.day
        OR d.selected IS DISTINCT FROM o.selected OR d.one_tabm IS DISTINCT FROM o.one_tabm
        OR f.v80 IS DISTINCT FROM c.three_tabm_runway
        OR f.{arm}_component_delta IS DISTINCT FROM CAST(d.three_tabm AS DOUBLE)-CAST(o.three_tabm AS DOUBLE)
        OR f.{arm} IS DISTINCT FROM round_even(greatest(0.,CAST(c.three_tabm_runway AS DOUBLE)
            +CAST(d.three_tabm AS DOUBLE)-CAST(o.three_tabm AS DOUBLE)),0)
        OR (d.three_tabm=o.three_tabm AND f.{arm} IS DISTINCT FROM f.v80))
        FROM f FULL JOIN o USING(mvt_id) FULL JOIN c USING(mvt_id)
        FULL JOIN d USING(mvt_id) FULL JOIN r USING(mvt_id)'''


def metrics_sql(columns):
    return 'SELECT month,'+','.join(f'sqrt(avg(pow(CAST({c} AS DOUBLE)-y,2))) AS {c}' for c in columns)+' FROM f GROUP BY month ORDER BY month'


def days_sql():
    terms=','.join(f'sum(pow(CAST({c} AS DOUBLE)-y,2)) AS {c}' for c in ['v80',*ARMS])
    return f'''SELECT month,day,count(*) AS n,{terms},min(ordinal) AS first_position
        FROM (SELECT *,row_number() OVER() AS ordinal FROM f)
        GROUP BY month,day ORDER BY month,first_position'''


def bootstrap(day,metrics):
    paired={}
    for reference in REFERENCES:
        random=np.random.Generator(np.random.PCG64(0));draws=np.zeros(2000)
        for month in MONTHS:
            part=day[day.month.eq(month)];n=part.n.to_numpy()
            a=part[reference].to_numpy();b=part.target_consistency.to_numpy()
            take=random.integers(0,len(part),size=(2000,len(part)))
            denom=n[take].sum(axis=1)
            draws+=(np.sqrt(b[take].sum(axis=1)/denom)-np.sqrt(a[take].sum(axis=1)/denom))/2
        ci=np.percentile(draws,[2.5,97.5]);change=float(np.mean([metrics[m]['target_consistency']-metrics[m][reference] for m in MONTHS]))
        paired[reference]=dict(change=change,ci95=ci.tolist(),decision='improvement' if ci[1]<0 else 'regression' if ci[0]>0 else 'inconclusive')
    return paired


def gate(metrics,paired):
    both={r:all(metrics[m]['target_consistency']<metrics[m][r] for m in MONTHS) for r in REFERENCES}
    significant={r:paired[r]['ci95'][1]<0 for r in REFERENCES}
    gain=np.mean([metrics[m]['v80']-metrics[m]['target_consistency'] for m in MONTHS])>=.1
    return dict(both_months=both,intervals_below_zero=significant,minimum_gain=bool(gain),
                passed=all(both.values()) and all(significant.values()) and bool(gain))


def checkpoint_check(record,teacher,steps):
    import torch
    torch.set_num_threads(1)
    saved=torch.load(verify(record['model']),map_location='cpu',weights_only=False)
    original=torch.load(verify(teacher['model']),map_location='cpu',weights_only=False)
    for key in ['identity','design','month','arm','curve']:assert saved[key]==record[key]
    states=saved['state_dict'];reference=original['state_dict']
    assert states.keys()==reference.keys()
    for key,value in states.items():
        assert value.shape==reference[key].shape and value.dtype==reference[key].dtype
        assert torch.isfinite(value).all() and torch.isfinite(reference[key]).all()
    terminal=saved['terminal_training_state'];assert terminal['updates']==2*steps
    def finite(value):
        if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
        elif isinstance(value,dict):
            for child in value.values():finite(child)
        elif isinstance(value,(list,tuple)):
            for child in value:finite(child)
        elif isinstance(value,(float,np.floating)):assert np.isfinite(value)
    finite(terminal)
    opt=terminal['optimizer'];assert len(opt['param_groups'])==1
    group=opt['param_groups'][0]
    for key,value in dict(lr=3e-5,weight_decay=.0003,betas=(.9,.999),eps=1e-8,amsgrad=False).items():assert group[key]==value
    assert set(opt['state'])==set(group['params']) and 0<len(group['params'])<=len(states)
    for state in opt['state'].values():assert state['step'].item()==2*steps
    sampler=terminal['sampler']
    if record['arm']=='supervised_control':assert sampler is None
    else:
        pool='train' if record['arm']=='training_consistency' else 'held'
        n=teacher['eligible'][pool];draws=2*steps*512;cycles=math.ceil(draws/n)
        assert sampler['draws']==draws and sampler['cycles']==cycles and sampler['cursor']==(draws-1)%n+1
        order=np.asarray(sampler['order']);assert len(order)==len(np.unique(order))==n
        assert (order>=0).all() and (order<record[pool+'_rows']).all()
        eligible=np.sort(order);random=np.random.Generator(np.random.PCG64(20261013))
        for _ in range(cycles):expected=random.permutation(eligible)
        np.testing.assert_array_equal(order,expected);assert sampler['rng']==random.bit_generator.state
    return dict(state_tensors=len(states),optimizer_parameters=len(group['params']),all_tensors_and_optimizer_finite=True,teacher_parameter_schema_exact=True,
                optimizer_updates=2*steps,sampler_counts_and_PCG64_terminal_exact=True,CPU_only_no_inference=True)


def fit_receipts():
    identity=read(TRAIN/'identity.json');fit=read(TRAIN/'audit.json')
    assert fit['status']=='pass' and fit['identity']==art(TRAIN/'identity.json')
    assert fit['all_held_labels_nan'] and fit['all_held_reload_exact']
    assert identity['design']['arms']==ARMS and identity['design']['epochs']==2
    assert identity['design']['no_hpo'] and identity['design']['no_stopping']
    assert identity['protocol']==art(PROTOCOL)
    for record in [identity['source'],identity['parent_identity'],identity['auditor'],identity['source_review']]:verify(record)
    for key in ['preflight','prefit_audit','allocation']:
        assert fit[key]==art(TRAIN/(key+'.json'));verify(fit[key])
    pref,prefit=read(TRAIN/'preflight.json'),read(TRAIN/'prefit_audit.json')
    assert pref['status']==prefit['status']=='pass' and pref['identity']==prefit['identity']==fit['identity']
    assert prefit['preflight']==fit['preflight'] and prefit['teachers']==pref['teachers']
    members=[];seen=set();model_paths=set();predictions={};checkpoints=[]
    for month in MONTHS:
        for arm in ARMS:
            path=TRAIN/arm/'networks'/f'{month}_s42.json';record=read(path);members.append(art(path))
            assert record['identity']==fit['identity'] and record['design']==identity['design']
            assert record['month']==month and record['arm']==arm and record['seed']==42 and record['epochs']==2
            assert record['inputs']==identity['folds'][month] and record['preprocessing']==identity['folds'][month]['preprocessing']
            assert record['all_held_reloaded_exact'] and (month,arm) not in seen;seen.add((month,arm))
            for key in ['model','prediction','preprocessing','checkpoint','teacher']:verify(record[key])
            assert record['model']['path'] not in model_paths;model_paths.add(record['model']['path'])
            assert record['prediction']['path']==str(path.with_suffix('.parquet').relative_to(ROOT))
            trained=read(verify(record['checkpoint']))
            for key in ['identity','month','arm','model','teacher']:assert trained[key]==record[key]
            assert trained['updates_completed_before_inference']
            for key in ['preflight','prefit_audit','allocation']:assert trained[key]==fit[key]
            teacher=read(verify(record['teacher']))
            assert teacher['identity']==fit['identity'] and teacher['month']==month
            assert teacher['preprocessing']==record['preprocessing']
            for key in ['train_mean','held_mean','model','control_prediction']:verify(teacher[key])
            assert record['scope']==teacher['scope']
            scope=record['scope'];assert scope['held_months']==[month] and scope['held_labels_absent_from_cache']
            assert scope['adapter_targets_all_nan'] and scope['held_metadata_columns']==['mvt_id','month']
            assert record['train_rows']==scope['train_rows'] and record['held_rows']==scope['held_rows']
            assert [c['epoch'] for c in record['curve']]==[1,2]
            steps=math.ceil(record['train_rows']/2048)
            for curve in record['curve']:
                assert curve['updates']==steps and np.isfinite(curve['train_mse_std'])
                assert curve['consistency_rows']==(0 if arm=='supervised_control' else steps*512)
                if arm=='supervised_control':assert curve['consistency_mse_std'] is None
                else:assert np.isfinite(curve['consistency_mse_std'])
            checkpoints.append(dict(month=month,arm=arm,**checkpoint_check(record,teacher,steps)))
            predictions[(month,arm)]=(record['prediction'],record['held_rows'])
    assert len(members)==len(fit['members'])==6 and sorted(members,key=lambda x:x['path'])==sorted(fit['members'],key=lambda x:x['path'])
    return members,predictions,checkpoints


def audit():
    start=time.monotonic();assert not (OUT/'independent_audit.json').exists()
    identity,result=read(OUT/'identity.json'),read(OUT/'result.json')
    assert identity['auditor']==art(Path(__file__)) and identity['protocol']==art(PROTOCOL) and identity['before_fitting']
    assert result['identity']==art(OUT/'identity.json') and result['fit_audit']==art(TRAIN/'audit.json')
    pinned={r['path']:r for r in identity['sources']}
    for record in identity['sources']:verify(record)
    for path in [SCORER,Path(__file__),PROTOCOL,*RAW,OLD_POOL_AUDIT,CURRENT/'identity.json',CURRENT_SUBMIT/'independent_audit.json']:
        assert pinned[str(path.relative_to(ROOT))]==art(path)
    design=identity['design']
    assert design['arms']==ARMS and design['selected_arm']=='target_consistency'
    assert design['comparisons']==REFERENCES and design['minimum_cv_gain_seconds']==.1
    assert design['no_control_promotion'] and design['no_weight_search'] and design['no_upload']
    assert result['exact_v80_replay'] and result['all_row_component_transport_SQL_exact']
    assert result['no_control_promotion'] and result['no_production_or_upload']
    members,predictions,checkpoints=fit_receipts()  # Close the fit-receipt gate before raw truth is read.
    old_audit=read(OLD_POOL_AUDIT);assert old_audit['status']=='pass' and old_audit['rows']==344419
    for record in old_audit['inputs']:verify(record)
    assert art(ORIGINAL/'scored.parquet') in old_audit['inputs']
    current_audit=read(CURRENT_SUBMIT/'independent_audit.json');assert current_audit['status']=='pass' and current_audit['cv_rows']==344419
    current_identity=read(verify(current_audit['identity']));current_result=read(verify(current_audit['result']))
    assert current_identity['cv']==current_result['cv_source']==art(CURRENT/'result.json')
    assert read(CURRENT/'result.json')['identity']==art(CURRENT/'identity.json')
    for folder in [ORIGINAL,CURRENT]:
        saved=read(folder/'result.json');assert saved['scored']==art(folder/'scored.parquet')
        for record in saved['sources']:verify(record)
    metrics={};components={}
    with duckdb.connect(config={'threads':1,'memory_limit':'768MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,path in [('f',verify(result['scored'])),('o',ORIGINAL/'scored.parquet'),('c',CURRENT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
            assert con.execute(f'SELECT count(*),count(distinct mvt_id) FROM {name}').fetchone()==(344419,344419)
        con.read_parquet([str(p) for p in RAW]).create_view('all_raw')
        con.execute('''CREATE TEMP VIEW r AS SELECT MVT_ID_mvt AS mvt_id,TAXITIME_SEC_mvt AS y,
            strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS month,strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS day
            FROM all_raw WHERE PHASE_mvt='DEP' ''')
        assert con.execute('SELECT count(*),count(distinct mvt_id) FROM r').fetchone()==(344419,344419)
        assert con.execute('SELECT count(*) FROM r WHERE mvt_id IS NULL OR y IS NULL OR NOT isfinite(y) OR month IS NULL OR day IS NULL').fetchone()[0]==0
        for arm in ARMS:
            folder=OUT/arm/'component';comp=read(verify(result['components'][arm]));checked=read(folder/'audit.json')
            assert result['components'][arm]==art(folder/'result.json') and checked['status']=='pass' and checked['result']==art(folder/'result.json')
            assert comp['exact_v78_replayed'] and comp['exact_one_tabm_replayed'] and comp['design']==design
            assert comp['scored']==art(folder/'scored.parquet')
            for record in comp['sources']:verify(record)
            for month in MONTHS:
                assert predictions[(month,arm)][0] in comp['sources']
                # Retained seed1042/2042 prediction identities remain the original pool's.
                for k in [1,2]:
                    control=read(TRAIN/'identity.json')['controls'][month][k]
                    assert read(verify(control))['prediction'] in comp['sources']
            con.read_parquet(str(folder/'scored.parquet')).create_view('d',replace=True)
            assert con.execute('SELECT count(*),count(distinct mvt_id) FROM d').fetchone()==(344419,344419)
            assert con.execute(transport_sql(arm)).fetchone()==(344419,344419,0)
            for table,columns in [('f',['v80',arm]),('d',['selected','one_tabm','three_tabm'])]:
                terms=' OR '.join(f'{c} IS NULL OR NOT isfinite({c}) OR {c}<0 OR {c} IS DISTINCT FROM round_even({c},0)' for c in columns)
                assert con.execute(f'SELECT count(*) FROM {table} WHERE {terms}').fetchone()[0]==0
            components[arm]=dict(result=art(folder/'result.json'),audit=art(folder/'audit.json'),prediction=comp['scored'])
        for (month,arm),(record,n) in predictions.items():
            con.read_parquet(str(verify(record))).create_view('p',replace=True)
            assert con.execute('SELECT count(*),count(distinct mvt_id) FROM p').fetchone()==(n,n)
            expected=con.execute('SELECT count(*) FROM r WHERE month=?',[month]).fetchone()[0];assert expected==n
            assert con.execute('''SELECT count(*),count(*) FILTER(WHERE p.mvt_id IS NULL OR r.mvt_id IS NULL
                OR p.base IS NULL OR NOT isfinite(p.base) OR p.base<0)
                FROM p FULL JOIN (SELECT * FROM r WHERE month=?) r USING(mvt_id)''',[month]).fetchone()==(n,0)
        for row in con.execute(metrics_sql(['v80',*ARMS])).fetchall():metrics[row[0]]=dict(zip(['v80',*ARMS],row[1:]))
        assert sorted(metrics)==MONTHS
        day=con.execute(days_sql()).fetchdf()
    cv={c:float(np.mean([metrics[m][c] for m in MONTHS])) for c in ['v80',*ARMS]}
    for month in MONTHS:
        for col in ['v80',*ARMS]:close(metrics[month][col],result['scores'][month][col])
    for col in cv:close(cv[col],result['cv'][col])
    close(cv['v80'],289.7627782102906);close(cv['v80'],current_audit['cv']['three_tabm_runway'])
    paired=bootstrap(day,metrics)
    for ref in REFERENCES:
        close(paired[ref]['change'],result['paired'][ref]['change']);close(paired[ref]['ci95'],result['paired'][ref]['ci95'])
        assert paired[ref]['decision']==result['paired'][ref]['decision']
    expected_gate=gate(metrics,paired);assert expected_gate==result['gate']
    report=dict(status='pass',source=art(Path(__file__)),identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),
        prediction=result['scored'],fit_audit=result['fit_audit'],members=members,checkpoints=checkpoints,components=components,
        rows=344419,raw_truth_files=[art(p) for p in RAW],all_raw_departures_ID_truth_UTC_calendar_exact=True,
        all_three_v80_component_transports_SQL_exact=True,original_selected_and_one_TabM_controls_exact=True,
        scores=metrics,cv=cv,independent_paired=paired,gate=expected_gate,bootstrap=dict(seed=0,reps=2000,unit='whole days within month',day_order='first occurrence in saved output'),
        no_control_promotion=True,no_threshold_or_weight_search=True,no_model_inference=True,checkpoint_loads_CPU_only=True,no_upload=True,
        reused_transductive_development=True,independent_confirmation=False,
        scope='Independent saved-row arithmetic and receipt audit. Inherited native correction algorithm covered by frozen source and exact controls, not independently reimplemented.',
        seconds=time.monotonic()-start)
    with (OUT/'independent_audit.json').open('x') as stream:stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass',audit=art(OUT/'independent_audit.json'),cv=cv,gate=expected_gate)))


if __name__=='__main__':
    with threadpool_limits(1):audit()
