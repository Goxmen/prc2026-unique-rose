"""Independent completed-screen audit; never fits, infers, selects or uploads.

The original laptop receipts establish full GPU prediction reloads. This audit
checks immutable provenance, all checkpoint tensors/curves, exact architecture
additions, every raw held label/ID/calendar, native component transport, SQL
scores and independently reconstructed paired-day uncertainty.
"""
from pathlib import Path
import hashlib
import json
import math

import duckdb
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_daily_cohort'
STACK=RUN/'comparison'
REPORT=ROOT/'VALIDATION/reports/cohort_learning_20261012'
OLD=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_stack'
CURRENT=ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo'
FOLDS={
    '2025-01':[f'2025-{m:02d}' for m in range(3,12)],
    '2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12'],
}
EPOCHS={'2025-01':21,'2025-07':17}
ROWS={'2025-01':153706,'2025-07':190713}
EXPECTED={
    'VALIDATION/nn_public_tabm_daily_cohort.py':'b646a9fefec584b3e1526ea8d11be1b3241922869e165043a34c9a5270ab5b83',
    'VALIDATION/nn_public_tabm_daily_cohort_model.py':'fef74a36c5e38d3b31fa794c73d0a5b316397823ef7e28f8f52526473a1b4092',
    'VALIDATION/daily_cohort_bank.py':'41fa243dfff8855797159d622feb2ac5ef85894ff8cdb8e7d5d69402c4239b1a',
    'VALIDATION/nn_public_tabm_daily_cohort_score.py':'b00d5b971cb89bf26e45917f216399aa621147f498e42aff101aa08baebc40f6',
    'VALIDATION/runs/v3_20261012_nn_public_tabm_daily_cohort/identity.json':'ae492d62698617a8c735713cdbfb90855d7f0b2d505a9549fa58e2781c9482aa',
    'VALIDATION/runs/v3_20261012_nn_public_tabm_daily_cohort/comparison/identity.json':'6b1b92b71fffbcb243aab0af8616f015d3adae7401634055c07ea260e263b824',
    'VALIDATION/reports/cohort_learning_20261012/independent_pilot_review.json':'6188170d43544d948a7fd44fd46776025e1f332844321cdfd08b449dc30d2ace',
    'VALIDATION/reports/cohort_learning_20261012/independent_bank_audit.json':'0bd0b6b7677dd6ed1be624b227702eba9b1feba7f175dd5de2f95ab799fcf4ba',
    'VALIDATION/reports/cohort_learning_20261012/independent_context_model_audit.json':'0da11b6d0d9526c980812fc725eeff26ac5a1d06cc06d922c604a05a447c8000',
    'VALIDATION/reports/cohort_learning_20261012/cuda_repeated_control_diagnostic.json':'cf48dff162b05d884be6f7441e3f8a3a967659333279c2b353c224380453d0ba',
}
VERIFIED={}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def art(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path))


def verify(record):
    path=(ROOT/record['path']).resolve();assert path.is_relative_to(ROOT)
    key=str(path.relative_to(ROOT));expected=record['sha256']
    if key in VERIFIED:assert VERIFIED[key]==expected,key
    else:
        assert sha(path)==expected,key
        VERIFIED[key]=expected
    if 'bytes' in record:assert path.stat().st_size==record['bytes']
    return path


def checks():
    """Verify prefit proofs without rerunning bank construction or inference."""
    for path,digest in EXPECTED.items():verify(dict(path=path,sha256=digest))
    identity=read(RUN/'identity.json');frozen=read(STACK/'identity.json')
    assert frozen['before_fitting']
    assert identity['design']['epochs']==EPOCHS and identity['design']['seed']==42
    assert identity['design']['effective_batch']==2048 and identity['design']['microbatch']==512
    assert identity['design']['inference_chunk']==512 and identity['design']['retained_pool_members']==[1042,2042]
    assert identity['design']['no_loss_change'] and identity['design']['no_parameter_scan']
    for a in identity['files']+frozen['sources']:verify(a)
    bank=read(verify(identity['donor_bank']))
    assert bank['no_labels_loaded'] and bank['rows']==2085047
    assert bank['outputs']==identity['donor_bank_outputs']
    for a in bank['sources']+bank['outputs']:verify(a)
    cpu=read(RUN/'cpu_checks.json')
    assert cpu==identity['checks'] and cpu['training_AST_restores_exactly']
    for key in ['source','model_source','bank_source']:verify(cpu[key])
    needed=['initialization_exact','original_forward_exact','original_gradient_exact',
        'original_dropout_and_constructor_RNG_exact','first_original_AdamW_update_exact',
        'active_donor_gradients_finite_nonzero','query_alias_and_padding_invariant','empty_branch_exact_zero']
    assert all(cpu['model'][k] for k in needed)
    preflight=read(RUN/'preflight.json')
    assert preflight['identity']==art(RUN/'identity.json') and preflight['training_ast_exact']
    assert {x['device'] for x in preflight['checks']}=={'cpu','cuda'}
    assert all(x[k] for x in preflight['checks'] for k in needed)
    backend=preflight['parity_backend']
    assert backend['scope']=='model_type.check CPU/CUDA only'
    assert backend['deterministic'] and not backend['benchmark']
    assert backend['restored'] and backend['original_flags']==backend['restored_flags']
    assert backend['capacity_and_fitting_unchanged']
    diagnostic=read(REPORT/'cuda_repeated_control_diagnostic.json')
    assert diagnostic['no_labels'] and diagnostic['no_fit'] and diagnostic['backend_flags_restored']
    for record in [diagnostic['source'],diagnostic['model'],diagnostic['old_model']]:verify(record)
    for case in diagnostic['modes']['scoped_deterministic_cudnn']['cases']:
        assert case['constructor_rng_exact'] and case['original_parameters_initial_exact']
        for control in case['comparisons'].values():
            assert control['rng_exact'] and control['output']['mismatches']==0
            assert control['original_path_output_after_step']['mismatches']==0
            for field in ['gradients','parameters_after_step','optimizer_updates']:
                assert all(x['mismatches']==0 for x in control[field].values())
    assert {x['month'] for x in preflight['capacity']}==set(EPOCHS)
    for x in preflight['capacity']:
        assert 0<x['peak_allocated_bytes']<7*(1<<30)
        assert x['donor_groups']==x['training_and_inference_width']==512
        assert x['donors_each']==bank['dimensions']['values'][1]
        assert x['scalar_bank_reserved']>0 and x['original_matrix_bytes_reserved']>0
        assert x['original_track_geometry_quality_reserved']>0
    pilot=read(REPORT/'independent_pilot_review.json')
    ba=read(REPORT/'independent_bank_audit.json')
    ma=read(REPORT/'independent_context_model_audit.json')
    assert pilot['status']=='pass_input_information_only' and ba['status']=='pass'
    assert ba['all_17_scaled_values_exact'] and ba['all_aliases_independent_SQL']
    assert ma['status']=='pass_context_math_only' and ma['source_sha256']==cpu['model_source']['sha256']
    assert ma['excluded_and_padding_gradient_cancellation_fp32_bound']
    return identity,frozen


def checkpoint_checks(identity):
    fit=read(RUN/'audit.json');returned=read(RUN/'return_audit.json')
    assert fit['status']==returned['status']=='pass'
    assert fit['identity']==returned['identity']==art(RUN/'identity.json')
    assert returned['execution']==identity['execution']
    assert fit['retained_controls_not_refitted'] and fit['all_held_reload_exact']
    for a in returned['files']+fit['members']:verify(a)
    returned_paths={a['path'] for a in returned['files']}
    assert str((RUN/'preflight.json').relative_to(ROOT)) in returned_paths
    members=[]
    for month,epochs in EPOCHS.items():
        path=RUN/f'k0/networks/{month}_tabm.json';record=read(path)
        assert record['identity']==art(RUN/'identity.json') and record['design']==identity['design']
        assert (record['seed'],record['member'],record['month'],record['epochs'])==(42,0,month,epochs)
        assert record['all_held_reloaded_exact'] and record['inputs']==identity['folds'][month]
        assert record['held_rows']==ROWS[month] and record['train_rows']==identity['folds'][month]['rows']
        assert record['preprocessing']==identity['folds'][month]['preprocessing']
        assert {k:record[k] for k in ['host','gpu','torch']}==identity['execution']
        for key in ['model','prediction','preprocessing','checkpoint']:verify(record[key])
        for key in ['model','prediction','checkpoint']:
            assert record[key]['path'] in returned_paths
        assert str(path.relative_to(ROOT)) in returned_paths
        checkpoint_receipt=read(verify(record['checkpoint']))
        assert checkpoint_receipt['identity']==art(RUN/'identity.json')
        assert checkpoint_receipt['design']==identity['design']
        assert checkpoint_receipt['updates_completed_before_inference']
        assert checkpoint_receipt['model']==record['model']
        assert checkpoint_receipt['inputs']==record['inputs']
        replay_path=RUN/f'control_replay/{month}.json';replay=read(replay_path)
        assert str(replay_path.relative_to(ROOT)) in returned_paths
        assert replay['identity']==art(RUN/'identity.json') and replay['rows']==ROWS[month]
        assert replay['all_held_old_and_zero_branch_exact'] and replay['original_random_state_restored']
        assert replay['control_record']==identity['controls'][month][0]
        verify(replay['control_record'])
        assert replay['scope']['train']['rows']==identity['folds'][month]['rows']
        assert replay['scope']['held']['rows']==ROWS[month]
        assert replay['scope']['train']['months']==sorted(FOLDS[month])
        assert replay['scope']['held']['months']==[month]
        assert all(replay['scope'][k]['same_UTC_day_donors'] for k in ['train','held'])
        model=torch.load(verify(record['model']),map_location='cpu',weights_only=False)
        assert model['identity']==art(RUN/'identity.json') and model['design']==identity['design']
        assert model['curve']==record['curve']
        assert [x['epoch'] for x in model['curve']]==list(range(1,epochs+1))
        assert all(math.isfinite(x['train_mse_std']) and x['train_mse_std']>=0
            and math.isfinite(x['seconds']) and x['seconds']>0 for x in model['curve'])
        old=torch.load(verify(identity['reused'][month]['model']),map_location='meta',weights_only=False)
        new_state,old_state=model['state_dict'],old['state_dict']
        assert set(old_state).issubset(new_state)
        expected={'donor.0.weight','donor.0.bias','donor.2.weight','donor.2.bias',
            'context_projection.weight','context_projection.bias'}
        assert set(new_state)-set(old_state)==expected
        assert all(new_state[k].shape==old_state[k].shape and new_state[k].dtype==old_state[k].dtype for k in old_state)
        joined=old_state['backbone.blocks.0.0.weight'].shape[-1]
        shapes={'donor.0.weight':(32,17),'donor.0.bias':(32,),
            'donor.2.weight':(32,32),'donor.2.bias':(32,),
            'context_projection.weight':(joined,65),'context_projection.bias':(joined,)}
        assert all(tuple(new_state[k].shape)==v for k,v in shapes.items())
        assert all(torch.isfinite(t).all() for t in new_state.values())
        assert new_state['context_projection.weight'].abs().sum()>0
        members.append(dict(record=art(path),control_replay=art(replay_path),
            tensors=len(new_state),new_parameters=sum(new_state[k].numel() for k in expected),
            complete_epochs=epochs,all_held_GPU_reload_receipt=True))
        del model,old,new_state,old_state
    return members


def statistical_checks(result):
    component=read(STACK/'component/result.json');ca=read(STACK/'component/audit.json')
    assert ca['status']=='pass' and ca['result']==art(STACK/'component/result.json')
    assert component['exact_v78_replayed'] and component['exact_one_tabm_replayed']
    for a in component['sources']+[component['scored'],result['scored']]:verify(a)
    raw_files=sorted((ROOT/'data').glob('training_2025-0[17]-01_*.parquet'))
    assert len(raw_files)==2
    rng=np.random.default_rng(0);draws=np.zeros(2000);scores={};raw_artifacts=[]
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB','preserve_insertion_order':True}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,path in [('f',STACK/'scored.parquet'),('d',STACK/'component/scored.parquet'),
                          ('o',OLD/'scored.parquet'),('c',CURRENT/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
            assert con.execute(f'select count(*),count(distinct mvt_id) from {name}').fetchone()==(344419,344419)
        con.read_parquet([str(p) for p in raw_files]).create_view('raw')
        con.execute('''create temp view truth as select MVT_ID_mvt as mvt_id,TAXITIME_SEC_mvt as y,
            strftime(MVT_TIME_UTC_mvt,'%Y-%m') as month,strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') as day
            from raw where PHASE_mvt='DEP' ''')
        assert con.execute('select count(*),count(distinct mvt_id) from truth').fetchone()==(344419,344419)
        assert con.execute('''select count(*) from f full join d using(mvt_id) full join o using(mvt_id)
            full join c using(mvt_id) full join truth t using(mvt_id)
            where f.mvt_id is null or d.mvt_id is null or o.mvt_id is null or c.mvt_id is null or t.mvt_id is null
            or f.y is distinct from t.y or d.y is distinct from t.y or o.y is distinct from t.y or c.y is distinct from t.y
            or f.month is distinct from t.month or d.month is distinct from t.month or o.month is distinct from t.month or c.month is distinct from t.month
            or cast(f.day as varchar) is distinct from t.day or cast(d.day as varchar) is distinct from t.day
            or cast(o.day as varchar) is distinct from t.day or cast(c.day as varchar) is distinct from t.day
            or d.selected is distinct from o.selected or d.one_tabm is distinct from o.one_tabm
            or f.control is distinct from c.three_tabm_runway
            or f.component_delta is distinct from d.three_tabm-o.three_tabm
            or f.candidate is distinct from round_even(greatest(0,c.three_tabm_runway+(d.three_tabm-o.three_tabm)),0)
            or not isfinite(f.y) or not isfinite(f.control) or not isfinite(f.candidate) or not isfinite(f.component_delta)
            or f.candidate<0 or f.candidate<>round_even(f.candidate,0)''').fetchone()[0]==0
        for month in sorted(EPOCHS):
            member=RUN/f'k0/networks/{month}_tabm.parquet'
            con.read_parquet(str(member)).create_view('member',replace=True)
            assert con.execute('select count(*),count(distinct mvt_id) from member').fetchone()==(ROWS[month],ROWS[month])
            assert con.execute('''select count(*) from member full join (select * from truth where month=?) t using(mvt_id)
                where member.mvt_id is null or t.mvt_id is null or member.base is null or not isfinite(member.base) or member.base<0''',[month]).fetchone()[0]==0
            actual=con.execute('''select count(*),sqrt(avg(pow(control-y,2))),sqrt(avg(pow(candidate-y,2)))
                from f where month=?''',[month]).fetchone()
            assert actual[0]==ROWS[month]
            scores[month]=dict(zip(['control','candidate'],actual[1:]))
            np.testing.assert_allclose(actual[1:],[result['scores'][month][k] for k in ['control','candidate']],atol=1e-8,rtol=0)
            for name,view in [('original', 'o'),('new','d')]:
                check=component if name=='new' else read(OLD/'result.json')
                for arm in ['selected','one_tabm','three_tabm']:
                    score=con.execute(f'select sqrt(avg(pow({arm}-y,2))) from {view} where month=?',[month]).fetchone()[0]
                    np.testing.assert_allclose(score,check['scores'][month][arm],atol=1e-8,rtol=0)
            # SQL forms day sufficient statistics independently. Preserve each
            # month's first-appearance day order, matching factorize's contract.
            days=np.asarray(con.execute('''select count(*),sum(pow(control-y,2)),sum(pow(candidate-y,2))
                from (select *,row_number() over() as pos from f) where month=?
                group by day order by min(pos)''',[month]).fetchall())
            assert len(days)==31
            chosen=rng.integers(0,len(days),(2000,len(days)))
            totals=days[chosen].sum(1)
            draws+=(np.sqrt(totals[:,2]/totals[:,0])-np.sqrt(totals[:,1]/totals[:,0]))/2.
        changed=con.execute('select count(*) from f where control!=candidate').fetchone()[0]
    cv={k:float(np.mean([s[k] for s in scores.values()])) for k in ['control','candidate']}
    for k in cv:np.testing.assert_allclose(cv[k],result['cv'][k],atol=1e-8,rtol=0)
    np.testing.assert_allclose(cv['control'],289.7627782102906,atol=1e-8,rtol=0)
    ci=np.percentile(draws,[2.5,97.5]);change=cv['candidate']-cv['control']
    np.testing.assert_allclose(ci,result['paired']['ci95'],atol=1e-8,rtol=0)
    np.testing.assert_allclose(change,result['paired']['change'],atol=1e-8,rtol=0)
    decision='improvement' if ci[1]<0 else 'regression' if ci[0]>0 else 'inconclusive'
    assert decision==result['paired']['decision']
    gate=bool(ci[1]<0 and all(s['candidate']<s['control'] for s in scores.values()))
    assert gate==result['gate']
    for p in raw_files:raw_artifacts.append(art(p))
    return dict(rows=344419,changed=changed,scores=scores,cv=cv,ci95=ci.tolist(),decision=decision,
        gate=gate,raw_truth_sources=raw_artifacts,independent_SQL_all_ID_truth_calendar_control_transport_scores=True,
        independent_paired_day_bootstrap=True)


def main():
    receipt=STACK/'independent_audit.json'
    assert not receipt.exists(),'Preserve the completed independent audit'
    required=[STACK/'result.json',STACK/'component/result.json',RUN/'audit.json',RUN/'return_audit.json']
    assert all(p.exists() for p in required),'Run only after fits, verified return and scoring are complete'
    identity,frozen=checks()
    result=read(STACK/'result.json')
    assert result['identity']==art(STACK/'identity.json') and frozen['before_fitting']
    members=checkpoint_checks(identity)
    statistics=statistical_checks(result)
    value=dict(status='pass',source=art(Path(__file__)),identity=art(RUN/'identity.json'),
        result=art(STACK/'result.json'),members=members,sources_verified=len(VERIFIED),
        exactly_expected_six_parameter_additions=True,all_original_parameter_shapes_unchanged=True,
        all_checkpoint_tensors_finite_and_epoch_curves_complete=True,
        immutable_prefit_bank_context_and_CPU_CUDA_proofs=True,
        all_held_original_and_zero_branch_GPU_control_replay=True,
        GPU_inference_scope='Full held-row checkpoint reloads verified by immutable original laptop fit receipts; no repeated inference',
        interpretation='Fixed screen on reused development folds; independent audit is not independent validation',
        no_fits_or_uploads=True,**statistics)
    receipt.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['status','rows','cv','ci95','decision','gate']}))


if __name__=='__main__':
    torch.set_num_threads(1)
    main()
