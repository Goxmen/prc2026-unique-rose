"""Independent saved-result SQL audit, without training or bootstrap reruns.

Run only after all continuation fits, compact return and both scorers finish.
Matrix validation is inherited from the laptop full-input receipts; no model
module is imported and no unavailable desktop training guard is bypassed.
"""
import json
import time

import duckdb
import audit_public_cohort_scores_20261012 as common

ROOT,RUNS=common.ROOT,common.RUNS
read,art,verify,metrics=common.read,common.artifact,common.verify,common.validate_metrics
TRAIN=RUNS/'v3_20261012_nn_public_tabm_pool3'
ORIGINAL=RUNS/'v3_20261012_nn_public_tabm'
FIRST=RUNS/'v3_20261012_nn_public_tabm_stack'
STACK=RUNS/'v3_20261012_nn_public_tabm_pool3_stack'
TRANSPORT=RUNS/'v3_20261012_tabm_pool3_delta_transport'
OUT=ROOT/'VALIDATION/reports/public_tabm_20261012/pool3_score_audit.json'
ARMS=['selected','one_tabm','three_tabm']


def main():
    started=time.monotonic();assert not OUT.exists(),'Preserve independent receipt'
    identity=read(TRAIN/'identity.json');original=read(ORIGINAL/'portable.json')
    fit=read(TRAIN/'audit.json');stack=read(STACK/'result.json');audit=read(STACK/'audit.json')
    frozen=read(TRANSPORT/'design.json');transport=read(TRANSPORT/'tabm_pool3/result.json')
    assert fit['status']==audit['status']=='pass'
    assert fit['identity']==art(TRAIN/'identity.json')
    assert identity['original_identity']==art(ORIGINAL/'portable.json')
    assert audit['result']==art(STACK/'result.json')
    assert identity['design']['seeds']==[42,1042,2042]
    assert identity['design']['new_seeds']==[1042,2042]
    assert identity['design']['reused']==42
    assert fit['original_seed42_reused_exactly'] and fit['full_input_hashes_verified']
    assert fit['original_preprocessing_exact'] and fit['all_held_reload_exact']
    assert stack['exact_v78_replayed'] and stack['exact_one_tabm_replayed']
    assert transport['exact_v78_control'] and transport['three_vs_one_predeclared']
    assert transport['identity']==art(TRANSPORT/'design.json') and transport['design']==frozen['design']
    assert frozen['design']['coefficient']==1 and frozen['design']['new_bounds'] is False
    assert frozen['design']['experiments']==dict(tabm_pool3=dict(directory=STACK.name,
        control='selected',arms=['one_tabm','three_tabm']))
    assert identity['selected_v78']==art(common.PARENT78/'result.json')
    for record in [identity['source'],*identity['files'],*stack['sources'],
                   *transport['sources'],*frozen['sources'],stack['scored'],transport['scored']]:verify(record)
    first=read(FIRST/'result.json');first_audit=read(FIRST/'audit.json')
    assert first_audit['status']=='pass' and first_audit['result']==art(FIRST/'result.json')
    assert first['scored']==art(FIRST/'scored.parquet')
    members=[]
    for month,epochs in [('2025-01',21),('2025-07',17)]:
        for k,seed in enumerate([42,1042,2042]):
            path=(ORIGINAL if k==0 else TRAIN/f'k{k}')/f'networks/{month}_tabm.json'
            record=read(path);members.append(art(path))
            assert record['month']==month and record['seed']==seed and record['epochs']==epochs
            assert [row['epoch'] for row in record['curve']]==list(range(1,epochs+1))
            assert record['inputs']==identity['folds'][month]
            assert record['preprocessing']==identity['folds'][month]['preprocessing']
            assert record['all_held_reloaded_exact']
            assert {key:record[key] for key in ['host','gpu','torch']}==identity['execution']
            if k==0:
                assert identity['reused'][month]['record']==art(path)
                assert record['identity']==art(ORIGINAL/'portable.json') and record['design']==original['design']
                for key in ['model','prediction','checkpoint']:
                    assert record[key]==identity['reused'][month][key]
            else:
                assert record['member']==k and record['identity']==art(TRAIN/'identity.json')
                assert record['design']==identity['design']
            for key in ['model','prediction','checkpoint','preprocessing']:verify(record[key])
            trained=read(ROOT/record['checkpoint']['path'])
            assert trained['model']==record['model'] and trained['preprocessing']==record['preprocessing']
            assert trained['identity']==record['identity'] and trained['design']==record['design']
            assert trained['updates_completed_before_inference']
    assert sorted(members,key=lambda x:x['path'])==sorted(fit['members'],key=lambda x:x['path'])
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        for view,path in [('f',STACK/'scored.parquet'),('old',common.PARENT78/'scored.parquet'),
            ('one',FIRST/'scored.parquet'),('p',common.PARENT79/'cv.parquet'),
            ('t',TRANSPORT/'tabm_pool3/scored.parquet')]:
            con.read_parquet(str(path)).create_view(view)
            n,u=con.execute(f'SELECT count(*),count(distinct mvt_id) FROM {view}').fetchone()
            assert n==u==344419
        formulas=' OR '.join(f't.{arm} IS DISTINCT FROM round_even(greatest(0,p.pred+f.{arm}-f.selected),0) '
            f'OR (f.{arm}=f.selected AND t.{arm} IS DISTINCT FROM p.pred)' for arm in ['one_tabm','three_tabm'])
        n,bad=con.execute(f'''SELECT count(*),sum(CASE WHEN
            f.y IS DISTINCT FROM old.y OR f.month IS DISTINCT FROM old.month OR f.day IS DISTINCT FROM old.day OR
            f.selected IS DISTINCT FROM old.candidate OR
            f.y IS DISTINCT FROM one.y OR f.month IS DISTINCT FROM one.month OR f.day IS DISTINCT FROM one.day OR
            f.one_tabm IS DISTINCT FROM one.replace_tabm OR
            f.y IS DISTINCT FROM p.y OR f.month IS DISTINCT FROM p.month OR f.day IS DISTINCT FROM p.day OR
            t.y IS DISTINCT FROM p.y OR t.month IS DISTINCT FROM p.month OR t.day IS DISTINCT FROM p.day OR
            t.control IS DISTINCT FROM p.pred OR {formulas} THEN 1 ELSE 0 END)
            FROM f JOIN old USING(mvt_id) JOIN one USING(mvt_id) JOIN p USING(mvt_id) JOIN t USING(mvt_id)''').fetchone()
        assert n==344419 and bad==0
        for view,columns in [('f',ARMS),('t',['control','one_tabm','three_tabm'])]:
            terms=' OR '.join(f'{c} IS NULL OR NOT isfinite({c}) OR {c}<0 OR {c}<>round_even({c},0)' for c in columns)
            assert con.execute(f'SELECT count(*) FROM {view} WHERE {terms}').fetchone()[0]==0
        primary=metrics(con,'f',stack,ARMS)
        secondary=metrics(con,'t',transport,['control','one_tabm','three_tabm'])
    cv,tcv=stack['cv'],transport['cv']
    effects=dict(v78_three_vs_one=cv['three_tabm']-cv['one_tabm'],v78_three_vs_original=cv['three_tabm']-cv['selected'],
        v79_three_vs_one=tcv['three_tabm']-tcv['one_tabm'],v79_three_vs_original=tcv['three_tabm']-tcv['control'])
    receipt=dict(status='pass',rows=n,seconds=time.monotonic()-started,source=art(__file__),helper=art(common.__file__),
        members=members,execution=identity['execution'],effects=effects,sql_scores=primary,sql_transport_scores=secondary,
        inputs=[art(p) for p in [TRAIN/'identity.json',TRAIN/'audit.json',ORIGINAL/'portable.json',
            FIRST/'result.json',FIRST/'audit.json',FIRST/'scored.parquet',STACK/'result.json',STACK/'audit.json',
            STACK/'scored.parquet',TRANSPORT/'design.json',TRANSPORT/'tabm_pool3/result.json',
            TRANSPORT/'tabm_pool3/scored.parquet',common.PARENT78/'scored.parquet',common.PARENT79/'cv.parquet']],
        full_row_truth_calendar_ids_v78_and_one_tabm_exact=True,all_row_transport_and_unchanged_recovery_exact=True,
        input_scope='Full matrices, semantic preprocessing and track cache verified by remote full-input fit receipts. '
            'This desktop audit verifies returned artifacts and arithmetic, without claiming unavailable local training paths passed.',
        interpretation='Fixed additional-seed development comparison; unchanged original seed42. All TabM seeds share one '
            'laptop/GPU, but replaced retained PLE members used the desktop. Architecture and historical machine/execution '
            'effects remain confounded; reused development data is not independent confirmation.',
        scope='Independent sources/member bindings, IDs, SQL RMSE and serialization. Original correction stack is covered '
            'by reviewed frozen scorer and exact replay; no new inference, fit or bootstrap rerun.')
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass',rows=n,effects=effects,seconds=receipt['seconds']),indent=2))


if __name__=='__main__':main()
