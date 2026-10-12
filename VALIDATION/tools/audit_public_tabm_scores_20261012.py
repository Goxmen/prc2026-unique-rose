"""Independent saved-result audit; no fits, stack replay or bootstrap rerun.

Preserves same-laptop family contrasts separately from historical-control
machine/execution differences. Run only after the scoring CPU handoff.
"""
import json
from pathlib import Path
import time

import duckdb
import audit_public_cohort_scores_20261012 as common

ROOT,RUNS=common.ROOT,common.RUNS
read,artifact,verify,metrics=common.read,common.artifact,common.verify,common.validate_metrics
TRAIN=RUNS/'v3_20261012_nn_public_tabm'
STACK=RUNS/'v3_20261012_nn_public_tabm_stack'
TRANSPORT=RUNS/'v3_20261012_delta_transport'
ARMS=['historical_single','laptop_control','tabm_single','selected','replace_control','replace_tabm']
OUT=ROOT/'VALIDATION/reports/public_tabm_20261012/score_audit.json'


def main():
    started=time.monotonic();assert not OUT.exists(),'Preserve independent receipt'
    identity=read(TRAIN/'portable.json');fit=read(TRAIN/'audit.json')
    returned_path=ROOT/'tmp/research_20261012/model_diversity/tabm_return_audit.json'
    returned=read(returned_path)
    assert returned['status']=='pass' and returned['fit_process_exit']==returned['transport_process_exit']==0
    assert returned['portable_manifest']==artifact(TRAIN/'portable.json')
    assert returned['fit_audit']==artifact(TRAIN/'audit.json')
    assert fit['full_input_hashes_verified'] and fit['original_preprocessing_exact']
    matrix_scope={month:dict(record=fold['matrix'],available_on_desktop=(ROOT/fold['matrix']['path']).exists())
        for month,fold in identity['folds'].items()}
    stack=read(STACK/'result.json');audit=read(STACK/'audit.json')
    frozen=read(TRANSPORT/'design.json');transport=read(TRANSPORT/'tabm/result.json')
    assert fit['status']==audit['status']=='pass'
    assert fit['identity']==artifact(TRAIN/'portable.json')
    assert audit['result']==artifact(STACK/'result.json')
    assert stack['exact_v78_replayed'] and transport['exact_v78_control']
    assert transport['identity']==artifact(TRANSPORT/'design.json')
    assert transport['design']==frozen['design']
    assert frozen['design']['coefficient']==1 and frozen['design']['new_bounds'] is False
    assert frozen['design']['experiments']['tabm']==dict(
        directory=STACK.name,control='selected',arms=['replace_tabm','replace_control'])
    for item in [identity['source'],*identity['files'],*stack['sources'],
                 *transport['sources'],*frozen['sources'],stack['scored'],transport['scored']]:verify(item)
    assert identity['selected_v78']==artifact(common.PARENT78/'result.json')
    members=[];machines=set()
    for month,epochs in [('2025-01',21),('2025-07',17)]:
        for arm in ['tabm','control']:
            path=TRAIN/f'networks/{month}_{arm}.json';record=read(path)
            assert record['identity']==artifact(TRAIN/'portable.json')
            assert record['design']==identity['design'] and record['inputs']==identity['folds'][month]
            assert record['month']==month and record['arm']==arm and record['seed']==42
            assert record['epochs']==epochs and len(record['curve'])==epochs
            assert record['all_held_reloaded_exact']
            assert record['preprocessing']==identity['folds'][month]['preprocessing']
            for key in ['model','prediction','preprocessing','checkpoint']:verify(record[key])
            trained=read(ROOT/record['checkpoint']['path'])
            assert trained['model']==record['model'] and trained['preprocessing']==record['preprocessing']
            assert trained['updates_completed_before_inference']
            members.append(artifact(path));machines.add((record['host'],record['gpu']))
    assert len(machines)==1,'Fair candidate/control must use one host/GPU'
    assert sorted(fit['members'],key=lambda r:r['path'])==sorted(members,key=lambda r:r['path'])
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        for name,path in [('f',STACK/'scored.parquet'),('old',common.PARENT78/'scored.parquet'),
                          ('p',common.PARENT79/'cv.parquet'),('t',TRANSPORT/'tabm/scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
            n,u=con.execute(f'SELECT count(*),count(distinct mvt_id) FROM {name}').fetchone()
            assert n==u==344419
        bad_arms=' OR '.join(f't.{arm} IS DISTINCT FROM round_even(greatest(0,p.pred+f.{arm}-f.selected),0) '
            f'OR (f.{arm}=f.selected AND t.{arm} IS DISTINCT FROM p.pred)' for arm in ['replace_tabm','replace_control'])
        n,bad=con.execute(f'''SELECT count(*),sum(CASE WHEN
            f.y IS DISTINCT FROM old.y OR f.month IS DISTINCT FROM old.month OR
            f.day IS DISTINCT FROM old.day OR f.selected IS DISTINCT FROM old.candidate OR
            f.y IS DISTINCT FROM p.y OR f.month IS DISTINCT FROM p.month OR f.day IS DISTINCT FROM p.day OR
            t.y IS DISTINCT FROM p.y OR t.month IS DISTINCT FROM p.month OR t.day IS DISTINCT FROM p.day OR
            t.control IS DISTINCT FROM p.pred OR {bad_arms} THEN 1 ELSE 0 END)
            FROM f JOIN old USING(mvt_id) JOIN p USING(mvt_id) JOIN t USING(mvt_id)''').fetchone()
        assert n==344419 and bad==0
        for view,columns in [('f',ARMS),('t',['control','replace_tabm','replace_control'])]:
            terms=' OR '.join(f'{c} IS NULL OR NOT isfinite({c}) OR {c}<0 OR {c}<>round_even({c},0)' for c in columns)
            assert con.execute(f'SELECT count(*) FROM {view} WHERE {terms}').fetchone()[0]==0
        primary=metrics(con,'f',stack,ARMS)
        secondary=metrics(con,'t',transport,['control','replace_tabm','replace_control'])
    cv,tcv=stack['cv'],transport['cv']
    effects=dict(actual_v78_replacement=cv['replace_tabm']-cv['selected'],
        matched_laptop_replacement=cv['replace_tabm']-cv['replace_control'],
        matched_laptop_single=cv['tabm_single']-cv['laptop_control'],
        historical_machine_execution_single=cv['laptop_control']-cv['historical_single'],
        historical_machine_execution_replacement=cv['replace_control']-cv['selected'],
        actual_v79_transport=tcv['replace_tabm']-tcv['control'],
        matched_laptop_v79_transport=tcv['replace_tabm']-tcv['replace_control'])
    receipt=dict(status='pass',rows=n,seconds=time.monotonic()-started,source=artifact(__file__),
        helper=artifact(common.__file__),members=members,machine=list(machines)[0],
        remote_return_receipt=artifact(returned_path),matrix_scope=matrix_scope,
        input_validation_scope='Full training matrix byte hashes and semantic fingerprints were verified on the laptop. '
            'This independent desktop audit verifies returned sources, models, predictions, preprocessing, '
            'checkpoint receipts and scored arithmetic. It does not claim the desktop training-input guard passed; '
            'the unchanged laptop-specific matrix paths are absent locally.',
        inputs=[artifact(p) for p in [TRAIN/'portable.json',TRAIN/'audit.json',
            STACK/'result.json',STACK/'audit.json',STACK/'scored.parquet',TRANSPORT/'design.json',
            TRANSPORT/'tabm/result.json',TRANSPORT/'tabm/scored.parquet',
            common.PARENT78/'scored.parquet',common.PARENT79/'cv.parquet']],
        sql_scores=primary,sql_transport_scores=secondary,effects=effects,
        full_row_truth_calendar_ids_and_v78_exact=True,all_row_v79_transport_and_zero_delta_exact=True,
        interpretation='Matched same-laptop contrasts measure the fixed family/decay comparison under '
            'common microbatch execution. Historical differences also include GPU/execution effects; '
            'they are not a clean architecture estimate. All comparisons reuse development data.',
        scope='Independent artifact binding, SQL error arithmetic and serialization. Inherited corrections '
            'are covered by reviewed frozen scorer and exact parent replay, not independently reimplemented. '
            'No repeated model inference, fit, bootstrap or independent-confirmation claim.')
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass',rows=n,seconds=receipt['seconds'],effects=effects),indent=2))


if __name__=='__main__':main()
