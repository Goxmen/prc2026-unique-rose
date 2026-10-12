"""Independent saved-output audit; no model fit/reload or bootstrap repetition."""
import json
import time

import duckdb
import audit_public_cohort_scores_20261012 as common

ROOT,RUNS=common.ROOT,common.RUNS
read,art,verify,metrics=common.read,common.artifact,common.verify,common.validate_metrics
RUN=RUNS/'v3_20261012_linear_leaf_main'
BASE=RUNS/'v3_20261011_public_daily_taxi_lgb'
OUT=ROOT/'VALIDATION/reports/linear_leaf_20261012/independent_score_audit.json'
ARMS=['selected','constant','candidate']


def main():
    start=time.monotonic();assert not OUT.exists(),'Preserve completed receipt'
    identity=read(RUN/'identity.json');design=read(RUN/'design.json');preflight=read(RUN/'preflight.json')
    result=read(RUN/'result.json');audit=read(RUN/'audit.json')
    frozen=read(RUN/'v79/design.json');transport=read(RUN/'v79/result.json');ta=read(RUN/'v79/audit.json')
    assert result['design']==identity['design']==design['design']
    assert result['identity']==audit['identity']==preflight['identity']==art(RUN/'identity.json')
    assert preflight['status']==audit['status']==ta['status']=='pass'
    assert audit['result']==art(RUN/'result.json') and ta['result']==art(RUN/'v79/result.json')
    assert result['exact_v77_and_v78_controls'] and audit['all_parent_controls_exact']
    assert audit['all_daily_shifts_and_v78_constraints_replayed']
    assert transport['identity']==art(RUN/'v79/design.json') and transport['design']==frozen['design']
    assert frozen['primary_design']==art(RUN/'design.json')
    assert frozen['design']['coefficient']==1 and frozen['design']['new_bounds'] is False
    assert frozen['design']['arms']==['constant','candidate']
    for item in [*design['sources'],*frozen['sources'],*result['artifacts'],*result['model_records'],
                 *transport['sources'],transport['scored']]:verify(item)
    # Preserve the large-input guard's scope without repeating multi-GB matrix scans.
    for item in identity['sources']:
        if item['path'].endswith(('.py','.json')):verify(item)
    members=[];machines=set()
    for month in ['2025-01','2025-07']:
        original=read(BASE/'predictions'/f'{month}_s42.json')
        scaler_path=RUN/'scalers'/f'{month}.json';scaler=read(scaler_path)
        assert scaler['identity']==art(RUN/'identity.json') and scaler['month']==month
        assert scaler['columns']==original['columns'] and len(scaler['columns'])==184
        assert scaler['categoricals']==original['categoricals']
        for arm in ['constant','candidate']:
            rp=RUN/'predictions'/f'{month}_{arm}.json';record=read(rp)
            staged_path=RUN/'models'/f'{month}_{arm}.json';staged=read(staged_path)
            assert record['staged_model']==art(staged_path)
            assert record['identity']==staged['identity']==art(RUN/'identity.json')
            assert record['month']==staged['month']==month and record['arm']==staged['arm']==arm
            assert record['all_held_reload_exact'] and staged['model_saved_before_held_inference']
            assert record['scaler']==staged['scaler']==art(scaler_path)
            assert record['held_rows']==staged['held_rows']==scaler['held_rows']
            assert staged['train_rows']==scaler['train_rows'] and staged['columns']==scaler['columns']
            assert staged['categoricals']==scaler['categoricals'] and staged['rounds']==10000
            expected={**original['params'],'linear_tree':arm=='candidate'}
            if arm=='candidate':expected['linear_lambda']=1.
            assert staged['params']==expected
            for item in [record['prediction'],staged['model']]:verify(item)
            machines.add(staged['machine'].lower());members.append(art(rp))
    assert len(machines)==1
    assert sorted(members,key=lambda x:x['path'])==sorted(result['model_records'],key=lambda x:x['path'])
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        for view,path in [('f',RUN/'scored.parquet'),('old',common.PARENT78/'scored.parquet'),
                         ('p',common.PARENT79/'cv.parquet'),('t',RUN/'v79/scored.parquet')]:
            con.read_parquet(str(path)).create_view(view)
            n,u=con.execute(f'SELECT count(*),count(distinct mvt_id) FROM {view}').fetchone()
            assert n==u==344419
        formulas=' OR '.join(f't.{arm} IS DISTINCT FROM round_even(greatest(0,p.pred+f.{arm}-f.selected),0) '
            f'OR (f.{arm}=f.selected AND t.{arm} IS DISTINCT FROM p.pred)' for arm in ['constant','candidate'])
        n,bad=con.execute(f'''SELECT count(*),sum(CASE WHEN
            f.y IS DISTINCT FROM old.y OR f.month IS DISTINCT FROM old.month OR f.day IS DISTINCT FROM old.day OR
            f.selected IS DISTINCT FROM old.candidate OR f.increment IS DISTINCT FROM old.increment OR
            f.y IS DISTINCT FROM p.y OR f.month IS DISTINCT FROM p.month OR f.day IS DISTINCT FROM p.day OR
            t.y IS DISTINCT FROM p.y OR t.month IS DISTINCT FROM p.month OR t.day IS DISTINCT FROM p.day OR
            t.selected IS DISTINCT FROM p.pred OR {formulas} THEN 1 ELSE 0 END)
            FROM f JOIN old USING(mvt_id) JOIN p USING(mvt_id) JOIN t USING(mvt_id)''').fetchone()
        assert n==344419 and bad==0
        terms=' OR '.join(f'{c} IS NULL OR NOT isfinite({c}) OR {c}<0 OR {c}<>round_even({c},0)' for c in ARMS)
        for view in ['f','t']:assert con.execute(f'SELECT count(*) FROM {view} WHERE {terms}').fetchone()[0]==0
        primary=metrics(con,'f',result,ARMS);secondary=metrics(con,'t',transport,ARMS)
    effects={kind:dict(matched=value['cv']['candidate']-value['cv']['constant'],
        candidate_vs_selected=value['cv']['candidate']-value['cv']['selected'],
        constant_vs_selected=value['cv']['constant']-value['cv']['selected'])
        for kind,value in [('v78',result),('v79',transport)]}
    value=dict(status='pass',source=art(__file__),helper=art(common.__file__),rows=n,seconds=time.monotonic()-start,
        machine=list(machines)[0],members=members,effects=effects,sql_scores=primary,sql_transport_scores=secondary,
        inputs=[art(p) for p in [RUN/'identity.json',RUN/'design.json',RUN/'preflight.json',RUN/'result.json',
            RUN/'audit.json',RUN/'scored.parquet',RUN/'v79/design.json',RUN/'v79/result.json',RUN/'v79/audit.json',
            RUN/'v79/scored.parquet',common.PARENT78/'scored.parquet',common.PARENT79/'cv.parquet']],
        checks=['All4 staged model/prediction/scaler receipts and exact original recipe changes',
            'Complete IDs/truth/calendar and exact retained v78 selected values',
            'All-row fixed v79 transport, no-change recovery, nonnegative integer serialization',
            'Independent monthly SQL RMSE and equally averaged Jan/Jul scores'],
        scope='Saved artifacts and arithmetic; original full-input/scaler validation and exact model reloads are inherited '
            'from their retained receipts. No repeated large-matrix scan, model inference, correction implementation or bootstrap.',
        interpretation='Candidate-minus-constant is the matched same-desktop comparison. Both arms versus selected remain '
            'separate normalization/control comparisons on reused development data; not independent confirmation.')
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass',rows=n,effects=effects,seconds=value['seconds']),indent=2))


if __name__=='__main__':main()
