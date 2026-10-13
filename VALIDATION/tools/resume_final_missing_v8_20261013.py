"""Continue interrupted v8 final inference from the saved model and exact old vector.

Stages freeze, run, infer. No model fitting, training preparation or old model
inference is implemented. The numerical final loop is AST-identical to the
frozen producer. Prior execution and completed old artifacts remain unchanged.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'VALIDATION'))
import final_missing_early_reconstruct as parent
from final_missing_early_reconstruct import (OUT, DESIGN, MANIFEST, read, art, save,
    backend, metadata, load_training, connect, load_phase, lgb, np, pd)
from records import runtime_identity
from threadpoolctl import threadpool_limits
RESUME = OUT/'resume_v8_20261013'
IDENTITY = RESUME/'identity.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def guard():
    identity=parent.guard()
    assert runtime_identity()==identity['runtime']
    frozen=read(IDENTITY)
    for record in frozen['sources']:
        assert record==art(ROOT/record['path']), record['path']
    assert read(OUT/'v8/trained.json')['runtime']==identity['runtime']
    assert not Path('/proc/84832').exists() and not Path('/proc/86367').exists(), 'Old paused handles require inspection'
    return frozen


def freeze():
    assert not IDENTITY.exists()
    assert not any((OUT/'v8'/name).exists() for name in ['final.parquet','final.json','result.json','independent_audit.json'])
    identity=parent.guard()
    assert runtime_identity()==identity['runtime']==read(OUT/'v8/trained.json')['runtime']
    old=read(OUT/'v8/old.json')
    assert old['exact_original_integer_replay'] and old['rows']==344841 and old['changed_rows']==0
    assert old['prediction']==art(OUT/'v8/old.parquet')
    assert read(OUT/'v8/trained.json')['model']==art(OUT/'v8/model.txt')
    trees=[ast.parse(Path(path).read_text()) for path in [parent.__file__,__file__]]
    functions=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
               for tree,name in zip(trees,['infer','resume_final'])]
    loops=[next(n for n in node.body if isinstance(n,ast.For)) for node in functions]
    original_body=ast.dump(ast.Module(body=loops[0].body,type_ignores=[]),include_attributes=False)
    new_body=ast.dump(ast.Module(body=loops[1].body,type_ignores=[]),include_attributes=False)
    assert original_body==new_body
    assert ast.literal_eval(loops[1].iter)==['final']
    paths=[Path(__file__),Path(parent.__file__),OUT/'identity.json',OUT/'output_auditor_identity.json',
        OUT/'completion_v8/plan.json',OUT/'completion_v8/execution.json',OUT/'completion_v8/allocation.json',
        ROOT/'VALIDATION/reports/USER_PAUSE_20261012.json',
        ROOT/'VALIDATION/tools/audit_final_missing_early_20261012.py']
    paths += [OUT/'v8'/name for name in ['model.txt','trained.json','preflight.json','training.pkl','fit_started.json','old.parquet','old.json']]
    RESUME.mkdir(parents=True,exist_ok=True)
    save(IDENTITY,dict(state='frozen_saved_model_continuation',version=8,sources=[art(p) for p in paths],
        numerical_final_loop_AST_exact=True,loop_AST_sha256=hashlib.sha256(original_body.encode()).hexdigest(),
        previous_worker_missing=True,completed_old_retained=True,model_reused_without_fit=True,
        unfinished_final_has_no_disk_artifact=True,steps=['final_inference','independent_audit'],
        authorization='Root explicitly allocated desktop on October13 to safe saved-v8 inference/audit continuation',
        no_training=True,no_old_inference=True,no_upload=True,runtime=identity['runtime']))
    print(json.dumps(dict(state='frozen',identity=art(IDENTITY),no_fit=True)))


def verified_old():
    record=read(OUT/'v8/old.json')
    assert record['prediction']==art(OUT/'v8/old.parquet')
    old=pd.read_parquet(OUT/'v8/old.parquet')
    template=pd.read_parquet(ROOT/'data/submitting.parquet',columns=['MVT_ID_mvt']).MVT_ID_mvt
    assert len(old)==344841 and old.mvt_id.is_unique
    np.testing.assert_array_equal(old.mvt_id,template)
    original=read(MANIFEST)['8']
    assert parent.sha(ROOT/original['file'])==original['sha256']
    target=pd.read_parquet(ROOT/original['file']).set_index('MVT_ID_mvt').loc[old.mvt_id,'TAXITIME_SEC_mvt']
    np.testing.assert_array_equal(old.submitted,target)
    np.testing.assert_array_equal(old.submitted,np.rint(old.pred))
    return {'old':record}


def resume_final(version):
    guard()
    directory = OUT / f'v{version}'
    assert not (directory / 'result.json').exists(), 'Preserve completed or failed replay'
    trained = read(directory / 'trained.json')
    assert trained['identity'] == art(OUT / 'identity.json')
    assert trained['model'] == art(directory / 'model.txt') and trained['rounds'] == 800
    prepared = read(directory / 'preflight.json')
    assert prepared['training'] == art(directory / 'training.pkl')
    data = pd.read_pickle(directory / 'training.pkl')
    cats = data['cats']
    del data
    gc.collect()
    module, meta = (backend(), metadata(version))
    model = lgb.Booster(model_file=str(directory / 'model.txt'))
    replay = lgb.Booster(model_file=str(directory / 'model.txt'))
    dep, arr = load_training()
    reports = verified_old()
    for view in ['final']:
        source = ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet')
        template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
        ids = pd.read_parquet(template, columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
        con = connect()
        con.execute('SET threads=1')
        raw, arrivals = (load_phase(con, str(source), 'DEP'), load_phase(con, str(source), 'ARR'))
        con.close()
        assert raw.mvt_id.is_unique and set(raw.mvt_id) == set(ids)
        frames = []
        for month in sorted(pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m').unique()):
            rows = module.month_rows(raw, month)
            V = module.validation_matrix(dep, arr, rows, module.month_rows(arrivals, month), DESIGN['freeze'], cats, meta['config'])
            assert list(V) == meta['features']
            pred = module.predict(model, V)
            np.testing.assert_array_equal(pred, module.predict(replay, V))
            frames.append(pd.DataFrame(dict(mvt_id=rows.mvt_id.to_numpy(), month=month, pred=pred, submitted=np.rint(pred))))
        frame = pd.concat(frames, ignore_index=True).set_index('mvt_id').loc[ids].reset_index()
        assert np.isfinite(frame[['pred', 'submitted']]).all().all()
        assert (frame[['pred', 'submitted']] >= 0).all().all()
        frame.to_parquet(directory / f'{view}.parquet', index=False)
        report = dict(prediction=art(directory / f'{view}.parquet'), rows=len(frame), months=frame.month.value_counts().sort_index().to_dict(), template_order_exact=True, all_model_reload_predictions_exact=True, original_feature_order_exact=True)
        if view == 'old':
            cache = read(MANIFEST)[str(version)]
            original = pd.read_parquet(ROOT / cache['file']).set_index('MVT_ID_mvt').loc[ids, 'TAXITIME_SEC_mvt']
            diff = frame.submitted.to_numpy() - original.to_numpy()
            report.update(exact_original_integer_replay=bool(np.all(diff == 0)), changed_rows=int(np.count_nonzero(diff)), rms_difference=float(np.sqrt(np.mean(diff ** 2))), maximum_absolute_difference=float(np.max(abs(diff))))
        reports[view] = report
        save(directory / f'{view}.json', report)
        del raw, arrivals, frames, frame
        gc.collect()
        if view == 'old' and (not report['exact_original_integer_replay']):
            save(directory / 'result.json', dict(state='closed_nonexact_reconstruction', identity=art(OUT / 'identity.json'), version=version, views=reports, no_final_inference=True, no_sweep=True, no_upload=True))
            print(json.dumps(dict(stage='closed_nonexact_reconstruction', version=version, replay=report)), flush=True)
            return
    save(directory / 'result.json', dict(state='exact_old_replay_final_inference_complete', identity=art(OUT / 'identity.json'), version=version, views=reports, historical_model_source_exact=version == 4, original_model_hash_unavailable=True, independent_final_audit_pending=True, no_upload=True))
    print(json.dumps(dict(stage='exact_old_replay_final_inference_complete', version=version)), flush=True)


def run():
    frozen=guard()
    path=RESUME/'execution.json'
    assert not path.exists(), 'Preserve continuation receipt; no automatic restart'
    assert not any((OUT/'v8'/name).exists() for name in ['final.parquet','final.json','result.json','independent_audit.json'])
    state=dict(state='running',pid=os.getpid(),started_at=now(),identity=art(IDENTITY),steps=[],
               no_fit=True,no_old_inference=True,no_upload=True)
    save(path,state)
    try:
        for stage,command in [('final_inference',[str(Path(__file__)), 'infer']),
                              ('independent_audit',['VALIDATION/tools/audit_final_missing_early_20261012.py','--version','8'])]:
            guard()
            assert state['identity']==art(IDENTITY)
            state.update(active_stage=stage);save(path,state)
            with (RESUME/(stage+'.log')).open('x') as log:
                child=subprocess.Popen([str(ROOT/'.venv/bin/python'),*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;save(path,state);code=child.wait()
            assert code==0, f'{stage} exited {code}; preserve saved model and all finished artifacts'
            if stage=='final_inference':
                result=read(OUT/'v8/result.json')
                assert result['state']=='exact_old_replay_final_inference_complete'
                assert result['views']['old']==read(OUT/'v8/old.json')
                assert result['views']['final']['rows']==670790
                evidence=art(OUT/'v8/result.json')
            else:
                assert read(OUT/'v8/independent_audit.json')['status']=='pass'
                evidence=art(OUT/'v8/independent_audit.json')
            state['steps'].append(dict(stage=stage,evidence=evidence,finished_at=now()));save(path,state)
        guard()
        state.update(state='complete',active_stage=None,child_pid=None,finished_at=now())
        save(path,state)
        print(json.dumps(dict(state='complete',receipt=art(path))))
    except BaseException as exc:
        state.update(state='failed',failure=repr(exc),traceback=traceback.format_exc(),finished_at=now());save(path,state)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','run','infer'])
    stage=parser.parse_args().stage
    with threadpool_limits(limits=1,user_api='blas'):
        if stage=='infer':
            guard();resume_final(8)
        else:
            globals()[stage]()
