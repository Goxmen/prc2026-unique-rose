"""Fixed Gaussian likelihood screen; evaluate the unchanged conditional mean."""
import argparse
import ast
import inspect
import gc
from pathlib import Path
import pickle
import types

import duckdb
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_scratch_jan_jul as single
import nn_public_gaussian as model_type
import public_stack_cv as joint
import public_refinements_submission_cv as rounded
from records import write_json

ROOT=single.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_nn_public_gaussian'
MANIFEST=OUT/'identity.json'
STOP=single.STOP
art,verify,read=single.artifact,single.verify,single.shared.ts.read
DESIGN=dict(single.DESIGN,variant='public',change='twice Gaussian likelihood with learned per-row observation variance',
    training=model_type.SETTINGS['training'],model=model_type.SETTINGS,added_columns=model_type.features.FEATURES, no_new_inputs=True,
    primary='matched seed42 physical network inside production-rounded public stack',
    secondary='replace only seed42 of selected three-member pool; unchanged1042/2042',
    no_production=True,no_upload=True)


def clone(function,**namespace):
    return types.FunctionType(function.__code__,{**function.__globals__,**namespace},
        function.__name__,function.__defaults__,function.__closure__)


def paths(month):
    return OUT/'networks'/f'{month}_s42.parquet',OUT/'models'/f'{month}_s42.pt',OUT/'networks'/f'{month}_s42.json'


def completed(month,identity):
    return clone(single.completed,OUT=OUT,MANIFEST=MANIFEST,DESIGN=DESIGN,paths=paths)(month,identity)


def control(month,identity):
    return clone(single.completed,DESIGN=dict(single.DESIGN,variant='public'))(month,identity)


def variance_predictions(model,data,tracks,rows):
    was_training=model.training
    model.eval(); values=[]
    with torch.no_grad():
        for start in range(0,len(rows),2048):
            model(*(x[start:start+2048] for x in data.held),*tracks.batch(rows[start:start+2048]))
            values.append(model.log_variance[:,0].cpu().numpy())
    model.train(was_training)
    result=np.concatenate(values)
    assert np.isfinite(result).all() and (result>=-6).all() and (result<=4).all()
    return result


def training_function():
    function=model_type.training_function(single.base.train,dict(OUT=OUT/'runtime',prototype=model_type))
    def train(data,tracks,train_rows,held_rows,stop):
        month=next(m for m,n in STOP.items() if n==stop)
        recovery=OUT/'models'/f'{month}_recovery.pt'
        assert not recovery.exists(),'Completed unrecorded fit needs recovery, not retraining'
        # Replay the saved late-only model on the complete unchanged held inputs.
        old_record=control(month,read(single.MANIFEST))
        old=model_type.parent.SparseFusion(data.cards,data.bins,data.train[2].shape[1]).to(data.device)
        old.load_state_dict(torch.load(verify(old_record['model']),map_location=data.device,weights_only=False)['state_dict'])
        old_prediction=single.base.predict(old,data,tracks,held_rows)
        saved=pd.read_parquet(verify(old_record['prediction']))
        np.testing.assert_array_equal(old_prediction.astype('float32'),saved.base)
        (OUT/'models').mkdir(exist_ok=True)
        write_json(OUT/'models'/f'{month}_control_reload.json',dict(identity=art(MANIFEST),
            model=old_record['model'],prediction=old_record['prediction'],rows=len(saved),all_held_exact=True))
        del old,old_prediction,saved;gc.collect();torch.cuda.empty_cache()
        model,prediction,curve=function(data,tracks,train_rows,held_rows,stop)
        # Preserve the completed fit before any additional diagnostic can fail.
        (OUT/'models').mkdir(exist_ok=True)
        temporary=recovery.with_suffix('.pt.tmp')
        torch.save(dict(state_dict=model.state_dict(),curve=curve,design=DESIGN),temporary);temporary.replace(recovery)
        restored=model_type.SparseFusion(data.cards,data.bins,data.train[2].shape[1]).to(data.device)
        restored.load_state_dict(torch.load(recovery,map_location=data.device,weights_only=False)['state_dict'])
        actual=single.base.predict(restored,data,tracks,held_rows)
        np.testing.assert_array_equal(actual,prediction)
        original_variance=variance_predictions(model,data,tracks,held_rows)
        restored_variance=variance_predictions(restored,data,tracks,held_rows)
        np.testing.assert_array_equal(restored_variance,original_variance)
        variance_path=OUT/'networks'/f'{month}_s42_variance.parquet'
        pd.DataFrame(dict(held_position=np.arange(len(prediction)),log_variance=original_variance)).to_parquet(variance_path,index=False)
        write_json(OUT/'models'/f'{month}_reload.json',dict(identity=art(MANIFEST),month=month,
            rows=len(prediction),all_predictions_reloaded_exact=True,recovery=art(recovery),
            all_variances_reloaded_exact=True,variances=art(variance_path),
            variance_floor_rows=int((original_variance==-6).sum()),variance_ceiling_rows=int((original_variance==4).sum()),
            sigma_seconds_quantiles=np.quantile(np.exp(original_variance/2)*data.sd,[0,.1,.5,.9,1]).tolist()))
        return model,prediction,curve
    return train


def guard():
    previous=read(single.MANIFEST)
    assert previous['source']==art(single.__file__)
    for p in previous['helpers']:verify(p)
    for field in ['public_features','feature_audit','metadata','metadata_values']:verify(previous[field])
    assert read(single.OUT/'audit.json')['status']=='pass'
    assert read(rounded.OUT/'audit.json')['status']=='pass'
    controls={m:art(single.paths(m)[2]) for m in STOP}
    for month in STOP:assert control(month,previous) is not None
    training_function()
    value=dict(design=DESIGN,source=art(__file__),cache=previous['cache'],folds=previous['folds'],
        public_features=previous['public_features'],feature_audit=previous['feature_audit'],
        metadata=previous['metadata'],metadata_values=previous['metadata_values'],
        control_identity=art(single.MANIFEST),controls=controls,control_audit=art(single.OUT/'audit.json'),
        selected_result=previous['selected_result'],selected_errors=previous['selected_errors'],
        helpers=[art(m.__file__) for m in [single,model_type,model_type.parent,model_type.features,
            single.base,single.adapter,joint,rounded,rounded.original.constraints,rounded.original.constraints.exclusion]],
        rome_predictions=art(joint.neural.cat.OUT/'scored.parquet'),
        selected_predictions=art(joint.OUT/'scored.parquet'),rounded=art(rounded.OUT/'scored.parquet'),
        checks=model_type.check('cpu'),training_ast_except_loss_and_metric_exact=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('**/*.pt'));write_json(MANIFEST,value)
    return value


def fit():
    identity=guard();single.base.guard(single.base.cache_identity())
    # The preparation/save wrapper is unchanged except for its truthful preflight label.
    tree=ast.parse(inspect.getsource(single.fit))
    flags=[n for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='training_bytecode_exact']
    assert len(flags)==1 and isinstance(flags[0].value,ast.Constant) and flags[0].value.value is True
    flags[0].arg='training_ast_except_loss_and_metric_exact'
    namespace={**single.fit.__globals__, 'OUT':OUT,'MANIFEST':MANIFEST,'DESIGN':DESIGN,
        'airport_tune':model_type,'training_function':training_function,'paths':paths,'completed':completed}
    exec(compile(ast.fix_missing_locations(tree),__file__+':fit_wrapper','exec'),namespace)
    namespace['fit'](list(STOP),identity)


def audit_models():
    identity=guard();assert not (OUT/'audit.json').exists(),'Preserve completed audit'
    single.shared.recipe()
    index=pd.read_parquet(single.base.seq.OUT/'full_index.parquet').set_index('mvt_id',drop=False)
    points=np.memmap(single.base.seq.OUT/'full_points.bin',mode='r',dtype=np.int32).reshape(-1,6)
    public=pd.read_parquet(verify(identity['public_features'])).set_index('mvt_id');checks=[]
    for month in STOP:
        record=completed(month,identity);assert record is not None
        receipt=read(OUT/'models'/f'{month}_reload.json');assert receipt['all_predictions_reloaded_exact']
        verify(receipt['recovery'])
        assert receipt['all_variances_reloaded_exact']
        saved_variance=pd.read_parquet(verify(receipt['variances']))
        np.testing.assert_array_equal(saved_variance.held_position,np.arange(len(saved_variance)))
        old_receipt=read(OUT/'models'/f'{month}_control_reload.json')
        assert old_receipt['identity']==art(MANIFEST) and old_receipt['all_held_exact']
        verify(old_receipt['model']);verify(old_receipt['prediction'])
        with verify(record['preprocessing']).open('rb') as file:pre=pickle.load(file)
        data=pd.read_pickle(verify(identity['folds'][month]['matrix']));ids=np.asarray(data['ids']);scope=index.loc[ids]
        extra=[np.flatnonzero(mask)[:8] for mask in [scope['count'].eq(0),scope['count'].eq(1),
            scope['count'].between(2,31),scope['count'].ge(32),scope.raw_ground_count.gt(0),scope.clock_ms.isna()]]
        positions=np.unique(np.concatenate([np.linspace(0,len(ids)-1,512,dtype=int),*extra]))
        chosen=ids[positions];X=data['V'][pre['columns']].iloc[positions].copy();del data,scope;gc.collect()
        cats,num=pre['encoder'].transform(X)
        extra=model_type.transform(public.loc[chosen])
        np.testing.assert_array_equal(extra[:,:12],model_type.parent.transform(public.loc[chosen]))
        values=[cats,num[:,pre['ple']],np.concatenate([num[:,pre['plain']],extra],axis=1)]
        tensors=[torch.from_numpy(np.ascontiguousarray(v)).cuda() for v in values]
        selected=index.loc[chosen].reset_index(drop=True)
        tracks=single.base.tensor_seq.RaggedTracks(selected,points,'cuda')
        track=tracks.batch(np.arange(len(chosen)))
        for tensor,expected in zip(track,single.base.seq.encoded_batch(selected.iloc[:32],points)):
            np.testing.assert_allclose(tensor[:32].cpu().numpy(),expected,rtol=2e-6,atol=2e-7)
        model=model_type.SparseFusion(pre['cards'],pre['bins'],len(pre['plain'])+12).cuda().eval()
        checkpoint=torch.load(verify(record['model']),map_location='cuda',weights_only=False)
        assert checkpoint['design']==DESIGN;model.load_state_dict(checkpoint['state_dict'])
        with torch.no_grad():standardized=model(*tensors,*track)[:,0].cpu().numpy().astype(float)
        actual_variance=model.log_variance[:,0].detach().cpu().numpy()
        np.testing.assert_allclose(actual_variance,saved_variance.log_variance.to_numpy()[positions],atol=1e-4,rtol=0)
        prediction=np.maximum(standardized*pre['sd']+pre['mu'],0)
        saved=pd.read_parquet(verify(record['prediction'])).set_index('mvt_id').loc[chosen,'base'].to_numpy(float)
        np.testing.assert_allclose(prediction,saved,atol=.01,rtol=0)
        checks.append(dict(month=month,sampled_rows=len(chosen),maximum_difference=float(np.max(abs(prediction-saved))),
            maximum_log_variance_difference=float(np.max(abs(actual_variance-saved_variance.log_variance.to_numpy()[positions]))),
            all_variances_reloaded_exact=True,variance_receipt=receipt['variances'],
            all_held_reload_receipt=art(OUT/'models'/f'{month}_reload.json'),
            all_control_reload_receipt=art(OUT/'models'/f'{month}_control_reload.json'),raw_scalar_rows=32))
        del model,checkpoint,pre,X,tracks,track,tensors;gc.collect();torch.cuda.empty_cache()
    write_json(OUT/'audit.json',dict(status='pass',identity=art(MANIFEST),source=art(__file__),checks=checks));print(checks,flush=True)


def score():
    identity=guard();assert read(OUT/'audit.json')['status']=='pass'
    assert not (OUT/'result.json').exists(),'Preserve completed score'
    cat=joint.neural.cat;replay=cat.r20.physical.replay
    sc,rome,_,trees=replay.baseline.baseline();pool_identity=joint.neural.guard()
    prior=pd.read_parquet(verify(identity['rome_predictions'])).set_index('mvt_id')
    expected=pd.read_parquet(verify(identity['selected_predictions'])).set_index('mvt_id')
    rounding=pd.read_parquet(verify(identity['rounded'])).set_index('mvt_id')
    raw=pd.read_parquet(cat.r20.features.base.OUT/'rows2025.parquet').set_index('mvt_id')
    constraints=cat.r20.constraints;matches=pd.read_parquet(constraints.exclusion.OUT/'matches2025.parquet');frames=[]
    for month in STOP:
        d=sc.d[month];ids=d['ids']
        members=[replay.values(verify(joint.neural.completed(month,k,pool_identity)['prediction']),ids,'base') for k in range(3)]
        new=replay.values(verify(completed(month,identity)['prediction']),ids,'base')
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        main=replay.values(joint.main.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        tree=np.mean([main,trees['airport'][42][month],trees['catboost'][42][month]],axis=0)
        probability=.5*prior.loc[ids,'newcat'].to_numpy()+.5*rome[month]
        allowed=matches[matches.donor_month.isin(single.shared.ts.FOLDS[month]+[month])]
        outputs={}
        for name,physical in [('control',members[0]),('candidate',new),('selected',np.mean(members,axis=0)),
                              ('replace_seed42',np.mean([new,*members[1:]],axis=0))]:
            direct=sc.final(month,.71*.5*(calls+physical)+.29*tree,probability)
            if name=='selected':np.testing.assert_array_equal(direct,expected.loc[ids,'combined'])
            p=constraints.matched_bound(raw.loc[ids].reset_index(),
                rounding.loc[ids,'control_boundary'].to_numpy()+direct-prior.loc[ids,'control'].to_numpy())
            p,_=constraints.exclusion.apply(ids,np.maximum(p,0),allowed)
            outputs[name]=np.rint(np.maximum(p,0))
            if name=='selected':np.testing.assert_array_equal(outputs[name],rounding.loc[ids,'control'])
        frames.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],**outputs)))
    f=pd.concat(frames,ignore_index=True);assert len(f)==344419 and f.mvt_id.is_unique
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in outputs} for m,g in f.groupby('month')}
    paired={f'{b}_vs_{a}':single.shared.ts.paired_bootstrap(f.day,f.month,(f[a]-f.y)**2,(f[b]-f.y)**2)
            for a,b in [('control','candidate'),('selected','replace_seed42')]}
    f.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,identity=art(MANIFEST),predictions=art(OUT/'scored.parquet'),scores=scores,
        cv={c:float(np.mean([v[c] for v in scores.values()])) for c in outputs},paired=paired,
        actual_selected_pool_and_rounding_exact=True)
    write_json(OUT/'result.json',result);print(result,flush=True)


def audit_scores():
    guard();assert not (OUT/'score_audit.json').exists(),'Preserve completed audit'
    result=read(OUT/'result.json');verify(result['predictions'])
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('c',OUT/'scored.parquet'),('r',joint.main.features.base.OUT/'rows2025.parquet'),
            ('p',rounded.OUT/'scored.parquet')]:con.read_parquet(str(path)).create_view(name)
        n,u,errors=con.execute('''select count(*),count(distinct c.mvt_id),sum(case when
            c.y is distinct from r.y or c.month is distinct from r.month or c.day is distinct from r.day
            or c.selected is distinct from p.control then 1 else 0 end)
            from c join r using(mvt_id) join p using(mvt_id)''').fetchone()
        assert n==u==344419 and errors==0
        for col in result['cv']:
            values=con.execute(f'select month,sqrt(avg(pow({col}-y,2))) from c group by month').fetchall()
            for month,value in values:np.testing.assert_allclose(value,result['scores'][month][col],atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.mean([v for m,v in values]),result['cv'][col],atol=1e-8,rtol=0)
    value=dict(status='pass',rows=n,independent_sql_truth_calendar_control_scores=True,
        identity=art(MANIFEST),source=art(__file__),result=art(OUT/'result.json'))
    write_json(OUT/'score_audit.json',value);print(value,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['guard','fit','audit_models','score','audit_scores'])
    a=p.parse_args();torch.set_num_threads(2 if a.stage in ['fit','audit_models'] else 1)
    with threadpool_limits(2 if a.stage in ['fit','audit_models'] else 1):globals()[a.stage]()
