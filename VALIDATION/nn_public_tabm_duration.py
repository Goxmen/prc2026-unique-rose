"""Fixed24-epoch TabM screen with same-trajectory21/17 checkpoint controls.

No stopping search. The original optimizer/update function receives only a
checkpoint callback after each epoch; its AST otherwise remains identical.
Desktop input paths are explicitly rebound to the audited equivalent cache.
"""
import argparse,ast,copy,inspect,os,textwrap,types
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_tabm_jan_jul as original

ROOT=original.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_duration'
MANIFEST=OUT/'identity.json'
STOP={month:24 for month in original.STOP}
art,read,verify,write=original.art,original.read,original.verify,original.write_json
DESIGN=dict(parent='selected equal three-TabM pool plus robust same-date runway correction',
    change='seed42 completes original24-epoch OneCycle schedule, rather than inherited PLE21/17 stops',
    controls='same-trajectory21/17 snapshots, plus retained laptop seed42 and other two members',
    epochs=STOP,checkpoint_epochs=original.STOP,seed=42,
    model=original.model_type.SETTINGS,effective_batch=2048,microbatch=512,
    architecture_optimizer_inputs_and_loss='unchanged original public/physical TabM',
    checkpoint='after original epoch update; torch RNG unchanged; no early held inference',
    primary='one-of-three24-epoch replacement versus same-desktop early snapshot',
    selected='same replacement versus actual three-TabM pool, including unchanged runway/v79 transport',
    gate='both primary and actual selected comparisons improve in both months with paired-day upper endpoint below zero',
    machine='desktop RTX3080; same-trajectory duration control, explicit historical laptop difference',
    no_epoch_search=True,no_inner_fit=True,no_new_feature=True,no_new_weights=True,no_upload=True,
    caveat='Reused development months; does not imply optimal training duration or a gain for every seed')


def snapshot(model,month,epoch,curve):
    if epoch!=original.STOP[month]:return
    path=OUT/'early_models'/f'{month}.pt'
    assert not path.exists(),'Preserve recorded intermediate weights'
    path.parent.mkdir(exist_ok=True)
    cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
    temp=path.with_suffix('.tmp')
    torch.save(dict(identity=art(MANIFEST),design=DESIGN,epoch=epoch,
                    curve=copy.deepcopy(curve),state_dict=model.state_dict()),temp)
    temp.replace(path)
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())


def training_function(callback=snapshot):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.train)))
    epoch_loop=next(node for node in tree.body[0].body if isinstance(node,ast.For))
    i=next(i for i,node in enumerate(epoch_loop.body) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='curve.append')
    marker=ast.parse('snapshot(model,month,epoch+1,curve)').body[0]
    epoch_loop.body.insert(i+1,marker)
    restored=copy.deepcopy(tree);del next(n for n in restored.body[0].body if isinstance(n,ast.For)).body[i+1]
    assert ast.dump(restored,include_attributes=False)==ast.dump(ast.parse(textwrap.dedent(inspect.getsource(original.train))),include_attributes=False)
    scope=dict(original.train.__globals__,OUT=OUT,snapshot=callback)
    exec(compile(ast.fix_missing_locations(tree),'<tabm-duration-checkpoint-only>','exec'),scope)
    return scope['train']


def early_inference(month,d,tracks,rows,ids,identity):
    mp=OUT/'early_models'/f'{month}.pt';pp=OUT/'networks'/f'{month}_early.parquet'
    rp=pp.with_suffix('.json');checkpoint=torch.load(mp,map_location='cuda',weights_only=False)
    assert checkpoint['identity']==art(MANIFEST) and checkpoint['design']==DESIGN
    assert checkpoint['epoch']==original.STOP[month] and len(checkpoint['curve'])==original.STOP[month]
    model=original.make('tabm',d.cards,d.bins,d.train[2].shape[1]).cuda()
    model.load_state_dict(checkpoint['state_dict']);prediction=original.predict(model,d,tracks,rows)
    # Independent instance and disk reload, all held rows.
    restored=original.make('tabm',d.cards,d.bins,d.train[2].shape[1]).cuda()
    restored.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
    np.testing.assert_array_equal(prediction,original.predict(restored,d,tracks,rows))
    if pp.exists():
        f=pd.read_parquet(pp);np.testing.assert_array_equal(f.mvt_id,ids);np.testing.assert_array_equal(f.base,prediction.astype('float32'))
    else:original.original.shared.nni.save_network(pp,ids,prediction)
    write(rp,dict(identity=art(MANIFEST),design=DESIGN,month=month,seed=42,epochs=original.STOP[month],
        model=art(mp),prediction=art(pp),preprocessing=identity['folds'][month]['preprocessing'],
        all_held_reloaded_exact=True,rows=len(ids),same_trajectory_as_final=True))


def namespace():
    scope=dict(original.__dict__,OUT=OUT,MANIFEST=MANIFEST,STOP=STOP,ARMS=['tabm'],
        DESIGN=DESIGN,guard=guard,train=training_function(),__file__=__file__)
    for name in ['paths','completed','preflight']:
        fn=getattr(original,name);scope[name]=types.FunctionType(fn.__code__,scope,name,fn.__defaults__,fn.__closure__)
    original_completed=scope['completed']
    def completed(month,arm,identity):
        value=original_completed(month,arm,identity)
        early=OUT/'networks'/f'{month}_early.json'
        if value is None or not early.exists():return None
        r=read(early);assert r['identity']==art(MANIFEST) and r['epochs']==original.STOP[month]
        for key in ['model','prediction','preprocessing']:verify(r[key])
        return value
    scope['completed']=completed;scope['early_inference']=early_inference
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.fit)))
    # The sole fit-body addition runs AFTER the completed final model is recorded.
    class Insert(ast.NodeTransformer):
        count=0
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='write_json' and node.value.args and ast.unparse(node.value.args[0])=='rp':
                self.count+=1
                return [node,ast.parse('early_inference(month,d,tracks,va,held_ids,identity)').body[0]]
            return self.generic_visit(node)
    edit=Insert();tree=edit.visit(tree);assert edit.count==1
    exec(compile(ast.fix_missing_locations(tree),'<tabm-duration-early-inference>','exec'),scope)
    return scope


def freeze():
    assert not MANIFEST.exists()
    old=read(original.MANIFEST);desktop=read(original.original.MANIFEST)
    assert old['source']==art(original.__file__) and desktop['source']==art(original.original.__file__)
    for record in old['files']:verify(record)
    assert read(ROOT/'tmp/research_20261010/desktop_input_fingerprints.json')==read(ROOT/'tmp/research_20261010/laptop_input_fingerprints.json')
    value=copy.deepcopy(old)
    value.update(design=DESIGN,source=art(__file__),original_identity=art(original.MANIFEST),
        desktop_identity=art(original.original.MANIFEST),
        execution=dict(host=os.uname().nodename,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)))
    assert value['execution']['gpu']=='NVIDIA GeForce RTX 3080'
    for month in STOP:
        value['folds'][month]['matrix']=desktop['folds'][month]['matrix']
        verify(value['folds'][month]['matrix'])
    additional=[Path(__file__),ROOT/'VALIDATION/nn_public_tabm_duration_score.py',original.MANIFEST,original.original.MANIFEST,
        ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_stack/result.json',
        ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_stack/scored.parquet',
        ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo/result.json',
        ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo/scored.parquet']
    value['files']=list({r['path']:r for r in [*old['files'],*[art(p) for p in additional]]}.values())
    value['source_mapping']='Explicit audited desktop matrix paths; identical original156 semantic fingerprints and saved preprocessing; laptop paths are not bypassed'
    training_function();namespace();write(MANIFEST,value)
    print(dict(frozen=True,identity=art(MANIFEST),new_trajectories=2,same_trajectory_controls=2),flush=True)


def guard(full=False):
    value=read(MANIFEST);assert value['design']==DESIGN and value['source']==art(__file__)
    for record in value['files']:verify(record)
    assert dict(host=os.uname().nodename,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))==value['execution']
    if full:
        for fold in value['folds'].values():verify(fold['matrix'])
        manifest=verify(value['cache']);cache=read(manifest)
        assert cache['outputs']==value['cache']['outputs']
        for name,digest in cache['outputs'].items():assert original.sha256(manifest.parent/name)==digest
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','preflight','fit'])
    torch.set_num_threads(2)
    with threadpool_limits(2):
        stage=parser.parse_args().stage
        if stage=='freeze':freeze()
        else:namespace()[stage]()
