"""Fixed two-seed continuation of the selected full-input TabM family.

Retain seed42 exactly. Add1042/2042 with the same21/17 epochs and update recipe.
The only training-function specialization is the permutation generator seed.
No fit is started by preparation; no original runner/source is changed.
"""
import argparse,ast,gc,inspect,os,pickle,tempfile,textwrap,types
from pathlib import Path
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import nn_public_tabm_jan_jul as original
from records import write_json

ROOT=original.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3'
MANIFEST=OUT/'identity.json'
SEEDS=[42,1042,2042]
STOP=original.STOP
art,read,verify=original.art,original.read,original.verify
DESIGN=dict(parent='selected one-of-three TabM replacement',model=original.model_type.SETTINGS,
    seeds=SEEDS,reused=42,new_seeds=[1042,2042],epochs=STOP,schedule_epochs=24,
    effective_batch=original.BATCH,microbatch=original.MICRO,inference_chunk=original.PRED_CHUNK,
    changes='only seed specialization of original model initialization and permutation/dropout stream',
    loss='unchanged individual K32-head MSE and actual-row-weighted accumulation',
    primary='equal TabM42/1042/2042 versus immutable one-TabM/two-PLE pool',
    secondary='same all-TabM pool versus original three PLE members',
    stack='exact original v78 stack and separate fixed coefficient-one v79 transport',
    controls='reuse saved original desktop PLE42/1042/2042; no repeated controls',
    host='laptop RTX5060, same host as completed TabM42',
    caveat='additional seeds reuse desktop controls with execution/hardware confounding; reused development data',
    preprocessing='exact saved original156 plus unchanged12 public transform; same physical track cache',
    no_new_stopping=True,no_new_weight=True,no_seed_selection=True,no_hpo=True,no_production_fit=True,no_upload=True)


def trainer(k):
    assert k in [0,1,2]
    source=textwrap.dedent(inspect.getsource(original.train));tree=ast.parse(source)
    class Seed(ast.NodeTransformer):
        count=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Attribute) and node.func.attr=='manual_seed':
                assert len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==42
                node.args[0]=ast.Name(id='RUN_SEED',ctx=ast.Load());self.count+=1
            return node
    edit=Seed();tree=edit.visit(tree);assert edit.count==1
    class Restore(ast.NodeTransformer):
        def visit_Name(self,node):return ast.copy_location(ast.Constant(value=42),node) if node.id=='RUN_SEED' else node
    restored=Restore().visit(ast.parse(ast.unparse(tree)))
    assert ast.dump(restored,include_attributes=False)==ast.dump(ast.parse(source),include_attributes=False)
    namespace=dict(original.train.__globals__,OUT=OUT/f'k{k}',RUN_SEED=SEEDS[k])
    exec(compile(ast.fix_missing_locations(tree),'<fixed-tabm-seed-specialization>','exec'),namespace)
    assert original.OUT!=OUT/f'k{k}' and 'RUN_SEED' not in original.train.__globals__
    return namespace['train']


def paths(month,k):
    if k==0:return original.paths(month,'tabm')
    run=OUT/f'k{k}';stem=f'{month}_tabm'
    return run/'models'/f'{stem}.pt',run/'networks'/f'{stem}.parquet',run/'networks'/f'{stem}.json'


def freeze():
    assert not MANIFEST.exists(),'Preserve frozen design'
    prior=read(original.MANIFEST)
    assert prior['source']==art(original.__file__) and prior['design']==original.DESIGN
    for rec in prior['files']:verify(rec)
    audit=read(original.OUT/'audit.json');assert audit['status']=='pass' and audit['identity']==art(original.MANIFEST)
    for rec in audit['members']:verify(rec)
    selected=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_stack'
    assert read(selected/'result.json')['paired']['replace_tabm_vs_selected']['decision']=='improvement'
    assert read(selected/'audit.json')['result']==art(selected/'result.json')
    files={Path(__file__),original.MANIFEST,original.OUT/'audit.json',original.OUT/'preflight.json',selected/'result.json',selected/'audit.json',selected/'scored.parquet',
        ROOT/'VALIDATION/nn_public_tabm_pool3_stack.py',ROOT/'VALIDATION/tabm_pool3_v79_transport.py',
        ROOT/'VALIDATION/tests/test_public_tabm_pool3.py',ROOT/'VALIDATION/reports/public_tabm_20261012/score_audit.json'}
    files.update(verify(rec) for rec in prior['files'])
    reused={};execution=None
    for month in STOP:
        record=original.completed(month,'tabm',prior);assert record['all_held_reloaded_exact']
        actual_execution={key:record[key] for key in ['host','gpu','torch']}
        if execution is None:execution=actual_execution
        else:assert execution==actual_execution
        mp,pp,rp=paths(month,0);reused[month]=dict(record=art(rp),model=art(mp),prediction=art(pp),checkpoint=record['checkpoint'])
        files.update([mp,pp,rp,verify(record['checkpoint'])]);trainer(0);trainer(1);trainer(2)
    value=dict(design=DESIGN,source=art(__file__),original_identity=art(original.MANIFEST),
        reused=reused,execution=execution,folds=prior['folds'],cache=prior['cache'],selected_v78=prior['selected_v78'],
        **{k:prior[k] for k in ['public_features','feature_audit','metadata','metadata_values']},
        files=[art(p) for p in sorted(files)])
    OUT.mkdir(parents=True,exist_ok=True);write_json(MANIFEST,value)
    (OUT/'transfer_files.txt').write_text('\n'.join(sorted({r['path'] for r in value['files']}|{str(MANIFEST.relative_to(ROOT))}))+'\n')
    print(dict(frozen=True,identity=art(MANIFEST),new_fits=4,reused=2),flush=True)


def guard(full=False):
    value=read(MANIFEST);assert value['design']==DESIGN and value['source']==art(__file__)
    assert value['original_identity']==art(original.MANIFEST)
    for rec in value['files']:verify(rec)
    if full:
        for fold in value['folds'].values():verify(fold['matrix'])
        manifest=read(verify(value['cache']))
        assert manifest['rows']==value['cache']['rows'] and manifest['outputs']==value['cache']['outputs']
        assert manifest['settings']['original_sequence']==original.original.base.seq.SETTINGS
        for name,digest in manifest['outputs'].items():verify(dict(path=str((ROOT/value['cache']['path']).parent.joinpath(name).relative_to(ROOT)),sha256=digest))
    return value


def completed(month,k,identity):
    if k==0:
        original_record=original.completed(month,'tabm',read(original.MANIFEST))
        assert identity['reused'][month]['record']==art(paths(month,0)[2]);return original_record
    mp,pp,rp=paths(month,k)
    if not rp.exists():return None
    record=read(rp)
    assert record['identity']==art(MANIFEST) and record['design']==DESIGN
    assert record['seed']==SEEDS[k] and record['member']==k and record['month']==month and record['epochs']==STOP[month]
    assert record['inputs']==identity['folds'][month] and record['all_held_reloaded_exact']
    assert record['preprocessing']==identity['folds'][month]['preprocessing']
    assert {key:record[key] for key in ['host','gpu','torch']}==identity['execution']
    assert record['model']==art(mp) and record['prediction']==art(pp)
    for field in ['model','prediction','preprocessing','checkpoint']:verify(record[field])
    return record


def fit():
    identity=guard(full=True)
    assert dict(host=os.uname().nodename,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))==identity['execution']
    # Capacity is unchanged from the audited same-host seed42 experiment.
    assert read(original.OUT/'preflight.json')['identity']==art(original.MANIFEST)
    original.original.shared.recipe();meta,_=original.original.metadata.load()
    directory=(ROOT/identity['cache']['path']).parent
    index=pd.read_parquet(directory/'full_index.parquet')
    points=np.memmap(directory/'full_points.bin',mode='r',dtype=np.int32).reshape(-1,6)
    tracks=original.original.base.tensor_seq.RaggedTracks(index,points,'cuda')
    public=pd.read_parquet(verify(identity['public_features'])).set_index('mvt_id')
    for month in STOP:
        pending=[k for k in [1,2] if completed(month,k,identity) is None]
        if not pending:continue
        data=pd.read_pickle(verify(identity['folds'][month]['matrix']));names=data['baseline'];assert len(names)==len(set(names))==156
        train_ids,held_ids=np.asarray(data['train_ids']).copy(),np.asarray(data['ids']).copy()
        tm,vm=meta.loc[train_ids],meta.loc[held_ids]
        assert set(tm.month.astype(str))==set(original.original.shared.ts.FOLDS[month]) and set(vm.month.astype(str))=={month}
        assert not np.intersect1d(train_ids,held_ids).size
        np.testing.assert_array_equal(data['y'],tm.taxi_sec)
        item=dict(X=data['X'][names],Vn=data['V'][names],y=data['y'],raw=vm.reset_index(),tmonth=tm.month.astype(str).to_numpy(),meta=dict(month=month))
        del data,tm,vm;gc.collect()
        with tempfile.TemporaryDirectory(prefix='encoding_',dir=OUT) as scratch:d=original.original.adapter.ChunkedDirectData(item,'cuda',scratch)
        state=d.preprocessing();state['design']=original.original.base.DESIGN
        prepath=verify(identity['folds'][month]['preprocessing'])
        with prepath.open('rb') as f:original.original.base.preprocess_guard.equal_preprocessing(pickle.load(f),state)
        for part,ids in [('train',train_ids),('held',held_ids)]:
            extra=torch.from_numpy(original.model_type.transform(public.loc[ids])).to(d.device)
            values=getattr(d,part);setattr(d,part,(*values[:2],torch.cat([values[2],extra],dim=1)))
        del item,extra,values;gc.collect();tr,va=tracks.align(train_ids),tracks.align(held_ids)
        for k in pending:
            seed=SEEDS[k];run=OUT/f'k{k}'
            for folder in ['models','networks','trained','runtime']:(run/folder).mkdir(parents=True,exist_ok=True)
            torch.manual_seed(seed);np.random.seed(seed)
            model=original.make('tabm',d.cards,d.bins,d.train[2].shape[1]).cuda()
            mp,pp,rp=paths(month,k);receipt=run/'trained'/f'{month}_tabm.json'
            if receipt.exists():
                rec=read(receipt)
                assert rec['identity']==art(MANIFEST) and rec['design']==DESIGN and rec['inputs']==identity['folds'][month]
                assert rec['seed']==seed and rec['member']==k and rec['month']==month and rec['epochs']==STOP[month]
                assert rec['model']==art(mp) and rec['preprocessing']==art(prepath) and rec['updates_completed_before_inference']
                checkpoint=torch.load(verify(rec['model']),map_location='cuda',weights_only=False)
                assert checkpoint['identity']==art(MANIFEST) and checkpoint['design']==DESIGN
                model.load_state_dict(checkpoint['state_dict']);curve=checkpoint['curve']
                assert [v['epoch'] for v in curve]==list(range(1,STOP[month]+1))
            else:
                assert not mp.exists() and not pp.exists(),'Unrecorded artifacts: inspect instead of refitting'
                torch.cuda.reset_peak_memory_stats();curve=trainer(k)(model,d,tracks,tr,STOP[month],'tabm',month)
                tmp=mp.with_suffix('.pt.tmp');torch.save(dict(identity=art(MANIFEST),design=DESIGN,state_dict=model.state_dict(),curve=curve),tmp);tmp.replace(mp)
                write_json(receipt,dict(identity=art(MANIFEST),design=DESIGN,month=month,member=k,seed=seed,epochs=STOP[month],
                    inputs=identity['folds'][month],model=art(mp),preprocessing=art(prepath),updates_completed_before_inference=True))
            prediction=original.predict(model,d,tracks,va);assert np.isfinite(prediction).all() and (prediction>=0).all()
            if pp.exists():
                saved=pd.read_parquet(pp);np.testing.assert_array_equal(saved.mvt_id,held_ids);np.testing.assert_array_equal(saved.base,prediction.astype('float32'))
            else:original.original.shared.nni.save_network(pp,held_ids,prediction)
            restored=original.make('tabm',d.cards,d.bins,d.train[2].shape[1]).cuda()
            restored.load_state_dict(torch.load(mp,map_location='cuda',weights_only=False)['state_dict'])
            actual=original.predict(restored,d,tracks,va);np.testing.assert_array_equal(actual,prediction)
            write_json(rp,dict(identity=art(MANIFEST),design=DESIGN,month=month,member=k,seed=seed,epochs=STOP[month],
                inputs=identity['folds'][month],model=art(mp),prediction=art(pp),preprocessing=art(prepath),checkpoint=art(receipt),
                train_rows=len(train_ids),held_rows=len(held_ids),curve=curve,all_held_reloaded_exact=True,
                host=os.uname().nodename,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__)))
            original.original.shared.log(f'{month} TabM seed {seed}: saved and every held row reloaded exactly')
            del model,restored,prediction,actual;gc.collect();torch.cuda.empty_cache()
        del d,tr,va;gc.collect();torch.cuda.empty_cache()
    write_json(OUT/'audit.json',dict(status='pass',identity=art(MANIFEST),members=[art(paths(m,k)[2]) for m in STOP for k in range(3)],
        original_seed42_reused_exactly=True,full_input_hashes_verified=True,original_preprocessing_exact=True,all_held_reload_exact=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','fit'])
    torch.set_num_threads(2)
    with threadpool_limits(2):globals()[parser.parse_args().stage]()
