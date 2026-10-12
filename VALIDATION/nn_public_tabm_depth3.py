"""One fixed capacity screen: three TabM blocks instead of two.

Same seed42, inputs, physical encoder, K32 heads, optimizer, batches, and
21/17 stopping points. Retain the two other selected TabM members unchanged.
No architecture search, production or uploads. Sources and selection are
frozen before either held-out prediction exists.
"""
import argparse,ast,copy,gc,inspect,os,pickle,textwrap,types
from pathlib import Path
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import nn_public_tabm_pool3 as parent

ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_depth3'
MANIFEST=OUT/'identity.json'
art,read,verify,write=parent.art,parent.read,parent.verify,parent.write_json
SETTINGS=dict(parent.original.model_type.SETTINGS,n_blocks=3)
DESIGN=dict(parent='v80 equal three-TabM pool and selected runway evidence',
    architecture=SETTINGS,change='n_blocks2 to3 for seed42; retain1042/2042 unchanged',
    seed=42,epochs=parent.STOP,schedule_epochs=24,effective_batch=parent.original.BATCH,
    microbatch=parent.original.MICRO,inference_chunk=parent.original.PRED_CHUNK,
    primary='Exact v80 with only this one member replaced; unchanged stack and runway stage',
    gate='Improve both months and paired-day95% upper endpoint below0 versus exactv80',
    continuation='Only a passing fixed comparison may justify confirmation with other seeds',
    caveat='Reused Jan/Jul development data; one architecture comparison, no independent confirmation',
    no_epoch_selection=True,no_new_features=True,no_new_weight=True,no_production=True,no_upload=True)


def clone(function,**values):
    return types.FunctionType(function.__code__,dict(function.__globals__,**values),
        function.__name__,function.__defaults__,function.__closure__)


class DeepTabM(parent.original.model_type.PublicTabM):
    # Keeping the original closure makes its zero-argument super() initialize
    # nn.Module for this subclass. No retained module globals are modified.
    __init__=clone(parent.original.model_type.PublicTabM.__init__,SETTINGS=SETTINGS)


def make(arm,cards,bins,n_plain):
    assert arm=='tabm'
    return DeepTabM(cards,bins,n_plain)


def paths(month,k):
    assert k==0
    run=OUT/'k0';stem=f'{month}_tabm'
    return run/'models'/f'{stem}.pt',run/'networks'/f'{stem}.parquet',run/'networks'/f'{stem}.json'


def completed(month,k,identity):
    mp,pp,rp=paths(month,k)
    if not rp.exists():return None
    r=read(rp)
    assert r['identity']==art(MANIFEST) and r['design']==DESIGN and r['seed']==42
    assert r['member']==0 and r['month']==month and r['epochs']==parent.STOP[month]
    assert r['inputs']==identity['folds'][month] and r['all_held_reloaded_exact']
    assert r['model']==art(mp) and r['prediction']==art(pp)
    assert {key:r[key] for key in ['host','gpu','torch']}==identity['execution']
    for key in ['model','prediction','preprocessing','checkpoint']:verify(r[key])
    return r


def trainer(k):
    return clone(parent.trainer,OUT=OUT,SEEDS=[42])(k)


def training_function():
    source=textwrap.dedent(inspect.getsource(parent.fit));tree=ast.parse(source)
    class Select(ast.NodeTransformer):
        lists=0;ranges=0;flags=0
        def visit_List(self,node):
            if ast.dump(node,include_attributes=False)==ast.dump(ast.parse('[1,2]',mode='eval').body,include_attributes=False):
                self.lists+=1;return ast.copy_location(ast.parse('[0]',mode='eval').body,node)
            return self.generic_visit(node)
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==3:
                node.args[0]=ast.Constant(value=1);self.ranges+=1
            for arg in node.keywords:
                if arg.arg=='original_seed42_reused_exactly':
                    arg.arg='retained_controls_not_refitted';self.flags+=1
            return node
    edit=Select();tree=edit.visit(tree);assert (edit.lists,edit.ranges,edit.flags)==(1,1,1)
    # Reverse all three administrative edits and prove the training body unchanged.
    restored=ast.unparse(tree).replace('pending = [k for k in [0]', 'pending = [k for k in [1, 2]')
    restored=restored.replace('for k in range(1)', 'for k in range(3)').replace('retained_controls_not_refitted=', 'original_seed42_reused_exactly=')
    assert ast.dump(ast.parse(restored),include_attributes=False)==ast.dump(ast.parse(source),include_attributes=False)
    original=types.SimpleNamespace(**vars(parent.original));original.make=make
    ns=dict(parent.fit.__globals__,original=original,OUT=OUT,MANIFEST=MANIFEST,DESIGN=DESIGN,
        SEEDS=[42],guard=guard,completed=completed,paths=paths,trainer=trainer)
    exec(compile(ast.fix_missing_locations(tree),'<fixed-depth3-administrative-wrapper>','exec'),ns)
    return ns['fit']


def freeze():
    assert not MANIFEST.exists()
    prior=parent.guard();assert read(parent.OUT/'audit.json')['status']=='pass'
    training_function();trainer(0)
    assert parent.original.model_type.SETTINGS['n_blocks']==2 and SETTINGS['n_blocks']==3
    assert {k:v for k,v in SETTINGS.items() if k!='n_blocks'}=={k:v for k,v in parent.original.model_type.SETTINGS.items() if k!='n_blocks'}
    paths_=[Path(__file__),parent.MANIFEST,parent.OUT/'audit.json',Path(parent.__file__),Path(parent.original.__file__),Path(parent.original.model_type.__file__)]
    controls={}
    for month in parent.STOP:
        controls[month]=[]
        for k in range(3):
            r=parent.completed(month,k,prior);assert r is not None
            controls[month].append(art(parent.paths(month,k)[2]));paths_+=[verify(r['model']),verify(r['prediction']),parent.paths(month,k)[2]]
    reference=ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo'
    paths_ += [reference/n for n in ['identity.json','result.json','scored.parquet']]
    files={a['path']:a for a in [*prior['files'],*[art(p) for p in paths_]]}
    value=dict(prior,design=DESIGN,source=art(__file__),files=list(files.values()),controls=controls,
        parent_identity=art(parent.MANIFEST),checks=dict(training_AST_restores_exactly=True,only_architecture_setting_changed='n_blocks'))
    OUT.mkdir(parents=True,exist_ok=True);write(MANIFEST,value)
    print(dict(frozen=True,new_fits=2,retained_controls=6,epochs=parent.STOP),flush=True)


def guard(full=False):
    m=read(MANIFEST);assert m['design']==DESIGN and m['source']==art(__file__)
    assert m['parent_identity']==art(parent.MANIFEST)
    for a in m['files']:verify(a)
    if full:
        for fold in m['folds'].values():verify(fold['matrix'])
        c=read(verify(m['cache']))
        for name,digest in c['outputs'].items():verify(dict(path=str(Path(m['cache']['path']).parent/name),sha256=digest))
    return m


def preflight():
    m=guard(full=True);assert not (OUT/'preflight.json').exists()
    assert dict(host=os.uname().nodename,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))==m['execution']
    original=parent.original
    check=clone(original.model_type.check,PublicTabM=DeepTabM,SETTINGS=SETTINGS)
    checks=[check('cpu'),check('cuda')];bench=[]
    for month in parent.STOP:
        with verify(m['folds'][month]['preprocessing']).open('rb') as f:pre=pickle.load(f)
        rows=m['folds'][month]['rows']+m['folds'][month]['held_rows']
        reserve=rows*(8*len(pre['cards'])+4*(len(pre['ple'])+len(pre['plain'])+12)+4)
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        storage=torch.empty(reserve,dtype=torch.uint8,device='cuda')
        torch.manual_seed(42);model=make('tabm',pre['cards'],pre['bins'],len(pre['plain'])+12).cuda().train()
        tab,track=original.original.base.prototype.synthetic(pre['cards'],pre['bins'],len(pre['plain'])+12,original.MICRO,'cuda')
        track=(track[0],torch.ones_like(track[1]),track[2]);target=torch.zeros(original.MICRO,device='cuda')
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(original.BATCH//original.MICRO):original.model_type.loss(model(*tab,*track),target).div(original.BATCH//original.MICRO).backward()
            optimizer.step()
        torch.cuda.synchronize();bench.append(dict(month=month,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        del storage,model,tab,track,target,optimizer;gc.collect();torch.cuda.empty_cache()
    write(OUT/'preflight.json',dict(identity=art(MANIFEST),checks=checks,capacity=bench,training_ast_exact=True))
    print(dict(preflight='pass',capacity=bench),flush=True)


def fit():
    assert read(OUT/'preflight.json')['identity']==art(MANIFEST)
    training_function()()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['freeze','preflight','fit'])
    torch.set_num_threads(2)
    with threadpool_limits(2):globals()[ap.parse_args().stage]()
