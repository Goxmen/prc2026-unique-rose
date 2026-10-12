"""Fixed zero-initialized full-day peer encoder, one selected TabM member.

Original row permutation, optimizer, individual-head loss, batches, schedules,
physical tracks and public inputs remain unchanged. No capacity or gate search.
"""
import argparse
import ast
import gc
import inspect
import os
import pickle
import textwrap
import time
import types
from pathlib import Path

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_tabm_pool3 as parent
import nn_public_tabm_daily_cohort_model as model_type
import daily_cohort_bank as bank

ROOT = parent.ROOT
OUT = ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_daily_cohort'
MANIFEST = OUT/'identity.json'
art, read, verify, write = parent.art, parent.read, parent.verify, parent.write_json
DESIGN = dict(parent='v80 selected three TabM seeds and same-date runway evidence',
    change='zero-initialized complete airport/UTC-day input-only peer encoder for seed42 only',
    architecture=model_type.SETTINGS, seed=42, epochs=parent.STOP, schedule_epochs=24,
    effective_batch=parent.original.BATCH, microbatch=parent.original.MICRO,
    inference_chunk=parent.original.PRED_CHUNK, retained_pool_members=[1042, 2042],
    control='same-laptop saved seed42; replay every held prediction plus zero-branch outputs before each fit',
    cohort_identity='existing UTC takeoff-day feature grouping; exact published cohort/calendar identity not established',
    scaler='old training-only preprocessing exact; donor fixed physical transforms with no fitted statistics',
    randomness='unchanged initialization, row-permutation/dropout streams; no added stochastic operation',
    training='unchanged original train AST; only nonrandom scalar packet added through track adapter',
    primary='one replacement in three-TabM native component; integer component delta once onto exact v80',
    gate='both months improve and paired-day95% upper endpoint below0 versus exactv80',
    continuation='passing fixed comparison required before any extra seeds; otherwise close',
    source_scope='training donor day belongs to exact allowed months; held donor day belongs only to held month',
    caution='repeatedly reused development data; no independent confirmation',
    no_loss_change=True, no_weight_search=True, no_epoch_selection=True,
    no_parameter_scan=True, no_production=True, no_upload=True)


def clone(function, **values):
    return types.FunctionType(function.__code__, dict(function.__globals__, **values),
        function.__name__, function.__defaults__, function.__closure__)


def make(arm, cards, bins, n_plain):
    assert arm == 'tabm'
    return model_type.DailyTabM(cards, bins, n_plain)


def paths(month, k):
    assert k == 0
    run = OUT/'k0'; stem = f'{month}_tabm'
    return run/'models'/f'{stem}.pt', run/'networks'/f'{stem}.parquet', run/'networks'/f'{stem}.json'


def completed(month, k, identity):
    mp, pp, rp = paths(month, k)
    if not rp.exists(): return None
    r = read(rp)
    assert r['identity'] == art(MANIFEST) and r['design'] == DESIGN and r['seed'] == 42
    assert r['member'] == 0 and r['month'] == month and r['epochs'] == parent.STOP[month]
    assert r['inputs'] == identity['folds'][month] and r['all_held_reloaded_exact']
    assert r['model'] == art(mp) and r['prediction'] == art(pp)
    assert {key:r[key] for key in ['host','gpu','torch']} == identity['execution']
    for key in ['model','prediction','preprocessing','checkpoint']: verify(r[key])
    replay = read(OUT/'control_replay'/f'{month}.json')
    assert replay['identity'] == art(MANIFEST) and replay['all_held_old_and_zero_branch_exact']
    return r


def make_tracks(index, points, device):
    old = parent.original.original.base.tensor_seq.RaggedTracks(index, points, device)
    return bank.Tracks(old, index, device)


class OldAdapter(torch.nn.Module):
    def __init__(self, original):
        super().__init__(); self.original = original
    def forward(self, *values):
        return self.original(*values[:6])


def replay_control(d, tracks, rows, month):
    """Same-host prediction replay and zero-projection replay, without RNG drift."""
    identity = read(MANIFEST)
    target = OUT/'control_replay'/f'{month}.json'
    train_ids = tracks.ids[np.asarray(rows)]
    held_rows = tracks.alignments[-1]
    held_ids = tracks.ids[held_rows]
    scopes = dict(train=tracks.bank.scope(train_ids, parent.original.original.shared.ts.FOLDS[month]),
        held=tracks.bank.scope(held_ids, [month]))
    control = parent.completed(month, 0, read(parent.MANIFEST))
    if target.exists():
        prior = read(target)
        assert prior['identity'] == art(MANIFEST) and prior['scope'] == scopes
        assert prior['control_record'] == art(parent.paths(month, 0)[2])
        assert prior['all_held_old_and_zero_branch_exact']
        return
    cpu_rng = torch.get_rng_state().clone(); cuda_rng = torch.cuda.get_rng_state().clone()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        old = parent.original.model_type.PublicTabM(d.cards, d.bins, d.train[2].shape[1]).cuda()
        checkpoint = torch.load(verify(control['model']), map_location='cuda', weights_only=False)
        old.load_state_dict(checkpoint['state_dict'])
        zero = make('tabm', d.cards, d.bins, d.train[2].shape[1]).cuda()
        mismatch = zero.load_state_dict(checkpoint['state_dict'], strict=False)
        assert not mismatch.unexpected_keys
        assert set(mismatch.missing_keys) == {name for name in zero.state_dict()
            if name.startswith('donor.') or name.startswith('context_projection.')}
        expected = pd.read_parquet(verify(control['prediction'])).set_index('mvt_id').loc[held_ids, 'base'].to_numpy()
        actual = parent.original.predict(OldAdapter(old), d, tracks, held_rows)
        np.testing.assert_array_equal(actual.astype(np.float32), expected)
        candidate = parent.original.predict(zero, d, tracks, held_rows)
        np.testing.assert_array_equal(candidate, actual)
        del old, zero, checkpoint, actual, candidate, expected
    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
    gc.collect(); torch.cuda.empty_cache()
    write(target, dict(identity=art(MANIFEST), scope=scopes, control_record=art(parent.paths(month, 0)[2]),
        all_held_old_and_zero_branch_exact=True, original_random_state_restored=True, rows=len(held_ids)))


def trainer(k):
    assert k == 0
    original = clone(parent.trainer, OUT=OUT, SEEDS=[42])(0)
    def train(model, d, tracks, rows, epochs, arm, month):
        replay_control(d, tracks, rows, month)
        return original(model, d, tracks, rows, epochs, arm, month)
    return train


def training_function():
    source = textwrap.dedent(inspect.getsource(parent.fit)); tree = ast.parse(source)
    class Select(ast.NodeTransformer):
        lists=0; ranges=0; flags=0
        def visit_List(self, node):
            if ast.dump(node) == ast.dump(ast.parse('[1,2]', mode='eval').body):
                self.lists += 1
                return ast.copy_location(ast.parse('[0]', mode='eval').body, node)
            return self.generic_visit(node)
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value==3:
                node.args[0]=ast.Constant(value=1); self.ranges+=1
            for arg in node.keywords:
                if arg.arg=='original_seed42_reused_exactly':
                    arg.arg='retained_controls_not_refitted'; self.flags+=1
            return node
    edit=Select(); tree=edit.visit(tree); assert (edit.lists,edit.ranges,edit.flags)==(1,1,1)
    restored=ast.unparse(tree).replace('pending = [k for k in [0]', 'pending = [k for k in [1, 2]')
    restored=restored.replace('for k in range(1)', 'for k in range(3)').replace('retained_controls_not_refitted=', 'original_seed42_reused_exactly=')
    assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(source))
    original=types.SimpleNamespace(**vars(parent.original)); original.make=make
    original.original=types.SimpleNamespace(**vars(parent.original.original))
    original.original.base=types.SimpleNamespace(**vars(parent.original.original.base))
    original.original.base.tensor_seq=types.SimpleNamespace(**vars(parent.original.original.base.tensor_seq))
    original.original.base.tensor_seq.RaggedTracks=make_tracks
    ns=dict(parent.fit.__globals__,original=original,OUT=OUT,MANIFEST=MANIFEST,DESIGN=DESIGN,
        SEEDS=[42],guard=guard,completed=completed,paths=paths,trainer=trainer)
    exec(compile(ast.fix_missing_locations(tree),'<fixed-daily-cohort-administrative-wrapper>','exec'),ns)
    return ns['fit']


def cpu_check():
    training_function(); trainer(0)
    result=dict(source=art(__file__),model_source=art(model_type.__file__),bank_source=art(bank.__file__),
        model=model_type.check('cpu'), bank=bank.checks(), training_AST_restores_exactly=True,
        original_microbatch=parent.original.MICRO, original_predict_chunk=parent.original.PRED_CHUNK)
    target=OUT/'cpu_checks.json'
    if target.exists(): assert read(target)==result
    else: write(target,result)
    print(result,flush=True)


def freeze():
    assert not MANIFEST.exists()
    cpu=read(OUT/'cpu_checks.json')
    assert cpu['source']==art(__file__) and cpu['model_source']==art(model_type.__file__) and cpu['bank_source']==art(bank.__file__)
    prior=parent.guard(); packed=bank.guard(full=True)
    assert read(parent.OUT/'audit.json')['status']=='pass'
    bank_audit=ROOT/'VALIDATION/reports/cohort_learning_20261012/independent_bank_audit.json'
    assert read(bank_audit)['status']=='pass' and read(bank_audit)['all_aliases_independent_SQL']
    training_function(); trainer(0)
    files=[Path(__file__),Path(model_type.__file__),Path(bank.__file__),OUT/'cpu_checks.json',
        bank.OUT/'manifest.json',parent.MANIFEST,parent.OUT/'audit.json',Path(parent.__file__),
        ROOT/'VALIDATION/nn_public_tabm_daily_cohort_score.py',bank_audit,
        ROOT/'VALIDATION/tools/audit_daily_cohort_bank_20261012.py']
    files += [bank.verify(r) for r in packed['outputs']]
    controls={}
    for month in parent.STOP:
        controls[month]=[]
        for k in range(3):
            r=parent.completed(month,k,prior); assert r is not None
            controls[month].append(art(parent.paths(month,k)[2]))
            files += [verify(r['model']),verify(r['prediction']),parent.paths(month,k)[2]]
    reference=ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_other_runway_combo'
    files += [reference/n for n in ['identity.json','result.json','scored.parquet']]
    records={a['path']:a for a in [*prior['files'],*[art(p) for p in files]]}
    value=dict(prior,design=DESIGN,source=art(__file__),files=list(records.values()),controls=controls,
        parent_identity=art(parent.MANIFEST),donor_bank=art(bank.OUT/'manifest.json'),
        donor_bank_outputs=packed['outputs'],checks=cpu)
    write(MANIFEST,value)
    (OUT/'transfer_files.txt').write_text('\n'.join(sorted(set(records)|{str(MANIFEST.relative_to(ROOT))}))+'\n')
    print(dict(frozen=True,new_fits=2,retained_controls=6,epochs=parent.STOP,files=len(records)),flush=True)


def guard(full=False):
    m=read(MANIFEST); assert m['design']==DESIGN and m['source']==art(__file__)
    assert m['parent_identity']==art(parent.MANIFEST)
    for a in m['files']: verify(a)
    assert m['donor_bank']==art(bank.OUT/'manifest.json')
    if full:
        for fold in m['folds'].values(): verify(fold['matrix'])
        c=read(verify(m['cache']))
        for name,digest in c['outputs'].items(): verify(dict(path=str(Path(m['cache']['path']).parent/name),sha256=digest))
    return m


def preflight():
    m=guard(full=True); assert not (OUT/'preflight.json').exists()
    assert dict(host=os.uname().nodename,gpu=torch.cuda.get_device_name(),torch=str(torch.__version__))==m['execution']
    original=parent.original; packed=read(bank.OUT/'manifest.json')
    checks=[model_type.check('cpu'),model_type.check('cuda')]; bench=[]
    maximum=packed['dimensions']['values'][1]
    bank_bytes=int(np.prod(packed['dimensions']['values'])*4+np.prod(packed['dimensions']['mask']))
    track_bytes=int(m['cache']['rows']*(7*8+9*4))
    for month in parent.STOP:
        with verify(m['folds'][month]['preprocessing']).open('rb') as f: pre=pickle.load(f)
        rows=m['folds'][month]['rows']+m['folds'][month]['held_rows']
        reserve=rows*(8*len(pre['cards'])+4*(len(pre['ple'])+len(pre['plain'])+12)+4)
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        storage=torch.empty(reserve+bank_bytes+track_bytes,dtype=torch.uint8,device='cuda')
        torch.manual_seed(42); model=make('tabm',pre['cards'],pre['bins'],len(pre['plain'])+12).cuda().train()
        tab,track=original.original.base.prototype.synthetic(pre['cards'],pre['bins'],len(pre['plain'])+12,original.MICRO,'cuda')
        track=(track[0],torch.ones_like(track[1]),track[2]); target=torch.zeros(original.MICRO,device='cuda')
        packet=(torch.zeros((original.MICRO,maximum,bank.DIM),device='cuda'),
            torch.ones((original.MICRO,maximum),dtype=torch.bool,device='cuda'),
            torch.arange(original.MICRO,device='cuda'),torch.zeros((original.MICRO,1),dtype=torch.long,device='cuda'))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0003)
        times=[]
        for _ in range(3):
            torch.cuda.synchronize(); start=time.monotonic(); optimizer.zero_grad(set_to_none=True)
            for _ in range(original.BATCH//original.MICRO):
                original.model_type.loss(model(*tab,*track,packet),target).div(original.BATCH//original.MICRO).backward()
            optimizer.step(); torch.cuda.synchronize(); times.append(time.monotonic()-start)
        peak=torch.cuda.max_memory_allocated(); assert peak < 7*(1<<30), 'Reject, do not resize after freeze'
        model.eval()
        with torch.no_grad(): assert torch.isfinite(model(*tab,*track,packet)).all()
        bench.append(dict(month=month,peak_allocated_bytes=peak,step_seconds=float(np.mean(times[1:])),
            donor_groups=original.MICRO,donors_each=maximum,scalar_bank_reserved=bank_bytes,
            original_matrix_bytes_reserved=reserve,original_track_geometry_quality_reserved=track_bytes,
            training_and_inference_width=original.MICRO))
        del storage,model,tab,track,packet,target,optimizer; gc.collect(); torch.cuda.empty_cache()
    write(OUT/'preflight.json',dict(identity=art(MANIFEST),checks=checks,capacity=bench,training_ast_exact=True))
    print(dict(preflight='pass',capacity=bench),flush=True)


def fit():
    assert read(OUT/'preflight.json')['identity']==art(MANIFEST)
    training_function()()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['cpu_check','freeze','preflight','fit'])
    torch.set_num_threads(2)
    with threadpool_limits(2): globals()[ap.parse_args().stage]()
