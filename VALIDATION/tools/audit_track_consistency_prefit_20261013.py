"""Independent fixed-consistency prefit audit; never trains or runs inference.

The actual held replay is established by immutable laptop receipts. Here its
saved standardized means are independently converted to the old float32 output,
and every training/held ID, month, target scale and eligibility count is checked.
Only training labels in the existing fold pickle are decoded. Metadata is
projected to IDs/month; no raw held label is read. One fold is resident at a time.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

OUT=ROOT/'VALIDATION/runs/v3_20261013_nn_public_tabm_track_consistency'
REPORT=ROOT/'VALIDATION/reports/cohort_learning_20261013/track_consistency_independent_fixture.json'
RUNNER=ROOT/'VALIDATION/nn_public_tabm_track_consistency.py'
VIEW=ROOT/'VALIDATION/track_consistency_view.py'
PROTOCOL=ROOT/'VALIDATION/reports/cohort_learning_20261013/target_track_consistency_protocol.json'
FOLDS={'2025-01':[f'2025-{m:02d}' for m in range(3,12)],
       '2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
ROWS={'2025-01':(1596201,153706),'2025-07':(1140233,190713)}
VERIFIED={}


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def art(p):
    p=Path(p).resolve();return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))


def verify(a):
    p=(ROOT/a['path']).resolve();assert p.is_relative_to(ROOT)
    if a['path'] in VERIFIED:assert VERIFIED[a['path']]==a['sha256']
    else:
        assert sha(p)==a['sha256'],a['path'];VERIFIED[a['path']]=a['sha256']
    if 'bytes' in a:assert p.stat().st_size==a['bytes']
    return p


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)


def digest(ids):return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()


def check():
    """All129 lengths, independent scalar indices and analytic mean-loss gradient."""
    import track_consistency_view as v
    torch.set_num_threads(1)
    counts=np.arange(129);points=torch.arange(129*128*16,dtype=torch.float32).reshape(129,128,16)/100
    mask=torch.arange(128)[None,:]<torch.from_numpy(counts)[:,None]
    quality=torch.arange(129*9,dtype=torch.float32).reshape(129,9)/100
    actual=v.thin(points,mask,quality)
    for i,n in enumerate(counts):
        k=max(2,(int(n)+1)//2) if n>=3 else int(n)
        indices=[j*(int(n)-1)//max(1,k-1) for j in range(k)]
        assert actual[1][i].sum().item()==k
        assert torch.equal(actual[0][i,:k],points[i,indices])
        assert torch.count_nonzero(actual[0][i,k:])==0
        assert torch.equal(actual[2][i,[0,2,3,4,5,6,7,8]],quality[i,[0,2,3,4,5,6,7,8]])
        assert actual[2][i,1].item()==k/128
    # Small differentiable32-head model: no inference through the real backbone.
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3,dtype=torch.float64));self.dropout=torch.nn.Dropout(.5)
        def forward(self,x,points,mask,quality):
            value=self.dropout(x)*self.weight
            return value[:,None].expand(-1,32)
    model=Toy().train();model.dropout.eval()
    x=torch.tensor([1.,2.,3.],dtype=torch.float64);target=torch.tensor([.5,-1.,.3],dtype=torch.float64)
    flags=[m.training for m in model.modules()];state=torch.get_rng_state().clone()
    track=tuple(a[:3] for a in (points,mask,quality))
    loss=v.consistency_loss(model,(x,),track,target);loss.backward()
    expected=np.mean((.3*x.numpy()-target.numpy())**2)
    expected_grad=2*np.mean(x.numpy()*(.3*x.numpy()-target.numpy()))
    np.testing.assert_allclose(loss.item(),expected,atol=1e-14,rtol=0)
    np.testing.assert_allclose(model.weight.grad.item(),expected_grad,atol=1e-14,rtol=0)
    assert torch.equal(state,torch.get_rng_state()) and flags==[m.training for m in model.modules()]
    try:
        with v.eval_preserving_state(model):
            torch.rand(3);raise RuntimeError('fixture')
    except RuntimeError:pass
    assert torch.equal(state,torch.get_rng_state()) and flags==[m.training for m in model.modules()]
    eligible=np.array([2,4,7,11,13]);sampler=v.CycleSampler(eligible)
    expected_rng=np.random.Generator(np.random.PCG64(20261013))
    expected=np.concatenate([expected_rng.permutation(eligible) for _ in range(4)])[:17]
    np.testing.assert_array_equal(np.r_[sampler.take(4),sampler.take(13)],expected)
    write(REPORT,dict(status='pass',source=art(__file__),view=art(VIEW),all_129_lengths_scalar_exact=True,
        analytic_mean_loss_and_gradient_exact=True,exception_restores_all_module_flags_and_rng=True,
        independent_PCG64_cycle_exact=True,no_real_data_or_model_inference=True))
    print(json.dumps(dict(status='pass',report=art(REPORT))))


def audit():
    identity_path=OUT/'identity.json';identity=read(identity_path)
    assert identity['source']==art(RUNNER) and identity['protocol']==art(PROTOCOL)
    assert identity['auditor']==art(__file__)
    for a in identity['files']:verify(a)
    source_review=read(verify(identity['source_review']))
    assert source_review['status']=='pass' and source_review['source']==art(RUNNER)
    assert source_review['view']==art(VIEW) and source_review['auditor']==art(__file__)
    assert source_review['independent_fixture']==art(REPORT)
    fixture=read(REPORT)
    assert fixture['status']=='pass' and fixture['source']==art(__file__) and fixture['view']==art(VIEW)
    design=identity['design']
    assert design['arms']==['supervised_control','training_consistency','target_consistency']
    for key,value in dict(epochs=2,lr=3e-5,weight_decay=.0003,batch=2048,microbatch=512,
                          consistency_weight=1.,consistency_batch=512,seed=42,sampling_seed=20261013).items():
        assert design[key]==value,(key,design[key])
    assert design['no_hpo'] and design['no_stopping'] and design['no_extra_seed'] and design['no_upload']
    cpu=read(OUT/'cpu_checks.json');pref=read(OUT/'preflight.json')
    assert cpu['source']==art(RUNNER) and cpu['view']==art(VIEW) and cpu['protocol']==art(PROTOCOL)
    assert cpu['poison']['held_label_poison_and_removal_invariant']
    assert cpu['zero_parity']['zero_weight_output_gradient_rng_update_exact']
    assert pref['status']=='pass' and pref['identity']==art(identity_path)
    assert pref['execution']==identity['execution'] and pref['cpu_checks']==art(OUT/'cpu_checks.json')
    comparison_path=OUT/'comparison/identity.json';comparison=read(comparison_path)
    assert comparison['before_fitting'] and comparison['protocol']==art(PROTOCOL)
    assert art(identity_path) in comparison['sources']
    for a in comparison['sources']:verify(a)
    assert pref['restored'] and pref['deterministic_cudnn_check_only'] and pref['no_candidate_fit']
    assert len(pref['original_backend_flags'])==2
    assert len(pref['checks'])==2 and all(x['device']=='cuda' for x in pref['checks'])
    assert pref['checks'][1]['zero_weight_output_gradient_rng_update_exact']
    required=['ordered_selection_exact','endpoints_retained','point_values_and_clocks_exact','raw_descriptors_exact',
        'retained_slots_updated','selected_physical_adjacency_recomputed','preceding_raw_gap_preserved',
        'batch_permutation_padding_invariance','sampler_chunk_invariance','eval_flags_and_rng_restored',
        'finite_nonzero_student_gradients','teacher_no_gradients']
    for checks in [cpu['view_check'],pref['checks'][0]]:
        assert all(checks[k] for k in required) and math.isfinite(checks['loss']) and checks['loss']>=0
    assert len(pref['teachers'])==2 and len(pref['capacity'])==2
    assert {x['month'] for x in pref['capacity']}==set(FOLDS)
    assert not list(OUT.glob('*/models/*.pt')),'Independent prefit gate must predate candidate fitting'
    cache=read(verify(identity['cache']));folder=(ROOT/identity['cache']['path']).parent
    assert cache['outputs']==identity['cache']['outputs'] and cache['rows']==2085047
    for name,h in cache['outputs'].items():verify(dict(path=str((folder/name).relative_to(ROOT)),sha256=h))
    index=pd.read_parquet(folder/'full_index.parquet',columns=['mvt_id','count','start']).set_index('mvt_id')
    assert len(index)==2085047 and index.index.is_unique and index['count'].between(0,128).all()
    assert (index.start>=0).all() and (index.start+index['count']<=cache['stored_points']).all()
    metadata=pd.read_parquet(verify(identity['metadata_values']),columns=['mvt_id','month']).set_index('mvt_id')
    assert metadata.index.is_unique
    records=[]
    for month in FOLDS:
        fold=identity['folds'][month]
        # Independently decode the immutable original matrix once, rather than
        # trusting scope assertions or rereading unavailable held truth.
        data=pd.read_pickle(verify(fold['matrix']))
        assert set(data)=={'X','V','y','train_ids','ids','baseline'}
        ids=np.asarray(data['train_ids'],dtype=np.int64);held=np.asarray(data['ids'],dtype=np.int64)
        assert (len(ids),len(held))==ROWS[month] and (fold['rows'],fold['held_rows'])==ROWS[month]
        assert len(np.unique(ids))==len(ids) and len(np.unique(held))==len(held)
        assert not np.intersect1d(ids,held).size
        assert sorted(metadata.loc[ids,'month'].astype(str).unique())==sorted(FOLDS[month])
        assert list(metadata.loc[held,'month'].astype(str).unique())==[month]
        cols=data['baseline'];assert len(cols)==len(set(cols))==156
        assert not set(cols)&{'taxi_sec','TAXITIME_SEC_mvt','BLOCK_TIME_UTC_mvt'}
        assert len(data['X'])==len(ids) and len(data['V'])==len(held)
        assert set(cols).issubset(data['X'].columns) and set(cols).issubset(data['V'].columns)
        y=np.asarray(data['y'],dtype=float);assert y.shape==(len(ids),) and np.isfinite(y).all()
        target=np.minimum(y,10800);mu=float(target.mean());sd=float(target.std());assert sd>0
        with verify(fold['preprocessing']).open('rb') as f:pre=pickle.load(f)
        assert pre['mu']==mu and pre['sd']==sd and pre['columns']==cols
        tp=OUT/'teacher'/month/'receipt.json';record=read(tp)
        assert art(tp) in pref['teachers'] and record['month']==month and record['identity']==art(identity_path)
        assert record['inputs']==fold and record['preprocessing']==fold['preprocessing']
        control=read(verify(identity['controls'][month][0]))
        assert record['model']==control['model'] and record['control_prediction']==control['prediction']
        assert record['mu']==mu and record['sd']==sd
        for k in ['all_held_original_exact','repeated_first_chunk_exact','teacher_frozen','full_inputs','unclipped_standardized_mean']:assert record[k]
        scope=dict(train_rows=len(ids),held_rows=len(held),train_ids_sha256=digest(ids),held_ids_sha256=digest(held),
            train_months=sorted(FOLDS[month]),held_months=[month],held_labels_absent_from_cache=True,
            held_metadata_columns=['mvt_id','month'],adapter_targets_all_nan=True)
        assert record['scope']==scope
        eligible={k:int((index.loc[x,'count']>=3).sum()) for k,x in [('train',ids),('held',held)]}
        assert record['eligible']==eligible and min(eligible.values())>0
        for key,n in [('train_mean',len(ids)),('held_mean',len(held))]:
            arr=np.load(verify(record[key]),mmap_mode='r',allow_pickle=False)
            assert arr.shape==(n,) and arr.dtype==np.float32 and np.isfinite(arr).all()
            del arr
        means=np.load(verify(record['held_mean']),mmap_mode='r',allow_pickle=False)
        original=pd.read_parquet(verify(record['control_prediction']))
        np.testing.assert_array_equal(original.mvt_id.to_numpy(np.int64),held)
        np.testing.assert_array_equal(original.base.to_numpy(),np.maximum(means.astype(float)*sd+mu,0).astype(np.float32))
        capacity=next(x for x in pref['capacity'] if x['month']==month)
        assert capacity['actual_train_held_tensors_and_track_geometry_reserved']
        assert 0<capacity['peak_allocated_bytes']<7*(1<<30)
        assert capacity['logical_update_seconds']>0 and capacity['input_scope']==scope and capacity['eligible']==eligible
        np.testing.assert_allclose(capacity['estimated_single_arm_minutes'],capacity['logical_update_seconds']*2*math.ceil(len(ids)/2048)/60,atol=1e-12,rtol=0)
        records.append(dict(month=month,teacher=art(tp),scope=scope,eligible=eligible,
            target_mu=mu,target_sd=sd,all_held_float32_teacher_replay_exact=True,
            contiguous_prefix_masks_proved_by_frozen_RaggedTracks_source=True,peak_allocated_bytes=capacity['peak_allocated_bytes']))
        del data,ids,held,y,target,pre,means,original;gc.collect()
    write(OUT/'prefit_audit.json',dict(status='pass',identity=art(identity_path),preflight=art(OUT/'preflight.json'),
        teachers=pref['teachers'],source=art(__file__),fixture=art(REPORT),folds=records,
        comparison=art(comparison_path),
        verified_files=len(VERIFIED),no_held_label_column_decoded=True,no_fit_or_model_inference=True,
        GPU_replay_and_preprocessing_execution='Original immutable laptop receipts; saved means independently checked here',
        limitation='Branch-view regularization and transductive reused development; no sparse-sensor invariance claim'))
    print(json.dumps(dict(status='pass',audit=art(OUT/'prefit_audit.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['check','audit'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
