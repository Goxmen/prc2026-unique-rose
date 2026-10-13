"""Fixed target-input branch-view regularization; no data/labels loaded here.

Full source descriptors remain observed. Only the encoded sequence selection and
its retained-slot count change. This is not a complete sensor outage simulation.
"""
import contextlib
import copy
import math
import numpy as np
import torch
import nn_public_tabm as original

SETTINGS=dict(selection='floor(j*(n-1)/(k-1)), k=max(2,ceil(n/2)), n>=3',
    unchanged='full source descriptors, tabular inputs, selected point values and all movement clocks',
    updated_quality='retained_slots_fraction only',teacher='frozen unclipped standardized ensemble mean',
    loss='MSE of student ensemble mean versus teacher mean',weight=1.,consistency_batch=512,
    sampler='PCG64 seed20261013, permutation cycles, fixed draw count')


def thin(points, mask, quality):
    assert points.ndim==3 and points.shape[1:]==(128,16)
    assert mask.shape==points.shape[:2] and quality.shape==(len(points),9)
    n=mask.sum(1)
    k=torch.where(n>=3,((n+1)//2).clamp(min=2),n)
    slot=torch.arange(128,device=points.device)[None,:]
    selected=(slot*(n[:,None]-1).clamp(min=0))//(k[:,None]-1).clamp(min=1)
    valid=slot<k[:,None]
    selected=selected.clamp(0,127)
    value=points.gather(1,selected[:,:,None].expand(-1,-1,16))*valid[:,:,None]
    q=quality.clone();q[:,1]=k.to(q.dtype)/128
    return value,valid,q


@contextlib.contextmanager
def eval_preserving_state(model):
    flags=[(module,module.training) for module in model.modules()]
    parameter=next(model.parameters())
    devices=[parameter.device.index if parameter.device.index is not None else torch.cuda.current_device()] if parameter.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        model.eval()
        try:yield
        finally:
            for module,flag in flags:module.training=flag


def consistency_loss(model, tab, track, teacher):
    assert teacher.ndim==1 and not teacher.requires_grad
    with eval_preserving_state(model):
        prediction=original.mean(model(*tab,*thin(*track)))
    return torch.nn.functional.mse_loss(prediction,teacher)


def backward_consistency(model, tab, track, teacher, weight=1.):
    # The zero arm is a strict no-op used by the preflight parity check.
    # Production calls always use the frozen weight1; no runtime tuning option.
    if weight==0:return 0.
    assert weight==1.
    loss=consistency_loss(model,tab,track,teacher)
    assert torch.isfinite(loss)
    loss.backward()
    return float(loss.detach())


class CycleSampler:
    def __init__(self, eligible, seed=20261013):
        self.eligible=np.asarray(eligible,dtype=np.int64)
        assert len(self.eligible)>0 and len(np.unique(self.eligible))==len(self.eligible)
        self.rng=np.random.Generator(np.random.PCG64(seed));self.order=np.empty(0,dtype=np.int64);self.cursor=0;self.cycles=0;self.draws=0
    def take(self,count=512):
        result=[];remain=count
        while remain:
            if self.cursor==len(self.order):
                self.order=self.rng.permutation(self.eligible);self.cursor=0;self.cycles+=1
            size=min(remain,len(self.order)-self.cursor)
            result.append(self.order[self.cursor:self.cursor+size]);self.cursor+=size;remain-=size
        self.draws+=count
        return np.concatenate(result)


def check(device='cpu'):
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,17
    torch.manual_seed(42);model=original.PublicTabM(cards,bins,plain).to(device)
    tab,track=original.prototype.synthetic(cards,bins,plain,12,device)
    # Semantic fixture uses ordered point codes and original descriptor values.
    points,mask,quality=track
    counts=torch.tensor([0,1,2,3,4,5,6,7,8,31,127,128],device=device)
    mask=torch.arange(128,device=device)[None,:]<counts[:,None]
    points=torch.arange(12*128*16,device=device,dtype=torch.float32).reshape(12,128,16)/10000
    quality[:,1]=counts/128
    actual=thin(points,mask,quality)
    for i,n in enumerate(counts.cpu().tolist()):
        k=max(2,math.ceil(n/2)) if n>=3 else n
        selected=np.floor(np.arange(k)*(n-1)/max(k-1,1)).astype(int)
        torch.testing.assert_close(actual[0][i,:k],points[i,selected],rtol=0,atol=0)
        assert int(actual[1][i].sum())==k and torch.count_nonzero(actual[0][i,k:])==0
        torch.testing.assert_close(actual[2][i,torch.arange(9,device=device)!=1],quality[i,torch.arange(9,device=device)!=1],atol=0,rtol=0)
        assert actual[2][i,1]==k/128
        if n>=2:
            torch.testing.assert_close(actual[0][i,0],points[i,0],rtol=0,atol=0)
            torch.testing.assert_close(actual[0][i,k-1],points[i,n-1],rtol=0,atol=0)
    permutation=torch.arange(11,-1,-1,device=device)
    reordered=thin(points[permutation],mask[permutation],quality[permutation])
    for a,b in zip(actual,reordered):torch.testing.assert_close(a[permutation],b,atol=0,rtol=0)
    altered=points.clone();altered[~mask]=999
    for a,b in zip(actual,thin(altered,mask,quality)):torch.testing.assert_close(a,b,atol=0,rtol=0)
    parts=[thin(points[:5],mask[:5],quality[:5]),thin(points[5:],mask[5:],quality[5:])]
    for j in range(3):torch.testing.assert_close(actual[j],torch.cat([p[j] for p in parts]),atol=0,rtol=0)
    # Selected adjacency is recomputed by the unchanged physical encoder, while
    # preceding RAW gap remains the original source channel in each kept point.
    physical_points=torch.zeros((1,128,16),device=device)
    physical_mask=torch.arange(128,device=device)[None,:]<5
    physical_points[0,:5,0]=torch.tensor([.1,.09,.08,.07,.06],device=device)
    physical_points[0,:5,3]=torch.arange(5,device=device)/100
    physical_points[0,:5,1]=.123
    p,m,q=thin(physical_points,physical_mask,torch.zeros((1,9),device=device))
    f=original.physical.physical_features(p,m)
    dt=(p[0,0,0]-p[0,1,0])*7200
    expected=(p[0,1,3]-p[0,0,3])*20000/dt/50
    torch.testing.assert_close(f[0,1,8],expected,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(f[0,1,10],torch.log1p(dt)/math.log1p(7260),atol=1e-7,rtol=1e-6)
    assert f[0,1,11]==1 and p[0,1,1]==physical_points[0,2,1]
    first=CycleSampler([2,5,9]);second=CycleSampler([2,5,9]);one=first.take(17);two=np.r_[second.take(4),second.take(13)]
    np.testing.assert_array_equal(one,two);assert first.draws==second.draws==17
    # Real architecture, synthetic bounded physical values, frozen teacher output.
    tab,track=original.prototype.synthetic(cards,bins,plain,12,device)
    teacher=copy.deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():target=original.mean(teacher(*tab,*track)).detach()
    model.train();model.track.blocks[0].eval()  # Mixed modes must restore exactly.
    flags=[m.training for m in model.modules()];cpu=torch.get_rng_state().clone()
    gpu=torch.cuda.get_rng_state().clone() if device!='cpu' else None
    loss=consistency_loss(model,tab,track,target);loss.backward()
    assert flags==[m.training for m in model.modules()] and torch.equal(cpu,torch.get_rng_state())
    if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state())
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
    assert all(p.grad is None for p in teacher.parameters())
    # Empty/single-point representation remains original semantics; exact no-thin
    # evaluation is delegated to retained teacher full-row replay preflight.
    return dict(device=device,ordered_selection_exact=True,endpoints_retained=True,
        point_values_and_clocks_exact=True,raw_descriptors_exact=True,retained_slots_updated=True,
        selected_physical_adjacency_recomputed=True,preceding_raw_gap_preserved=True,
        batch_permutation_padding_invariance=True,sampler_chunk_invariance=True,
        eval_flags_and_rng_restored=True,finite_nonzero_student_gradients=True,
        teacher_no_gradients=True,loss=float(loss.detach()))
