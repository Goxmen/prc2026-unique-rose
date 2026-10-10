"""Fixed local-distance and motion features inside the existing track encoder.

Uses only the already-audited point tensor. Original weights, initialization,
RNG and missing-track gates are retained. A zero-initialized parallel stem
learns 14 fixed nonlinear transforms; no hyperparameter search or new sources.
"""
import math
import torch
from torch import nn
import nn_sparse_track_prototype as parent

FEATURES = ['stand_near50m','stand_near200m','stand_near1000m',
            'runway_near100m','runway_near500m','runway_near2000m',
            'stand_log_distance','runway_log_distance',
            'east_velocity','north_velocity','sampled_gap','sampled_pair_known',
            'actual_clock_minute_scale','slow_speed']
SETTINGS = dict(features=FEATURES, stand_scales_m=[50,200,1000],
    runway_scales_m=[100,500,2000],velocity_scale_mps=50,
    clock_scale_sec=60, slow_speed_scale_kt=5,
    initialization='parallel linear stem zero; original encoder and RNG preserved',
    new_data=False, additional_dropout=0)


def physical_features(points, mask):
    x=points*mask.unsqueeze(-1)
    stand=x[:,:,10].clamp(0,1)*40000
    runway=x[:,:,12].clamp(0,1)*40000
    stand_known=(x[:,:,11]>.5)&mask
    runway_known=(x[:,:,13]>.5)&mask
    features=[torch.exp(-stand/scale)*stand_known for scale in SETTINGS['stand_scales_m']]
    features += [torch.exp(-runway/scale)*runway_known for scale in SETTINGS['runway_scales_m']]
    features += [torch.log1p(stand)/math.log1p(40000)*stand_known,
                 torch.log1p(runway)/math.log1p(40000)*runway_known]
    dt=(x[:,:-1,0]-x[:,1:,0])*7200
    pair=mask[:,:-1]&mask[:,1:]&(dt>0)
    for col in [3,4]:
        velocity=((x[:,1:,col]-x[:,:-1,col])*20000/dt.clamp(min=.001)/50).clamp(-1,1)
        features.append(torch.nn.functional.pad(velocity*pair,(1,0)))
    gap=torch.log1p(dt.clamp(0,7260))/math.log1p(7260)*pair
    features += [torch.nn.functional.pad(gap,(1,0)),torch.nn.functional.pad(pair,(1,0))]
    clock_sec=torch.sinh(x[:,:,14].clamp(-1,1)*8)*1800
    features.append(torch.asinh(clock_sec/60).clamp(-8,8)/8*(x[:,:,15]>.5))
    features.append(torch.exp(-x[:,:,5].clamp(0,1)*300/5)*(x[:,:,6]>.5))
    return torch.stack(features,dim=-1).to(points.dtype)*mask.unsqueeze(-1)


class PhysicalTrackEncoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.stem=original.stem
        self.blocks=original.blocks
        self.physics_stem=nn.Linear(len(FEATURES),parent.SETTINGS['width'],bias=False)
        nn.init.zeros_(self.physics_stem.weight)

    def forward(self,points,mask):
        present=mask.unsqueeze(-1)
        value=torch.nn.functional.silu(self.stem(points*present)+
                    self.physics_stem(physical_features(points,mask)))*present
        for block in self.blocks:value=block(value,present)
        mean=value.sum(1)/mask.sum(1).clamp(min=1).unsqueeze(1)
        maximum=value.masked_fill(~present,-torch.inf).amax(1)
        maximum=torch.where(mask.any(1).unsqueeze(1),maximum,torch.zeros_like(maximum))
        return torch.cat([mean,maximum],dim=1)


class SparseFusion(parent.SparseFusion):
    def __init__(self,cards,bins,n_plain):
        super().__init__(cards,bins,n_plain)
        with torch.random.fork_rng(devices=[]):
            self.track=PhysicalTrackEncoder(self.track)


def check(device='cpu'):
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    parent.shared.recipe()
    torch.manual_seed(42);old=parent.SparseFusion(cards,bins,plain).to(device)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(42);new=SparseFusion(cards,bins,plain).to(device)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in old.state_dict().items():assert torch.equal(v,new.state_dict()[k]),k
    tab,track=parent.synthetic(cards,bins,plain,24,device)
    with torch.no_grad():
        old.fusion[-1].weight.fill_(.01);old.fusion[-1].bias.fill_(.03)
        new.load_state_dict(old.state_dict(),strict=False)
    for training in [False,True]:
        old.train(training);new.train(training)
        torch.manual_seed(913);expected=old(*tab,*track)
        cpu_rng=torch.get_rng_state().clone()
        gpu_rng=torch.cuda.get_rng_state(device).clone() if device!='cpu' else None
        torch.manual_seed(913);actual=new(*tab,*track)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.equal(cpu_rng,torch.get_rng_state())
        if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    new.eval();old.eval()
    with torch.no_grad():new.track.physics_stem.weight.fill_(.02)
    actual=new(*tab,*track)
    absent=track[1].sum(1)<2
    torch.testing.assert_close(actual[absent],old(*tab,*track)[absent],rtol=0,atol=0)
    altered=track[0].clone();altered[~track[1]]=999
    torch.testing.assert_close(new(*tab,altered,*track[1:]),actual,rtol=0,atol=0)
    actual.square().mean().backward()
    gradient=new.track.physics_stem.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert torch.isfinite(physical_features(*track[:2])).all()
    return dict(device=device,original_weights_and_rng_exact=True,nonzero_initial_path_exact=True,
                dropout_rng_exact=True,absent_single_gate_exact=True,padding_invariant=True,
                finite_nonzero_gradient=True)
