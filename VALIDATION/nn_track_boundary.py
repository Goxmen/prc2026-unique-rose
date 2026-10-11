"""Dense off-block-boundary supervision for the retained physical track encoder.

The inference architecture is unchanged: a zero-initialized training-only
point head sees the shared 32-channel physical representation. Its targets
come exclusively from ordinary training labels and cached point ages.
"""
import torch
from torch import nn
import nn_track_physical_scales as physical

SETTINGS=dict(parent=physical.SETTINGS,change='training-only per-point off-block boundary head',
    auxiliary_weight=.1,label_smoothing_sec=30.,point_age_scale_sec=7200.,
    label='sigmoid((clipped recorded taxi duration - point age)/30)',
    reduction='mean over valid points per usable flight, then mean over flights',
    usable='at least two points; original ordinary training population',
    initialization='zero linear point head; old parameters and RNG unchanged')

class BoundaryTrackEncoder(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.stem=original.stem;self.blocks=original.blocks;self.physics_stem=original.physics_stem
        self.boundary_head=nn.Linear(32,1)
        nn.init.zeros_(self.boundary_head.weight);nn.init.zeros_(self.boundary_head.bias)
        self.boundary_logits=None

    def forward(self,points,mask):
        present=mask.unsqueeze(-1)
        value=torch.nn.functional.silu(self.stem(points*present)+
            self.physics_stem(physical.physical_features(points,mask)))*present
        for block in self.blocks:value=block(value,present)
        self.boundary_logits=self.boundary_head(value).squeeze(-1) if self.training else None
        mean=value.sum(1)/mask.sum(1).clamp(min=1).unsqueeze(1)
        maximum=value.masked_fill(~present,-torch.inf).amax(1)
        maximum=torch.where(mask.any(1).unsqueeze(1),maximum,torch.zeros_like(maximum))
        return torch.cat([mean,maximum],dim=1)

class SparseFusion(physical.SparseFusion):
    def __init__(self,cards,bins,n_plain):
        super().__init__(cards,bins,n_plain)
        with torch.random.fork_rng(devices=[]):self.track=BoundaryTrackEncoder(self.track)

def labels(points,target_sec):
    age=points[:,:,0]*SETTINGS['point_age_scale_sec']
    return torch.sigmoid((target_sec[:,None]-age)/SETTINGS['label_smoothing_sec'])

def auxiliary_loss(model,points,mask,target_sec):
    logits=model.track.boundary_logits
    assert logits is not None
    target=labels(points,target_sec)
    usable=mask.sum(1)>=2
    loss=nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
    per_flight=(loss*mask).sum(1)/mask.sum(1).clamp(min=1)
    return (per_flight*usable).sum()/usable.sum().clamp(min=1)

def check(device='cpu'):
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    physical.parent.shared.recipe()
    torch.manual_seed(42);old=physical.SparseFusion(cards,bins,plain).to(device);rng=torch.get_rng_state().clone()
    torch.manual_seed(42);new=SparseFusion(cards,bins,plain).to(device)
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[name]),name
    tab,track=physical.parent.synthetic(cards,bins,plain,24,device)
    with torch.no_grad():
        old.fusion[-1].weight.fill_(.01);old.fusion[-1].bias.fill_(.03);old.track.physics_stem.weight.fill_(.02)
        new.load_state_dict(old.state_dict(),strict=False)
    for training in [False,True]:
        old.train(training);new.train(training)
        torch.manual_seed(913);expected=old(*tab,*track);rng=torch.get_rng_state().clone()
        gpu=torch.cuda.get_rng_state(device).clone() if device!='cpu' else None
        torch.manual_seed(913);actual=new(*tab,*track)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0);assert torch.equal(rng,torch.get_rng_state())
        if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state(device))
    target=torch.full((24,),1200.,device=device)
    aux=auxiliary_loss(new,track[0],track[1],target)
    assert torch.isfinite(aux);aux.backward()
    grad=new.track.boundary_head.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    ages=torch.tensor([600.,1200.,1800.],device=device)
    sample=torch.zeros((1,3,16),device=device);sample[0,:,0]=ages/7200
    expected_label=torch.tensor([[1.,.5,0.]],device=device)
    torch.testing.assert_close(labels(sample,target[:1]),expected_label,rtol=0,atol=1e-8)
    new.eval();actual=new(*tab,*track)
    altered=track[0].clone();altered[~track[1]]=999
    torch.testing.assert_close(new(*tab,altered,*track[1:]),actual,rtol=0,atol=0)
    assert new.track.boundary_logits is None
    return dict(device=device,original_weights_rng_and_outputs_exact=True,padding_invariant=True,
        label_orientation_and_midpoint=True,finite_nonzero_auxiliary_gradient=True,inference_has_no_auxiliary_output=True)
