"""Fixed full-public/physical TabM family, without changing retained parents.

Each of32 heads receives its own MSE supervision. Only inference averages heads.
Raw track encoder and all input transforms are the selected public recipe.
This is a new architecture, not a claim of identical N09 initialization.
"""
import os
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
import torch
from torch import nn
import tabm
import nn_public_scratch as control

physical=control.physical
prototype=physical.parent
nni=prototype.shared.nni
features=control.features
transform=control.transform
SETTINGS=dict(original_columns=156,public_columns=features.FEATURES,
    public_transform=control.SETTINGS,track='unchanged physical CNN, masked mean/max64',
    track_missing='representation exactly0 for fewer than2points; retain9 quality inputs',
    k=32,n_blocks=2,d_block=512,dropout=.1,activation='ReLU',arch_type='tabm',
    start_scaling_init='normal',lr=.002,weight_decay=.0003,
    loss='mean squared error over rows AND individual32 heads',
    inference='arithmetic head mean in standardized units, then seconds and nonnegative clip',
    initialization='fresh seed42; original Inputs; one shared physical encoder; no pretrained weights')


class PublicTabM(nn.Module):
    def __init__(self,cards,bins,n_plain):
        super().__init__()
        assert n_plain>=12
        self.original_plain=n_plain-12
        self.inputs=nni.Inputs(cards,bins,self.original_plain)
        self.track=physical.PhysicalTrackEncoder(prototype.TrackEncoder())
        chunks=self.inputs.chunks+[1]*12+[64]+[1]*len(prototype.seq.QUALITY_NAMES)
        self.view=tabm.EnsembleView(k=SETTINGS['k'])
        self.backbone=tabm.make_tabm_backbone(d_in=sum(chunks),
            **{k:SETTINGS[k] for k in ['n_blocks','d_block','dropout','activation','k','arch_type','start_scaling_init']},
            start_scaling_init_chunks=chunks)
        self.output=tabm.LinearEnsemble(SETTINGS['d_block'],1,k=SETTINGS['k'])

    def forward(self,cat,g,plain,points,mask,quality):
        tab=self.inputs(cat,g,plain[:,:self.original_plain])
        representation=self.track(points,mask)*(mask.sum(1)>=2).unsqueeze(1)
        joined=torch.cat([tab,plain[:,self.original_plain:],representation,quality],dim=1)
        return self.output(self.backbone(self.view(joined))).squeeze(-1)


def loss(output,target):
    assert output.ndim==2 and target.ndim==1 and len(output)==len(target)
    return torch.nn.functional.mse_loss(output,target[:,None].expand_as(output))


def mean(output):
    assert output.ndim==2
    return output.mean(dim=1)


def check(device='cpu'):
    prototype.shared.recipe()
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    torch.manual_seed(42)
    model=PublicTabM(cards,bins,plain+12).to(device).eval()
    tab,track=prototype.synthetic(cards,bins,plain+12,12,device)
    output=model(*tab,*track)
    assert output.shape==(12,32) and torch.isfinite(output).all()
    altered=track[0].clone();altered[~track[1]]=999
    torch.testing.assert_close(model(*tab,altered,*track[1:]),output,atol=0,rtol=0)
    # An observed single point has no track representation; its position cannot affect output.
    altered=track[0].clone();absent=track[1].sum(1)<2;altered[absent]=.3
    torch.testing.assert_close(model(*tab,altered,*track[1:])[absent],output[absent],atol=0,rtol=0)
    target=torch.zeros(12,device=device)
    error=loss(output,target);error.backward()
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
    assert model.track.physics_stem.weight.grad.abs().sum()>0
    assert model.output.weight.grad.abs().sum()>0
    heads=torch.tensor([[-1.,1.]],device=device,requires_grad=True)
    assert loss(heads,torch.zeros(1,device=device)).item()==1.
    assert loss(mean(heads)[:,None],torch.zeros(1,device=device)).item()==0.
    # Partial-microbatch weighting must equal one full-batch MSE and its gradient.
    x=torch.arange(35.,device=device).reshape(7,5)/17
    full=nn.Linear(5,32).to(device);split=nn.Linear(5,32).to(device);split.load_state_dict(full.state_dict())
    y=torch.linspace(-1,1,7,device=device)
    loss(full(x),y).backward()
    for start in range(0,7,3):loss(split(x[start:start+3]),y[start:start+3]).mul(len(x[start:start+3])/7).backward()
    for a,b in zip(full.parameters(),split.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=2e-7,rtol=1e-6)
    return dict(device=device,heads=32,member_loss_not_mean_loss=True,padding_ignored=True,
        fewer_than_two_track_representation_zero=True,all_gradients_finite=True,
        physical_gradient_nonzero=True,partial_microbatch_gradient_weighting=True)
