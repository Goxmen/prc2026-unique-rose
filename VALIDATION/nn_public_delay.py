"""Zero-initialized public daily-information residual on saved physical networks."""
import numpy as np
import torch
from torch import nn
import nn_track_physical_scales as physical
import public_delay_features as features
SCALES=[5.,600.,20.,72000.,100.,100.,100.,3600.,3600.,1.,1.,1.]
SETTINGS=dict(features=features.FEATURES,scales=SCALES,transform='asinh(x/scale)/4,clip[-5,5],missing0',
    epochs=2,old_learning_rate=2e-5,new_learning_rate=1e-3,batch=2048,
    public_head='256 original hidden +12 public inputs ->32 SiLU ->1 zero final',
    parent='saved physical seed42 fold models',control='retained global continuation, same epochs/old learning rate/RNG')

def transform(frame):
    x=frame[features.FEATURES].to_numpy(float)
    return np.nan_to_num(np.clip(np.arcsinh(x/np.asarray(SCALES))/4,-5,5),nan=0,posinf=5,neginf=-5).astype('float32')

class PublicFusion(physical.SparseFusion):
    def __init__(self,cards,bins,n_plain):
        super().__init__(cards,bins,n_plain);self.original_plain=n_plain
        with torch.random.fork_rng(devices=[]):
            self.public=nn.Sequential(nn.Linear(256+12,32),nn.SiLU(),nn.Linear(32,1))
            nn.init.zeros_(self.public[-1].weight);nn.init.zeros_(self.public[-1].bias)
    def initialize(self,state):
        result=self.load_state_dict(state,strict=False)
        assert not result.unexpected_keys and set(result.missing_keys)=={'public.0.weight','public.0.bias','public.2.weight','public.2.bias'}
    def forward(self,cat,g,plain,points,mask,quality):
        hidden=self.tabular.inputs(cat,g,plain[:,:self.original_plain])
        for layer in list(self.tabular.mlp.children())[:-1]:hidden=layer(hidden)
        base=self.tabular.mlp[-1](hidden)
        representation=self.track(points,mask)
        delta=self.fusion(torch.cat([hidden,representation,quality],dim=1))
        return base+delta*(mask.sum(1)>=2).unsqueeze(1)+self.public(torch.cat([hidden,plain[:,self.original_plain:]],dim=1))

def check(device='cpu'):
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    physical.parent.shared.recipe();torch.manual_seed(42)
    old=physical.SparseFusion(cards,bins,plain).to(device)
    new=PublicFusion(cards,bins,plain).to(device);new.initialize(old.state_dict())
    tab,track=physical.parent.synthetic(cards,bins,plain,24,device)
    extra=torch.zeros(24,12,device=device);newtab=(*tab[:2],torch.cat([tab[2],extra],dim=1))
    for train in [False,True]:
        old.train(train);new.train(train);torch.manual_seed(222)
        expected=old(*tab,*track);rng=torch.get_rng_state().clone()
        torch.manual_seed(222);actual=new(*newtab,*track)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.equal(rng,torch.get_rng_state())
    old.eval();new.eval()
    with torch.no_grad():new.public[-1].bias.fill_(.1)
    torch.testing.assert_close(new(*newtab,*track),old(*tab,*track)+.1,rtol=0,atol=0)
    new(*newtab,*track).square().mean().backward()
    assert torch.isfinite(new.public[-1].weight.grad).all() and new.public[-1].weight.grad.abs().sum()>0
    return dict(device=device,initial_outputs_exact=True,original_dropout_rng_exact=True,new_head_active_without_track=True,finite_gradient=True)
