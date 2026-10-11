"""Extend the existing public physical network with ten daily taxi inputs."""
import numpy as np
import torch
from torch import nn

import nn_public_scratch as parent
import public_daily_taxi_features as daily

physical=parent.physical
SCALES=[600.,600.,300.,200.,600.,600.,300.,600.,1.,300.]
SETTINGS=dict(parent=parent.SETTINGS,added=daily.FEATURES,scales=SCALES,
    transform='existing12 unchanged; new10 asinh(x/scale)/4 clipped[-5,5],missing0',
    architecture='widen existing public first layer268 to278; preserve all existing weights; new columns zero',
    training='original sparse training function; seed42,21/17 fixed epochs; no continuation or EMA')


def transform(frame):
    old=parent.transform(frame)
    extra=frame[daily.FEATURES].to_numpy(float)
    extra=np.nan_to_num(np.clip(np.arcsinh(extra/np.asarray(SCALES))/4,-5,5),nan=0,posinf=5,neginf=-5)
    return np.column_stack([old,extra]).astype('float32')


class SparseFusion(parent.SparseFusion):
    def __init__(self,cards,bins,n_plain):
        assert n_plain>=22
        super().__init__(cards,bins,n_plain-10)
        original=self.public[0]
        assert original.in_features==268
        with torch.random.fork_rng(devices=[]):
            expanded=nn.Linear(278,32)
        with torch.no_grad():
            expanded.weight[:,:268].copy_(original.weight)
            expanded.weight[:,268:].zero_()
            expanded.bias.copy_(original.bias)
        self.public[0]=expanded


def check(device='cpu'):
    physical.parent.shared.recipe()
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    torch.manual_seed(42);old=parent.SparseFusion(cards,bins,plain+12).to(device)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(42);new=SparseFusion(cards,bins,plain+22).to(device)
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in old.state_dict().items():
        other=new.state_dict()[name]
        if name=='public.0.weight':
            assert torch.equal(value,other[:,:268]) and not other[:,268:].count_nonzero()
        else:assert torch.equal(value,other),name
    tab,track=physical.parent.synthetic(cards,bins,plain,24,device)
    public=torch.linspace(-2,2,12,device=device).repeat(24,1)
    daily_values=torch.linspace(-1,1,10,device=device).repeat(24,1)
    oldtab=(*tab[:2],torch.cat([tab[2],public],dim=1))
    newtab=(*tab[:2],torch.cat([tab[2],public,daily_values],dim=1))
    for training in [False,True]:
        old.train(training);new.train(training)
        torch.manual_seed(917);expected=old(*oldtab,*track)
        cpu_rng=torch.get_rng_state().clone()
        gpu_rng=torch.cuda.get_rng_state(device).clone() if device!='cpu' else None
        torch.manual_seed(917);actual=new(*newtab,*track)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.equal(cpu_rng,torch.get_rng_state())
        if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    old.eval();new.eval()
    with torch.no_grad():
        old.public[-1].weight.fill_(.02);new.public[-1].weight.fill_(.02)
    torch.testing.assert_close(new(*newtab,*track),old(*oldtab,*track),rtol=1e-6,atol=1e-7)
    new(*newtab,*track).square().mean().backward()
    gradient=new.public[0].weight.grad[:,268:]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    return dict(device=device,all_original_weights_exact=True,new_columns_zero=True,
        original_initial_output_and_rng_exact=True,active_old_public_head_preserved=True,
        added_columns_have_finite_nonzero_gradient=True)
