"""The original callsign PLE with a zero-initialized public-information head."""
import torch
from torch import nn

import nn_callsign_clocks_jan_jul as calls
import nn_public_delay as public

transform = public.transform
FEATURES = public.features.FEATURES
SETTINGS = dict(original_inputs=162, public_inputs=FEATURES, scales=public.SCALES,
    transform=public.SETTINGS['transform'], head='256 hidden plus12 public inputs ->32 SiLU ->1',
    initialization='original callsign seed42 plus zero public output; original RNG unchanged',
    training='original direct target, optimizer, batches and fixed21/17 epochs')


class PublicCallsNet(nn.Module):
    def __init__(self,cards,bins,n_plain):
        super().__init__()
        assert n_plain>=12
        self.original_plain=n_plain-12
        self.tabular=calls.shared.nni.PleNet(cards,bins,self.original_plain)
        with torch.random.fork_rng(devices=[]):
            self.public=nn.Sequential(nn.Linear(256+12,32),nn.SiLU(),nn.Linear(32,1))
            nn.init.zeros_(self.public[-1].weight)
            nn.init.zeros_(self.public[-1].bias)

    def forward(self,cat,g,plain):
        hidden=self.tabular.inputs(cat,g,plain[:,:self.original_plain])
        for layer in list(self.tabular.mlp.children())[:-1]:hidden=layer(hidden)
        return self.tabular.mlp[-1](hidden)+self.public(torch.cat([hidden,plain[:,self.original_plain:]],dim=1))


def check(device='cpu'):
    calls.shared.recipe()
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    torch.manual_seed(42);old=calls.shared.nni.PleNet(cards,bins,plain).to(device)
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(42);new=PublicCallsNet(cards,bins,plain+12).to(device)
    assert torch.equal(expected_rng,torch.get_rng_state())
    for name,value in old.state_dict().items():assert torch.equal(value,new.tabular.state_dict()[name]),name
    generator=torch.Generator(device=device).manual_seed(1337)
    cat=torch.stack([torch.randint(c,(48,),generator=generator,device=device) for c in cards],1)
    g=torch.rand(48,3,generator=generator,device=device)
    x=torch.rand(48,plain,generator=generator,device=device)
    pub=torch.linspace(-2,2,12,device=device).repeat(48,1)
    for training in [False,True]:
        old.train(training);new.train(training)
        torch.manual_seed(914);expected=old(cat,g,x)
        cpu_rng=torch.get_rng_state().clone()
        gpu_rng=torch.cuda.get_rng_state(device).clone() if device!='cpu' else None
        torch.manual_seed(914);actual=new(cat,g,torch.cat([x,pub],1))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.equal(cpu_rng,torch.get_rng_state())
        if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    new.eval();old.eval()
    with torch.no_grad():new.public[-1].bias.fill_(.1)
    torch.testing.assert_close(new(cat,g,torch.cat([x,pub],1)),old(cat,g,x)+.1,rtol=0,atol=0)
    new(cat,g,torch.cat([x,pub],1)).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in new.parameters())
    assert new.public[-1].weight.grad.abs().sum()>0
    return dict(device=device,original_weights_exact=True,initial_output_exact=True,
        initialization_and_dropout_rng_exact=True,public_head_active=True,finite_nonzero_gradient=True)
