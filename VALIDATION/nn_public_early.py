"""Feed the existing twelve public inputs into the first tabular hidden layer."""
import torch
from torch import nn

import nn_public_scratch as parent

physical = parent.physical
features = parent.features
transform = parent.transform
SETTINGS = dict(parent=parent.SETTINGS,
    change='add zero-initialized twelve-input projection before first tabular activation',
    architecture='existing first linear output plus Linear(12,512,bias=False); original late public head retained',
    added_parameters=6144, inputs='same twelve public inputs and original transform, no new data',
    initialization='all new weights zero; every original parameter/output/gradient/RNG unchanged',
    training='same original sparse update,21/17 epochs,seed42,original batch and schedule')


class SparseFusion(parent.SparseFusion):
    def __init__(self,cards,bins,n_plain):
        super().__init__(cards,bins,n_plain)
        first = self.tabular.mlp[0]
        assert isinstance(first,nn.Linear) and first.out_features==512
        with torch.random.fork_rng(devices=[]):
            self.public_early = nn.Linear(12,first.out_features,bias=False)
            nn.init.zeros_(self.public_early.weight)

    def forward(self,cat,g,plain,points,mask,quality):
        public = plain[:,self.original_plain:]
        assert public.shape[1]==12
        inputs = self.tabular.inputs(cat,g,plain[:,:self.original_plain])
        hidden = self.tabular.mlp[0](inputs)+self.public_early(public)
        for layer in list(self.tabular.mlp.children())[1:-1]:
            hidden = layer(hidden)
        base = self.tabular.mlp[-1](hidden)
        representation = self.track(points,mask)
        delta = self.fusion(torch.cat([hidden,representation,quality],dim=1))
        return base+delta*(mask.sum(1)>=2).unsqueeze(1)+self.public(torch.cat([hidden,public],dim=1))


def check(device='cpu'):
    physical.parent.shared.recipe()
    cards,bins,plain = [3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    torch.manual_seed(42);old=parent.SparseFusion(cards,bins,plain+12).to(device)
    rng=torch.get_rng_state().clone()
    torch.manual_seed(42);new=SparseFusion(cards,bins,plain+12).to(device)
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in old.state_dict().items():
        assert torch.equal(value,new.state_dict()[name]),name
    assert torch.count_nonzero(new.public_early.weight)==0
    assert sum(p.numel() for p in new.public_early.parameters())==6144
    tab,track=physical.parent.synthetic(cards,bins,plain,24,device)
    extra=torch.linspace(-2,2,12,device=device).repeat(24,1)
    tab=(*tab[:2],torch.cat([tab[2],extra],dim=1))
    # Activate existing paths too, so equivalence does not rely on zero old heads.
    with torch.no_grad():
        for model in [old,new]:
            model.public[-1].weight.fill_(.02)
            model.fusion[-1].weight.fill_(.01)
            model.track.physics_stem.weight.fill_(.02)
    for training in [False,True]:
        old.train(training);new.train(training)
        torch.manual_seed(912);expected=old(*tab,*track)
        cpu_rng=torch.get_rng_state().clone()
        gpu_rng=torch.cuda.get_rng_state(device).clone() if device!='cpu' else None
        torch.manual_seed(912);actual=new(*tab,*track)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.equal(cpu_rng,torch.get_rng_state())
        if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
        old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
        expected.square().mean().backward();actual.square().mean().backward()
        params=dict(new.named_parameters())
        for name,value in old.named_parameters():
            other=params[name]
            assert (value.grad is None)==(other.grad is None),name
            if value.grad is not None:
                torch.testing.assert_close(value.grad,other.grad,rtol=0,atol=0)
        gradient=new.public_early.weight.grad
        assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    new.eval()
    with torch.no_grad():
        new.public_early.weight[0,0]=.05
        actual=new(*tab,*track)
        assert torch.max(abs(actual-old.eval()(*tab,*track)))>0
        altered=track[0].clone();altered[~track[1]]=999
        torch.testing.assert_close(new(*tab,altered,*track[1:]),actual,rtol=0,atol=0)
    return dict(device=device,original_parameters_exact=True,new_weights_zero=True,
        initial_outputs_and_original_gradients_exact=True,initialization_and_dropout_rng_exact=True,
        old_nonzero_heads_preserved=True,new_projection_has_nonzero_finite_gradient=True,
        nonzero_projection_changes_output=True,track_padding_invariant=True,added_parameters=6144)
