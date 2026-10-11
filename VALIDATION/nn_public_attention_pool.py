"""Learn one masked attention score over the retained physical track states.

Zero scores preserve the original mean/max pool exactly. The original mean
receives only the weighted-minus-uniform residual; max pooling is unchanged.
This adds32 parameters, with no additional source, width or recurrent layer.
"""
import torch
from torch import nn

import nn_public_scratch as parent

physical,features,transform=parent.physical,parent.features,parent.transform
SETTINGS=dict(parent=parent.SETTINGS,
    change='learn one scalar masked attention score in place of uniform physical mean pooling',
    attention='Linear32to1 biasFalse zero; softmax over valid points; max pool retained',
    added_parameters=32,initialization='forked RNG; zero weighted-minus-uniform residual',
    training='unchanged168 inputs,seed42,21/17epochs,original MSE and optimizer',
    no_width_change=True,no_time_weight=True,no_extra_dropout=True)


def attention_delta(value,mask,score):
    logits=score(value).squeeze(-1)
    logits=logits.masked_fill(~mask,-torch.inf)
    maximum=logits.max(1,keepdim=True).values
    maximum=torch.where(mask.any(1,keepdim=True),maximum,torch.zeros_like(maximum))
    unnormalized=torch.exp(logits-maximum)*mask
    weights=unnormalized/unnormalized.sum(1,keepdim=True).clamp(min=1e-30)
    uniform=mask.to(value.dtype)/mask.sum(1,keepdim=True).clamp(min=1)
    return ((weights-uniform).unsqueeze(-1)*value*mask.unsqueeze(-1)).sum(1)


class AttentionTrackEncoder(nn.Module):
    def __init__(self,original):
        super().__init__()
        self.stem,self.blocks,self.physics_stem=original.stem,original.blocks,original.physics_stem
        self.attention_score=nn.Linear(physical.parent.SETTINGS['width'],1,bias=False)
        nn.init.zeros_(self.attention_score.weight)

    def forward(self,points,mask):
        present=mask.unsqueeze(-1)
        value=nn.functional.silu(self.stem(points*present)+self.physics_stem(physical.physical_features(points,mask)))*present
        for block in self.blocks:value=block(value,present)
        mean=value.sum(1)/mask.sum(1).clamp(min=1).unsqueeze(1)
        maximum=value.masked_fill(~present,-torch.inf).amax(1)
        maximum=torch.where(mask.any(1).unsqueeze(1),maximum,torch.zeros_like(maximum))
        return torch.cat([mean+attention_delta(value,mask,self.attention_score),maximum],dim=1)


class SparseFusion(parent.SparseFusion):
    def __init__(self,cards,bins,n_plain):
        super().__init__(cards,bins,n_plain)
        with torch.random.fork_rng(devices=[]):self.track=AttentionTrackEncoder(self.track)


def pooling_check(device):
    values=torch.arange(6*9*3,device=device,dtype=torch.float64).reshape(6,9,3)/97-1
    counts=torch.tensor([0,1,2,4,8,9],device=device)
    mask=torch.arange(9,device=device)[None,:]<counts[:,None]
    score=nn.Linear(3,1,bias=False).to(device=device,dtype=torch.float64)
    with torch.no_grad():score.weight.copy_(torch.tensor([[.3,-.7,.8]],device=device,dtype=torch.float64))
    actual=attention_delta(values,mask,score);expected=[]
    for row,n in enumerate(counts.tolist()):
        if not n:expected.append(torch.zeros(3,device=device,dtype=torch.float64));continue
        h=values[row,:n];s=h@score.weight[0];weights=(s-s.max()).exp();weights/=weights.sum()
        expected.append(sum((weights[j]-1/n)*h[j] for j in range(n)))
    torch.testing.assert_close(actual,torch.stack(expected),rtol=0,atol=1e-15)
    altered=values.clone();altered[~mask]=999
    torch.testing.assert_close(attention_delta(altered,mask,score),actual,rtol=0,atol=0)
    order=torch.arange(8,-1,-1,device=device)
    torch.testing.assert_close(attention_delta(values[:,order],mask[:,order],score),actual,rtol=0,atol=1e-15)
    score.weight.data.zero_();torch.testing.assert_close(attention_delta(values,mask,score),torch.zeros_like(actual),rtol=0,atol=0)
    return dict(scalar_softmax_weights=True,all_missing_and_single_point=True,padding_ignored=True,
        local_state_permutation_invariant=True,zero_logits_exact_uniform=True)


def check(device='cpu'):
    physical.parent.shared.recipe();cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    torch.manual_seed(42);old=parent.SparseFusion(cards,bins,plain+12).to(device);rng=torch.get_rng_state().clone()
    torch.manual_seed(42);new=SparseFusion(cards,bins,plain+12).to(device)
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[name]),name
    added=sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters());assert added==32
    tab,track=physical.parent.synthetic(cards,bins,plain,24,device)
    tab=(*tab[:2],torch.cat([tab[2],torch.linspace(-2,2,12,device=device).repeat(24,1)],dim=1))
    with torch.no_grad():
        for model in [old,new]:
            model.public[-1].weight.fill_(.02);model.fusion[-1].weight.fill_(.01);model.track.physics_stem.weight.fill_(.02)
    worst=0.
    for training in [False,True]:
        old.train(training);new.train(training);torch.manual_seed(912);expected=old(*tab,*track)
        cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state(device).clone() if device!='cpu' else None
        torch.manual_seed(912);actual=new(*tab,*track);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.equal(cpu_rng,torch.get_rng_state())
        if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
        if training:
            old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True);expected.square().mean().backward();actual.square().mean().backward()
            parameters=dict(new.named_parameters())
            for name,value in old.named_parameters():
                other=parameters[name];assert (value.grad is None)==(other.grad is None)
                if value.grad is not None:
                    tolerance=0.
                    if device!='cpu' and name.startswith('track.'):
                        tolerance=8*torch.finfo(value.grad.dtype).eps*float(torch.maximum(value.grad.abs(),other.grad.abs()).max())
                    torch.testing.assert_close(value.grad,other.grad,rtol=0,atol=tolerance)
                    worst=max(worst,float((value.grad-other.grad).abs().max()))
            grad=new.track.attention_score.weight.grad;assert torch.isfinite(grad).all() and grad.abs().sum()>0
    old.eval();new.eval()
    with torch.no_grad():
        new.track.attention_score.weight.fill_(.03);actual=new(*tab,*track);reference=old(*tab,*track)
        assert (actual-reference).abs().max()>0
        absent=track[1].sum(1)<2;torch.testing.assert_close(actual[absent],reference[absent],rtol=0,atol=0)
        altered=track[0].clone();altered[~track[1]]=999
        torch.testing.assert_close(new(*tab,altered,*track[1:]),actual,rtol=0,atol=0)
    return dict(device=device,added_parameters=added,original_weights_outputs_rng_exact=True,
        original_gradients_checked=True,maximum_gradient_error=worst,missing_track_and_padding_exact=True,
        attention_has_finite_nonzero_gradient=True,pooling=pooling_check(device))
