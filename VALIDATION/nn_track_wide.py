"""Double the successful physical track encoder's width, keeping its inputs.

The tabular/public branches, three temporal blocks, pooling, fusion hidden
width, optimizer and schedule stay fixed. Branch initialization is isolated
from the original tabular dropout RNG. No pretrained weights are used.
"""
import torch
from torch import nn
import nn_public_scratch as parent

physical, features, transform = parent.physical, parent.features, parent.transform
SETTINGS = dict(parent=parent.SETTINGS, track_width=64, previous_track_width=32,
    temporal_blocks=3, dilations=[1,4,16], fusion_hidden=128,
    change='track stem, physical stem and temporal channels32 to64; fusion input dimension follows',
    unchanged='tabular/public branches, all inputs, masks, pooling, epochs, batches and optimizer',
    initialization='fresh wider branch inside fork_rng; zero final track residual',
    no_width_search=True, no_epoch_search=True)


class WidePhysicalEncoder(physical.PhysicalTrackEncoder):
    def __init__(self):
        nn.Module.__init__(self)
        width=SETTINGS['track_width']
        self.stem=nn.Linear(len(physical.parent.seq.CHANNEL_NAMES),width)
        self.blocks=nn.ModuleList([physical.parent.TemporalBlock(width,d) for d in SETTINGS['dilations']])
        self.physics_stem=nn.Linear(len(physical.FEATURES),width,bias=False)
        nn.init.zeros_(self.physics_stem.weight)


class SparseFusion(parent.SparseFusion):
    def __init__(self,cards,bins,n_plain):
        super().__init__(cards,bins,n_plain)
        with torch.random.fork_rng(devices=[]):
            self.track=WidePhysicalEncoder()
            self.fusion=nn.Sequential(nn.Linear(256+2*SETTINGS['track_width']+len(physical.parent.seq.QUALITY_NAMES),128),
                nn.SiLU(),nn.Linear(128,1))
            nn.init.zeros_(self.fusion[-1].weight)
            nn.init.zeros_(self.fusion[-1].bias)


def check(device='cpu'):
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    physical.parent.shared.recipe()
    torch.manual_seed(42);old=parent.SparseFusion(cards,bins,plain+12).to(device)
    expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(42);model=SparseFusion(cards,bins,plain+12).to(device)
    assert torch.equal(expected_rng,torch.get_rng_state())
    for prefix in ['tabular.','public.']:
        for name,value in old.state_dict().items():
            if name.startswith(prefix):assert torch.equal(value,model.state_dict()[name]),name
    assert model.track.stem.out_features==64 and old.track.stem.out_features==32
    tab,track=physical.parent.synthetic(cards,bins,plain,48,device)
    public=torch.linspace(-2,2,12,device=device).repeat(len(track[1]),1)
    extended=(*tab[:2],torch.cat([tab[2],public],dim=1))
    for training in [False,True]:
        model.train(training);old.train(training)
        torch.manual_seed(914);expected=old(*extended,*track)
        cpu_rng=torch.get_rng_state().clone()
        gpu_rng=torch.cuda.get_rng_state(device).clone() if device!='cpu' else None
        torch.manual_seed(914);actual=model(*extended,*track)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.equal(cpu_rng,torch.get_rng_state())
        if gpu_rng is not None:assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
    model.eval();old.eval()
    with torch.no_grad():
        model.fusion[-1].weight.fill_(.01)
        model.fusion[-1].bias.fill_(.03)
        model.track.physics_stem.weight.fill_(.02)
        model.public[-1].weight.fill_(.01)
        old.public.load_state_dict(model.public.state_dict())
    actual=model(*extended,*track)
    absent=track[1].sum(1)<2
    torch.testing.assert_close(actual[absent],old(*extended,*track)[absent],rtol=0,atol=0)
    altered=track[0].clone();altered[~track[1]]=999
    torch.testing.assert_close(model(*extended,altered,*track[1:]),actual,rtol=0,atol=0)
    actual.square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.track.physics_stem.weight.grad.abs().sum()>0
    assert model.track.blocks[0].convs[0].weight.grad.abs().sum()>0
    assert model.public[-1].weight.grad.abs().sum()>0
    return dict(device=device,width64=True,tabular_and_public_parameters_exact=True,
        initialization_and_dropout_rng_exact=True,initial_output_exact=True,
        missing_single_track_gate_exact=True,padding_invariant=True,finite_nonzero_gradients=True,
        original_parameters=sum(p.numel() for p in old.parameters()),
        candidate_parameters=sum(p.numel() for p in model.parameters()))
