"""Shadow parameter averaging with the original AdamW updates untouched."""
import numpy as np
import torch
from torch import nn

import nn_public_scratch as parent

SparseFusion = parent.SparseFusion
physical, features, transform = parent.physical, parent.features, parent.transform
DECAY = .999
SETTINGS = dict(parent=parent.SETTINGS, change='exponential average of every learned parameter',
    decay=DECAY, initialized='copy of initial parameters', update='after every original AdamW step',
    buffers='unchanged original buffers; no running batch statistics in this model',
    training='original gradients, optimizer, dropout, batches, schedule and21/17 epochs',
    inference='averaged parameters only', no_decay_search=True)


class ShadowAdamW(torch.optim.AdamW):
    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self.original_parameters = [p for group in self.param_groups for p in group['params']]
        self.averages = [p.detach().clone() for p in self.original_parameters]
        self.averaged_steps = 0

    def step(self, closure=None):
        result = super().step(closure)
        with torch.no_grad():
            torch._foreach_mul_(self.averages, DECAY)
            torch._foreach_add_(self.averages, self.original_parameters, alpha=1-DECAY)
        self.averaged_steps += 1
        return result

    def copy_average(self, model):
        assert [id(p) for p in model.parameters()] == [id(p) for p in self.original_parameters]
        assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())
        with torch.no_grad():
            for parameter, average in zip(self.original_parameters, self.averages):
                parameter.copy_(average)


def check(device='cpu'):
    """Two aligned trajectories plus an independent scalar recurrence oracle."""
    torch.manual_seed(173)
    old = nn.Sequential(nn.Linear(4,8),nn.SiLU(),nn.Dropout(.2),nn.Linear(8,1)).to(device)
    torch.manual_seed(173)
    new = nn.Sequential(nn.Linear(4,8),nn.SiLU(),nn.Dropout(.2),nn.Linear(8,1)).to(device)
    original = torch.optim.AdamW(old.parameters(), lr=.002, weight_decay=1e-5)
    shadow = ShadowAdamW(new.parameters(), lr=.002, weight_decay=1e-5)
    schedules = [torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.002,total_steps=12) for opt in [original,shadow]]
    oracle = [p.detach().cpu().numpy().astype(float).copy() for p in new.parameters()]
    x = torch.arange(64,dtype=torch.float32,device=device).reshape(16,4)/32
    y = torch.sin(x.sum(1,keepdim=True))
    for step in range(12):
        for model, optimizer, schedule in zip([old,new],[original,shadow],schedules):
            torch.manual_seed(1000+step)
            loss = (model(x)-y).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); schedule.step()
            if model is old:
                rng = torch.get_rng_state().clone()
                gpu_rng = torch.cuda.get_rng_state(device).clone() if device!='cpu' else None
            else:
                assert torch.equal(rng,torch.get_rng_state())
                if gpu_rng is not None: assert torch.equal(gpu_rng,torch.cuda.get_rng_state(device))
        for i,(a,b) in enumerate(zip(old.parameters(),new.parameters())):
            assert torch.equal(a,b) and torch.equal(a.grad,b.grad)
            for key,value in original.state[a].items():
                if torch.is_tensor(value): assert torch.equal(value,shadow.state[b][key])
                else: assert value==shadow.state[b][key]
            oracle[i] = DECAY*oracle[i]+(1-DECAY)*b.detach().cpu().numpy().astype(float)
        assert schedules[0].get_last_lr()==schedules[1].get_last_lr()
    assert shadow.averaged_steps==12
    maximum = max(float(np.max(np.abs(a.detach().cpu().numpy()-b))) for a,b in zip(shadow.averages,oracle))
    assert maximum<1e-6
    buffers = {name:v.clone() for name,v in new.named_buffers()}
    shadow.copy_average(new)
    for parameter,average in zip(new.parameters(),shadow.averages): assert torch.equal(parameter,average)
    for name,value in new.named_buffers(): assert torch.equal(value,buffers[name])
    assert any(not torch.equal(a,b) for a,b in zip(old.parameters(),new.parameters()))
    return dict(device=device,original_updates_gradients_moments_exact=True,
        original_rng_and_schedule_exact=True,steps=12,independent_recurrence_max_error=maximum,
        average_installed_exactly=True,buffers_unchanged=True)
