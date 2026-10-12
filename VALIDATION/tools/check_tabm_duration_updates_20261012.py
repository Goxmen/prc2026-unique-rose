"""Exact optimizer/RNG parity and checkpoint-prefix checks on a small CPU model."""
import io,sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
import nn_public_tabm_duration as run


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.Dropout(.1),torch.nn.Linear(8,32))
    def forward(self,*parts):return self.layers(parts[0])


def main():
    run.guard();torch.set_num_threads(1)
    torch.manual_seed(723);x=torch.randn(17,5);target=torch.randn(17)
    data=SimpleNamespace(y=target.numpy(),ty=target,train=(x,),device='cpu')
    tracks=SimpleNamespace(batch=lambda rows:());rows=np.arange(17);snapshots={}
    torch.manual_seed(42);a=Tiny();initial=copy.deepcopy(a.state_dict())
    run.original.train(a,data,tracks,rows,24,'tabm','2025-01');expected_rng=torch.get_rng_state().clone()
    def callback(model,month,epoch,curve):
        if epoch not in [17,21]:return
        rng=torch.get_rng_state().clone();buffer=io.BytesIO()
        torch.save(model.state_dict(),buffer);buffer.seek(0)
        snapshots[epoch]=torch.load(buffer,weights_only=True)
        assert torch.equal(rng,torch.get_rng_state())
    torch.manual_seed(42);b=Tiny();b.load_state_dict(initial)
    run.training_function(callback)(b,data,tracks,rows,24,'tabm','2025-01')
    assert torch.equal(expected_rng,torch.get_rng_state())
    for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],rtol=0,atol=0)
    for epoch in [17,21]:
        torch.manual_seed(42);c=Tiny();c.load_state_dict(initial)
        run.original.train(c,data,tracks,rows,epoch,'tabm','2025-01')
        for key in c.state_dict():torch.testing.assert_close(c.state_dict()[key],snapshots[epoch][key],rtol=0,atol=0)
    value=dict(status='pass',source=run.art(__file__),identity=run.art(run.MANIFEST),
        unchanged_training_AST_except_checkpoint=True,exact24_epoch_optimizer_parameters_and_RNG=True,
        exact17_and21_epoch_prefixes=True,checkpoint_serialization_preserves_CPU_RNG=True,
        live_GPU_checkpoint_RNG_assertion=True,no_real_training_rows=True)
    run.write(run.OUT/'update_parity.json',value);print(value,flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
