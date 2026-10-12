"""Tiny full-model update parity, including partial batches and exact RNG."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import nn_public_cohort_jan_jul as run


class Tracks:
    def __init__(self,values):self.values=values
    def batch(self,rows):return tuple(x[rows] for x in self.values)


def execution_check(device='cpu'):
    run.original.shared.recipe()
    model_type=run.model_type;cards=[3,5];bins=[torch.tensor([-2.,0.,2.])]*2
    bank=model_type.ContextBank(*model_type.synthetic_bank(),'synthetic').to(device)
    train,train_track=run.original.base.prototype.synthetic(cards,bins,15,12,device)
    held,held_track=run.original.base.prototype.synthetic(cards,bins,15,7,device)
    train[2][:,-1]=torch.arange(12,device=device);held[2][:,-1]=torch.arange(12,19,device=device)
    tracks=Tracks(tuple(torch.cat([a,b]) for a,b in zip(train_track,held_track)))
    generator=torch.Generator(device=device).manual_seed(512)
    data=types.SimpleNamespace(cards=cards,bins=bins,train=train,held=held,device=device,
        y=np.arange(12,dtype=float),yv=np.arange(7,dtype=float),mu=1000.,sd=500.,
        ty=torch.randn(12,generator=generator,device=device))
    tr,va=np.arange(12),np.arange(12,19)
    views=[run.fast.RankingView('held',0,7,tracks,va)]
    original=run.original.base.train
    with tempfile.TemporaryDirectory(prefix='cohort_parity_') as directory, \
            patch.dict(run.original.shared.nni.NEW,{'batch':5}):
        namespace=dict(original.__globals__)
        namespace.update(OUT=Path(directory),prototype=types.SimpleNamespace(SparseFusion=
            lambda c,b,p:model_type.SparseFusion(c,b,p,bank)))
        reference=types.FunctionType(original.__code__,namespace,'reference',original.__defaults__,original.__closure__)
        before,prediction,curve=reference(data,tracks,tr,va,2)
        cpu=torch.get_rng_state().clone();numpy=np.random.get_state()
        cuda=torch.cuda.get_rng_state(device).clone() if device.startswith('cuda') else None
        with patch.object(run.fast,'MONITOR_ROWS',3):
            after,new_curve=run.training_function(bank)(data,tracks,tr,views,42,2)
        assert torch.equal(cpu,torch.get_rng_state())
        for a,b in zip(numpy,np.random.get_state()):np.testing.assert_array_equal(a,b)
        if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state(device))
        assert [x['train_mse_std'] for x in curve]==[x['train_mse_std'] for x in new_curve]
        for name,value in before.state_dict().items():
            torch.testing.assert_close(value,after.state_dict()[name],rtol=0,atol=0)
        np.testing.assert_array_equal(prediction,run.predict(after,data,tracks,va))
    assert run.original.base.train is original
    return dict(device=device,partial_batch_updates_exact=True,loss_weights_predictions_exact=True,
        numpy_cpu_cuda_rng_exact=True,original_namespace_unchanged=True)


class CohortTests(unittest.TestCase):
    def test_original_update_parity(self):
        torch.set_num_threads(1);execution_check()
    def test_model_and_lookup_contract(self):
        torch.set_num_threads(1);run.model_type.check('cpu');run.context.checks()


if __name__=='__main__':unittest.main()
