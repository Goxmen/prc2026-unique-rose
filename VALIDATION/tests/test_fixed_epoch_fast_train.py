"""Exact optimized-training parity; only unlabelled prediction work differs."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import fixed_epoch_fast_train as fast
import nn_callsign_clocks_jan_jul as original
from VALIDATION.tests.test_chunked_direct_data import fixture


def data(device):
    item=fixture()
    # Three batches, including an incomplete final batch, with the real N09
    # network/batch/optimizer/scheduler rather than a simplified test trainer.
    item['X']=pd.concat([item['X']]*16,ignore_index=True)
    item['y']=np.tile(item['y'],16)
    item['tmonth']=np.tile(item['tmonth'],16)
    item['Vn']=pd.concat([item['Vn']]*25,ignore_index=True)
    item['raw']=pd.DataFrame({'taxi_sec':np.full(len(item['Vn']),np.nan)})
    return original.DirectData(item,device)


def check_training(device):
    fast.shared.recipe()
    d=data(device)
    rows=np.arange(len(d.y))
    reference,pred1,curve1=fast.shared.train(d,rows,42,2,(d.held,d.yv,d.val_offset))
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    numpy_rng=np.random.get_state()
    candidate,pred2,curve2=fast.train(d,rows,42,2,(d.held,d.yv,d.val_offset))
    for name,value in reference.state_dict().items():
        if not torch.equal(value,candidate.state_dict()[name]):
            raise AssertionError(f'Weight changed: {name}')
    if not torch.equal(cpu_rng,torch.get_rng_state()):
        raise AssertionError('CPU RNG changed')
    if cuda_rng is not None and not torch.equal(cuda_rng,torch.cuda.get_rng_state()):
        raise AssertionError('CUDA RNG changed')
    now=np.random.get_state()
    np.testing.assert_array_equal(numpy_rng[1],now[1])
    if numpy_rng[0]!=now[0] or numpy_rng[2:]!=now[2:]:
        raise AssertionError('NumPy RNG changed')
    np.testing.assert_array_equal([x['train_mse_std'] for x in curve1],[x['train_mse_std'] for x in curve2])
    np.testing.assert_allclose(pred1,pred2,rtol=0,atol=0.01)
    if not candidate.training:
        raise AssertionError('Prediction must restore train mode exactly')
    return d,candidate,dict(device=device,weights_exact=True,rng_exact=True,training_losses_exact=True,
                           training_rows=len(d.y),ranking_rows=len(d.yv),epochs=2,
                           prediction_max_abs_sec=float(np.max(np.abs(pred1-pred2))))


class FastTrainingTests(unittest.TestCase):
    def test_exact_fixed_training_cpu(self):
        check_training('cpu')


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
