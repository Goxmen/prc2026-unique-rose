"""Execution-only portability must preserve exact training updates and RNG."""
from pathlib import Path
import os
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nn_clock_bypass_portable as worker
import nn_sparse_track_prototype as synthetic


class PortableClockTests(unittest.TestCase):
    def test_full_width_initialization_and_bounded_arithmetic(self):
        torch.set_num_threads(2)
        device = os.environ.get('CLOCK_PARITY_DEVICE', 'cpu')
        path = Path(os.environ.get('CLOCK_PREPROCESSING', str(
            worker.ROOT / 'VALIDATION/runs/v3_20261010_nn_sparse_track_jan_jul/preprocessing/2025-01.pkl')))
        with path.open('rb') as handle:
            state = pickle.load(handle)
        result = worker.checks.self_check(state['cards'], state['bins'], len(state['plain']), device)
        self.assertTrue(result['initial_original_gradients_exact'])
        self.assertTrue(result['initial_eval_and_train_outputs_exact'])
        self.assertLessEqual(result['saturated_gate_arithmetic']['max_error_std'], 1e-6)

    def test_exact_candidate_and_direct_updates_with_partial_batches(self):
        torch.set_num_threads(2)
        worker.original.shared.recipe()
        device = os.environ.get('CLOCK_PARITY_DEVICE', 'cpu')
        cards, bins, plain = [3, 5], [torch.tensor([-2., 0., 2.])]*2, 3
        train, _ = synthetic.synthetic(cards, bins, plain, 12, device)
        held, _ = synthetic.synthetic(cards, bins, plain, 7, device)
        def extra(count):
            clocks = torch.arange(count*4, device=device).reshape(count, 4).float()*250
            masks = torch.ones(count, 4, device=device, dtype=torch.bool)
            masks[0] = False
            return clocks, masks
        gen = torch.Generator(device=device).manual_seed(314)
        d = SimpleNamespace(cards=cards, bins=bins, device=device, train=(*train, *extra(12)),
            held=(*held, *extra(7)), y=np.arange(12.), yv=np.arange(7.),
            ty=torch.randn(12, device=device, generator=gen), mu=1000., sd=500., val_offset=np.zeros(7))
        with patch.dict(worker.original.shared.nni.NEW, {'batch': 5}), patch.object(worker, 'PREDICTION_CHUNK', 3):
            for arm in worker.ARMS:
                if arm == 'bypass':
                    ref, ref_pred, ref_curve = worker.original.train(d, 2)
                else:
                    direct = SimpleNamespace(**{**vars(d), 'train': train, 'held': held})
                    ref, ref_pred, ref_curve = worker.original.shared.train(direct, np.arange(12), 42, 2,
                        (held, d.yv, d.val_offset))
                cpu = torch.get_rng_state().clone()
                np_state = np.random.get_state()
                gpu = torch.cuda.get_rng_state(device).clone() if device.startswith('cuda') else None
                model, pred, curve = worker.training_function(arm)(d, 2)
                self.assertTrue(torch.equal(cpu, torch.get_rng_state()), arm)
                for a, b in zip(np_state, np.random.get_state()):
                    np.testing.assert_array_equal(a, b)
                if gpu is not None:
                    self.assertTrue(torch.equal(gpu, torch.cuda.get_rng_state(device)))
                candidate = model if arm == 'bypass' else model.tabular
                for name, value in ref.state_dict().items():
                    self.assertTrue(torch.equal(value, candidate.state_dict()[name]), arm+'/'+name)
                self.assertEqual([v['train_mse_std'] for v in ref_curve],
                                 [v['train_mse_std'] for v in curve])
                np.testing.assert_allclose(ref_pred, pred, rtol=1e-6, atol=1e-4)
        self.assertIs(worker.original.train.__globals__['ClockBypass'], worker.original.ClockBypass)
        self.assertEqual(worker.original.shared.predict.__defaults__[-1], 65536)


if __name__ == '__main__':
    unittest.main()
