"""Production execution parity and overlapping-ID ranking namespaces."""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fixed_epoch_sparse_train as fast
import nn_sparse_track_pool3 as pool


class Tracks:
    def __init__(self, ids, values, ranking_view=None):
        self.ids = {int(value): i for i, value in enumerate(ids)}
        self.values = values
        self.ranking_view = ranking_view

    def align(self, ids):
        return np.array([self.ids[int(value)] for value in ids], dtype=np.int64)

    def batch(self, rows):
        return tuple(value[rows] for value in self.values)


class Echo(torch.nn.Module):
    def forward(self, cat, g, plain, points, mask, quality):
        return points[:, 0, :1]+plain[:, :1]


class FixedSparseExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_overlapping_ids_use_distinct_points_and_tabular_slices(self):
        ids = np.array([101, 102])
        def store(values, view):
            return Tracks(ids, (torch.tensor(values, dtype=torch.float32)[:, None, None],
                                torch.ones(2, 1, dtype=torch.bool), torch.zeros(2, 1)), view)
        stores = {'old': store([10., 20.], 'old'), 'final': store([100., 200.], 'final')}
        data = SimpleNamespace(held=(torch.zeros(4, 1), torch.zeros(4, 1),
                                     torch.tensor([[1.], [2.], [3.], [4.]])), mu=0., sd=1.)
        views = fast.ranking_views(data, {'old': ids, 'final': ids}, stores)
        np.testing.assert_array_equal(fast.predict_view(Echo(), data, views[0]), [11., 22.])
        np.testing.assert_array_equal(fast.predict_view(Echo(), data, views[1]), [103., 204.])
        with self.assertRaisesRegex(ValueError, 'ranking_view manifest'):
            fast.ranking_views(data, {'old': ids, 'final': ids}, {'old': stores['old'], 'final': stores['old']})

    def test_exact_weights_loss_rng_and_final_predictions(self):
        pool.shared.recipe()
        device = os.environ.get('SPARSE_PARITY_DEVICE', 'cpu')
        cards = [3, 5]
        bins = [torch.tensor([-2., 0., 2.])] * 2
        train, track_values = pool.base.prototype.synthetic(cards, bins, 3, 12, device)
        held, held_values = pool.base.prototype.synthetic(cards, bins, 3, 7, device)
        train_tracks = Tracks(np.arange(12), track_values)
        held_tracks = Tracks(np.arange(7), held_values, 'old')
        combined_tracks = Tracks(np.arange(19), tuple(torch.cat([a, b])
            for a, b in zip(track_values, held_values)))
        generator = torch.Generator(device=device).manual_seed(512)
        data = SimpleNamespace(cards=cards, bins=bins, train=train, held=held, device=device,
            y=np.arange(12, dtype=float), yv=np.arange(7, dtype=float), mu=1000., sd=500.,
            ty=torch.randn(12, generator=generator, device=device))
        train_rows = train_tracks.align(np.arange(12))
        held_rows = combined_tracks.align(np.arange(12, 19))
        views = fast.ranking_views(data, {'old': np.arange(7)}, {'old': held_tracks})
        # CUDA convolution backward is nondeterministic even for original/original
        # repeats. Scope deterministic kernels to this exact arithmetic test;
        # production retains its original backend flags and all assertions stay exact.
        with tempfile.TemporaryDirectory(prefix='sparse_execution_parity_') as directory, \
                patch.dict(pool.shared.nni.NEW, {'batch': 5}), \
                torch.backends.cudnn.flags(deterministic=True, benchmark=False):
            reference = pool.compile_training(directory)
            old, old_pred, old_curve = reference(data, combined_tracks, train_rows, held_rows, 2, 1042)
            old_rng = torch.get_rng_state().clone()
            old_numpy = np.random.get_state()
            old_cuda = torch.cuda.get_rng_state(device).clone() if device.startswith('cuda') else None
            with patch.object(fast, 'MONITOR_ROWS', 3):
                new, predictions, new_curve = fast.train(data, train_tracks, train_rows, views, 1042, 2)
            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
            for old_value, new_value in zip(old_numpy, np.random.get_state()):
                np.testing.assert_array_equal(old_value, new_value)
            if old_cuda is not None:
                self.assertTrue(torch.equal(old_cuda, torch.cuda.get_rng_state(device)))
            self.assertEqual([row['train_mse_std'] for row in old_curve],
                             [row['train_mse_std'] for row in new_curve])
            for name, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
            np.testing.assert_array_equal(old_pred, predictions['old'])


if __name__ == '__main__':
    unittest.main()
