"""Exact full-versus-chunked encoding, bins, targets and RNG parity."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chunked_direct_data as chunked
import nn_callsign_pool_resume as equality


def fixture():
    n = 257
    x = np.arange(n, dtype=float)
    train = pd.DataFrame({
        'airport': pd.Categorical(np.where(x < 150, 'B', 'A'), categories=['unused', 'A', 'B']),
        'operator': pd.Categorical(np.where(x < 12, 'rare', 'common'), categories=['common', 'rare', 'unused']),
        'varied': np.where(x % 11 == 0, np.nan, np.sin(x) * (x+1)),
        'infinite': np.where(x % 9 == 0, np.inf, np.where(x % 13 == 0, -np.inf, x)),
        'constant': np.full(n, 4., np.float32),
        'missing': np.full(n, np.nan),
        'ties': np.floor(x/9),
        'boolean': x % 2 == 0,
    }, index=np.arange(n)*3+7)
    held = train.iloc[::5].copy()
    held['airport'] = pd.Categorical(['new'] * len(held), categories=['B', 'new', 'A'])
    held.loc[held.index[2], 'operator'] = np.nan
    held.loc[held.index[3], 'varied'] = np.inf
    target = 400 + x*17
    target[::31] = 22000
    months = np.where(x % 3 == 0, '2025-11', '2025-03')
    return dict(X=train, Vn=held, y=target, raw=pd.DataFrame({'taxi_sec': np.nan}, index=held.index),
                tmonth=months, meta={'month': '2025-01'})


class ChunkedDataTests(unittest.TestCase):
    def test_original_transform_exact_with_chunks(self):
        item = fixture()
        encoder = chunked.original.shared.nnj.Encoder(item['X'])
        expected_cat, expected_num = encoder.transform(item['X'])
        with tempfile.TemporaryDirectory() as temp:
            cat, num = chunked.encode(encoder, item['X'], temp, 'train', chunk_rows=17)
            np.testing.assert_array_equal(cat, expected_cat, strict=True)
            np.testing.assert_array_equal(num, expected_num, strict=True)
            self.assertTrue(num.flags.c_contiguous)
            self.assertTrue(cat.flags.c_contiguous)
            cat._mmap.close()
            num._mmap.close()

    def test_complete_direct_data_and_preprocessing_rng_exact(self):
        chunked.original.shared.recipe()
        item = fixture()
        torch.manual_seed(718)
        np.random.seed(719)
        original_cpu_rng = torch.get_rng_state().clone()
        original_np_rng = np.random.get_state()
        reference = chunked.original.DirectData(item, 'cpu')
        with tempfile.TemporaryDirectory() as temp:
            actual = chunked.ChunkedDirectData(item, 'cpu', temp, chunk_rows=17)
        for name in ['train', 'held']:
            for a, b in zip(getattr(reference, name), getattr(actual, name)):
                self.assertTrue(torch.equal(a, b), name)
                self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(a.stride(), b.stride())
        for name in ['ty']:
            self.assertTrue(torch.equal(getattr(reference, name), getattr(actual, name)))
        for name in ['offset', 'val_offset', 'inner_train', 'inner_val', 'y', 'yv']:
            np.testing.assert_array_equal(getattr(reference, name), getattr(actual, name), strict=True)
        equality.equal_preprocessing(reference.preprocessing(), actual.preprocessing())
        self.assertTrue(torch.equal(torch.get_rng_state(), original_cpu_rng))
        after_np = np.random.get_state()
        self.assertEqual(original_np_rng[0], after_np[0])
        np.testing.assert_array_equal(original_np_rng[1], after_np[1])
        self.assertEqual(original_np_rng[2:], after_np[2:])
        torch.manual_seed(42)
        model = chunked.original.shared.nni.PleNet(reference.cards, reference.bins, reference.train[2].shape[1]).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(*reference.held), model(*actual.held)))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
