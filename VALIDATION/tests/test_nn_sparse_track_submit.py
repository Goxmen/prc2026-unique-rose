"""Bounded production namespace regression with real ragged transforms."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nn_sparse_track_submit as submit


class ProductionSparseTests(unittest.TestCase):
    def test_saved_training_survives_partial_inference_without_refit(self):
        with tempfile.TemporaryDirectory(dir=submit.ROOT / 'tmp', prefix='sparse_resume_') as folder:
            target = Path(folder)
            with patch.object(submit, 'OUT', target), patch.object(submit, 'IDENTITY', target / 'identity.json'):
                submit.parent.write_json(submit.IDENTITY, {'synthetic': True})
                model, _ = submit.member_paths(0)
                model.parent.mkdir(parents=True)
                model.write_bytes(b'saved fixed updates')
                pre = target / 'preprocessing/full_year.pkl'
                pre.parent.mkdir()
                pre.write_bytes(b'saved preprocessing')
                record = dict(identity=submit.parent.artifact(submit.IDENTITY), seed=42,
                    epochs=15, config={'frozen': True}, model=submit.parent.artifact(model),
                    preprocessing=submit.parent.artifact(pre),
                    training_curve=[{'epoch': i+1} for i in range(15)])
                submit.parent.write_json(submit.training_path(0), record)
                identity = {'config': {'frozen': True}}
                self.assertEqual(submit.trained_member(0, identity), record)
                self.assertIsNone(submit.completed_member(0, identity))
                partial = target / 'networks/old_s42_k0.parquet'
                partial.parent.mkdir()
                partial.write_bytes(b'partial ranking output')
                self.assertIsNone(submit.completed_member(0, identity))
                model.write_bytes(b'changed model bytes')
                with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                    submit.trained_member(0, identity)

    def test_same_id_jittered_clock_uses_separate_raw_cache(self):
        torch.set_num_threads(2)
        start = int(pd.Timestamp('2026-07-10T12:00:00Z').timestamp()*1000)
        query = pd.DataFrame([dict(mvt_id=101, airport='TEST', hex=123,
            to_ms=start, clock_ms=start-600000, day='2026-07-10',
            origin_lat=50., origin_lon=4., stand_lat=50., stand_lon=4.,
            runway_lat=np.nan, runway_lon=np.nan)])
        points = pd.DataFrame(dict(airport=['TEST']*2, hex=[123]*2,
            ts_ms=[start-10000, start], lat=[50000000, 50000001],
            lon=[4000000, 4000001], alt=[-123]*2, gs=[0, 10]))
        with tempfile.TemporaryDirectory(prefix='sparse_submit_namespace_') as folder:
            stores = {}
            for view, shift in [('old', 0), ('final', 12000)]:
                own = query.assign(to_ms=start+shift)
                index, raw = submit.cv.seq.select_sequences(own, points, {'2026-07-10'})
                target = Path(folder) / view
                target.mkdir()
                index.to_parquet(target / 'full_index.parquet', index=False)
                raw.astype('<i4').tofile(target / 'full_points.bin')
                identity = dict(ranking_view=view, rows=1, stored_points=len(raw))
                stores[view] = submit.load_tracks(target, identity, 'cpu', view)
                with self.assertRaisesRegex(ValueError, 'ranking_view'):
                    submit.load_tracks(target, identity, 'cpu', 'wrong')
            old = stores['old'].batch(stores['old'].align([101]))[0].numpy()
            final = stores['final'].batch(stores['final'].align([101]))[0].numpy()
            np.testing.assert_allclose(final[0, :2, 0]-old[0, :2, 0], 12000/7200000,
                                       rtol=0, atol=1e-9)
            # Genuine AOBT is unchanged: point-relative-to-AOBT channel is equal.
            np.testing.assert_array_equal(final[0, :2, 14], old[0, :2, 14])
            self.assertNotEqual(stores['old'].manifest_identity, stores['final'].manifest_identity)


if __name__ == '__main__':
    unittest.main()
