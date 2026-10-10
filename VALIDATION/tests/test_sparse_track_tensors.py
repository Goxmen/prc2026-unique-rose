from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sparse_track_sequence as seq
import sparse_track_tensors as tensors
from test_sparse_track_sequence import query,positions


class TestSparseTensors(unittest.TestCase):
    def test_full_empty_cache_masks_and_missing_id(self):
        q=pd.DataFrame([query(1,hex=np.nan),query(2,hex=np.nan)])
        index,points=seq.select_sequences(q,pd.DataFrame(),{'2025-01-10'})
        tracks=tensors.RaggedTracks(index,points,'cpu')
        rows=tracks.align([2,1])
        features,mask,quality=tracks.batch(rows)
        self.assertFalse(features.any());self.assertFalse(mask.any())
        self.assertTrue(torch.isfinite(quality).all())
        with self.assertRaises(ValueError):tracks.align([3])

    def test_vectorized_geometry_clock_missingness_and_alignment(self):
        q=query();q['clock_ms']=q['to_ms']-1800000
        queries=pd.DataFrame([q,query(2,hex=np.nan),query(3,stand_lat=50.001,stand_lon=4.002)])
        p=positions([q['to_ms']-3600000,q['to_ms']-1800000,q['to_ms']],
                    alt=[-123,-2,400],gs=[50,-1,100])
        index,points=seq.select_sequences(queries,p,{'2025-01-09','2025-01-10'})
        tracks=tensors.RaggedTracks(index,points,'cpu')
        rows=tracks.align([3,1,2])
        expected=seq.encoded_batch(index.iloc[rows],points)
        for got,want in zip(tracks.batch(rows),expected):
            np.testing.assert_allclose(got.numpy(),want,atol=2e-7,rtol=2e-6)

    def test_recorded_cache_identity_is_manifest_not_last_geometry_map(self):
        import nn_sparse_track_jan_jul as runner
        import build_sparse_track_full as full
        digest=lambda path:'hash:'+str(path)
        manifest=dict(source_sha256=digest(runner.ROOT/'VALIDATION/build_sparse_track_full.py'),
            settings=dict(original_sequence=seq.SETTINGS,full_days=full.DAYS,labels_used=False),
            outputs={name:digest(seq.OUT/name) for name in ['full_index.parquet','full_points.bin']},
            helpers={},identity_sha256=digest(seq.IDENTITY),stand_map_sha256=digest(seq.sf.COORDS),
            runway_map_sha256=digest(seq.sf.ENDS),movement_sources={},positions={},rows=2,stored_points=3)
        with patch.object(runner.shared.ts,'read',return_value=manifest),patch.object(runner,'sha256',side_effect=digest):
            result=runner.cache_identity()
        self.assertEqual(result['path'],str((seq.OUT/'full.json').relative_to(runner.ROOT)))
        self.assertEqual(result['sha256'],digest(seq.OUT/'full.json'))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
