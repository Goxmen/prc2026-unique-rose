"""Portable callsign packet semantics and isolated view reconstruction."""
import gzip
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import callsign_submit_portable as worker


class PortableCallsTests(unittest.TestCase):
    def setup_frames(self):
        names = ['airport', 'hour']
        train = pd.DataFrame({'airport': pd.Categorical(['A', 'B'], categories=['A', 'B']),
                              'hour': np.array([1, 2], np.float32)}, index=[90, 91])
        old = pd.DataFrame({'airport': pd.Categorical(['A', 'B'], categories=['A', 'B']),
                            'hour': np.array([3, 4], np.float32)}, index=[7, 8])
        final = pd.DataFrame({'airport': pd.Categorical(['B', 'C'], categories=['C', 'B']),
                              'hour': np.array([5, 6], np.float32)}, index=[2, 1])
        data = dict(Xn=train, Vn=old, rank_ids=np.array([101, 102], np.int64))
        ids = np.array([11, 12], np.int64)
        packets = dict(old=dict(view='old', Vn=old, rank_ids=data['rank_ids']),
                       final=dict(view='final', Vn=final, rank_ids=np.array([101, 102], np.int64)))
        tables = {view: pd.DataFrame({c: np.full(2, value, np.float32)
                    for c in worker.parent.cv.FEATURES}, index=pd.Index(keys, name='mvt_id'))
                  for view, keys, value in [('train', ids, 0), ('old', [101, 102], 1),
                                            ('final', [101, 102], 2)]}
        return data, ids, names, packets, tables

    def test_exact_parent_assembly_and_isolated_overlapping_ids(self):
        data, ids, names, packets, tables = self.setup_frames()
        X, V, view_ids, sizes = worker.assemble_inputs(data, ids, names, tables, packets)
        expected_x = worker.parent.cv.add_features(data['Xn'][names], ids, tables['train'])
        expected_parts = [worker.parent.cv.add_features(packets[v]['Vn'][names],
                           packets[v]['rank_ids'], tables[v]) for v in worker.parent.VIEWS]
        expected_v = pd.concat(worker.parent.fs.widen(expected_parts), ignore_index=True)
        pd.testing.assert_frame_equal(X, expected_x, check_exact=True)
        pd.testing.assert_frame_equal(V, expected_v, check_exact=True)
        self.assertEqual(sizes, {'old': 2, 'final': 2})
        np.testing.assert_array_equal(view_ids['old'], view_ids['final'])
        np.testing.assert_array_equal(V.cc_call_start_sec, [1, 1, 2, 2])
        self.assertEqual(V.airport.astype(object).tolist(), ['A', 'B', 'B', 'C'])

    def test_old_view_reuses_cache_and_final_serialization_preserves_categories(self):
        data, _, names, packets, _ = self.setup_frames()
        manifest = dict(columns=names, views={'old': {'rows': 2}, 'final': {'rows': 2}})
        old = worker.load_packet('old', manifest, data=data)
        pd.testing.assert_frame_equal(old['Vn'], data['Vn'], check_exact=True)
        with tempfile.TemporaryDirectory(dir=worker.ROOT / 'tmp', prefix='callsign_packet_test_') as temp:
            path = Path(temp) / 'final.pkl.gz'
            with gzip.open(path, 'wb') as stream:
                pickle.dump(packets['final'], stream, protocol=pickle.HIGHEST_PROTOCOL)
            manifest['views']['final']['packet'] = worker.parent.artifact(path)
            final = worker.load_packet('final', manifest)
            pd.testing.assert_frame_equal(final['Vn'], packets['final']['Vn'], check_exact=True)
            self.assertEqual(final['Vn'].airport.cat.categories.tolist(), ['C', 'B'])
            manifest['views']['final']['rows'] = 3
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                worker.load_packet('final', manifest)


if __name__ == '__main__':
    unittest.main()
