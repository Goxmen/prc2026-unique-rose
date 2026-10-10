"""Production identity namespace, raw-clock jitter and boundary-availability checks."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_sparse_track_submit as b


def query(mid=1, stamp='2026-01-01T00:01:00Z', clock=np.nan, hex=123):
    t = int(pd.Timestamp(stamp).timestamp()*1000)
    return pd.DataFrame([dict(mvt_id=mid, airport='TEST', stand='A', runway='01',
        to_ms=t, clock_ms=clock, hex=hex, day=stamp[:10], stand_lat=np.nan,
        stand_lon=np.nan, runway_lat=50., runway_lon=4., origin_lat=50., origin_lon=4.)])


class TestProductionSequence(unittest.TestCase):
    def test_identity_rejects_cross_view_jitter_and_airport(self):
        q = query().drop(columns=['hex', 'day', 'stand_lat', 'stand_lon',
                                  'runway_lat', 'runway_lon', 'origin_lat', 'origin_lon'])
        identity = q[['mvt_id', 'airport', 'to_ms']].assign(phase='DEP', hex=123)
        joined = b.join_identity(q, identity)
        self.assertEqual(joined.hex.iloc[0], 123)
        self.assertTrue(joined.clock_ms.isna().all())
        for changed in [identity.assign(to_ms=identity.to_ms+12000),
                        identity.assign(airport='OTHER'), identity.assign(phase='ARR'),
                        pd.concat([identity, identity], ignore_index=True)]:
            with self.assertRaises(ValueError):
                b.join_identity(q, changed)

    def test_allowed_schema_and_raw_missing_clock(self):
        q = query()
        with patch.dict(b.ROWS, old=1), patch.dict(b.MONTHS, old=['2026-01']):
            b.validate_query(q, 'old')
            with self.assertRaises(ValueError):
                b.validate_query(q.assign(TAXITIME_SEC_mvt=12), 'old')
            with self.assertRaises(ValueError):
                b.validate_query(q.assign(day='2026-07-01'), 'old')
            with self.assertRaises(ValueError):
                b.validate_query(q.assign(hex=123.5), 'old')

    def test_shared_id_gets_separate_jittered_window_and_missing_flags(self):
        old = query(stamp='2026-07-31T23:59:48Z')
        final = query(stamp='2026-08-01T00:00:00Z')
        # Frozen selection independently reflects final takeoff jitter, including
        # window membership, relative timestamps and source availability.
        t = old.to_ms.iloc[0]
        position = pd.DataFrame(dict(airport=['TEST']*3, hex=[123]*3,
            ts_ms=[t-7200000, t-1000, t], lat=[50000000]*3, lon=[4000000]*3,
            alt=[-123]*3, gs=[50]*3))
        io, po = b.seq.select_sequences(old, position, {'2026-07-31'})
        iff, pf = b.seq.select_sequences(final, position, {'2026-07-31'})
        self.assertEqual(io['count'].iloc[0], 3)
        self.assertEqual(iff['count'].iloc[0], 2)
        np.testing.assert_array_equal(po[:, 0], [7200000, 1000, 0])
        np.testing.assert_array_equal(pf[:, 0], [13000, 12000])
        np.testing.assert_array_equal(pf[:, 5], [-1, 1000])
        self.assertEqual(io.right_source_missing.iloc[0], 1)
        self.assertEqual(iff.right_source_missing.iloc[0], 1)
        missing, points = b.seq.select_sequences(final.assign(hex=np.nan), position, {'2026-07-31'})
        self.assertEqual(len(points), 0)
        self.assertEqual(missing.right_source_missing.iloc[0], 1)
        result = b.overlap(old, final)
        self.assertEqual(result['takeoff_jitter_ms_max'], 12000)
        self.assertFalse(result['arrays_reused_between_views'])
        with self.assertRaises(ValueError):
            b.overlap(old, final.assign(to_ms=final.to_ms+1))

    def test_missing_boundary_cannot_silently_appear(self):
        with tempfile.TemporaryDirectory(dir=b.ROOT/'tmp', prefix='submit_seq_') as folder:
            heat = Path(folder)
            (heat/'positions').mkdir()
            path = heat/'positions/2026-02-28.parquet'
            path.write_bytes(b'fixed source')
            with patch.object(b.seq, 'HEATMAP', heat):
                positions, absent = b.position_inventory(['2026-02-28'])
                manifest = dict(positions=positions, missing_position_days=absent,
                    views={'final': dict(days=['2026-02-28'])})
                b.validate_positions(manifest)
                (heat/'positions/2026-03-01.parquet').write_bytes(b'new source')
                with self.assertRaises(ValueError):
                    b.validate_positions(manifest)

    def test_transfer_allowlist_excludes_large_original_caches(self):
        manifest = dict(helpers={'VALIDATION/sparse_track_sequence.py': 'a'},
            training_contract=dict(manifest=dict(path='data/derived/train/full.json'),
                                   design=dict(path='data/derived/train/full_design.json')),
            maps=[dict(path='data/maps.parquet')], positions=[dict(path='data/positions/day.parquet')],
            views={'old': dict(query=dict(path='data/derived/output/old/queries.parquet'),
                               identity=dict(path='data/identity.parquet'),
                               raw_ranking=dict(path='data/ranking.parquet'))})
        files = b.portable_files(b.OUT, manifest)
        self.assertIn('data/derived/output/old/queries.parquet', files)
        self.assertNotIn('data/identity.parquet', files)
        self.assertNotIn('data/ranking.parquet', files)
        self.assertNotIn('data/derived/train/full_points.bin', files)


if __name__ == '__main__':
    unittest.main()
