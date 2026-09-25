"""Guard stand-name matching and map parsing for the stand coordinates (idea 34)."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stand_geometry as sg


class CanonTests(unittest.TestCase):
    def test_spellings_of_one_stand_agree(self):
        for a, b in [('H01', 'H1'), ('Gate E 58', 'E58'), ('Stand W 42', 'W42'),
                     ('K08-(E)', 'K8'), ('Gate 126A', '126A'), ('b02', 'B2'), ('Ramp 36', '36')]:
            self.assertEqual(sg.canon(a), sg.canon(b), (a, b))

    def test_zeros_inside_a_number_are_kept(self):
        self.assertEqual(sg.canon('100'), '100')
        self.assertEqual(sg.canon('V105'), 'V105')
        self.assertEqual(sg.canon('0'), '0')

    def test_ranges_and_blanks_are_not_stands(self):
        self.assertIsNone(sg.canon('Ramp 1 - 2'))
        self.assertIsNone(sg.canon(None))
        self.assertIsNone(sg.canon(float('nan')))
        self.assertIsNone(sg.canon('  '))

    def test_label_words_are_only_stripped_as_words(self):
        self.assertEqual(sg.canon('GAC 103'), 'GAC103')
        self.assertEqual(sg.canon('Gateway 1'), 'GATEWAY1')


class MatchTests(unittest.TestCase):
    idx = {'210A': (48.0, 11.0), '210B': (48.0002, 11.0), '151': (48.01, 11.0),
           'E58': (48.02, 11.0), 'E5': (48.03, 11.0)}

    def test_exact(self):
        self.assertEqual(sg.match('E5', self.idx, 'exact'), (48.03, 11.0))

    def test_substands_average_their_halves(self):
        lat, lon = sg.match('210', self.idx, 'substands')
        self.assertAlmostEqual(lat, 48.0001)
        self.assertAlmostEqual(lon, 11.0)

    def test_substands_need_exactly_one_trailing_letter(self):
        self.assertIsNone(sg.match('21', self.idx, 'substands'))
        self.assertIsNone(sg.match('E', self.idx, 'substands'))

    def test_substands_far_apart_are_not_one_stand(self):
        far = {'5A': (48.0, 11.0), '5B': (48.01, 11.0)}
        self.assertIsNone(sg.match('5', far, 'substands'))

    def test_parent_drops_only_a_letter_after_a_digit(self):
        self.assertEqual(sg.match('151A', self.idx, 'parent'), (48.01, 11.0))
        self.assertIsNone(sg.match('151', self.idx, 'parent'))
        self.assertIsNone(sg.match('A', self.idx, 'parent'))

    def test_index_drops_one_name_for_two_places(self):
        stands = pd.DataFrame({'name': ['A1', 'A01', 'B2', 'Stand B2'],
                               'lat': [48.0, 48.0002, 48.0, 48.02], 'lon': [11.0] * 4})
        idx = sg.index(stands)
        self.assertAlmostEqual(idx['A1'][0], 48.0001)
        self.assertNotIn('B2', idx)


class LocateTests(unittest.TestCase):
    maps = {'xplane': {'210A': (48.0, 11.0), '210B': (48.0002, 11.0), '103': (47.456, 8.556)},
            'osm': {'210': (48.1, 11.1), '103': (47.454, 8.573)}}

    def test_exact_in_either_map_beats_approximate(self):
        got = sg.locate('210', self.maps)
        self.assertEqual((got['source'], got['rule']), ('osm', 'exact'))

    def test_xplane_first_unless_overridden(self):
        self.assertEqual(sg.locate('103', self.maps, 'EDDF')['source'], 'xplane')
        self.assertEqual(sg.locate('103', self.maps, 'LSZH')['source'], 'osm')

    def test_unknown_and_missing_have_no_position(self):
        for name in ('UNKNOWN', None, 'Z99'):
            self.assertTrue(np.isnan(sg.locate(name, self.maps)['lat']), name)


class ParseTests(unittest.TestCase):
    def test_apt_stands_and_both_runway_ends(self):
        text = '\n'.join([
            '1300  50.03350000  008.55000000 159.00 gate heavy|jets Gate A 14',
            '100 60.00 1 0 0.25 1 3 0 07C  50.03  8.53  120 0 3 0 1 0 25C  50.04  8.58  0 0 3 0 1 0',
            '1201 50.0 8.5 both 1 node'])
        stands, ends = sg.parse_apt(text)
        self.assertEqual(stands.name.tolist(), ['Gate A 14'])
        self.assertEqual(sg.canon(stands.name[0]), 'A14')
        self.assertEqual(ends.runway.tolist(), ['07C', '25C'])
        self.assertEqual(ends.displaced_m.tolist(), [120.0, 0.0])

    def test_osm_nodes_ways_and_ref_lists(self):
        data = {'elements': [
            {'type': 'node', 'lat': 1.0, 'lon': 2.0, 'tags': {'aeroway': 'parking_position', 'ref': 'A1;A2'}},
            {'type': 'way', 'geometry': [{'lat': 0.0, 'lon': 0.0}, {'lat': 2.0, 'lon': 4.0}],
             'tags': {'aeroway': 'parking_position', 'ref': 'B1'}},
            {'type': 'node', 'lat': 9.0, 'lon': 9.0, 'tags': {'aeroway': 'parking_position'}},
            {'type': 'way', 'geometry': [], 'tags': {'aeroway': 'apron', 'ref': 'X'}}]}
        got = sg.parse_osm(data).values.tolist()
        self.assertEqual(got, [['A1', 1.0, 2.0], ['A2', 1.0, 2.0], ['B1', 1.0, 2.0]])


@unittest.skipUnless((sg.OUT_DIR / 'stand_coords.parquet').exists(), 'stand table not built')
class BuiltTableTests(unittest.TestCase):
    """The resolved conflicts and the 2026 stands, on the real table."""

    @classmethod
    def setUpClass(cls):
        t = pd.read_parquet(sg.OUT_DIR / 'stand_coords.parquet')
        cls.t = t.set_index(['airport', 'stand'])

    def test_overrides_took_effect(self):
        for ap, st in [('LSZH', '101'), ('LSZH', '105'), ('EDDM', '103'), ('EDDM', '105')]:
            self.assertEqual(self.t.loc[(ap, st), 'source'], 'osm', (ap, st))

    def test_new_2026_stands_have_positions(self):
        for ap, st in [('EDDM', '103'), ('EDDM', '108'), ('EDDF', 'J5'), ('EDDF', 'J10')]:
            self.assertTrue(np.isfinite(self.t.loc[(ap, st), 'lat']), (ap, st))

    def test_one_row_per_named_stand(self):
        named = self.t.reset_index().dropna(subset=['stand'])
        self.assertFalse(named.duplicated(['airport', 'stand']).any())


if __name__ == '__main__':
    unittest.main()
