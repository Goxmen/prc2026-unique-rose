"""Checks for the Rome signal inputs, the fitted corrections and their prediction path."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rome_hybrid as h
import rome_signals as s


class Stub:
    def __init__(self, values):
        self.values = np.asarray(values, float)
        self.seen = []

    def predict(self, X, num_iteration=None, num_threads=None):
        self.seen.append(list(X.columns))
        return self.values[X.index.to_numpy()]


def frames(gaps, airports, id_missing, takeoff='2025-07-01 12:00'):
    n = len(gaps)
    t = pd.Timestamp(takeoff, tz='UTC')
    raw = pd.DataFrame({
        'ADEP_mvt': airports,
        'FLIGHT_ID_mvt': [np.nan if m else 1.0 for m in id_missing],
        'MVT_TIME_UTC_mvt': [t] * n,
        'SCHED_TIME_UTC_mvt': [t - pd.Timedelta(seconds=g) for g in gaps],
        'raw_flight_id_missing': id_missing, 'raw_plan_block_missing': id_missing,
    })
    V = pd.DataFrame(0.0, index=raw.index, columns=h.regressor_features('clock'))
    V[h.GAP] = np.asarray(gaps, float)
    return V, raw


class FlightNumberTests(unittest.TestCase):
    def test_airline_code_reads_icao_or_iata_prefix(self):
        codes = s.airline_code(['ITY630', 'ITYTY630', 'LH1109', 'U21234', 'KL0565', None, 'U2'])
        self.assertEqual(codes.iloc[:5].tolist(), ['ITY', 'ITY', 'LH', 'U2', 'KL'])
        self.assertTrue(codes.iloc[5:].isna().all())

    def test_format_flags_mangled_clean_and_missing(self):
        # T78T7850 repeats its prefix, but a prefix with digits is not an ICAO airline code
        flights = ['ITYTY630', 'ARGG1141', 'WZZZZ401', 'T78T7850', 'WMT5640', 'AFR1234A', 'LH1109', None]
        self.assertEqual(s.flight_number_format(flights).tolist(), [1, 1, 1, 0, 0, 0, 0, 2])


class HandlerTests(unittest.TestCase):
    def test_share_is_scoped_to_airport_airline_and_month_and_shrunk(self):
        sched = pd.Timestamp('2025-07-10 10:00', tz='UTC')
        rows = [('LIRF', 'ISR1', sched, sched, 0.0)] * 3 + [('LIRF', 'ISR2', sched, sched, 600.0)]
        rows += [('LIRF', 'KLM1', sched, sched, 600.0)] * 4
        rows += [('LIRF', 'ISR3', sched + pd.Timedelta(days=30), sched + pd.Timedelta(days=30), 0.0)] * 5
        rows += [('EGLL', 'ISR4', sched, sched, 0.0)] * 5
        arr = pd.DataFrame(rows, columns=['ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'taxi_sec'])
        # landing at 09:59 with a 121 s taxi-in blocks in 61 s after schedule: not a match
        arr.loc[len(arr)] = ['LIRF', 'ISR5', sched - pd.Timedelta(seconds=60), sched, 121.0]
        dep = pd.DataFrame({'ADEP_mvt': ['LIRF', 'LIRF', 'LIRF', 'LIRF'],
                            'FLIGHT_mvt': ['ISRSR342', 'KLM1', 'DAL1', None],
                            'MVT_TIME_UTC_mvt': [sched + pd.Timedelta(days=1)] * 4})
        out = s.handler_rates(dep, arr)
        prior = 3 / 9  # July LIRF arrivals: 3 ISR matches of 9 (the 60 s + 120 s landing does not match)
        np.testing.assert_allclose(out.handler_arr_n.iloc[:3], [5, 4, 0])
        np.testing.assert_allclose(out.handler_arr_sched_share.iloc[:3],
                                   [(3 + 10 * prior) / 15, (0 + 10 * prior) / 14, prior], rtol=1e-6)
        self.assertTrue(out.iloc[3].isna().all())

    def test_add_signals_appends_categorical_airline_and_refuses_duplicates(self):
        t = pd.Timestamp('2025-07-10 10:00', tz='UTC')
        raw = pd.DataFrame({'ADEP_mvt': ['LIRF'], 'FLIGHT_mvt': ['ITYTY630'], 'MVT_TIME_UTC_mvt': [t]})
        arr = pd.DataFrame({'ADES_mvt': ['LIRF'], 'FLIGHT_mvt': ['ITY631'], 'MVT_TIME_UTC_mvt': [t],
                            'SCHED_TIME_UTC_mvt': [t], 'taxi_sec': [300.0]})
        X = s.add_signals(pd.DataFrame({'x': [1.0]}), raw, arr)
        self.assertEqual(X.columns.tolist(), ['x', *s.SIGNALS])
        self.assertEqual(X[s.AIRLINE].dtype, 'category')
        self.assertEqual(X[s.FORMAT].iloc[0], 1)
        with self.assertRaises(ValueError):
            s.add_signals(X, raw, arr)


class CorrectionTests(unittest.TestCase):
    def test_parameters_use_only_late_and_dayshift_rows(self):
        # LIRF no ID: late (gap 7200, label 1000), substituted (label = gap), too-short gap,
        # with-ID late row, next-day 15 h rule row shifted, next-day 15 h rule row substituted
        gaps = [7200, 7200, 3000, 7200, 54000, 54000]
        V, raw = frames(gaps, ['LIRF'] * 6, [True, True, True, False, True, True], takeoff='2025-07-02 06:00')
        y = [1000, 7200, 1000, 1500, 86400 + 900, 54000]
        p = s.fit_parameters(V, y, raw)
        self.assertEqual((p['late_rows'], p['late_mean']), (1, 1000.0))
        self.assertEqual((p['dayshift_rows'], p['dayshift_shifted'], p['dayshift_weight']), (2, 1, 0.5))

    def test_all_switches_off_reproduces_rome_hybrid(self):
        gaps = [10800, 18000, 18000, 18000, 1200, 5000]
        airports = ['LIRF', 'LIRF', 'LIRF', 'EGLL', 'LIRF', 'LIRF']
        V, raw = frames(gaps, airports, [True, True, False, True, False, True])
        ordinary, classifier = Stub([900, 2000, 900, 900, 900, 7000]), Stub([0.6, 0.9, 0.5, 0.99, 0.25, 0.3])
        a = h.predict(ordinary, classifier, V, raw, 'clock', threads=1)
        b = s.predict(ordinary, classifier, V, raw, 'clock', threads=1)
        for key in ['ordinary', 'p', 'blended', 'final', 'rule_triggered']:
            np.testing.assert_array_equal(a[key], b[key])
        self.assertFalse(b['late_rows'].any() or b['dayshift_rows'].any())

    def test_late_estimate_replaces_ordinary_only_on_no_id_lirf_rows_between_one_and_four_hours(self):
        gaps = [5000, 4000, 5000, 5000, 3000, 18000]
        V, raw = frames(gaps, ['LIRF', 'LIRF', 'LIRF', 'EGLL', 'LIRF', 'LIRF'], [True, True, False, True, True, True])
        ordinary, classifier = Stub([7000] * 6), Stub([0.25] * 6)
        out = s.predict(ordinary, classifier, V, raw, 'clock', threads=1, late_mean=1000.0)
        self.assertEqual(out['late_rows'].tolist(), [True, True, False, False, False, False])
        # 5000 s gap: late part min(1000, 1400) = 1000; 4000 s gap: min(1000, 400) = 400
        np.testing.assert_allclose(out['blended'][:2], [0.25 * 5000 + 0.75 * 1000, 0.25 * 4000 + 0.75 * 400])
        np.testing.assert_allclose(out['blended'][2:5], [0.25 * 5000 + 0.75 * 7000, 7000, 0.25 * 3000 + 0.75 * 7000])
        self.assertEqual(out['final'][5], 18000)

    def test_dayshift_blends_next_day_rule_rows_from_fourteen_hours(self):
        # 22:30 local schedule, 15 h late -> next local day; 5 h late -> same day; 15 h but with ID
        gaps = [54000, 18000, 54000]
        V, raw = frames(gaps, ['LIRF'] * 3, [True, True, False], takeoff='2025-07-02 11:30')
        ordinary, classifier = Stub([1000] * 3), Stub([0.5] * 3)
        out = s.predict(ordinary, classifier, V, raw, 'clock', threads=1, dayshift_weight=0.6)
        self.assertEqual(out['dayshift_rows'].tolist(), [True, False, False])
        self.assertAlmostEqual(out['final'][0], 0.6 * (86400 + 1000) + 0.4 * 54000)
        self.assertEqual(out['final'][1], 18000)

    def test_classifier_matrix_appends_extra_inputs(self):
        V, raw = frames([1200], ['LIRF'], [True])
        V[s.FORMAT] = 1.0
        X = s.classifier_matrix(V, raw, 'clock', (s.FORMAT,))
        self.assertEqual(X.columns.tolist(), h.classifier_matrix(V, raw, 'clock').columns.tolist() + [s.FORMAT])


if __name__ == '__main__':
    unittest.main()
