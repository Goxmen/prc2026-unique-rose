"""Fixed-rule boundaries, raw provenance, saved-row alignment and recipe guards."""

import argparse
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import postprocess
import submit
from prediction_rules import apply_rule
from records import MODEL_FILES, ROOT, sha256
from VALIDATION.tests.test_v3 import config


def raw_rows():
    t = pd.Timestamp('2025-07-01 12:00', tz='UTC')
    return pd.DataFrame({'mvt_id': range(6), 'ADEP_mvt': ['LIRF']*4+['LFPG','LIRF'],
                         'FLIGHT_ID_mvt': [np.nan, np.nan, 2., np.nan, np.nan, np.nan],
                         'MVT_TIME_UTC_mvt': t,
                         'SCHED_TIME_UTC_mvt': [t-pd.Timedelta(seconds=x) for x in
                                                [21600,21599,30000,40000,50000,60000]]})


class PredictionRuleTests(unittest.TestCase):
    def test_exact_boundary_and_original_id(self):
        raw = raw_rows()
        raw.loc[5, 'SCHED_TIME_UTC_mvt'] = pd.NaT
        # A recovered plan must not disable the original missing-ID trigger.
        raw['ADEP_flt'] = ['LIRF',None,None,'LIRF',None,None]
        original = raw.copy(deep=True)
        pred = np.array([100.,200.,300.,400.,500.,600.])
        out, audit = apply_rule(pred, raw, 'rome_6h')
        np.testing.assert_array_equal(out, [21600,200,300,40000,500,600])
        self.assertEqual(audit.rule_triggered.tolist(), [True,False,False,True,False,False])
        pd.testing.assert_frame_equal(raw, original)
        np.testing.assert_array_equal(pred, [100,200,300,400,500,600])

    def test_hidden_labels_and_plan_fields_do_not_control_rule(self):
        raw = raw_rows()
        a = apply_rule(np.ones(6), raw, 'rome_6h')
        raw['taxi_sec'] = -1e20
        raw['BLOCK_TIME_UTC_mvt'] = pd.Timestamp('2030-01-01', tz='UTC')
        raw['raw_plan_block_missing'] = False
        b = apply_rule(np.ones(6), raw, 'rome_6h')
        np.testing.assert_array_equal(a[0], b[0])
        pd.testing.assert_frame_equal(a[1], b[1])

    def test_off_identity_and_invalid_predictions(self):
        raw = raw_rows()
        p = np.arange(6.)
        out, audit = apply_rule(p, raw)
        np.testing.assert_array_equal(out, p)
        self.assertFalse(audit.rule_triggered.any())
        for pred in [p[:2], [np.nan]*6]:
            with self.assertRaises(ValueError):
                apply_rule(pred, raw, 'rome_6h')
        with self.assertRaises(ValueError):
            apply_rule(p, raw, 'rome_5h')

    def test_saved_predictions_align_by_id_and_preserve_targets(self):
        raw = raw_rows().iloc[:3].set_index('mvt_id')
        frame = pd.DataFrame({'mvt_id': [2,0,1], 'airport': 'LIRF',
                              'mvt_time': raw.MVT_TIME_UTC_mvt.iloc[0],
                              'y': [50.,21600.,100.], 'pred': [100.,500.,600.]})
        out, changed = postprocess.transform_predictions(frame, raw, 'rome_6h')
        np.testing.assert_array_equal(out.pred, [100,21600,600])
        pd.testing.assert_series_equal(out.y, frame.y)
        self.assertEqual(changed.mvt_id.tolist(), [0])
        self.assertLess(changed.delta_squared_error.iloc[0], 0)
        wrong = frame.copy()
        wrong.loc[0,'airport'] = 'LFPG'
        with self.assertRaisesRegex(ValueError, 'airports'):
            postprocess.transform_predictions(wrong, raw, 'rome_6h')

    def test_linked_recipe_inherits_rule_and_rejects_overrides_or_source_drift(self):
        source = {'files': {p: sha256(ROOT/p) for p in MODEL_FILES}}
        ref = {'run_id': 'q', 'protocol': 'v3', 'status': 'complete',
               'config': config(recovery='history', prediction_rule='rome_6h'),
               'source': {'files': {}}, 'model_source': source,
               'rule_source': {'pipeline/prediction_rules.py': sha256(ROOT/'pipeline/prediction_rules.py')}}
        args = argparse.Namespace(run_id='q', seed=None, **dict.fromkeys(submit.DEFAULTS))
        with patch.object(submit, 'read_run', return_value=(ref, Path('.'))):
            cfg, _, _ = submit.resolve_config(args)
            self.assertEqual(cfg['prediction_rule'], 'rome_6h')
            args.prediction_rule = 'off'
            with self.assertRaisesRegex(ValueError, 'differs'):
                submit.resolve_config(args)
            args.prediction_rule = None
            ref['rule_source']['pipeline/prediction_rules.py'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'rule source changed'):
                submit.resolve_config(args)
            ref['model_source']['files']['VALIDATION/modeling.py'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'modeling.py changed'):
                submit.resolve_config(args)


if __name__ == '__main__':
    unittest.main()
