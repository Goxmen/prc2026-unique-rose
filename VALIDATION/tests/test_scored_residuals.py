"""Numerical and inventory invariants for the residual report (no model fits)."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from plot_scored_residuals import aggregate, with_live_inventory, ledger


class ResidualReportTests(unittest.TestCase):
    def test_eight_column_submission_ledger_and_missing_predictions(self):
        with TemporaryDirectory() as tmp:
            Path(tmp,'SUBMISSIONS.md').write_text(
                '| 1 | **Airport-mean** | mean | — | — | — | 661.5737 | v1 |\n'
                '| 12 | **B0-800** | model | 366.7 | 334.2 | 348.9 | 406.4281 | v12 |\n'
                '| 10 | rejected | model | — | — | — | Rejected | v10 |\n')
            with patch('plot_scored_residuals.ROOT',Path(tmp)):
                rows=ledger()
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['coverage'],'unavailable')
        self.assertEqual(rows[1]['coverage'],'original')
        self.assertEqual(rows[1]['lb'],406.4281)
        self.assertEqual(rows[1]['cv_combined'],348.9)

    def test_group_contributions_reconstruct_total_mse(self):
        d = pd.DataFrame({'airport':['A','A','B'], 'y':[1.,2.,3.],
                          'date':['2025-07-01','2025-07-02','2025-07-01'],
                          'band':pd.Categorical(['small','big','small'],categories=['small','big'])})
        # Opposing seeds must retain error; averaging their predictions would hide it.
        e = np.array([[2., -4., 3.], [-2., 4., 3.]])
        model = dict(err=e[0], sq=(e**2).mean(axis=0), bias=e.mean(axis=0))
        rows = aggregate(d, model, 'band')
        self.assertEqual([r['key'] for r in rows], ['small','big'])
        self.assertAlmostEqual(sum(r['mse_contribution'] for r in rows), 29/3)
        self.assertAlmostEqual(sum(r['error_share'] for r in rows), 100)
        self.assertEqual(sum(r['n'] for r in rows), 3)
        self.assertEqual(rows[0]['days'], 1)
        scoped = aggregate(d, model, 'band', 'A')
        self.assertAlmostEqual(sum(r['mse_contribution'] for r in scoped), 10)
        self.assertEqual(sum(r['n'] for r in scoped), 2)

    def test_live_inventory_includes_unknown_successful_upload(self):
        old = [dict(version='v34',lb=280.4658)]
        live = {'history':[dict(version=34,status='Succeeded',score=280.4658),
                           dict(version=37,status='Succeeded',score=280.1743),
                           dict(version=38,status='Rejected',score=None)]}
        rows = with_live_inventory(old,live)
        self.assertEqual([r['version'] for r in rows],['v34','v37'])
        self.assertIn('unverified', rows[-1]['coverage'])
        self.assertIsNone(rows[-1]['cv_combined'])

    def test_conflicting_official_score_is_not_silently_accepted(self):
        with self.assertRaises(AssertionError):
            with_live_inventory([dict(version='v34',lb=280.4658)],
                {'history':[dict(version=34,status='Succeeded',score=281.)]})


if __name__ == '__main__':
    unittest.main()
