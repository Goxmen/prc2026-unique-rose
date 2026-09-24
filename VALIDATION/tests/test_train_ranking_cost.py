"""Guard the error denominators, population joins and neighbour-label isolation."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import analyze_train_ranking_cost as cost


class CostTests(unittest.TestCase):
    def test_cost_uses_squared_errors_and_original_population(self):
        f = pd.DataFrame({'mean_sq':[1.,9.,16.], 'date':pd.to_datetime(['2025-07-01']*3)})
        r = cost.cost_stats(f,np.array([True,True,False]))
        self.assertAlmostEqual(r['rmse'],np.sqrt(5))
        self.assertAlmostEqual(r['sse_pct'],100*10/26)
        self.assertEqual(r['complement_rmse'],4)
        self.assertEqual(r['n'],2)
        self.assertIsNone(cost.cost_stats(f,np.zeros(3,bool))['rmse'])

    def test_stand_novelty_is_airport_scoped_and_missing_is_separate(self):
        train = pd.DataFrame({'airport':['A','B'],'stand':['1','2']})
        query = pd.DataFrame({'airport':['A']*3,'stand':['1','2',None]})
        np.testing.assert_array_equal(cost.unseen_stands(query,train),[False,True,False])

    def test_identity_rejects_mismatched_target(self):
        f = pd.DataFrame({'airport':['A'],'mvt_time':pd.to_datetime(['2025-07-01']),
                          'y':[10.],'mean_sq':[4.]},index=[1.])
        cost.check_identity(f,f)
        other=f.copy();other['y']=11
        with self.assertRaises(AssertionError): cost.check_identity(f,other)

    def neighbours(self):
        donors = pd.DataFrame({'airport':['A','A'],'date':pd.to_datetime(['2025-07-01','2025-07-02']),
                               'mean_sq':[1.,9.],'fold':[0,0]},index=[1.,2.])
        q = pd.DataFrame({'airport':['A'],'date':pd.to_datetime(['2025-07-15']),
                          'label':[0],'fold':[1], 'raw_neighbour1':[1.], 'raw_neighbour2':[2.]},index=[3.])
        return q,donors

    def test_neighbour_proxy_averages_losses_and_requires_every_donor(self):
        q,d=self.neighbours()
        self.assertEqual(cost.neighbour_risk(q,d,'raw',2)[0],5.)
        with self.assertRaises(AssertionError): cost.neighbour_risk(q,d.iloc[:1],'raw',2)

    def test_same_date_block_donor_is_rejected_even_when_not_adjacent(self):
        q,d=self.neighbours();q['fold']=0
        with self.assertRaises(AssertionError): cost.neighbour_risk(q,d,'raw',2)


if __name__ == '__main__': unittest.main()
