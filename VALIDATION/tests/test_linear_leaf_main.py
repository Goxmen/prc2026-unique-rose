import sys
from pathlib import Path
import unittest

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import linear_leaf_main_jan_jul as screen
import linear_leaf_v79_transport as transport


class LinearComparisonTest(unittest.TestCase):
    def test_only_intended_algorithm_parameters_change(self):
        p = dict(seed=42, num_leaves=127, num_threads=4, learning_rate=.02,
            lambda_l2=100., bagging_fraction=.8, feature_fraction=.7)
        before = p.copy(); constant = screen.params_for(p, 'constant'); linear = screen.params_for(p, 'candidate')
        self.assertEqual(p, before)
        self.assertEqual(set(k for k in set(constant)|set(linear) if constant.get(k) != linear.get(k)),
            {'linear_tree','linear_lambda'})
        self.assertEqual(linear['linear_lambda'], 1.)

    def test_scaling_uses_training_and_preserves_unknown_categories_and_missingness(self):
        train = pd.DataFrame(dict(x=[1.,2.,3.,np.nan], z=[0.,0.,0.,0.],
            c=pd.Categorical(['a','b','a',None], categories=['b','a','new'])), index=[9,4,3,1])
        held = train.copy(); held.loc[9,'x'] = 10000.; held.loc[4,'c'] = 'new'
        stats = screen.scaling.scaler_fit(train,['c']); out = screen.scaling.scaler_apply(held,stats)
        self.assertEqual(stats['x']['mean'],2.)
        self.assertGreater(out.loc[9,'x'],10000.)
        pd.testing.assert_series_equal(out.c,held.c)
        np.testing.assert_array_equal(out.isna(),held.isna())
        self.assertEqual(list(out.index),list(held.index))

    def test_v79_delta_transport_keeps_zero_delta_and_sql_rounding(self):
        parent=np.array([0.,1.,2.,10.,20.]); selected=np.array([5.,6.,7.,8.,9.])
        arm=np.array([5.,4.5,7.5,8.5,7.5])
        actual=transport.transport(parent,selected,arm)
        frame=pd.DataFrame(dict(parent=parent,selected=selected,arm=arm))
        with duckdb.connect(config={'threads':1}) as con:
            con.register('x',frame)
            expected=con.sql('select round_even(greatest(0,parent+arm-selected),0) as candidate from x').df().candidate
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(actual[arm==selected],parent[arm==selected])
        with self.assertRaises(AssertionError):transport.transport([1.5],[1.],[1.])


if __name__=='__main__':
    unittest.main()
