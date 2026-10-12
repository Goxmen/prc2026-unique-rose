import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import probe_movement_id_arrival_clock_20261012 as p


class ArrivalIDClock(unittest.TestCase):
    def test_linear_clock_and_no_extrapolation(self):
        ids=np.arange(10,130,10)
        out=p.clocks([5,55,65,125],ids,ids*100.,ids*100.+300)
        np.testing.assert_array_equal(out['available'],[False,True,True,False])
        np.testing.assert_array_equal(out['landing'][1:3],[5500,6500])
        np.testing.assert_array_equal(out['inblock'][1:3],[5800,6800])
        self.assertTrue(np.isnan(out['inblock'][[0,3]]).all())

    def test_median_not_mean_and_disjoint_identity(self):
        ids=np.arange(10,130,10);times=np.arange(12,dtype=float)*60
        a=p.clocks([55],ids,times,times)
        changed=times.copy();changed[0]=-1e6;changed[9]=1e6
        b=p.clocks([55],ids,changed,changed)
        np.testing.assert_array_equal(a['inblock'],b['inblock'])
        with self.assertRaises(AssertionError):p.clocks([50],ids,times,times)

    def test_auc_ties_and_orientation(self):
        self.assertEqual(p.auc([0,1,0,1],[0,1,0,1]),1.)
        self.assertEqual(p.auc([0,1,0,1],[1,0,1,0]),0.)
        self.assertEqual(p.auc([0,1],[4,4]),.5)
        self.assertIsNone(p.auc([0,0],[1,2]))


if __name__=='__main__':unittest.main()
