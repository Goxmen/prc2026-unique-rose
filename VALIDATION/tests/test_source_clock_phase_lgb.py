import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import source_clock_phase_lgb as screen


class PhaseInputTest(unittest.TestCase):
    def test_preserve_values_order_categories_and_missing(self):
        frame=pd.DataFrame(dict(a=pd.Categorical(['x','y'],categories=['y','x','unused']),
                                b=np.array([np.nan,0],dtype='float32')),index=[7,2])
        phase=pd.DataFrame({c:np.array([13,np.nan,0],dtype='float32') for c in screen.features.FEATURES},index=[19,21,23])
        got=screen.append_phases(frame,[23,21],phase)
        pd.testing.assert_frame_equal(got[frame.columns],frame,check_exact=True)
        self.assertTrue(got.iloc[0][screen.features.FEATURES].eq(0).all())
        self.assertTrue(got.iloc[1][screen.features.FEATURES].isna().all())
        with self.assertRaises(AssertionError):screen.append_phases(frame,[23,99],phase)


if __name__=='__main__':unittest.main()
