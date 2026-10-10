"""Fixed mixture/control and protected submission serialization contract."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import sparse_callsign_submit as b


class SparseCombinationTests(unittest.TestCase):
    def test_fixed_mixture_and_exact_zero_control(self):
        calls=pd.DataFrame(dict(mvt_id=[20,10],base=[100.25,200.75]))
        sparse=pd.DataFrame(dict(mvt_id=[20,10],base=[50.5,400.]))
        np.testing.assert_array_equal(b.mix_neural([20,10],calls,sparse,0),calls.base)
        np.testing.assert_array_equal(b.mix_neural([20,10],calls,sparse),[75.375,300.375])
        for weight in [.1,.25,1.]:
            with self.assertRaises(ValueError):b.mix_neural([20,10],calls,sparse,weight)
        with self.assertRaises(AssertionError):b.mix_neural([10,20],calls,sparse)
        with self.assertRaises(ValueError):b.mix_neural([20,10],calls,sparse.assign(base=[np.nan,1]))

    def test_template_order_rounding_and_protected_columns(self):
        template=pa.table({'MVT_ID_mvt':pa.array([10,20],type=pa.int64()),
                           'TAXITIME_SEC_mvt':pa.array([0,0],type=pa.int32()),
                           'unchanged':pa.array(['A','B'])})
        table=b.serialized(template,[20,10],np.array([100.5,201.5]))
        np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),[202,100])
        self.assertTrue(table['unchanged'].equals(template['unchanged']))
        frame=pd.DataFrame(dict(mvt_id=[20,10],pred=[100.5,201.5]))
        b.check_zero(frame,frame.copy(),table,table)
        with self.assertRaises(AssertionError):b.check_zero(frame.assign(pred=[100.6,201.5]),frame,table,table)
        bad=table.set_column(2,'unchanged',pa.array(['changed','B']))
        with self.assertRaises(ValueError):b.check_zero(frame,frame,table,bad)


if __name__=='__main__':unittest.main()
