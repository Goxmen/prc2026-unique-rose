"""Counterfactual dependency and provenance checks for the no-plan specialist."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import missing_plan_view as v


def fixture():
    cols=v.DROP+v.OP_OWN+v.AB_OWN+[f'untouched_{i}' for i in range(99)]
    assert len(cols)==156 and len(set(cols))==156
    X=pd.DataFrame(7.,index=[11,22,33,44],columns=cols)
    X['op_matched']=[1,1,0,1];X['ab_matched']=[1,1,0,1]
    X['op_deice_sec']=[0,0,0,0];X['ab_stand_n']=[0,0,0,0]
    meta=pd.DataFrame(dict(mvt_id=[101,102,103,104],
        op_key_ok=[True,False,True,True],ab_key_ok=[True,False,True,True],
        raw_aobt3_missing=[False,False,True,True],airport=['EHAM','LFPG','EDDF','LIRF'])).set_index('mvt_id')
    return X,meta


def test_whole_plan_removal_and_physical_provenance():
    X,m=fixture();before=X.copy(deep=True)
    got=v.transform(X,np.array([101,102,103,104]),m)
    assert len(got.columns)==121 and not set(v.DROP)&set(got)
    assert got.index.equals(X.index)
    pd.testing.assert_frame_equal(X,before)
    assert got.loc[[22,33],v.OP_OWN+v.AB_OWN].isna().all().all()
    assert got.loc[11,'op_deice_sec']==0 and got.loc[11,'ab_stand_n']==0
    assert got.loc[44,'op_deice_sec']==0
    pd.testing.assert_frame_equal(got.filter(like='untouched'),X.filter(like='untouched'))
    # Known no-match and changed/unverified keys have exactly the same view.
    pd.testing.assert_series_equal(got.loc[22],got.loc[33],check_names=False)
    np.testing.assert_array_equal(v.gate(m,[101,102,103,104]),[False,False,True,False])


def test_matcher_key_semantics_and_reordering():
    raw=pd.DataFrame(dict(CALLSIGN_flt=[' ABC ','abc','A B','',None,None,'DIFFERENT'],
                          FLIGHT_mvt=['ABC','ABC','AB','ABC','ABC',None,'ABC']))
    flags=v.key_flags(raw)
    np.testing.assert_array_equal(flags.op_key_ok,[True,False,False,True,True,True,False])
    np.testing.assert_array_equal(flags.ab_key_ok,[True,True,True,False,True,True,False])
    X,m=fixture()
    got=v.transform(X,[104,102,103,101],m)
    assert got.loc[11,'ab_stand_n']==0 and got.loc[22,v.AB_OWN].isna().all()


def test_missing_identity_or_wrong_input_contract_rejected():
    X,m=fixture()
    with unittest.TestCase().assertRaises(ValueError):v.transform(X,[101,102,103,999],m)
    with unittest.TestCase().assertRaises(ValueError):v.transform(X.drop(columns='operator'),[101,102,103,104],m)
    with unittest.TestCase().assertRaises(ValueError):v.transform(X,[101,102,103,104],pd.concat([m,m]))
    bad=m.copy();bad['op_key_ok']=bad.op_key_ok.astype('boolean');bad.loc[101,'op_key_ok']=pd.NA
    with unittest.TestCase().assertRaises(ValueError):v.transform(X,[101,102,103,104],bad)


def test_independent_opdi_and_adsb_key_guards():
    X,m=fixture();m.loc[101,'ab_key_ok']=False
    out=v.transform(X,[101,102,103,104],m)
    assert out.loc[11,v.AB_OWN].isna().all()
    pd.testing.assert_series_equal(out.loc[11,v.OP_OWN],X.loc[11,v.OP_OWN])


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(f) for name,f in globals().copy().items() if name.startswith('test_'))


if __name__=='__main__':
    unittest.main()
