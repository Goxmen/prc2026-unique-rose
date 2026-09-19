"""Guard diagnostic weighting and observable regime boundaries."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path[:0] = [str(Path(__file__).resolve().parents[1]),
               str(Path(__file__).resolve().parents[1] / 'tools')]
import reporting
from summarize_artifact_errors import influence, regime_labels


def loss(ids,dates,squares):
    meta=pd.DataFrame(dict(mvt_time=pd.to_datetime(dates,utc=True),airport='LIRF',y=100.0,
                           has_flight_plan=False,arrival_history_available=True,
                           arrival_history_days_available=30),index=pd.Index(ids,name='mvt_id'))
    return reporting.Loss(meta,np.asarray(squares,float),(42,142))


def check_repeated_december_origin_is_averaged_before_influence():
    november=loss([1],['2025-11-01'],[[4],[4]])
    december1=loss([2],['2025-12-01'],[[16],[36]])
    december2=loss([2],['2025-12-01'],[[36],[64]])
    control=reporting.group_loss({'B3':november,'B4':december1,'B7':december2},['B3','B4','B7'])
    assert control.sq.mean(axis=0).tolist()==[4,38]
    candidate=reporting.Loss(control.meta.copy(),np.asarray([[4,4],[4,4]],float),(42,142))
    groups={'R_A':control,'R_B_long':control}
    changed={'R_A':candidate,'R_B_long':candidate}
    result=influence(groups,changed,.44)
    assert result['R_combined']['day']=='2025-12-01'
    assert result['R_combined']['removed_rows_by_group']=={'R_A':1,'R_B_long':1}
    assert result['R_combined']['without_day_delta_rmse']==0
    assert np.isclose(result['R_combined']['full_delta_rmse'],2-np.sqrt(21))
    unchanged=influence(groups,groups,.44)
    assert unchanged['R_combined']['day'] is None
    assert unchanged['R_combined']['removed_rows_by_group']=={}


def check_regimes_use_local_date_and_raw_id_without_targets():
    frame=pd.DataFrame({
        'ADEP_mvt':['LIRF']*5+['LFPG','LFPG'],
        'FLIGHT_ID_mvt':[None,None,None,None,7,None,8],
        'MVT_TIME_UTC_mvt':['2025-09-28T22:38Z','2025-09-28T20:38Z','2025-09-28T11:00Z',
                            '2025-09-28T08:50Z','2025-09-28T11:00Z','2025-09-28T11:00Z','2025-09-28T11:00Z'],
        'SCHED_TIME_UTC_mvt':['2025-09-28T08:35Z']*7,
    })
    regimes=list(regime_labels(frame,[False,False,False,False,False,True,True]))
    assert regimes==['Rome next-day hard rule','Rome other hard rule','Rome no-ID 1–4h',
                     'Rome other no-ID','Other ID','Other airport no-ID','ID twin']


class ArtifactErrorSummaryTests(unittest.TestCase):
    def test_repeated_december_origin_is_averaged_before_influence(self):
        check_repeated_december_origin_is_averaged_before_influence()

    def test_regimes_use_local_date_and_raw_id_without_targets(self):
        check_regimes_use_local_date_and_raw_id_without_targets()


if __name__=='__main__':
    unittest.main()
