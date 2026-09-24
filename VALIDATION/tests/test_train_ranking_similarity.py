"""Guard date separation, prior correction and mixed-distance geometry."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import train_ranking_similarity as sim
from check_similarity_weather_scale import corrected_encoding


def toy(n):
    f = pd.DataFrame({c: (['A']*n if c in sim.CATEGORICAL else np.zeros(n))
                      for c in sum(sim.GROUPS.values(), [])})
    f['mvt_id'] = np.arange(n)
    f['airport'] = 'X'
    return f


class SimilarityTests(unittest.TestCase):
    def test_dates_tested_exactly_once_and_embargoed_across_month_boundary(self):
        dates = pd.Series(pd.date_range('2025-01-01','2025-02-28').repeat(3))
        times_tested = np.zeros(len(dates), int)
        for fold in range(4):
            role = sim.roles(dates, fold)
            times_tested += role == 'test'
            groups = {name:set(dates[role==name]) for name in ['fit','cal','test']}
            for a,b in [('fit','cal'),('fit','test'),('cal','test')]:
                self.assertTrue(all(abs(x-y)>pd.Timedelta(days=1) for x in groups[a] for y in groups[b]))
        np.testing.assert_array_equal(times_tested, 1)

    def test_balancing_removes_class_and_airport_frequency(self):
        airport = ['A']*12+['B']*8
        y = [0]*10+[1]*2+[0]*3+[1]*5
        w = sim.balanced_weights(airport,y)
        self.assertAlmostEqual(w.sum(),20)
        for a in ['A','B']:
            for label in [0,1]:
                mask=(np.array(airport)==a)&(np.array(y)==label)
                self.assertAlmostEqual(w[mask].sum(),5)

    def test_classifier_encoding_is_fit_only_and_stands_are_scoped(self):
        fit=pd.DataFrame({'airport':['A','B'],'stand':['1','1']})
        _,mapping=sim.model_matrix(fit,['airport','stand'])
        query=pd.DataFrame({'airport':['A','A'],'stand':['1','new']})
        encoded,_=sim.model_matrix(query,['airport','stand'],mapping)
        self.assertEqual(mapping['stand'],['A:1','B:1'])
        self.assertEqual(encoded.stand.iloc[0],'A:1')
        self.assertTrue(pd.isna(encoded.stand.iloc[1]))

    def test_nominal_codes_do_not_create_ordinal_distances(self):
        ref=toy(3);ref['dest']=['A','B','Z']
        query=toy(1);query['dest']='A'
        spec=sim.fit_distance(ref)
        _,_,d=next(sim.distance_blocks(sim.encode_distance(ref,spec),sim.encode_distance(query,spec)))
        self.assertEqual(d['raw'][0,0],0)
        self.assertEqual(d['raw'][0,1],d['raw'][0,2])

    def test_unseen_category_is_not_mistaken_for_missing(self):
        ref=toy(2);ref['dest']=['A',None]
        query=toy(1);query['dest']='NEW'
        spec=sim.fit_distance(ref)
        _,_,d=next(sim.distance_blocks(sim.encode_distance(ref,spec),sim.encode_distance(query,spec)))
        self.assertGreater(d['raw'][0,1],0)
        self.assertEqual(d['raw'][0,0],d['raw'][0,1])

    def test_missing_numeric_values_and_constant_scales_stay_finite(self):
        ref=toy(12);ref['wx_gust_kt']=np.nan
        query=toy(2);query.loc[0,'wx_gust_kt']=np.nan
        query.loc[1,'takeoff_minus_sched_min']=1e8
        result=sim.nearest(ref,query,k=10)
        for view in result:
            self.assertTrue(np.isfinite(result[view]['distance']).all())
            self.assertTrue((np.diff(result[view]['distance'],axis=1)>=0).all())
        # All 12 references tie; select the ten smallest IDs reproducibly.
        np.testing.assert_array_equal(result['raw']['index'][0],np.arange(10))
        self.assertGreater(result['raw']['distance'][1,0],result['raw']['distance'][0,0])

    def test_query_extremes_do_not_change_reference_scaling_or_other_queries(self):
        ref=toy(12);ref['wx_temp_c']=np.arange(12.)
        query=toy(2);query['wx_temp_c']=[5.,10.]
        before=sim.nearest(ref,query,k=5)
        query.loc[1,'wx_temp_c']=1e9
        after=sim.nearest(ref,query,k=5)
        for view in before:
            np.testing.assert_array_equal(before[view]['index'][0],after[view]['index'][0])
            np.testing.assert_allclose(before[view]['distance'][0],after[view]['distance'][0])

    def test_calibration_uses_balanced_priors_and_cannot_reverse_order(self):
        y=np.r_[np.zeros(100),np.ones(10)]
        weights=sim.balanced_weights(['A']*len(y),y)
        fitted=sim.fit_calibrator(np.zeros(len(y)),y,weights)
        self.assertAlmostEqual(sim.calibrated([0],fitted)[0],.5)
        self.assertGreaterEqual(fitted['slope'],0)
        self.assertTrue(np.all(np.diff(sim.calibrated([-1,0,1],fitted))>=0))

    def test_rare_weather_variation_is_invariant_to_metres_vs_kilometres(self):
        ref=toy(101);ref['wx_visibility_m']=10000.;ref.loc[0,'wx_visibility_m']=9000.
        query=toy(1);query['wx_visibility_m']=9500.
        old_r,old_q,_=corrected_encoding(ref,query)
        ref['wx_visibility_m']/=1000;query['wx_visibility_m']/=1000
        new_r,new_q,_=corrected_encoding(ref,query)
        for group in sim.GROUPS:
            np.testing.assert_allclose(old_r[group][1],new_r[group][1])
            np.testing.assert_allclose(old_q[group][1],new_q[group][1])

    def test_constant_weather_has_bounded_deviation_and_missing_donor_has_no_numeric_scale(self):
        ref=toy(12);ref['wx_ceiling_ft']=25000.;ref['wx_gust_kt']=np.nan
        query=toy(1);query['wx_ceiling_ft']=24900.;query['wx_gust_kt']=100.
        r,q,_=corrected_encoding(ref,query)
        names=sim.fit_distance(ref)['weather']['numeric'];n=len(names)
        z=q['weather'][1]*np.sqrt(2*n)
        self.assertEqual(z[0,names.index('wx_ceiling_ft')],1)
        self.assertEqual(z[0,names.index('wx_gust_kt')],0)
        self.assertEqual(r['weather'][1][0,n+names.index('wx_gust_kt')]*np.sqrt(2*n),1)


if __name__=='__main__':
    unittest.main()
