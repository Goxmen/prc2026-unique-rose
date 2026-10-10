"""Clock target encodings must exclude target months and missing-clock queries."""
from pathlib import Path
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import clock_offset_encoding as enc


def rows(month, offset, n=30, stand='A', operator='OP', runway='01', airport='EDDF', start=0):
    ids=np.arange(start,start+n)
    t=np.full(n,1_000_000.)+ids
    block=t-1000
    return pd.DataFrame(dict(mvt_id=ids,airport=airport,stand=stand,operator=operator,
        runway=runway,t=t,aobt=block-offset,block=block,schedule=block-1000,y=1000.,month=month))


class ClockOffsetTests(unittest.TestCase):
    def test_source_projection_and_exact_label_arithmetic(self):
        t=pd.Timestamp('2025-01-10T12:00:00Z')
        raw=pd.DataFrame(dict(MVT_ID_mvt=[1.],PHASE_mvt=['DEP'],ADEP_mvt=['EDDF'],
            STAND_mvt=['A'],AIRCRAFT_OPERATOR_flt=['OP'],RUNWAY_mvt=['01'],
            MVT_TIME_UTC_mvt=[t],AOBT_3_flt=[t-pd.Timedelta(seconds=1120)],
            BLOCK_TIME_UTC_mvt=[t-pd.Timedelta(seconds=1000)],
            SCHED_TIME_UTC_mvt=[t-pd.Timedelta(seconds=2000)],TAXITIME_SEC_mvt=[1000]))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'raw.parquet';raw.to_parquet(path)
            con=enc.connection()
            report=enc.read_sources(con,[path])
            self.assertEqual(report['label_relationship_max_abs_error_sec'],0)
            self.assertEqual(con.execute('SELECT month FROM events').fetchone()[0],'2025-01')
            raw['TAXITIME_SEC_mvt']=1001;raw.to_parquet(path)
            with self.assertRaises(ValueError):enc.read_sources(con,[path])
            con.close()

    def test_offset_sign_and_corrected_clock(self):
        ref=rows('2025-01',120)
        lookup=enc.fit_lookups(ref,['2025-01'])
        q=rows('2025-02',400,n=1,start=100)
        f=enc.apply_lookups(q,lookup).loc[100]
        np.testing.assert_allclose(f.iloc[:3],[120]*3)
        self.assertEqual(f.imp_minus_dlt_stand,1280.)
        self.assertTrue(all(dtype==np.dtype('float32') for dtype in enc.apply_lookups(q,lookup).dtypes))

    def test_support_cutoff_missing_keys_and_airport_fallback(self):
        ref=pd.concat([rows('2025-01',100,n=30,stand='A',start=0),
                       rows('2025-01',400,n=29,stand='B',start=100),
                       rows('2025-01',999,n=30,stand=None,start=200)],ignore_index=True)
        lookup=enc.fit_lookups(ref,['2025-01'])
        self.assertEqual(lookup.query("feature=='te_dlt_stand'").key.tolist(),['A'])
        q=pd.concat([rows('2025-02',10,n=1,stand=stand,start=1000+i)
                     for i,stand in enumerate(['A','B','UNSEEN',None,''])],ignore_index=True)
        got=enc.apply_lookups(q,lookup)
        np.testing.assert_array_equal(got.te_dlt_stand,[100,400,400,400,400])

    def test_clean_boundaries_and_original_actual_requirement(self):
        valid=rows('2025-01',90,n=30)
        invalid=rows('2025-01',900,n=6,start=100)
        invalid.loc[0,'aobt']=np.nan
        invalid.loc[1,'aobt']=invalid.loc[1,'block']-3600
        invalid.loc[2,'schedule']=invalid.loc[2,'block']-60
        invalid.loc[3,'y']=-1
        invalid.loc[4,'block']=np.nan
        invalid.loc[5,'schedule']=np.nan
        lookup=enc.fit_lookups(pd.concat([valid,invalid],ignore_index=True),['2025-01'])
        self.assertTrue(lookup.n.eq(30).all());self.assertTrue(lookup.offset_sec.eq(90).all())
        q=rows('2025-02',100,n=2,start=1000);q.loc[0,'aobt']=np.nan
        got=enc.apply_lookups(q,lookup)
        self.assertTrue(got.loc[1000].isna().all());self.assertTrue(got.loc[1001].notna().all())
        q['block']=1e15;q['y']=-1e15;q['schedule']=q['aobt']
        pd.testing.assert_frame_equal(got,enc.apply_lookups(q,lookup))

    def test_whole_own_month_exclusion_including_airport_prior(self):
        jan=rows('2025-01',110,n=30)
        jul=rows('2025-07',220,n=30,start=100)
        dec=rows('2025-12',330,n=30,start=200)
        all_rows=pd.concat([jan,jul,dec],ignore_index=True)
        q=jan.copy();q['stand']='UNSEEN'
        a=enc.apply_lookups(q,enc.fit_lookups(all_rows,['2025-07','2025-12']))
        # Mutating every label in the query month cannot alter its OOF values,
        # even when the query explicitly takes an airport fallback.
        all_rows.loc[all_rows.month.eq('2025-01'),'block']+=777
        b=enc.apply_lookups(q,enc.fit_lookups(all_rows,['2025-07','2025-12']))
        pd.testing.assert_frame_equal(a,b)
        np.testing.assert_allclose(a.iloc[:,:3],275.)

    def test_held_labels_excluded_and_query_order_preserved(self):
        train=rows('2025-01',60,n=30)
        held=rows('2025-07',110,n=3,start=100)
        full=pd.concat([train,held],ignore_index=True)
        shuffled=held.sample(frac=1,random_state=42)
        a=enc.apply_lookups(shuffled,enc.fit_lookups(full,['2025-01']))
        full.loc[full.month.eq('2025-07'),'block']-=888
        b=enc.apply_lookups(shuffled,enc.fit_lookups(full,['2025-01']))
        pd.testing.assert_frame_equal(a,b)
        np.testing.assert_array_equal(a.index,shuffled.mvt_id)
        unseen=held.copy();unseen['airport']='LTFM'
        self.assertTrue(enc.apply_lookups(unseen,enc.fit_lookups(train,['2025-01'])).isna().all().all())


if __name__=='__main__':unittest.main()
