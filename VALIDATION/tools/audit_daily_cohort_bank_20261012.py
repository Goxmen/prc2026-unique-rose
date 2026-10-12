"""Independent all-row audit of daily donor values, calendars and exclusions."""
from pathlib import Path
import hashlib
import json
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import daily_cohort_bank as candidate
import flight_cohort_context as old
import public_delay_features as public

OUT=ROOT/'VALIDATION/reports/cohort_learning_20261012'


def main():
    manifest=json.loads((candidate.OUT/'manifest.json').read_text())
    for a in manifest['sources']+manifest['outputs']+[manifest['source']]:
        assert hashlib.sha256((ROOT/a['path']).read_bytes()).hexdigest()==a['sha256']
    bank={p.stem:np.load(p,mmap_mode='r') for p in candidate.OUT.glob('*.npy')}
    meta=np.load(old.OUT/'train/meta.npy',mmap_mode='r')
    values=np.load(old.OUT/'train/values.npy',mmap_mode='r')
    clock=np.load(old.OUT/'train/clock.npy',mmap_mode='r')
    f=pd.read_parquet(public.OUT/'inputs2025.parquet',columns=['mvt_id','airport','day','gap','fltrule'])
    f=f.set_index('mvt_id').loc[meta[:,0]].reset_index()
    np.testing.assert_array_equal(bank['ids'],meta[:,0])
    assert len(f)==2085047 and f.mvt_id.is_unique and np.all(np.diff(bank['ids'])>0)
    days=np.floor_divide(meta[:,1],86400000)
    np.testing.assert_array_equal(bank['keys'][bank['group'],0],meta[:,2])
    np.testing.assert_array_equal(bank['keys'][bank['group'],1],days)
    np.testing.assert_array_equal(pd.to_datetime(days*86400000,unit='ms',utc=True).strftime('%Y-%m-%d'),f.day)
    eligible=~f.fltrule.eq('V').to_numpy()
    group,slot=np.where(bank['mask'])
    did=bank['donor_ids'][group,slot]
    positions=np.searchsorted(meta[:,0],did)
    np.testing.assert_array_equal(meta[positions,0],did)
    np.testing.assert_array_equal(np.sort(did),bank['ids'][eligible])
    np.testing.assert_array_equal(bank['keys'][group,0],meta[positions,2])
    np.testing.assert_array_equal(bank['keys'][group,1],days[positions])
    assert np.all(bank['donor_ids'][~bank['mask']]==-1)
    assert np.all(bank['values'][~bank['mask']]==0)
    for begin in range(0,len(positions),65536):
        p=positions[begin:begin+65536]
        x=values[p].copy()
        x[:,:4]=np.arcsinh(x[:,:4]/np.float32(3600))
        x[:,9:11]=np.minimum(10,np.maximum(-10,x[:,9:11]/np.float32(2000)))
        gap=f.gap.to_numpy()[p];finite=np.isfinite(gap)
        z=np.clip(np.arcsinh(np.where(finite,gap,0)/7200),-10,10)
        expected=np.column_stack([x,z,finite]).astype(np.float32)
        np.testing.assert_array_equal(bank['values'][group[begin:begin+len(p)],slot[begin:begin+len(p)]],expected)
    # Derive all expected aliases with separate relational joins, avoiding
    # the producer's group-by-group pairwise Boolean construction.
    q=pd.DataFrame(dict(row=np.arange(len(f)),id=meta[:,0],time=meta[:,1],airport=meta[:,2],
        day=days,stand=meta[:,3],runway=meta[:,4],hex=meta[:,5],clock=clock))
    d=q.iloc[positions].copy();d['slot']=slot
    erow,epos=np.where(bank['excluded']>=0)
    saved=pd.DataFrame(dict(row=erow,slot=bank['excluded'][erow,epos]))
    assert not saved.duplicated().any()
    with duckdb.connect(config={'threads':1,'memory_limit':'768MB'}) as con:
        for name,frame in [('q',q),('d',d),('saved',saved)]:con.register(name,frame)
        con.execute('''create temp table expected as
            select q.row,d.slot from q join d on q.id=d.id
            union select q.row,d.slot from q join d on q.airport=d.airport and q.day=d.day and q.hex=d.hex
                and q.hex>=0 and abs(q.time-d.time)<=120000
            union select q.row,d.slot from q join d on q.airport=d.airport and q.day=d.day and q.time=d.time
                and q.stand=d.stand and q.runway=d.runway and q.clock=d.clock
                and q.stand>0 and q.runway>0 and isfinite(q.clock)''')
        assert con.execute('select count(*) from (select * from expected except select * from saved)').fetchone()[0]==0
        assert con.execute('select count(*) from (select * from saved except select * from expected)').fetchone()[0]==0
        edge_count=con.execute('select count(*) from expected').fetchone()[0]
    assert edge_count==len(saved)
    result=dict(status='pass',rows=len(f),donors=len(did),groups=len(bank['keys']),
        exact_scaled_values=len(did)*17,exact_alias_edges=edge_count,
        all_source_output_hashes=True,all_source_ID_rule_and_calendar_reconstructed=True,
        all_17_scaled_values_exact=True,all_aliases_independent_SQL=True,
        padding_zero_and_canonical_donor_ID_membership=True,
        allowed_month_property='Every donor has exactly the query airport/UTC day; distinct allowed-month sets cannot mix',
        no_labels_or_predictions_read=True,model_and_trainer_review_separate=True)
    (OUT/'independent_bank_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
