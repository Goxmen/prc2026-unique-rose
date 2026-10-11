"""Input-only summaries of public daily pre-departure totals.

EUROCONTROL is credited in TARGET215_RESEARCH.md. Fixed physical references
900/1800 seconds bracket ordinary taxi times without fitting to row labels.
All arithmetic uses movement/schedule/AOBT_3 clocks and the public total only.
Departures marked VFR are excluded from population sums; unknown rules remain.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import duckdb,numpy as np,pandas as pd
import daily_delay_reconcile as base
from records import write_json
OUT=base.ROOT/'VALIDATION/runs/v3_20261011_public_delay_features'
FEATURES=['pub_dep_count_gap','pub_dep_mean','pub_dep_long_count','pub_dep_long_gap_sum',
    'pub_dep_resid900_mean','pub_dep_resid1800_mean','pub_dep_resid_clock_mean',
    'pub_dep_own900_left','pub_dep_own1800_left','pub_dep_own900_fraction',
    'pub_dep_own1800_fraction','pub_dep_long900_fraction']

def load(paths):
    with duckdb.connect(config={'threads':1}) as con:
        con.execute("SET TimeZone='UTC'")
        return con.execute('''select MVT_ID_mvt mvt_id,ADEP_mvt airport,
            strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS "day",
            epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt) gap,
            epoch(MVT_TIME_UTC_mvt-AOBT_3_flt) clock,
            FLIGHT_RULE_mvt fltrule from read_parquet(?) where PHASE_mvt='DEP' ''',[list(map(str,paths))]).fetchdf()

def build(rows,source):
    f=rows.copy();eligible=~f.fltrule.eq('V');long=f.gap.gt(3600)
    f['count']=eligible.astype(float);f['long_n']=(eligible&long).astype(float)
    f['long_gap']=np.where(eligible&long,f.gap,0)
    for ref in [900,1800]:f[f'contrib{ref}']=np.where(eligible,np.maximum(0,f.gap-ref),0)
    clock=np.clip(f.clock.fillna(1200).to_numpy(float),30,7200)
    f['contrib_clock']=np.where(eligible,np.maximum(0,f.gap-clock),0)
    f['long900']=np.where(long,f.contrib900,0)
    cols=['count','long_n','long_gap','contrib900','contrib1800','contrib_clock','long900']
    agg=f.groupby(['airport','day'],as_index=False)[cols].sum()
    agg=agg.merge(source,on=['airport','day'],validate='one_to_one')
    g=f[['mvt_id','airport','day','gap','contrib900','contrib1800']].merge(agg,on=['airport','day'],suffixes=('_own',''),validate='many_to_one')
    out=pd.DataFrame({'mvt_id':g.mvt_id})
    n=np.maximum(g['count'],1)
    out['pub_dep_count_gap']=g.n_public-g['count'];out['pub_dep_mean']=g.total/np.maximum(g.n_public,1)
    out['pub_dep_long_count']=g.long_n;out['pub_dep_long_gap_sum']=g.long_gap
    for ref in [900,1800]:
        residual=g.total-g[f'contrib{ref}']
        left=residual+g[f'contrib{ref}_own']
        out[f'pub_dep_resid{ref}_mean']=residual/n
        out[f'pub_dep_own{ref}_left']=left
        out[f'pub_dep_own{ref}_fraction']=left/np.maximum(g[f'contrib{ref}_own'],600)
    out['pub_dep_resid_clock_mean']=(g.total-g.contrib_clock)/n
    out['pub_dep_long900_fraction']=(g.total-(g.contrib900-g.long900))/np.maximum(g.long900,600)
    return out[['mvt_id']+FEATURES]

def checks():
    rows=pd.DataFrame(dict(mvt_id=[1,2,3],airport='LIRF',day='2025-01-01',gap=[1000.,5000.,8000.],clock=[700.,800.,900.],fltrule=['I',None,'V']))
    source=pd.DataFrame(dict(airport=['LIRF'],day=['2025-01-01'],n_public=[2.],total=[3500.]))
    out=build(rows,source).set_index('mvt_id')
    assert (out.pub_dep_count_gap==0).all() and (out.pub_dep_long_count==1).all()
    np.testing.assert_allclose(out.loc[2,'pub_dep_own900_left'],3400.)
    np.testing.assert_allclose(out.loc[2,'pub_dep_resid900_mean'],-350.)
    poison=rows.assign(y=1e12,BLOCK_TIME_UTC_mvt='2099-01-01')
    pd.testing.assert_frame_equal(build(poison,source),build(rows,source))
    pd.testing.assert_frame_equal(build(rows.sample(frac=1,random_state=42),source).sort_values('mvt_id').reset_index(drop=True),build(rows,source))
    return dict(explicit_vfr_excluded=True,unknown_included=True,hand_arithmetic=True,departure_label_isolation=True,row_order_invariant=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'manifest.json').exists()
    paths=sorted((base.ROOT/'data').glob('training_2025-*.parquet'))
    checks_result=checks();rows=load(paths);features=build(rows,base.public(2025))
    assert features.mvt_id.is_unique and len(features)==2085047
    features.to_parquet(OUT/'features2025.parquet',index=False)
    rows.to_parquet(OUT/'inputs2025.parquet',index=False)
    write_json(OUT/'manifest.json',dict(features=FEATURES,source=base.artifact(__file__),inputs=[base.artifact(p) for p in paths],
        public=base.artifact(base.SOURCE/'all_pre_departure_delays_2025.csv'),output=base.artifact(OUT/'features2025.parquet'),checks=checks_result))
    print(checks_result,len(features),flush=True)
if __name__=='__main__':main()
