"""Population-corrected daily posterior: exclude explicitly VFR departures.

The EUROCONTROL count is IFR. Unknown movement flight rules remain included:
training source identity shows most belong to the published population. This
screen changes only the population in the already tested posterior; no new
source-model settings or held-label tuning. Standalone parent remains v71 so
it can replace v72's component without conditioning twice on the same total.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd,duckdb
import daily_delay_posterior as posterior
from records import write_json
base=posterior.base
OUT=base.ROOT/'VALIDATION/runs/v3_20261011_daily_ifr'
DESIGN=dict(parent='v71',control='v72 full daily posterior',change='exclude explicit VFR departures from source population',
    unknown_rule='included',posterior=posterior.DESIGN,held_labels='final development scoring only',no_parameter_search=True)

def load_flags(paths):
    with duckdb.connect(config={'threads':1}) as con:
        return con.execute("select MVT_ID_mvt mvt_id,FLIGHT_RULE_mvt fltrule from read_parquet(?) where PHASE_mvt='DEP'",[list(map(str,paths))]).fetchdf()

def fit(rows,source):
    return posterior.fit(rows.loc[~rows.fltrule.eq('V')],source)

def apply(rows,pred,source,params):
    ix=np.flatnonzero(~rows.fltrule.eq('V'));out=np.asarray(pred,float).copy()
    out[ix],logs=posterior.apply(rows.iloc[ix].reset_index(drop=True),out[ix],source,params)
    assert np.array_equal(out[rows.fltrule.eq('V')],np.asarray(pred)[rows.fltrule.eq('V')])
    return out,logs

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'result.json').exists()
    paths=sorted((base.ROOT/'data').glob('training_2025-*.parquet'))
    rows=pd.read_parquet(base.OUT/'rows2025.parquet').merge(load_flags(paths),on='mvt_id',validate='one_to_one')
    rows.to_parquet(OUT/'rows2025.parquet',index=False)
    write_json(OUT/'design.json',dict(design=DESIGN,source=base.artifact(__file__),helpers=[base.artifact(posterior.__file__),base.artifact(base.__file__)],inputs=[base.artifact(p) for p in paths]))
    cv=pd.read_parquet(base.PARENT/'cv.parquet');aligned=rows.set_index('mvt_id').loc[cv.mvt_id].reset_index()
    old=pd.read_parquet(posterior.OUT/'screen.parquet');np.testing.assert_array_equal(old.mvt_id,cv.mvt_id)
    out=cv.pred.to_numpy(float).copy();params={};logs=[];source=base.public(2025)
    for month,allowed in base.FOLDS.items():
        ix=np.flatnonzero(cv.month.eq(month));params[month]=fit(rows.loc[rows.month.isin(allowed)],source)
        out[ix],log=apply(aligned.iloc[ix].reset_index(drop=True),out[ix],source,params[month]);logs.append(log)
    f=cv[['mvt_id','month','day','airport','y']].copy();f['parent']=cv.pred;f['old_daily']=old.pred;f['pred']=out
    f.to_parquet(OUT/'predictions.parquet',index=False);pd.concat(logs).to_parquet(OUT/'days.parquet',index=False)
    import tuning_seasonal as ts
    scores={m:{n:float(np.sqrt(np.mean((q[n]-q.y)**2))) for n in ['parent','old_daily','pred']} for m,q in f.groupby('month')}
    paired={n:ts.paired_bootstrap(f.day,f.month,(f[n]-f.y)**2,(f.pred-f.y)**2) for n in ['parent','old_daily']}
    write_json(OUT/'result.json',dict(design=DESIGN,scores=scores,cv=float(np.mean([q['pred'] for q in scores.values()])),paired=paired,parameters=params,source=base.artifact(__file__)))
    print(scores,paired,flush=True)
if __name__=='__main__':main()
