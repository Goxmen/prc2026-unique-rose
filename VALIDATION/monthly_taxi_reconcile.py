"""Fixed monthly-level screen with training-only composition calibration.

Source method: EUROCONTROL ATXOT_indicator_documentation_mar23.pdf, section3.4.
Inspiration: gracious-tornado's public monthly calibration, credited in report.
This implementation fits composition directly on training label aggregates,
uses local calendar months and conservative half-strength shifts. It does not
copy their source or require their model predictions.
"""
import argparse
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import daily_delay_reconcile as base
from records import write_json

OUT=base.ROOT/'VALIDATION/runs/v3_20261011_monthly_taxi'
PARENT=base.ROOT/'VALIDATION/runs/v3_20261011_expanded_affine/submission_v73'
ZONES=dict(EDDF='Europe/Berlin',EDDM='Europe/Berlin',EGLL='Europe/London',EHAM='Europe/Amsterdam',
    LEBL='Europe/Madrid',LEMD='Europe/Madrid',LFPG='Europe/Paris',LIRF='Europe/Rome',LSZH='Europe/Zurich',LTFM='Europe/Istanbul')
DESIGN=dict(parent='v73',weight=.5,maximum_invalid_share=.05,minimum_cell_rows=100,
    maximum_taxi_sec=7200.,maximum_abs_shift_sec=120.,composition='airport intercept + invalid share + no-reference share',
    regression='Huber 20 seconds; ridge 0.001; equal airport-month weighting',
    train_selection='stand and runway present, true taxi <=2h',
    inference_selection='stand and runway present, prediction <=2h',
    calendar='airport local month; cells below100 rows unchanged',
    limitation='Unobserved deicing, reference eligibility and helicopter exclusions; true/predicted truncation mismatch',
    validation='same allowed Jan/Jul training months; no selected airport gate')

def public(year):
    f=pd.read_csv(base.SOURCE/f'taxi_out_additional_time_{year}.csv')
    f=f[f.APT_ICAO.isin(ZONES)].copy()
    f['local_month']=f.YEAR.astype(str)+'-'+f.MONTH_NUM.astype(str).str.zfill(2)
    f['mean_sec']=60*(f.TOTAL_REF_TIME_MIN+f.TOTAL_ADD_TIME_MIN)/f.TOTAL_REF_NB_FL
    f['invalid']=(f.TF-f.VALID_FL)/f.TF;f['noref']=f.NO_REF/f.TF
    f=f.rename(columns={'APT_ICAO':'airport'})
    assert not f.duplicated(['airport','local_month']).any()
    return f[['airport','local_month','mean_sec','invalid','noref','TF','VALID_FL','TOTAL_REF_NB_FL']]

def load(paths,labelled=True):
    cols=['MVT_ID_mvt','ADEP_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','STAND_mvt','RUNWAY_mvt']
    if labelled:cols+=['TAXITIME_SEC_mvt']
    f=pd.concat([pd.read_parquet(p,columns=cols,filters=[('PHASE_mvt','==','DEP')]) for p in paths],ignore_index=True)
    f['local_month']=''
    for a,z in ZONES.items():
        m=f.ADEP_mvt.eq(a);f.loc[m,'local_month']=f.loc[m,'MVT_TIME_UTC_mvt'].dt.tz_convert(z).dt.strftime('%Y-%m')
    out=pd.DataFrame(dict(mvt_id=f.MVT_ID_mvt,airport=f.ADEP_mvt,local_month=f.local_month,
        utc_month=f.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m'),eligible=f.STAND_mvt.notna()&f.RUNWAY_mvt.notna()))
    if labelled:out['y']=f.TAXITIME_SEC_mvt.to_numpy(float)
    assert out.mvt_id.is_unique and out.local_month.str.len().eq(7).all()
    return out

def matrix(f):
    return np.column_stack([f.airport.eq(a).astype(float) for a in ZONES]+[f.invalid,f.noref])

def fit(rows,source):
    m=rows.eligible&rows.y.between(0,7200)
    cells=rows.loc[m].groupby(['airport','local_month'],as_index=False).agg(truth=('y','mean'),n=('y','size'))
    cells=cells.merge(source,on=['airport','local_month'],how='inner',validate='one_to_one')
    cells=cells[(cells.n>=100)&(cells.invalid<.05)].copy()
    X=matrix(cells);y=(cells.truth-cells.mean_sec).to_numpy(float)
    # Penalize scaled coefficients so weak weather/missing-reference correlations
    # do not receive a numerically arbitrary advantage from their small units.
    scale=np.r_[np.ones(10),.05,.1];A=X/scale
    fun=lambda b:np.r_[(A@b-y)/20.,np.sqrt(.001)*b/20.]
    result=least_squares(fun,np.zeros(12),loss='huber',f_scale=1.)
    assert result.success
    return dict(coefficients=(result.x/scale).tolist(),cells=cells.to_dict('records'),fit_message=result.message)

def apply(rows,pred,source,params):
    f=rows.copy();f['pred']=pred;eligible=f.eligible&(f.pred<=7200)
    c=f.loc[eligible].groupby(['airport','local_month'],as_index=False).agg(pred_mean=('pred','mean'),n=('pred','size'))
    c=c.merge(source,on=['airport','local_month'],how='left',validate='one_to_one')
    c['composition']=matrix(c)@np.asarray(params['coefficients'])
    c['raw_shift']=c.mean_sec+c.composition-c.pred_mean
    c['shift']=np.where((c.invalid<.05)&(c.n>=100),.5*c.raw_shift.clip(-120,120),0.)
    c['shift']=c['shift'].fillna(0)
    z=f[['airport','local_month']].merge(c[['airport','local_month','shift']],on=['airport','local_month'],how='left',validate='many_to_one')
    q=np.maximum(0,np.asarray(pred,float)+np.where(eligible,z['shift'].fillna(0),0))
    return q,c

def screen():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'result.json').exists(),'Preserve completed screen'
    paths=sorted((base.ROOT/'data').glob('training_2025-*.parquet'))
    write_json(OUT/'design.json',dict(design=DESIGN,source=base.artifact(__file__),inputs=[base.artifact(p) for p in paths],
        external=base.artifact(base.SOURCE/'taxi_out_additional_time_2025.csv'),parent=base.artifact(PARENT/'cv.parquet')))
    rows=load(paths);rows.to_parquet(OUT/'rows2025.parquet',index=False);source=public(2025)
    cv=pd.read_parquet(PARENT/'cv.parquet');aligned=rows.set_index('mvt_id').loc[cv.mvt_id].reset_index();np.testing.assert_array_equal(aligned.y,cv.y)
    out=cv.pred.to_numpy(float).copy();tables=[];parameters={}
    for month,allowed in base.FOLDS.items():
        parameters[month]=fit(rows[rows.utc_month.isin(allowed)],source)
        ix=np.flatnonzero(cv.month.eq(month));out[ix],cells=apply(aligned.iloc[ix].reset_index(drop=True),out[ix],source,parameters[month])
        tables.append(cells.assign(fold=month))
    f=cv[['mvt_id','month','day','airport','y']].copy();f['parent']=cv.pred;f['pred']=out
    f.to_parquet(OUT/'predictions.parquet',index=False);pd.concat(tables).to_parquet(OUT/'cells.parquet',index=False)
    import tuning_seasonal as ts
    scores={m:dict(parent=float(np.sqrt(np.mean((q.parent-q.y)**2))),candidate=float(np.sqrt(np.mean((q.pred-q.y)**2)))) for m,q in f.groupby('month')}
    paired=ts.paired_bootstrap(f.day,f.month,(f.parent-f.y)**2,(f.pred-f.y)**2)
    result=dict(design=DESIGN,months=scores,cv=float(np.mean([x['candidate'] for x in scores.values()])),paired=paired,
        parameters=parameters,source=base.artifact(__file__))
    write_json(OUT/'result.json',result);print(scores,paired,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['screen']);a=p.parse_args();globals()[a.stage]()
