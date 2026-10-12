"""Input-only census of taxi cells omitted by the same-date runway restriction."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import joint_public_other_runway_pilot as parent
import joint_public_moments as moments
from records import sha256,write_json

OUT=ROOT/'VALIDATION/runs/v3_20261012_cross_date_taxi_coverage'
VIEWS={'cv':ROOT/'VALIDATION/runs/v3_20261011_joint_public_moments',
       'ranking':ROOT/'VALIDATION/runs/v3_20261011_joint_public_moments_submit'}


def art(path):return dict(path=str(path.relative_to(ROOT)),sha256=sha256(path))


def main():
    assert not (OUT/'identity.json').exists(),'Preserve census'
    inputs={}
    for view,folder in VIEWS.items():
        # Production file naming is explicitly verified before reading any rows.
        paths={key:folder/(key+'.parquet') for key in ['inputs','taxi_source','predictions']}
        paths['parameters']=folder/'parameters.json'
        assert all(p.exists() for p in paths.values()),paths
        inputs[view]={key:art(p) for key,p in paths.items()}
    design=dict(scope='all original touched targets in Jan/Jul2025 and accepted Jan/Jul2026',
        change='only inspect additional local-day/runway taxi cells outside own local date that intersect the same assigned-day departure-delay cohort',
        eligibility='unchanged rawcount>=20,sourcecount>=20,expectedcount>=20,source/expected count in[.5,1.2]',
        expected_counts='unchanged original normal/discrete priors and frozen source parameters',
        outputs='eligible cells, target coverage, shared peer count; no covariance conditioning or posterior',
        no_labels=True,no_cv_score=True,no_model_fit=True,no_source_download=True,no_upload=True)
    write_json(OUT/'identity.json',dict(design=design,created_at=datetime.now(timezone.utc).isoformat(),
        source=art(Path(__file__).resolve()),sources=inputs,helpers=[art(Path(moments.__file__)),art(Path(parent.__file__))]))
    records=[]
    for view,source in inputs.items():
        rows=pd.read_parquet(ROOT/source['inputs']['path'],columns=parent.INPUTS)
        prior=pd.read_parquet(ROOT/source['predictions']['path'],columns=['mvt_id','touched']).set_index('mvt_id')
        taxi=pd.read_parquet(ROOT/source['taxi_source']['path']).set_index(['local_day','runway'])
        parameters=json.loads((ROOT/source['parameters']['path']).read_text())
        assert rows.mvt_id.is_unique and prior.index.is_unique and taxi.index.is_unique
        for month,f in rows.groupby('month',sort=True):
            f=f.reset_index(drop=True);gap=f.gap.to_numpy(float);p=f.prior.to_numpy(float)
            prm=parameters[month]
            means,_,_=moments.normal_moments(p,gap)
            for i in np.flatnonzero(gap>3600):
                y,w,_=moments.prior.support(gap[i],p[i],False,np.nan,'LIRF',prm['support'])
                means[i]=w@moments.contributions(y,gap[i])
            ok=(f.eligible&f.quality).to_numpy(bool);pos=np.flatnonzero(ok)
            groups={key:pos[ix] for key,ix in f.loc[ok].groupby(['local_day','runway']).indices.items()}
            allowed={}
            for key,ix in groups.items():
                if key not in taxi.index:continue
                n=float(means[ix,0].sum());sn=float(taxi.loc[key].source_n)
                if len(ix)>=20 and sn>=20 and n>=20 and .5<=sn/n<=1.2:allowed[key]=ix
            touched=prior.loc[f.mvt_id,'touched'].to_numpy(bool)
            cohort=f.loc[touched].groupby(parent.KEYS).mvt_id.agg(list)
            delay_ok=(~f.fltrule.eq('V')&f.delay_scope).to_numpy(bool)
            for key,targets in cohort.items():
                own_day,runway,assigned=key
                di=np.flatnonzero(delay_ok&f.assigned_day.eq(assigned).to_numpy())
                extra=[]
                for (day,rwy),ix in allowed.items():
                    if day==own_day:continue
                    shared=np.intersect1d(ix,di)
                    if not len(shared):continue
                    assert not np.intersect1d(f.mvt_id.iloc[ix],targets).size
                    extra.append(dict(local_day=day,runway=rwy,shared_rows=len(shared),
                        shared_long_rows=int((gap[shared]>3600).sum()),taxi_rows=len(ix),
                        expected_count=float(means[ix,0].sum()),source_count=float(taxi.loc[(day,rwy)].source_n)))
                records.append(dict(view=view,month=month,local_day=own_day,runway=runway,assigned_day=assigned,
                    target_ids=[int(v) for v in targets],targets=len(targets),extra_cells=extra))
    summary={}
    for view in VIEWS:
        selected=[r for r in records if r['view']==view]
        summary[view]=dict(cohorts=len(selected),cohorts_with_extra=sum(bool(r['extra_cells']) for r in selected),
            targets=sum(r['targets'] for r in selected),covered_targets=sum(r['targets'] for r in selected if r['extra_cells']),
            distinct_extra_cells=len({(e['local_day'],e['runway']) for r in selected for e in r['extra_cells']}))
    write_json(OUT/'result.json',dict(identity=art(OUT/'identity.json'),summary=summary,records=records,
        no_labels_or_posterior=True,interpretation='Coverage only. Other dates require their own source-error covariance treatment before any posterior experiment.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with threadpool_limits(1):main()
