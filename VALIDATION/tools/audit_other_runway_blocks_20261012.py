"""Six input-selected target checks via independent block assembly/direct solves."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joint_public_moments as moments
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'VALIDATION/runs/v3_20261011_joint_public_moments'
NEW=ROOT/'VALIDATION/runs/v3_20261012_joint_public_other_runway'
SELECTION=ROOT/'tmp/research_20261012/error_mechanisms/other_runway/design.json'
OUT=ROOT/'VALIDATION/reports/all_discrete_peer_20261012/other_runway_block_audit.json'
def read(p):return json.loads(Path(p).read_text())
def art(p):return dict(path=str(Path(p).relative_to(ROOT)),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def main():
    started=time.monotonic();assert not OUT.exists()
    inputs=pd.read_parquet(OLD/'inputs.parquet',columns=['mvt_id','month','gap','prior','eligible','quality','runway','local_day','assigned_day','fltrule','delay_scope'])
    old=pd.read_parquet(OLD/'predictions.parquet',columns=['mvt_id','touched']).set_index('mvt_id')
    actual=pd.read_parquet(NEW/'predictions.parquet').set_index('mvt_id')
    tx=pd.read_parquet(OLD/'taxi_source.parquet').set_index(['local_day','runway'])
    dep=pd.read_parquet(OLD/'delay_source.parquet').set_index('day');params=read(OLD/'parameters.json')
    selected=read(SELECTION)['selected'];checks=[]
    for month in ['2025-01','2025-07']:
        keys=[x for x in selected if x['month']==month];keys=[keys[j] for j in sorted(set([0,len(keys)//2,len(keys)-1]))]
        r=inputs[inputs.month.eq(month)].reset_index(drop=True);p=params[month]
        gap=r.gap.to_numpy(float);taxi=(r.eligible&r.quality).to_numpy(bool)
        delay=(~r.fltrule.eq('V')&r.delay_scope).to_numpy(bool)
        v,c,prob=moments.normal_moments(r.prior.to_numpy(float),gap);supports={}
        needed=(taxi&r.local_day.isin([x['local_day'] for x in keys]).to_numpy()) | (delay&r.assigned_day.isin([x['assigned_day'] for x in keys]).to_numpy())
        for j in np.flatnonzero(needed&(gap>3600)):
            y,w,_=moments.prior.support(gap[j],float(r.prior.iloc[j]),False,np.nan,'LIRF',p['support'])
            valid=(y>=0)&(y<=7200);vec=np.column_stack((valid,np.where(valid,y,0),np.maximum(gap[j]-y,0))).astype(float)
            v[j]=w@vec;center=vec-v[j];c[j]=(center.T*w)@center;supports[j]=(y,w,vec)
        for key in keys:
            own=(key['local_day'],key['runway']);dkey=key['assigned_day']
            dix=np.flatnonzero(delay&r.assigned_day.eq(dkey).to_numpy());dset=set(dix)
            groups={}
            for rw in sorted(r.loc[taxi&r.local_day.eq(own[0]).to_numpy(),'runway'].unique()):
                tk=(own[0],rw);ix=np.flatnonzero(taxi&r.local_day.eq(own[0]).to_numpy()&r.runway.eq(rw).to_numpy())
                if tk not in tx.index:continue
                count=float(v[ix,0].sum());src=tx.loc[tk]
                if len(ix)>=20 and src.source_n>=20 and count>=20 and .5<=src.source_n/count<=1.2:groups[tk]=ix
            extras=sorted(k for k,ix in groups.items() if k!=own and dset.intersection(ix))
            taxis=[own,*extras];dim=3+2*len(extras);mu=np.zeros(dim);cov=np.zeros((dim,dim));obs=np.zeros(dim);load=np.zeros(dim)
            ix_to_offset={int(j):(0 if n==0 else 3+2*(n-1)) for n,k in enumerate(taxis) for j in groups[k]}
            union=sorted(set(ix_to_offset)|dset)
            for j in union:
                projection=np.zeros((dim,3))
                if j in ix_to_offset:
                    offset=ix_to_offset[j];projection[offset,0]=projection[offset+1,1]=1
                if j in dset:projection[2,2]=1
                mu+=projection@v[j];cov+=projection@c[j]@projection.T
            for n,k in enumerate(taxis):
                offset=0 if n==0 else 3+2*(n-1);ix=groups[k];src=tx.loc[k]
                bias=np.asarray(p['taxi']['coefficients'])[0]+len(ix)/100*np.asarray(p['taxi']['coefficients'])[1]
                obs[offset:offset+2]=[src.source_n,src.source_n*src.source_mean]+bias
                cov[offset:offset+2,offset:offset+2]+=len(ix)*np.asarray(p['taxi']['covariance_per_row'])+np.diag([1.,3600.])
                load[offset+1]=v[ix,0].sum()
            ds=dep.loc[dkey];obs[2]=ds.total-p['delay']['bias']
            cov[2,2]+=3*(p['delay']['sigma']**2+(p['delay']['kappa']*(ds.n_public-len(dix)))**2)
            load[2]=-sum(prob[j] for j in dix if gap[j]<=3600);cov+=225*np.outer(load,load)
            eligible=r.local_day.eq(own[0])&r.runway.eq(own[1])&r.assigned_day.eq(dkey)&old.loc[r.mvt_id,'touched'].to_numpy(bool)
            flight=int(r.loc[eligible,'mvt_id'].min());j=int(np.flatnonzero(r.mvt_id.eq(flight))[0]);saved=actual.loc[flight]
            assert j in dset and j in groups[own] and all(j not in groups[x] for x in extras)
            mean=mu[:3]-v[j];a=cov[:3,:3]-c[j];q=0.;weight=1.
            if extras:
                cross=cov[:3,3:];b=cov[3:,3:];innovation=obs[3:]-mu[3:]
                q=float(innovation@np.linalg.solve(b,innovation));weight=min(1.,(5+len(innovation))/(3+q))
                mean=mean+weight*cross@np.linalg.solve(b,innovation)
                a=a-weight*cross@np.linalg.solve(b,cross.T)
            y,w,vec=supports[j];res=obs[:3]-mean-vec
            quad=np.sum(res*np.linalg.solve(.6*a,res.T).T,axis=1)
            # Normalizer/determinant are constant across this target's support.
            logw=np.log(np.maximum(w,1e-300))-4*np.log1p(np.maximum(quad,0)/5)
            computed=float(y@np.exp(logw-logsumexp(logw))) if extras else float(saved.old_joint)
            np.testing.assert_allclose(computed,saved.new_joint,atol=1e-7,rtol=0)
            np.testing.assert_allclose(weight,saved.weight,atol=1e-11,rtol=0)
            np.testing.assert_allclose(q,saved.extra_q,atol=1e-8,rtol=1e-11)
            checks.append(dict(month=month,mvt_id=flight,key=key,extra_cells=len(extras),rows=len(union),mean_abs_error_sec=abs(computed-float(saved.new_joint)),weight_abs_error=abs(weight-float(saved.weight)),q_abs_error=abs(q-float(saved.extra_q))))
    value=dict(status='pass',source=art(__file__),seconds=time.monotonic()-started,selection='first/middle/last previously input-selected cohort per month; minimum touched movement ID',checks=checks,inputs=[art(p) for p in [SELECTION,OLD/'inputs.parquet',OLD/'taxi_source.parquet',OLD/'delay_source.parquet',OLD/'parameters.json',NEW/'identity.json',NEW/'predictions.parquet']],scope='Independent per-flight projection covariance, direct block solves and t5 normalized support mean. Reuses audited original support and analytic ordinary moments. No labels, CV results or target selection by predictions.',posterior_atol_sec=1e-7)
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(json.dumps(value,indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
