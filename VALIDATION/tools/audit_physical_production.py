"""Reload retained production models on input-selected rows in both namespaces."""
import argparse,gc,os,pickle,sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import nn_track_physical_submit as runner
from records import write_json

def main(members):
    m=runner.guard();assert '5060' in torch.cuda.get_device_name()
    records={k:runner.completed(k,m) for k in members};assert all(records.values())
    with runner.verify(runner.trained(members[0],m)['preprocessing']).open('rb') as h:pre=pickle.load(h)
    checks=[]
    for view in ['old','final']:
        packet=runner.old.packets.load_packet(view,m['packet'])
        ids=packet['rank_ids'];folder=ROOT/m['tracks'][view]['directory']
        index=pd.read_parquet(folder/'full_index.parquet').set_index('mvt_id',drop=False).loc[ids]
        points=np.memmap(folder/'full_points.bin',mode='r',dtype=np.int32).reshape(-1,6)
        positions=[np.linspace(0,len(ids)-1,512,dtype=int)]
        for mask in [index['count'].eq(0),index['count'].eq(1),index['count'].between(2,31),
                     index['count'].ge(32),index.raw_ground_count.gt(0),index.clock_ms.isna()]:
            positions.append(np.flatnonzero(mask)[:8])
        for airport in sorted(index.airport.unique()):positions.append(np.flatnonzero(index.airport.eq(airport))[:8])
        at=np.unique(np.concatenate(positions));chosen=np.asarray(ids)[at]
        X=packet['Vn'].iloc[at][pre['columns']].copy();del packet;gc.collect()
        cat,num=pre['encoder'].transform(X)
        tensors=[torch.from_numpy(np.ascontiguousarray(a)).cuda() for a in [cat,num[:,pre['ple']],num[:,pre['plain']]]]
        selected=index.iloc[at].reset_index(drop=True)
        tracks=runner.cv.base.tensor_seq.RaggedTracks(selected,points,'cuda')
        track=tracks.batch(np.arange(len(selected)))
        for k,r in records.items():
            trained=runner.trained(k,m)
            model=runner.cv.physical_scales.SparseFusion(pre['cards'],pre['bins'],len(pre['plain'])).cuda().eval()
            checkpoint=torch.load(runner.verify(trained['model']),map_location='cuda',weights_only=False)
            assert checkpoint['design']==runner.DESIGN
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            with torch.no_grad():z=model(*tensors,*track)[:,0].cpu().numpy().astype(float)
            pred=np.maximum(z*pre['sd']+pre['mu'],0)
            saved=pd.read_parquet(runner.verify(r['predictions'][view]))
            np.testing.assert_array_equal(saved.mvt_id,ids)
            assert np.isfinite(saved.base).all() and saved.base.ge(0).all()
            expected=saved.base.to_numpy(float)[at]
            np.testing.assert_allclose(pred,expected,rtol=0,atol=.01)
            checks.append(dict(member=k,view=view,rows=len(at),airports=sorted(selected.airport.unique()),
                max_difference_sec=float(abs(pred-expected).max()),record=runner.art(runner.paths(k)[2])))
            del model,checkpoint;torch.cuda.empty_cache()
        del tensors,track,tracks,X,index,selected,points,saved;gc.collect();torch.cuda.empty_cache()
    result=dict(status='pass',members=members,checks=checks,source=runner.art(__file__),
        manifest=runner.art(runner.MANIFEST),reviewer='root; same-GPU reload from input-selected rows; not independent development data')
    path=runner.OUT/('audit_'+('_'.join(map(str,members)))+'.json')
    assert not path.exists(),'Retain completed production audit'
    write_json(path,result);print(result,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--members',nargs='+',type=int,default=[0]);a=ap.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):main(a.members)
