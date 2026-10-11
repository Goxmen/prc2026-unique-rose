"""Reload public-feature classifier models and verify every held prediction."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'VALIDATION'))
import lightgbm as lgb,numpy as np,pandas as pd
import rome_public_delay_jan_jul as r20
from records import write_json,sha256

def main():
    checks=[]
    for month in r20.shared.ts.SPLITS:
        d=pd.read_pickle(r20.OUT/'inputs'/f'{month}.pkl')
        path=r20.OUT/'predictions'/f'{month}_s42.parquet';record=r20.shared.ts.read(path.with_suffix('.json'))
        mp=r20.OUT/'models'/f'{month}_s42.txt'
        assert sha256(path)==record['prediction_sha256'] and sha256(mp)==record['model_sha256']
        model=lgb.Booster(model_file=str(mp));p=model.predict(d['B'],num_threads=1)
        f=pd.read_parquet(path);np.testing.assert_array_equal(f.mvt_id,d['ids'])
        np.testing.assert_allclose(p,f.p.to_numpy()[d['use']],rtol=0,atol=1e-12)
        assert (f.p.to_numpy()[~d['use']]==0).all()
        assert list(d['B'])==model.feature_name()
        checks.append(dict(month=month,held_rows=len(p),max_error=float(np.max(np.abs(p-f.p.to_numpy()[d['use']])))))
    result=dict(status='pass',r20_reload=checks,source=r20.features.base.artifact(__file__),score=r20.features.base.artifact(r20.OUT/'result.json'))
    write_json(r20.OUT/'audit.json',result);print(result,flush=True)
if __name__=='__main__':main()
