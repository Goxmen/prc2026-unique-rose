"""Reload every public-feature CatBoost CV model; compare all held probabilities."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'VALIDATION'))
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
import rome_public_delay_cat as run
from records import write_json

def main():
    reports=[]
    for month in run.parent.ROUNDS:
        d=pd.read_pickle(run.r20.OUT/'inputs'/f'{month}.pkl')
        cats=run.parent.validated_parent(month,False)[0]['categoricals'];X=run.parent.parent.cbj.frame(d['B'],cats)
        for seed in run.parent.SEEDS:
            path=run.OUT/'predictions'/f'{month}_s{seed}.parquet';rec=run.parent.shared.ts.read(path.with_suffix('.json'))
            mp=run.OUT/'models'/f'{month}_s{seed}.cbm'
            assert rec['model']==run.r20.features.base.artifact(mp) and rec['prediction']==run.r20.features.base.artifact(path)
            model=CatBoostClassifier();model.load_model(str(mp));p=model.predict_proba(Pool(X,cat_features=cats),thread_count=1)[:,1]
            f=pd.read_parquet(path);np.testing.assert_array_equal(f.mvt_id,d['ids']);np.testing.assert_array_equal(p,f.p.to_numpy()[d['use']])
            assert (f.p.to_numpy()[~d['use']]==0).all() and model.feature_names_==list(d['B'])
            reports.append(dict(month=month,seed=seed,rows=len(p),max_error=0.))
    result=dict(status='pass',models=reports,source=run.r20.features.base.artifact(__file__),score=run.r20.features.base.artifact(run.OUT/'result.json'))
    write_json(run.OUT/'audit.json',result);print(result,flush=True)
if __name__=='__main__':main()
