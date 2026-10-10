"""Refit legacy B0/R3 recipes on the current Jan/Jul folds for a new blend.

The old 10/11-month validations are not reused as today's 9/7-month folds.
Uses the archived, hash-verified Stage1 implementation and existing seasonal
runner, with the history cutoff fixed independently at each fold's train end.
No full-year refit, old ledger backfill, or upload. Runs one CPU fit at a time.
"""
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS','6')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import seasonal
import tuning_seasonal as ts
from movement_data import load_phase
from common import connect
from records import sha256,write_json,runtime_identity
OUT=ROOT/'VALIDATION/runs/v3_20261011_legacy_basic_jan_jul'

def main():
    batch,items=seasonal.references()
    blob,sources,reference=seasonal.model_source()
    config=dict(reference['config'],seeds=[42],rounds=800,sample=1.,threads=6,
                prediction_threads=6,fit_workers=1,features=batch['models']['recovery']['features'])
    for name in ('folds','train_months','label','smoke','recovery','checkpoints'):config.pop(name,None)
    config['smoke']=False
    jobs=[dict(id='janjul_'+m,strategy='current_9_7_month',train=ts.FOLDS[m],val=[m]) for m in ts.SPLITS]
    identity=dict(source=sha256(__file__),sources=sources,config=config,jobs=jobs,
                  freezes={m:ts.fold_freeze(m) for m in ts.SPLITS},
                  version_columns={'12':'B0-800','13':'Q0','14':'Q1','15':'Q0-4h','16':'Q1-4h'},
                  note='Fresh current-fold predictions only; no old submission ledger backfill')
    OUT.mkdir(exist_ok=True)
    p=OUT/'identity.json'
    if p.exists() and json.loads(p.read_text())!=identity:raise ValueError('Legacy identity changed')
    write_json(p,identity)
    progress=json.loads((OUT/'progress.json').read_text()) if (OUT/'progress.json').exists() else dict(models={},predictions={})
    seasonal.check_artifacts(OUT,progress)
    backend=seasonal.load_backend(blob,OUT)
    con=connect();dep=load_phase(con,'DEP');arr=load_phase(con,'ARR');con.close()
    with seasonal.run_lock(OUT):
        for job in jobs:
            month=job['val'][0]
            seasonal.FREEZE=ts.fold_freeze(month)
            seasonal.execute(dep,arr,[job],config,backend,OUT,progress)
    write_json(OUT/'complete.json',dict(identity_sha256=sha256(p),models=len(progress['models']),
        predictions=len(progress['predictions']),runtime=runtime_identity()))

if __name__=='__main__':main()
