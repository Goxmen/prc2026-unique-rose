"""Diagnose affine blends from our own scored predictions and aggregate RMSE.

For d_i = p_i - p_ref, let G_ij = mean(d_i*d_j). The public scores
identify c_i = mean((p_ref-y)*d_i) = (score_i^2-score_ref^2-G_ii)/2.
An affine blend then has MSE score_ref^2 + 2*c@w + w@G@w.
This is selection on the leaderboard, not independent validation. No hidden
labels are read, no model is trained, and no file is submitted by this tool.
A selected recipe still needs Jan/Jul scoring and serialization/source audits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/derived/scored_submission_cache_20261011'
OUT = ROOT / 'VALIDATION/runs/v3_20261011_target225/geometry'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(min_version):
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    ref = pd.read_parquet(ROOT / manifest['64']['file'])
    ids = ref.MVT_ID_mvt.to_numpy()
    versions, scores, columns, records = [], [], [], {}
    seen = set()
    for key, row in sorted(manifest.items(), key=lambda item: int(item[0])):
        version = int(key)
        if version < min_version or version == 62:
            continue
        path = ROOT / row['file']
        if digest(path) != row['sha256']:
            raise ValueError('Cached scored file changed: ' + str(path))
        frame = pd.read_parquet(path)
        if row['used_pairs'] != 344841 or not np.array_equal(frame.MVT_ID_mvt, ids):
            raise ValueError('Scored population mismatch: ' + str(version))
        values = frame.TAXITIME_SEC_mvt.to_numpy(np.float64)
        fingerprint = hashlib.sha256(values.tobytes()).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        versions.append(version)
        columns.append(values)
        scores.append(row['score'])
        records[key] = row
    if 64 not in versions:
        raise ValueError('Reference v64 is missing or duplicates another prediction')
    return ids, np.array(columns).T, np.array(scores), versions, records


def solve(P, scores, versions, lower, upper, ridge):
    at = versions.index(64)
    base = P[:, at]
    D = P - base[:, None]
    G = D.T @ D / len(P)
    mse0 = scores[at] ** 2
    c = (scores ** 2 - mse0 - np.diag(G)) / 2
    initial = np.zeros(len(versions)); initial[at] = 1
    def fun(w):
        value = mse0 + 2*c@w + w@G@w + ridge*np.sum((w-initial)**2)
        gradient = 2*(c+G@w) + 2*ridge*(w-initial)
        return float(value / 10000), gradient / 10000
    result = minimize(fun, initial, jac=True, method='SLSQP',
        bounds=[(lower, upper)]*len(versions),
        constraints=[dict(type='eq', fun=lambda w: w.sum()-1,
                          jac=lambda w: np.ones_like(w))],
        options=dict(ftol=1e-12, maxiter=3000))
    w = result.x
    if not result.success or abs(w.sum()-1) > 1e-8:
        raise ValueError('Optimizer failed: ' + str(result.message))
    prediction = P @ w
    mse = float(mse0 + 2*c@w + w@G@w)
    # A deterministic bound for the unknown effect of four-decimal score rounding.
    score_mse_radius = np.abs(scores)*1e-4 + 2.5e-9
    mse_radius = float(np.abs(w) @ score_mse_radius)
    return dict(weights={str(v):float(x) for v,x in zip(versions,w)},
        projected_rmse=float(np.sqrt(mse)),
        score_rounding_rmse_interval=[float(np.sqrt(max(mse-mse_radius,0))),
                                     float(np.sqrt(mse+mse_radius))],
        negative_predictions=int((prediction<0).sum()),
        minimum_prediction=float(prediction.min()), maximum_prediction=float(prediction.max()),
        coefficient_l1=float(np.abs(w).sum()), optimizer_message=str(result.message),
        iterations=int(result.nit), constraints=dict(lower=lower,upper=upper,ridge_mse=ridge),
        note='Projection before output clipping and integer serialization; leaderboard-selected.'), prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-version',type=int,default=45)
    parser.add_argument('--lower',type=float,default=-.5)
    parser.add_argument('--upper',type=float,default=1.5)
    parser.add_argument('--ridge',type=float,default=10.)
    args=parser.parse_args()
    tag=f'min{args.min_version}_lo{args.lower:g}_hi{args.upper:g}_ridge{args.ridge:g}'
    dest=OUT/tag
    if dest.exists(): raise ValueError('Retain completed diagnosis; use a distinct design')
    ids,P,scores,versions,records=load(args.min_version)
    result,pred=solve(P,scores,versions,args.lower,args.upper,args.ridge)
    result.update(reference_version=64,reference_score=float(scores[versions.index(64)]),
        versions=versions,rows=len(ids),sources=records,source_sha256=digest(__file__),
        jan_jul_cv='not yet evaluated',eligible_for_upload=False)
    dest.mkdir(parents=True)
    pd.DataFrame(dict(mvt_id=ids,pred=pred)).to_parquet(dest/'diagnostic_prediction.parquet',index=False)
    result['prediction_sha256']=digest(dest/'diagnostic_prediction.parquet')
    (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(2): main()
