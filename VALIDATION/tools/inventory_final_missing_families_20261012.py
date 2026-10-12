"""Compact next-family inventory from documented source paths and saved vectors.

No new model search, fits, source downloads, inference or candidate selection.
Family contribution norms describe missing portability directions, not gains.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/reports/final_missing_family_plan_20261012.json'
FAMILIES=[
 dict(name='raw_early',versions=list(range(2,10)),anchor=1,
      original_recipe='v2: clipped1..7200,3000 rounds. v3-v9: original27 inputs,800 rounds; v4 adds FPL flag and v5-v9 vary label ceiling.',
      missing_fits='Eight ordinary LightGBMs, with original seed/config; original v3-v9 full source hash is unavailable.',
      known_cv_runner='VALIDATION/legacy_raw_jan_jul.py',
      runtime='Original800-round fits approximately1 minute each after shared preparation; v2 several minutes. Estimate15-30minutes including replay.',
      priority=4,notes=['Use one source/config attempt per recipe; source approximation remains explicit until exact old prediction acceptance.',
                        'Original v4-v9 use8 threads, so coordinate host/thread allocation before execution.']),
 dict(name='stand_correction_family',versions=[30,33,34,37],anchor=38,
      original_recipe='v30 ordinary+classifier; v33 plan-window, v34 no-ID-cap, v37 dayshift-hedge are deterministic descendants.',
      missing_fits='One800-round ordinary and one300-round classifier; reuse this pair for all four versions.',
      known_cv_runner='VALIDATION/legacy_hybrid_jan_jul.py',
      runtime='Original ordinary+classifier about1-2minutes; allow10-20minutes including feature/calibration reconstruction and all-view replay.',
      priority=2,notes=['Retained v38 is not an exact substitute: its historical control differed on304637 old rows.',
                        'Replay v30 first before admitting any descendant. Reuse frozen correction parameters when available.']),
 dict(name='early_hybrids',versions=[26,27,28,29],anchor=38,
      original_recipe='v26 clock, v27 weather, v28 queue; they share one identical classifier hash. v29 changes ordinary and classifier inputs.',
      missing_fits='Four800-round ordinary models and two distinct small classifiers.',
      known_cv_runner='VALIDATION/legacy_hybrid_jan_jul.py',
      runtime='Recorded each pair1.22-1.40minutes; allow15-25minutes including shared feature preparation/replay.',
      priority=5,notes=['Do not fit three duplicate v26-v28 classifiers: original hash56c3a57e83b22446ef4cf580f43c2624bdee233cb9afaed8784f8b7b48d06ed4 is shared.']),
 dict(name='stand_coordinate_tuning',versions=[43,44],anchor=45,
      original_recipe='v43 original800-round ordinary model; v44 tuned5000-round ordinary model. Identical saved classifier is present under v46.',
      missing_fits='Two ordinary LightGBMs only; reuse exact classifier9501e34314f919425cc74cc345eef9562e895f1516002847fba1583afa97d31e.',
      known_cv_runner='VALIDATION/tuned_submit.py',
      runtime='Recorded ordinary fits97.4s and435.9s; allow15-25minutes including matched input preparation/replay.',
      priority=6,notes=['No repeated classifier fit is necessary.']),
 dict(name='single_plain_network',versions=[47],anchor=46,
      original_recipe='One original MLP in0.3neural/0.7tree mixture, using retained v46 trees and corrections.',
      missing_fits='One full-year neural model; original weights were not saved.',
      known_cv_runner='VALIDATION/nn_submit.py',
      runtime='Requires one GPU fit; record exact original stopping/schedule from retained manifest before allocation.',
      priority=8,notes=['Original recipe explicitly notes GPU nondeterminism; old-vector equality is acceptance, otherwise retain quantified approximation.']),
 dict(name='original_ple_pool',versions=[48,49,50],anchor=54,
      original_recipe='v48/v49 share five PLE models; v50 extends same134-input architecture to15 networks.',
      missing_fits='Fifteen full-year PLE fits total, reusing appropriate five for v48/v49; no repeat epoch selection.',
      known_cv_runner='VALIDATION/ideas_submit.py',
      runtime='GPU workload likely larger than tree families; derive epoch counts from saved group receipts, then benchmark first member once.',
      priority=3,notes=['All original weights absent in documented model directories; old prediction parts and stopping receipts remain.',
                        'First-five identity/order must be checked against v48 before assuming shared members.']),
 dict(name='combo_ple_pool',versions=[51,53],anchor=54,
      original_recipe='Shared15 PLE networks with22added inputs and original combo tree pool; v53 changes Rome classifier.',
      missing_fits='Fifteen original-combo neural fits; tree models are retained. v53 needs existing R20 final probabilities, with its one-flight old hold handled explicitly.',
      known_cv_runner='VALIDATION/ideas_submit.py',
      runtime='GPU workload; reuse saved stopping decisions and retained12 tree checkpoints, not refit tree controls.',
      priority=9,notes=['HPO v54 networks are not an exact substitute. Already-refitted October8 base networks only support v54 lineage.',
                        'v53 held original ITA flight202681908; define final-view semantics explicitly from the recorded rule.']),
 dict(name='flight_number_rome',versions=[56],anchor=54,
      original_recipe='v54 with1700-round R20+two flight-number copy inputs, retaining v54 probabilities for unseen numbers.',
      missing_fits='One Rome classifier; base/tree/network components already final-ready.',
      known_cv_runner='VALIDATION/rome_fn_submit.py',
      runtime='Small classifier fit; allow5-15minutes for feature/history reconstruction and replay.',
      priority=1,notes=['Original script did not save the new classifier. No need repeat its original R20 control fit: saved final R20 components already reproduce the parent.',
                        'Keep unseen-number fallback exactly; the no-fallback alternative was rejected.']),
 dict(name='weather_saved_models',versions=[60,61],anchor=58,
      original_recipe='v60 changes Rome weather classifier; v61 weather in allfive parts.',
      missing_fits='None: Rome/main/airport/CatBoost and all15 neural checkpoints are retained.',
      known_cv_runner='VALIDATION/open_meteo_submit.py',
      runtime='Inference only;15small network evaluations plus retained trees. Need extra20airport-month ERA5 cache files for February/June, then matched old replay.',
      priority=0,notes=['Existing hourly weather cache covers2025,January2026 andJuly2026 (June29onward only from July request). No repeated downloads of existing files.',
                        'New weather records must be hashed and independently joined; preserve old source namespace and all current weather controls.'])]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    diagnostic=ROOT/'VALIDATION/runs/v3_20261012_final_affine_recovery_diagnostic/result.json'
    d=json.loads(diagnostic.read_text());weights={int(k):v for k,v in d['effective_leaf_weights'].items()}
    basis=ROOT/'VALIDATION/runs/v3_20261012_target205_geometry/ranking_basis.parquet'
    cols=sorted({v for f in FAMILIES for v in f['versions']+[f['anchor']]})
    f=pd.read_parquet(basis,columns=['mvt_id',*[f'v{v}' for v in cols]])
    for family in FAMILIES:
        delta=sum(weights[v]*(f[f'v{v}'].to_numpy(float)-f[f'v{family["anchor"]}'].to_numpy(float)) for v in family['versions'])
        family['effective_weights']={str(v):weights[v] for v in family['versions']}
        family['weighted_residual_vs_portable_anchor']=dict(rms=float(np.sqrt(np.mean(delta**2))),
            mean=float(delta.mean()),absolute_q99=float(np.quantile(abs(delta),.99)),max_abs=float(max(abs(delta))))
    directories=['v3_20261003_ideas_submit/work/models','v3_20261008_open_meteo_submit/work/models',
        'v3_20261006_rome_fn_submit/submission_v56','v3_20261003_nn_improve_submit/submission_v48/models',
        'v3_20261002_nn_submit/submission_v47/models']
    retained={}
    for relative in directories:
        directory=ROOT/'VALIDATION/runs'/relative
        retained[relative]=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size)
            for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ['.pt','.cbm','.txt'] and 'source' not in p.relative_to(directory).parts]
    classifier=ROOT/'VALIDATION/runs/v3_20261002_adsblol_submit/submission_v46/models/refit2025__classifier__42.txt'
    assert sha(classifier)=='9501e34314f919425cc74cc345eef9562e895f1516002847fba1583afa97d31e'
    weather={}
    for p in sorted((ROOT/'data/external/conditions_20261008/weather').glob('EHAM*')):
        hours=json.loads(p.read_text())['hourly']['time'];weather[p.name]=dict(first=hours[0],last=hours[-1],rows=len(hours))
    out=dict(source_sha256=sha(__file__),diagnostic_sha256=sha(diagnostic),ranking_basis_sha256=sha(basis),
        purpose='Next missing-family recovery allocation; descriptive residual norms are not expected score gains',
        families=sorted(FAMILIES,key=lambda x:x['priority']),documented_model_directories=retained,
        retained_shared_classifier=dict(path=str(classifier.relative_to(ROOT)),sha256=sha(classifier)),
        weather_cache_example=weather,no_fit_started=True,no_new_download=True,
        acceptance='Every admitted exact recovery must replay all original old integers; nonexact cases remain explicitly approximated, with no parameter sweep.')
    OUT.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({x['name']:round(x['weighted_residual_vs_portable_anchor']['rms'],4) for x in out['families']}))


if __name__=='__main__':
    with threadpool_limits(1):main()
