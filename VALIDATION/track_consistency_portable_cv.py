"""Fixed secondary v82 portability gate for the target-track consistency screen.

Only retained predictions and deterministic corrections are evaluated. No model
inference, posterior inference, calibration, fit, weight search or upload.
The primary exact-v80 gate remains mandatory; its failure closes this stage
before any new prediction parquet is decoded.
"""
import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nn_public_tabm_track_consistency as candidate
import nn_public_tabm_track_consistency_score as primary
import portable_direct_cv_20261013 as direct
import portable_public_source_20261013 as portable
import tuning_seasonal as seasonal

ROOT = candidate.ROOT
OUT = candidate.OUT / 'portable_comparison'
MANIFEST = OUT / 'identity.json'
PLAN = ROOT / 'VALIDATION/reports/cohort_learning_20261013/target_track_consistency_production_design.json'
AUDITOR = ROOT / 'VALIDATION/tools/audit_track_consistency_portable_cv_20261013.py'
REVIEW = OUT / 'source_review.json'
ARMS = ['supervised_control', 'training_consistency', 'target_consistency']
MONTHS = ['2025-01', '2025-07']
ROWS = 344419
art, read, verify, write = candidate.art, candidate.read, candidate.verify, candidate.write
R = portable.R
DESIGN = dict(
    schema='track_consistency_portable_cv_v1', selected_arm='target_consistency',
    arms=ARMS, months=MONTHS, rows=ROWS,
    prefix='Retain audited v82 B/raw_posterior/T/D/half_monthly/full_monthly/M exactly.',
    direct='Replace seed42 only in original equal three-TabM pool; original Scorer.final and fixed calls/tree/probability.',
    suffix='Original serialized N, quality-gated daily correction Q, half fixed joint J, half fixed robust F.',
    gate='Primary pass AND target versus v82 mean<0, both months<0 and paired-day95%upper<0.',
    minimum_additional_gain=None, bootstrap_reps=2000, bootstrap_seed=0,
    primary_fail='Write closed receipt without decoding any new prediction vector.',
    comparisons_descriptive=['supervised_control', 'training_consistency'],
    no_control_promotion=True, no_model_inference=True, no_posterior_inference=True,
    no_fit=True, no_calibration=True, no_blend_search=True, no_upload=True,
    interpretation='Reused transductive Jan/Jul development, not independent confirmation.')


def freeze():
    assert not MANIFEST.exists()
    assert not (primary.OUT / 'result.json').exists(), 'Freeze before any new primary score'
    assert not (OUT / 'result.json').exists() and not (OUT / 'closed.json').exists()
    review = read(REVIEW)
    assert review['status'] == 'pass' and review['source'] == art(__file__)
    assert review['auditor'] == art(AUDITOR) and review['design'] == art(PLAN)
    plan = read(PLAN)
    assert plan['v82_portable_route']['secondary_gate']['fixed_before_scores']
    assert candidate.ARMS == primary.ARMS == ARMS
    dp = read(direct.MANIFEST)
    pp = read(portable.MANIFEST)
    old_result, old_audit = read(portable.OUT/'cv/result.json'), read(portable.OUT/'cv/audit.json')
    assert old_audit['status'] == 'pass' and old_audit['result'] == art(portable.OUT/'cv/result.json')
    assert old_result['identity'] == old_audit['identity'] == art(portable.MANIFEST)
    assert old_result['prediction'] == old_audit['prediction']
    paths = {Path(__file__), PLAN, AUDITOR, REVIEW, candidate.MANIFEST,
             primary.OUT/'identity.json', direct.MANIFEST, portable.MANIFEST,
             direct.OUT/'result.json', direct.OUT/'audit.json',
             portable.OUT/'cv/result.json', portable.OUT/'cv/audit.json',
             verify(old_result['prediction']), portable.daily.CACHE,
             ROOT/'VALIDATION/tools/audit_portable_public_source_20261013.py',
             portable.daily.SOURCE/'daily_runway.parquet', primary.AUDITOR}
    paths.update(verify(record) for record in dp['sources'])
    paths.update(verify(record) for record in pp['specs']['cv'].values())
    paths.update(verify(record) for record in read(primary.OUT/'identity.json')['sources'])
    for stem in ['all_pre_departure_delays', 'taxi_out_additional_time']:
        paths.add(portable.daily.base.SOURCE / f'{stem}_2025.csv')
    paths.update(ROOT/f'data/training_{month}-01_2025-{2 if month.endswith("01") else 8:02d}-01.parquet'
                 for month in MONTHS)
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT/'VALIDATION'):
                paths.add(path)
    write(MANIFEST, dict(design=DESIGN, source=art(__file__), auditor=art(AUDITOR),
        source_review=art(REVIEW), plan=art(PLAN), sources=[art(p) for p in sorted(paths)],
        before_any_new_score=True, before_new_prediction_decode=True,
        training_already_in_progress=True, parent=art(portable.MANIFEST),
        baseline_result=art(portable.OUT/'cv/result.json'),
        baseline_audit=art(portable.OUT/'cv/audit.json')))
    print(dict(identity=art(MANIFEST), sources=len(paths), no_numeric_evaluation=True), flush=True)


def guard():
    m = read(MANIFEST)
    assert m['design'] == DESIGN and m['source'] == art(__file__)
    assert m['auditor'] == art(AUDITOR) and m['plan'] == art(PLAN)
    for record in m['sources']:
        verify(record)
    return m


def alignment(frame, ids):
    assert frame.mvt_id.is_unique and set(frame.mvt_id) == set(ids)
    return frame.set_index('mvt_id').loc[ids].reset_index()


def collect_fit_receipts():
    """Authenticate all six immutable exports before opening any prediction."""
    trained = read(candidate.OUT/'audit.json')
    assert trained['status'] == 'pass' and trained['identity'] == art(candidate.MANIFEST)
    expected = [art(candidate.paths(month, arm)[2]) for month in MONTHS for arm in ARMS]
    assert len(trained['members']) == 6 and sorted(trained['members'],key=lambda r:r['path']) == sorted(expected,key=lambda r:r['path'])
    identity = read(candidate.MANIFEST)
    completed = {month:{} for month in MONTHS}
    for month in MONTHS:
        for arm in ARMS:
            record = candidate.completed(month, arm, identity)
            assert record is not None
            completed[month][arm] = record
    return completed, expected


def new_direct(packet, completed, identity):
    """Original direct operator on original held objects, one replaced member."""
    chunks = []
    for month in MONTHS:
        spec = identity['specs'][month]
        held = pd.read_pickle(verify(spec['held']))
        meta = read(verify(spec['meta']))
        ids = held['raw'].mvt_id.to_numpy()
        old = alignment(packet[packet.month.eq(month)], ids)
        np.testing.assert_array_equal(held['raw'].taxi_sec, old.y)
        sc = direct.hpo.Scorer.__new__(direct.hpo.Scorer)
        sc.d = {month:dict(held=held, meta=meta)}
        members = [direct.indexed(r['prediction']).loc[ids, 'base'].to_numpy(float) for r in spec['tabm']]
        zero_pool = np.mean(members, axis=0)
        np.testing.assert_array_equal(zero_pool, old.tabm_pool)
        result = pd.DataFrame(dict(mvt_id=ids))
        for arm in ['control', *ARMS]:
            if arm == 'control':
                pool = zero_pool
            else:
                replacement = pd.read_parquet(verify(completed[month][arm]['prediction']))
                replacement = alignment(replacement, ids)
                assert list(replacement) == ['mvt_id', 'base']
                pool = np.mean([replacement.base.to_numpy(float), *members[1:]], axis=0)
            ordinary = .71*.5*(old.calls.to_numpy(float)+pool)+.29*old.tree.to_numpy(float)
            value = sc.final(month, ordinary, old.probability.to_numpy(float))
            assert np.isfinite(value).all()
            if arm == 'control':
                np.testing.assert_array_equal(value, old.tabm_direct)
            result[arm+'_pool'] = pool
            result[arm+'_direct'] = value
        chunks.append(result)
        del held, meta, sc, old, members
        gc.collect()
    return pd.concat(chunks, ignore_index=True)


def suffix(ids, raw, qinputs, matches, increments, retained, prediction, monthly_source, taxi_source, parameters):
    nraw = retained.M.to_numpy(float)+prediction-retained.direct_control.to_numpy(float)
    n = portable.constrain('cv', raw, ids, nraw, matches)
    quality = qinputs[['airport','local_day']].assign(local_month=qinputs.local_day.str[:7]).merge(
        monthly_source[['airport','local_month','invalid']],on=['airport','local_month'],how='left',validate='many_to_one')
    gate = quality.invalid.lt(.05).to_numpy()
    shift = np.full(len(ids), np.nan)
    cells = []
    for month, at in raw.groupby('month', sort=True).indices.items():
        _, cell, shift[at] = portable.daily.apply(qinputs.iloc[at].reset_index(drop=True), n[at].astype(float),taxi_source,parameters[month])
        cells.append(cell.assign(fold=month))
    applied = np.where(gate, shift, 0.)
    q = portable.constrain('cv',raw,ids,n.astype(float)+applied,matches)
    j = portable.constrain('cv',raw,ids,q.astype(float)+.5*increments.increment.to_numpy(float),matches)
    f = portable.constrain('cv',raw,ids,j.astype(float)+.5*increments.robust_delta.to_numpy(float),matches)
    np.testing.assert_array_equal(f[~increments.robust_active],j[~increments.robust_active])
    for value in [prediction,nraw,n,shift,applied,q,j,f]:
        assert np.isfinite(value).all()
    for value in [n,q,j,f]:
        np.testing.assert_array_equal(value,R(value))
    return dict(N_raw=nraw,N=n,quality_eligible=gate,invalid_share=quality.invalid.to_numpy(),
                daily_shift=shift,applied_shift=applied,Q=q,J=j,submitted=f),pd.concat(cells,ignore_index=True)


def score():
    m = guard()
    assert not (OUT/'result.json').exists() and not (OUT/'closed.json').exists()
    assert not (OUT/'started.json').exists(), 'Partial numeric attempt; inspect, do not silently repeat'
    allocation = read(OUT/'allocation.json')
    assert allocation['authority'] == '/root' and allocation['identity'] == art(MANIFEST)
    assert allocation['state'] == 'released_for_fixed_portable_consistency_cv'
    completed, members = collect_fit_receipts()
    result, audit = read(primary.OUT/'result.json'), read(primary.OUT/'independent_audit.json')
    assert audit['status'] == 'pass' and audit['result'] == art(primary.OUT/'result.json')
    assert audit['identity'] == result['identity'] == art(primary.OUT/'identity.json')
    assert audit['gate'] == result['gate']
    prerequisites = dict(identity=art(MANIFEST), allocation=art(OUT/'allocation.json'),
        fit_audit=art(candidate.OUT/'audit.json'), members=members,
        primary_result=art(primary.OUT/'result.json'), primary_audit=art(primary.OUT/'independent_audit.json'))
    if not result['gate']['passed']:
        write(OUT/'closed.json',dict(**prerequisites,status='closed_primary_gate_failed',
            gate=False,new_prediction_vectors_decoded=False,no_alternative_arm=True))
        print(dict(closed='primary gate failed',no_new_prediction_decode=True),flush=True)
        return
    write(OUT/'started.json', prerequisites)
    parent_identity = read(portable.MANIFEST)
    base,raw,unused,packet,qinputs,matches,increments = portable.load('cv',parent_identity)
    del unused
    ids = base.mvt_id.to_numpy()
    retained = alignment(pd.read_parquet(verify(read(portable.OUT/'cv/result.json')['prediction'])),ids)
    assert len(ids) == len(set(ids)) == ROWS and set(raw.month) == set(MONTHS)
    for column in ['month','day','y']:
        np.testing.assert_array_equal(retained[column],base[column])
    np.testing.assert_array_equal(retained.direct_control,packet.control)
    np.testing.assert_array_equal(retained.tabm_direct,packet.tabm_direct)
    np.testing.assert_array_equal(retained.joint_increment,increments.increment)
    np.testing.assert_array_equal(retained.robust_delta,increments.robust_delta)
    np.testing.assert_array_equal(retained.robust_active,increments.robust_active)
    values = alignment(new_direct(packet,completed,read(direct.MANIFEST)),ids)
    monthly_source,taxi_source = portable.monthly.public(2025),portable.daily.public()
    parameters = read(verify(parent_identity['specs']['cv']['quality_parameters']))
    frame = retained.copy()
    frame = frame.rename(columns={'submitted':'v82'})
    cells = []
    for arm in ['control',*ARMS]:
        produced,cell = suffix(ids,raw,qinputs,matches,increments,retained,
            values[arm+'_direct'].to_numpy(float),monthly_source,taxi_source,parameters)
        if arm == 'control':
            for column,value in produced.items():
                np.testing.assert_array_equal(value,retained[column],err_msg=column)
        for column,value in produced.items():
            frame[arm+'_'+column] = value
        frame[arm+'_pool'] = values[arm+'_pool'].to_numpy()
        frame[arm+'_direct'] = values[arm+'_direct'].to_numpy()
        cells.append(cell.assign(arm=arm))
    for arm in ARMS:
        frame[arm] = frame[arm+'_submitted']
    scores = {month:{name:float(np.sqrt(np.mean((group[name]-group.y)**2)))
                    for name in ['v82',*ARMS]} for month,group in frame.groupby('month')}
    cv = {name:float(np.mean([value[name] for value in scores.values()])) for name in ['v82',*ARMS]}
    paired = {name:seasonal.paired_bootstrap(frame.day,frame.month,(frame[name]-frame.y)**2,
        (frame.target_consistency-frame.y)**2,reps=2000,seed=0) for name in ['v82',*DESIGN['comparisons_descriptive']]}
    both = all(s['target_consistency'] < s['v82'] for s in scores.values())
    mean = cv['target_consistency'] < cv['v82']
    upper = paired['v82']['ci95'][1] < 0
    gate = dict(primary=True,mean_improves=mean,both_months=both,interval_below_zero=upper,passed=mean and both and upper)
    frame.to_parquet(OUT/'scored.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'daily_cells.parquet',index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(OUT/'scored.parquet'),frame,check_exact=True)
    write(OUT/'result.json',dict(**prerequisites,status='complete',scored=art(OUT/'scored.parquet'),
        daily_cells=art(OUT/'daily_cells.parquet'),scores=scores,cv=cv,paired=paired,gate=gate,
        exact_v82_control_all_stages=True,unchanged_prefix=True,no_new_posterior_evaluation=True,
        independent_audit_pending=True,no_control_promotion=True,no_upload=True))
    print(dict(cv=cv,paired=paired,gate=gate),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','score'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
