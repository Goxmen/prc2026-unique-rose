"""Cache the selected three-TabM direct mixture from retained Jan/Jul predictions.

Usage: .venv/bin/python VALIDATION/portable_direct_cv_20261013.py freeze|build|audit
Freeze reads metadata/hashes only. Build requires a root allocation binding the
identity. No fit, model inference, coefficient solve, calibration or upload.
The original Scorer.final method receives its original held frame/parameters;
unused historical neural/tree pools are not loaded. Exact selected-v77/v78 and
three-TabM-v78 replay gates authenticate the resulting direct packet.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nn_public_tabm_pool3_stack as original
from records import sha256, write_json

ROOT = original.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261013_portable_direct_cv'
MANIFEST = OUT / 'identity.json'
current = original.replay_stack.current
neural = current.joint.neural
replay = neural.cat.r20.physical.replay
hpo = replay.baseline.hpo
DESIGN = dict(
    schema='portable_direct_cv_v1', folds=['2025-01', '2025-07'], rows=344419,
    neural='0.5*(retained calls15 + equal mean TabM42/1042/2042)',
    tree='equal mean selected main184, airport174, full174Cat',
    ordinary='0.71*neural + 0.29*tree',
    probability='0.5*selected public Rome Cat3 + 0.5*original R20',
    correction='Original Scorer.final with original held V/raw/parameters/corrections',
    control='Original cached physical/callsign direct control; no new base substituted',
    acceptance='Exact selected v77 before/shift/output, selected v78 and three-TabM v78 replay',
    no_fit=True, no_model_inference=True, no_solve=True, no_upload=True,
    caveat='Reused Jan/Jul development and original source/machine provenance; not independent holdout')


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def verify(record):
    path = ROOT / record['path']
    assert art(path) == record, record['path']
    return path


def freeze():
    assert not MANIFEST.exists(), 'Preserve frozen packet design'
    paths = {Path(__file__), Path(original.__file__)}
    for folder in [original.OUT, original.joint.OUT, current.OUT, current.quality.OUT]:
        result = read(folder / 'result.json')
        audit = read(folder / 'audit.json')
        assert audit['status'] == 'pass' and audit['result'] == art(folder / 'result.json')
        paths.update(folder / name for name in ['result.json', 'audit.json', 'scored.parquet'])
    prior_result = read(original.OUT / 'result.json')
    assert prior_result['scored'] == art(original.OUT / 'scored.parquet')
    paths.update(verify(record) for record in prior_result['sources'])
    common = dict(
        calls=replay.combo.sequence.CALLS_PRED,
        prior=neural.cat.OUT / 'scored.parquet',
        serialized=current.rounded.OUT / 'scored.parquet',
        selected77=current.OUT / 'scored.parquet',
        selected78=original.joint.OUT / 'scored.parquet',
        selected_tabm78=original.OUT / 'scored.parquet',
        quality=current.quality.OUT / 'scored.parquet',
        raw=current.daily.base.OUT / 'rows2025.parquet',
        daily_rows=current.daily.CACHE,
        daily_source=current.daily.SOURCE / 'daily_runway.parquet',
        daily_params=current.daily.OUT / 'parameters.json',
        matches=current.daily.constraints.exclusion.OUT / 'matches2025.parquet')
    paths.update(common.values())
    winner = hpo.winner('rome')
    rome_path = hpo.pred_path('rome', winner['trial'], 42)
    paths.add(rome_path)
    specs = {}
    for month in DESIGN['folds']:
        split = hpo.ts.split_dir(hpo.SPLITS_FROM, month)
        paths.update([split / 'heldout.pkl', split / 'meta.json'])
        ple, tabm = [], []
        for k in range(3):
            ple_receipt = neural.paths(month, k)[2]
            tabm_receipt = original.candidate.paths(month, k)[2]
            for receipt, target in [(ple_receipt, ple), (tabm_receipt, tabm)]:
                record = read(receipt)
                prediction = record['prediction']
                paths.update([receipt, verify(prediction)])
                target.append(dict(receipt=art(receipt), prediction=prediction))
        trees = [current.main.OUT / 'predictions' / f'{month}_s42.parquet',
                 current.rounded.original.airport.OUT / 'predictions' / f'{month}_s42.parquet',
                 current.cv.OUT / 'predictions' / f'{month}_s42.parquet']
        paths.update(trees)
        specs[month] = dict(held=art(split / 'heldout.pkl'), meta=art(split / 'meta.json'),
                           ple=ple, tabm=tabm, trees=[art(p) for p in trees])
    # Pin all imported local correction helpers, while avoiding training caches.
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT / 'VALIDATION'):
                paths.add(path)
    identity = dict(design=DESIGN, source=art(__file__), sources=[art(p) for p in sorted(paths)],
                    common={k:art(p) for k,p in common.items()}, specs=specs,
                    rome=art(rome_path), rome_checkpoint=winner['checkpoint'],
                    original_result=art(original.OUT / 'result.json'))
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST, identity)
    print(json.dumps(dict(identity=art(MANIFEST), sources=len(paths), no_numeric_execution=True)))


def guard():
    identity = read(MANIFEST)
    assert identity['design'] == DESIGN and identity['source'] == art(__file__)
    for record in identity['sources']:
        verify(record)
    return identity


def indexed(record):
    frame = pd.read_parquet(verify(record)).set_index('mvt_id')
    assert frame.index.is_unique
    return frame


def build():
    identity = guard()
    allocation = read(OUT / 'allocation.json')
    assert allocation['authority'] == '/root' and allocation['identity'] == art(MANIFEST)
    assert allocation['state'] == 'released_for_cached_direct_cv'
    assert not (OUT / 'packet.parquet').exists() and not (OUT / 'result.json').exists()
    common = identity['common']
    frames = {k:indexed(common[k]) for k in ['prior','serialized','selected77','selected78','selected_tabm78','quality','raw']}
    inputs = current.daily.rows().drop(columns='y').set_index('mvt_id')
    source, parameters = current.daily.public(), read(verify(common['daily_params']))
    matches = pd.read_parquet(verify(common['matches']))
    calls_frame = indexed(common['calls'])
    rome_frame = indexed(identity['rome'])
    output = []
    for month in DESIGN['folds']:
        spec = identity['specs'][month]
        held, meta = pd.read_pickle(verify(spec['held'])), read(verify(spec['meta']))
        raw = held['raw']; ids = raw.mvt_id.to_numpy()
        assert len(ids) == len(set(ids)) and raw.taxi_sec.notna().all()
        # Same method and input objects as Scorer.__init__; omit unused pools.
        sc = hpo.Scorer.__new__(hpo.Scorer)
        sc.d = {month:dict(held=held, meta=meta)}
        day = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime('%Y-%m-%d').to_numpy()
        calls = calls_frame.loc[ids, 'candidate_nn'].to_numpy(float)
        pool = {family:np.mean([indexed(member['prediction']).loc[ids, 'base'].to_numpy(float)
                              for member in spec[family]], axis=0) for family in ['ple','tabm']}
        tree = np.mean([indexed(record).loc[ids, 'pred'].to_numpy(float) for record in spec['trees']], axis=0)
        probability = .5 * frames['prior'].loc[ids, 'newcat'].to_numpy() + .5 * rome_frame.loc[ids, identity['rome_checkpoint']].to_numpy(float)
        control = frames['prior'].loc[ids, 'control'].to_numpy(float)
        permitted = matches[matches.donor_month.isin(current.daily.base.FOLDS[month] + [month])]
        values = {}
        for name in ['ple','tabm']:
            ordinary = .71 * .5 * (calls + pool[name]) + .29 * tree
            direct = sc.final(month, ordinary, probability)
            start = frames['serialized'].loc[ids, 'candidate_boundary'].to_numpy() + direct - control
            before = current.daily.constraints.matched_bound(frames['raw'].loc[ids].reset_index(), start)
            before, _ = current.daily.constraints.exclusion.apply(ids, np.maximum(before, 0), permitted)
            before = np.rint(np.maximum(before, 0))
            updated, _, shift = current.daily.apply(inputs.loc[ids].reset_index(), before, source, parameters[month])
            gated = np.where(frames['quality'].loc[ids, 'quality_eligible'], updated, before)
            p77 = current.daily.constrain(frames['raw'].loc[ids].reset_index(), ids,
                pd.Series(month, index=range(len(ids))), gated, matches)
            p78 = current.daily.constrain(frames['raw'].loc[ids].reset_index(), ids,
                pd.Series(month, index=range(len(ids))), p77 + .5 * frames['selected78'].loc[ids, 'increment'].to_numpy(), matches)
            if name == 'ple':
                for value, column in [(before,'combined_before'), (shift,'combined_shift'), (p77,'combined')]:
                    np.testing.assert_array_equal(value, frames['selected77'].loc[ids, column])
                np.testing.assert_array_equal(p77, frames['selected78'].loc[ids, 'control'])
                np.testing.assert_array_equal(p78, frames['selected78'].loc[ids, 'candidate'])
            else:
                np.testing.assert_array_equal(p78, frames['selected_tabm78'].loc[ids, 'three_tabm'])
            values.update({name+'_direct':direct, name+'_v77':p77, name+'_v78':p78})
        output.append(pd.DataFrame(dict(mvt_id=ids, month=month, day=day, y=raw.taxi_sec.to_numpy(float),
            control=control, calls=calls, tree=tree, probability=probability,
            ple_pool=pool['ple'], tabm_pool=pool['tabm'], **values)))
        del held, meta, raw, sc
        gc.collect()
    packet = pd.concat(output, ignore_index=True)
    assert len(packet) == packet.mvt_id.nunique() == DESIGN['rows']
    assert np.isfinite(packet.select_dtypes(include='number')).all().all()
    packet.to_parquet(OUT / 'packet.parquet', index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(OUT / 'packet.parquet'), packet, check_exact=True)
    write_json(OUT / 'result.json', dict(status='complete', identity=art(MANIFEST),
        allocation=art(OUT / 'allocation.json'), prediction=art(OUT / 'packet.parquet'), rows=len(packet),
        exact_selected_v77_all_stages=True, exact_selected_v78=True, exact_three_tabm_v78=True,
        no_fit=True, no_model_inference=True, no_solve=True, no_upload=True))
    print(json.dumps(dict(rows=len(packet), exact_replays=True, prediction=art(OUT / 'packet.parquet'))))


def audit():
    identity = guard(); result = read(OUT / 'result.json')
    assert result['identity'] == art(MANIFEST)
    verify(result['prediction'])
    assert not (OUT / 'audit.json').exists()
    common = identity['common']
    with duckdb.connect(config={'threads':1}) as con:
        for name, record in [('packet',result['prediction']), *[(k,common[k]) for k in
                              ['prior','selected77','selected78','selected_tabm78','raw']]]:
            con.read_parquet(str(verify(record))).create_view(name)
        n, unique, errors = con.execute('''SELECT count(*),count(DISTINCT p.mvt_id),sum(CASE WHEN
            p.y IS DISTINCT FROM r.y OR p.month IS DISTINCT FROM r.month OR p.day IS DISTINCT FROM r.day
            OR p.control IS DISTINCT FROM b.control OR p.ple_v77 IS DISTINCT FROM s.combined
            OR p.ple_v78 IS DISTINCT FROM j.candidate OR p.tabm_v78 IS DISTINCT FROM t.three_tabm
            THEN 1 ELSE 0 END) FROM packet p JOIN raw r USING(mvt_id) JOIN prior b USING(mvt_id)
            JOIN selected77 s USING(mvt_id) JOIN selected78 j USING(mvt_id) JOIN selected_tabm78 t USING(mvt_id)''').fetchone()
        assert n == unique == DESIGN['rows'] and errors == 0
    write_json(OUT / 'audit.json', dict(status='pass', identity=art(MANIFEST), result=art(OUT/'result.json'),
        source=art(__file__), rows=n, all_SQL_truth_calendar_original_control_selected_replays_exact=True,
        caveat='Direct postprocessor reused from pinned original code; gate replays original selected outputs, not an independent postprocessor derivation.',
        no_fit=True, no_model_inference=True, no_solve=True, no_upload=True))
    print(json.dumps(dict(status='pass', rows=n)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','build','audit'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
