"""Rebuild direct public model mixtures for both ranking views, without fitting.

These are inputs to later affine/public-source corrections, not submissions.
Every old direct value and selected component is checked against its retained
production artifact before any corresponding final view is admitted.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import callsign_submit as parent
import callsign_lgb_submit as main
import sparse_callsign_submit as sparse
import nn_track_physical_submit as physical
from records import sha256, write_json

ROOT = parent.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_public_direct_components'
MANIFEST = OUT / 'identity.json'
RUNS = ROOT / 'VALIDATION/runs'


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    p = Path(path)
    return dict(path=str(p.relative_to(ROOT)), sha256=sha256(p))


def verify(record):
    p = ROOT / record['path']
    assert sha256(p) == record['sha256'], p
    return p


def freeze():
    assert not MANIFEST.exists(), 'Preserve frozen direct assembly'
    sources = {Path(__file__)}
    def bind(path):
        sources.add(Path(path))
        return read(path)
    def ref(record):
        p = verify(record)
        sources.add(p)
        return record
    specs = {}
    for view in ['old', 'final']:
        template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
        sources.add(template)
        packet = parent.fs.WORK / f'inputs_{view}.pkl'
        packet_record = bind(parent.fs.OUT / f'inputs_{view}.json')
        assert sha256(packet) == packet_record['sha256']
        sources.add(packet)
        prior = bind(sparse.OUT / f'candidates/{view}/manifest.json')
        spec = dict(template=art(template), inputs=art(packet), reference=ref(prior['parts']),
                    physical=[], ple=[], tabm=[], public_rome=[])
        for k in range(3):
            record = bind(physical.paths(k)[2])
            spec['physical'].append(ref(record['predictions'][view]))
        for family, directory in [('ple', 'final_public_ple_inference'), ('tabm', 'final_public_tabm_inference')]:
            run = RUNS / ('v3_20261012_' + directory)
            identity = bind(run / 'identity.json')
            receipt = bind(run / f'{view}.json')
            assert receipt['identity'] == art(run / 'identity.json')
            assert receipt['all_old_predictions_exact']
            if view == 'final':
                assert receipt['all_final_reloads_exact']
            for k, member in enumerate(identity['members']):
                chosen = member['old_prediction'] if view == 'old' else art(run / f'predictions/final_k{k}.parquet')
                if view == 'final':
                    assert chosen in receipt['predictions']
                spec[family].append(ref(chosen))
        tree = RUNS / 'v3_20261012_final_public_tree_inference'
        tree_identity = bind(tree / 'identity.json')
        for key in ['main', 'airport']:
            receipt = bind(tree / key / f'{view}.json')
            assert receipt['identity'] == art(tree / 'identity.json') and receipt['old_replay_exact']
            if view == 'final':
                assert receipt['final_reload_exact']
            chosen = tree_identity['old_predictions'][key] if view == 'old' else art(tree / key / 'final.parquet')
            if view == 'final':
                assert chosen in receipt['predictions']
            spec['public_' + key] = ref(chosen)
        receipt = bind(tree / 'rome_cat' / f'{view}.json')
        assert receipt['identity'] == art(tree / 'identity.json')
        assert receipt['old_replay_exact'] and (view == 'old' or receipt['final_reload_exact'])
        for seed in [42, 142, 242]:
            chosen = tree_identity['models']['rome_cat'][str(seed)]['prediction'] if view == 'old' else art(tree / 'rome_cat' / f'final_s{seed}.parquet')
            if view == 'final':
                assert chosen in receipt['predictions']
            spec['public_rome'].append(ref(chosen))
        for key, stem, old_path in [
            ('daily_main', 'daily_main', RUNS / 'v3_20261011_daily_main_submit/predictions/old.parquet'),
            ('public_cat', 'public_catboost', RUNS / 'v3_20261011_catboost_full_inputs_submit/predictions/old.parquet')]:
            run = RUNS / ('v3_20261012_final_' + stem + '_inference')
            bind(run / 'identity.json')
            receipt = bind(run / f'{view}.json')
            assert receipt['identity'] == art(run / 'identity.json')
            if key == 'daily_main':
                assert receipt['all_original_predictions_exact'] and receipt['all_model_reload_predictions_exact']
            else:
                assert receipt['old_replay_exact'] and (view == 'old' or receipt['final_reload_exact'])
            if key == 'daily_main':
                chosen = receipt['prediction']
            elif view == 'old':
                chosen = bind(RUNS / 'v3_20261011_catboost_full_inputs_submit/predictions/old.json')['prediction']
            else:
                chosen = art(run / 'final.parquet')
                assert chosen in receipt['predictions']
            spec[key] = ref(chosen)
        original_rome = RUNS / 'v3_20261010_rome_catboost_submit'
        spec['original_rome_cat'] = ref(bind(original_rome / f'{view}.json')['prediction'])
        # These ordinary components are reused directly, never inferred again.
        if view == 'old':
            paths = [parent.hs.OUT / f'submission_v54/ranking_{m}.parquet' for m in parent.features.RANK_MONTHS[view]]
        else:
            paths = [parent.fs.WORK / 'parts/trees.parquet']
        spec['ordinary'] = [ref(art(p)) for p in paths]
        specs[view] = spec
    controls = dict(
        v76=RUNS / 'v3_20261011_public_neural_candidate/submission_v76/predictions.parquet',
        airport=RUNS / 'v3_20261011_public_daily_taxi_submit/predictions.parquet',
        cat=RUNS / 'v3_20261011_catboost_refinements_submit/predictions.parquet',
        main=RUNS / 'v3_20261011_daily_main_refinements_submit/predictions.parquet',
        tabm=RUNS / 'v3_20261012_nn_public_tabm_pool3_submit_stack/predictions.parquet')
    controls = {k:ref(art(p)) for k,p in controls.items()}
    # Pin every imported local helper used by retained correction bytecode.
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            p = Path(filename).resolve()
            if p.is_file() and p.is_relative_to(ROOT / 'VALIDATION'):
                sources.add(p)
    audit = bind(RUNS / 'v3_20261012_final_additional_public_components/audit.json')
    assert audit['status'] == 'pass'
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST, dict(specs=specs, controls=controls, sources=[art(p) for p in sorted(sources)],
        source=art(__file__), views=['old','final'], no_fit=True, no_affine_substitution=True,
        caveat='Retained base-network final components retain their October8 refit provenance.',
        acceptance='All old direct stages exact before final admission', no_submission=True))
    print(dict(frozen=True, sources=len(sources)), flush=True)


def aligned(record, ids):
    f = pd.read_parquet(verify(record)).set_index('mvt_id')
    assert f.index.is_unique and set(f.index) == set(ids)
    return f.loc[ids].reset_index()


def build(view):
    identity = read(MANIFEST)
    assert identity['source'] == art(__file__)
    for record in identity['sources']:
        verify(record)
    assert not (OUT / f'{view}.json').exists(), 'Preserve completed assembly'
    if view == 'final':
        assert read(OUT / 'old.json')['all_original_direct_values_exact']
    spec = identity['specs'][view]
    data = parent.ranking_data(view)
    ids = np.asarray(data['rank_ids'])
    reference = aligned(spec['reference'], ids)
    pools = {name: np.mean([aligned(r,ids).base.to_numpy(float) for r in spec[name]],axis=0)
             for name in ['physical','ple','tabm']}
    original_cat = aligned(spec['original_rome_cat'],ids).p.to_numpy(float)
    r20 = 2 * reference.p.to_numpy(float) - original_cat
    assert r20.min() > -1e-15 and r20.max() < 1+1e-15
    r20 = np.clip(r20,0,1)
    p = .5 * (np.mean([aligned(r,ids).p.to_numpy(float) for r in spec['public_rome']],axis=0) + r20)
    ordinary,_ = main.parent_components(view,ids)
    calls = reference.calls_nn.to_numpy(float)
    neural = {name:.5*(calls+pool) for name,pool in pools.items()}
    values = {name:aligned(spec[name],ids).pred.to_numpy(float)
              for name in ['public_main','public_airport','daily_main','public_cat']}
    trees = {
        'v76':np.column_stack([values['public_main'],ordinary.airport.to_numpy(float),ordinary.catboost.to_numpy(float)]).mean(axis=1),
        'airport':np.column_stack([values['public_main'],values['public_airport'],ordinary.catboost.to_numpy(float)]).mean(axis=1),
        'cat':np.column_stack([values['public_main'],values['public_airport'],values['public_cat']]).mean(axis=1),
        'main':np.column_stack([values['daily_main'],values['public_airport'],values['public_cat']]).mean(axis=1)}
    result = pd.DataFrame(dict(mvt_id=ids,calls_nn=calls,p=p,r20=r20,original_rome_cat=original_cat,
        **values,**{k+'_pool':v for k,v in pools.items()},**{k+'_neural':v for k,v in neural.items()}))
    direct,_ = parent.final_frame(data,neural['physical'],reference.gbm_base.to_numpy(float),reference.p.to_numpy(float))
    result['control'] = direct.pred.to_numpy()
    for name,tree in trees.items():
        direct,_ = parent.final_frame(data,neural['ple'],tree,p)
        result[name+'_tree'] = tree
        result[name+'_direct'] = direct.pred.to_numpy()
    direct,_ = parent.final_frame(data,neural['tabm'],trees['main'],p)
    result['tabm_direct'] = direct.pred.to_numpy()
    checked = 0
    if view == 'old':
        c = {k:aligned(r,ids) for k,r in identity['controls'].items()}
        pairs = [('control','v76','control'),('p','v76','p'),('calls_nn','v76','calls_nn'),
            ('physical_pool','v76','physical_old'),('ple_pool','v76','physical_new'),
            ('physical_neural','v76','neural_old'),('ple_neural','v76','neural_new'),
            ('v76_tree','v76','tree'),('v76_direct','v76','candidate'),
            ('airport_tree','airport','tree_new'),('airport_direct','airport','direct_new'),
            ('cat_tree','cat','candidate_tree'),('cat_direct','cat','candidate_direct'),
            ('main_tree','main','candidate_tree'),('main_direct','main','candidate_direct'),
            ('tabm_direct','tabm','tabm_direct')]
        for out,key,column in pairs:
            np.testing.assert_array_equal(result[out],c[key][column],err_msg=out)
            checked += len(ids)
    target_ids = pd.read_parquet(verify(spec['template']),columns=['MVT_ID_mvt']).MVT_ID_mvt
    result = result.set_index('mvt_id').loc[target_ids].reset_index()
    assert len(result) == result.mvt_id.nunique() == (344841 if view=='old' else 670790)
    assert np.isfinite(result.drop(columns='mvt_id')).all().all()
    result.to_parquet(OUT / f'{view}.parquet',index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(OUT/f'{view}.parquet'),result,check_exact=True)
    write_json(OUT / f'{view}.json',dict(identity=art(MANIFEST),view=view,rows=len(result),
        all_original_direct_values_exact=True,old_values_checked=checked,
        prediction=art(OUT/f'{view}.parquet'),template=spec['template'],
        no_fit=True,no_submission=True,independent_audit_pending=True))
    print(dict(view=view,rows=len(result),old_values_checked=checked),flush=True)
    del data,result
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','old','final'])
    args=parser.parse_args()
    with threadpool_limits(1):
        freeze() if args.stage=='freeze' else build(args.stage)
