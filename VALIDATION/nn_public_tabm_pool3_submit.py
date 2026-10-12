"""Conditional full-year continuation for the two predeclared TabM seeds.

Preparation requires completed, audited improvement over the one-TabM pool.
Seed42 is reused. Each new seed runs unchanged production fit bytecode in an
isolated member directory; k0 in inherited filenames denotes the local slot.
"""
import argparse
import copy
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_public_tabm_pool3 as cv
import nn_public_tabm_submit as first

ROOT = first.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_submit'
STACK = ROOT / 'VALIDATION/runs/v3_20261012_nn_public_tabm_pool3_stack'
TRANSPORT = ROOT / 'VALIDATION/runs/v3_20261012_tabm_pool3_delta_transport/tabm_pool3'
AUDIT = ROOT / 'VALIDATION/reports/public_tabm_20261012/pool3_score_audit.json'
art, read, verify, write = first.art, first.read, first.verify, first.write_json
FUNCTIONS = ['guard', 'preflight', 'trained', 'completed', 'fit']


def member(k):
    assert k in [1, 2]
    run = OUT / f'k{k}'
    seed = cv.SEEDS[k]
    design = {**first.DESIGN, 'choice': f'additional_full_year_tabm_seed{seed}',
              'seed': seed, 'member': k,
              'execution': 'unchanged production fit; reviewed CV permutation-seed specialization',
              'retained': 'seed42 production retained exactly; other TabM seed fitted separately',
              'export': 'three-seed equal pool exported by outer continuation runner',
              'no_extra_seed': False, 'no_whole_pool_substitution': False}
    namespace = dict(first.__dict__, OUT=run, MANIFEST=run / 'identity.json',
                     SEED=seed, DESIGN=design, __file__=__file__)
    train = cv.trainer(k)
    train_namespace = dict(train.__globals__, OUT=run)
    specialized = types.FunctionType(train.__code__, train_namespace, 'full_year_seed_train',
                                     train.__defaults__, train.__closure__)
    assert train_namespace['RUN_SEED'] == seed
    namespace['trainer'] = lambda: specialized
    for name in FUNCTIONS:
        original = getattr(first, name)
        namespace[name] = types.FunctionType(original.__code__, namespace, name,
                                             original.__defaults__, original.__closure__)
        assert namespace[name].__code__ is original.__code__
    assert first.SEED == 42 and first.OUT != run
    return namespace


def prepare():
    assert not (OUT / 'identity.json').exists(), 'Preserve frozen continuation'
    result, transported, audit = read(STACK / 'result.json'), read(TRANSPORT / 'result.json'), read(AUDIT)
    assert audit['status'] == 'pass'
    for record in audit['inputs']:
        verify(record)
    assert result['paired']['three_tabm_vs_one_tabm']['decision'] == 'improvement'
    assert transported['paired_three_vs_one']['decision'] == 'improvement'
    prior = read(first.MANIFEST)
    assert prior['source'] == art(first.__file__) and prior['design'] == first.DESIGN
    for record in prior['files']:
        verify(record)
    reused = first.completed(prior)
    assert reused is not None
    source_paths = [Path(__file__), Path(cv.__file__), cv.MANIFEST, first.MANIFEST,
                    first.OUT / 'trained/k0.json', first.OUT / 'networks/k0.json',
                    verify(reused['prediction']), verify(first.trained(prior)['model']),
                    STACK / 'result.json', STACK / 'audit.json', TRANSPORT / 'result.json', AUDIT]
    for key in ['source', 'files']:
        assert key in prior
    shared_sources = [art(p) for p in source_paths]
    members = {}
    for k in [1, 2]:
        n = member(k)
        value = copy.deepcopy(prior)
        value.update(design=n['DESIGN'], source=art(__file__),
                     cv_identity=art(cv.MANIFEST), selected_v78=art(STACK / 'result.json'),
                     selected_v79=art(TRANSPORT / 'result.json'),
                     checks=dict(production_fit_bytecode_exact=True, seed_specialization_ast_verified=True),
                     continuation_member=k, continuation_seed=cv.SEEDS[k],
                     files=list({r['path']: r for r in [*prior['files'], *shared_sources]}.values()))
        write(n['MANIFEST'], value)
        members[str(k)] = art(n['MANIFEST'])
    value = dict(source=art(__file__), sources=shared_sources, members=members,
                 seeds=cv.SEEDS, reused_seed42=dict(identity=art(first.MANIFEST),
                 trained=art(first.OUT / 'trained/k0.json'), record=art(first.OUT / 'networks/k0.json'),
                 prediction=reused['prediction']), epochs=15, ranking_rows=344841,
                 decision='Fixed three-TabM pool passes both audited comparisons against one-TabM',
                 no_upload=True)
    write(OUT / 'identity.json', value)
    paths = {r['path'] for r in [*shared_sources, *members.values()]}
    paths.add(str((OUT / 'identity.json').relative_to(ROOT)))
    # Large full-year inputs and track arrays are already cached on the laptop.
    (OUT / 'transfer_files.txt').write_text('\n'.join(sorted(paths)) + '\n')
    print(dict(prepared=True, seeds=[1042, 2042], identity=art(OUT / 'identity.json')), flush=True)


def guard():
    identity = read(OUT / 'identity.json')
    assert identity['source'] == art(__file__) and identity['seeds'] == [42, 1042, 2042]
    for record in [*identity['sources'], *identity['members'].values(), *identity['reused_seed42'].values()]:
        verify(record)
    return identity


def fit():
    identity = guard()
    expected = read(cv.MANIFEST)['execution']
    assert {key: value for key, value in dict(host=first.os.uname().nodename,
        gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)).items()} == expected
    for k in [1, 2]:
        n = member(k)
        assert identity['members'][str(k)] == art(n['MANIFEST'])
        if not (n['OUT'] / 'preflight.json').exists():
            n['preflight']()
        n['fit']()
    print('Both additional full-year seeds completed; seed42 reused', flush=True)


def export():
    identity = guard()
    assert not (OUT / 'export.json').exists()
    records = [identity['reused_seed42']['prediction']]
    for k in [1, 2]:
        n = member(k)
        record = n['completed'](n['guard'](full=False))
        assert record is not None
        records.append(record['prediction'])
    frames = [pd.read_parquet(verify(r)) for r in records]
    ids = frames[0].mvt_id
    assert len(ids) == 344841 and ids.is_unique
    for f in frames:
        np.testing.assert_array_equal(f.mvt_id, ids)
        assert np.isfinite(f.base).all() and (f.base >= 0).all()
    value = pd.DataFrame(dict(mvt_id=ids, **{f'tabm_k{k}': f.base for k, f in enumerate(frames)}))
    value['replacement_pool'] = np.mean([f.base.to_numpy(float) for f in frames], axis=0)
    value.to_parquet(OUT / 'replacement_pool.parquet', index=False)
    write(OUT / 'export.json', dict(identity=art(OUT / 'identity.json'),
          members=records, pool=art(OUT / 'replacement_pool.parquet'),
          rows=len(ids), seeds=cv.SEEDS, view='old', formula='float64 equal mean of float32 predictions',
          no_final_stack=True, no_upload=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'export'])
    torch.set_num_threads(2)
    with threadpool_limits(2):
        globals()[parser.parse_args().stage]()
