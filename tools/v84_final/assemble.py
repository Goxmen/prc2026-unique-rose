"""Assemble the owner-selected v84 final from saved prediction components.

The selected old-v84 recipe and unchanged fixed-v80 correction code determine
every coefficient, transform and rounding operation. No model training, new
calibration, original-ID splicing, competition upload or model selection occurs.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'VALIDATION/runs'
OUT = RUNS/'v3_20261014_v84_final'
OLD = RUNS/'v3_20261014_exact_v80_recovery/prediction_fidelity_recheck'
RECOVERY = OLD.parent
PRIOR = RUNS/'v3_20261013_v80_refit_final'
RECOVERED = RECOVERY/'newly_recovered_components.json'
EXPECTED_OLD = 'ab5749c68560aa02bae68aadf9e13b962a87f0c5227b2f2303303a41cff16763'
ASSEMBLER = ROOT/'tools/v80_refit/assemble.py'


def original_assembler():
    spec = importlib.util.spec_from_file_location('fixed_v80_assembly', ASSEMBLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def catalog(a):
    old_identity = a.read(OLD/'assembly_identity.json')
    old_catalog = a.read(OLD/'leaf_catalog.json')
    old_audit = a.read(OLD/'old/audit.json')
    if a.sha(OLD/'old/candidate.parquet') != EXPECTED_OLD or old_audit['status'] != 'pass':
        raise ValueError('Owner-selected old-v84 candidate changed')
    if old_audit['identity'] != a.art(OLD/'assembly_identity.json') or old_audit['candidate'] != a.art(OLD/'old/candidate.parquet'):
        raise ValueError('Old-v84 audit is not bound to selected identity/output')
    if old_identity['leaf_catalog'] != a.art(OLD/'leaf_catalog.json'):
        raise ValueError('Selected old-v84 component lineage changed')
    partial = a.read(a.CATALOG)['components']
    recovered = a.read(RECOVERED)['components']
    v3 = a.read(OUT/'v3/final.json')
    old3 = a.read(OUT/'v3/old.json')
    if v3['status'] != 'pass' or not v3['all_rows_fresh_inference'] or not v3['no_old_id_splicing']:
        raise ValueError('Corrected-source v3 final inference is not ready')
    if not old3['old_selected_replay_exact'] or old3['identity'] != v3['identity']:
        raise ValueError('Saved v3 model failed selected old replay')
    if v3['identity'] != a.art(OUT/'v3/identity.json'):
        raise ValueError('v3 final provenance changed')
    families = {name:a.read(PRIOR/name/'result.json') for name in ['trees','neural']}
    result = {}
    for version in old_identity['geometry71']['versions']:
        key = str(version)
        if version == 3:
            spec = dict(final=v3['prediction'], id_column='mvt_id', value_column='submitted',
                        receipt=a.art(OUT/'v3/final.json'), model=a.read(OUT/'v3/identity.json')['model'],
                        provenance='Saved corrected-source b04dd36 model; exact selected old replay, all final rows freshly inferred')
        elif key in recovered:
            spec = copy.deepcopy(recovered[key])
        elif key in partial:
            spec = copy.deepcopy(partial[key])
        else:
            family = 'trees' if version in a.TREE_VERSIONS else 'neural'
            if version not in (a.TREE_VERSIONS + a.NEURAL_VERSIONS):
                raise ValueError(f'Unrecognized selected leaf v{version}')
            spec = dict(final=families[family]['predictions']['final'], id_column='mvt_id',
                        value_column=f'v{version}', receipt=a.art(PRIOR/family/'result.json'),
                        provenance=f'Same retained {family} refit used by selected old-v84 recipe')
        spec['selected_old_component'] = old_catalog['components'][key]
        result[key] = spec
    return dict(schema='selected_v84_final_leaf_catalog_v1', components=result,
                old_v84=dict(candidate=a.art(OLD/'old/candidate.parquet'),
                             identity=a.art(OLD/'assembly_identity.json'),
                             audit=a.art(OLD/'old/audit.json'),
                             leaf_catalog=a.art(OLD/'leaf_catalog.json')),
                recovered_catalog=a.art(RECOVERED), no_fit=True, no_upload=True,
                all_model_weights_available='Requires separate weights inventory; not asserted by a prediction-component catalog')


def model_records(a, leaves):
    """Explicit selected recovery/refit weights; other families retain provenance.

    This list never treats saved predictions as model weights. Full availability
    is separately audited before the owner-authorized upload.
    """
    records = []
    for version, rec in leaves['components'].items():
        for key in ['model','classifier']:
            if key in rec:
                records.append(dict(version=int(version), role=key, artifact=rec[key]))
    for name in ['v26_ordinary','v26_classifier','v27_ordinary','v28_ordinary','v29_ordinary','v29_classifier']:
        receipt = PRIOR/'trees/models'/f'{name}.json'
        rec = a.read(receipt)
        records.append(dict(role=name, receipt=a.art(receipt), artifact=rec['model']))
    neural = a.read(PRIOR/'neural/result.json')
    for rec in neural['artifacts']:
        if rec['path'].endswith('.pt'):
            records.append(dict(role='selected_neural_refit_checkpoint', artifact=rec))
    for rec in records:
        a.verify(rec['artifact'])
    return records


def freeze(a):
    leaves = catalog(a)
    target = OUT/'assembly_identity.json'
    if target.exists():
        identity = a.read(target)
        if a.read(OUT/'leaf_catalog.json') != leaves:
            raise ValueError('Preserve existing final assembly leaf choices')
        for rec in identity['sources']:
            a.verify(rec)
        return identity, leaves
    identity = copy.deepcopy(a.read(OLD/'assembly_identity.json'))
    for rec in identity['sources']:
        a.verify(rec)
    a.write(OUT/'leaf_catalog.json', leaves)
    sources = [a.art(Path(__file__)), a.art(RECOVERED), a.art(OUT/'leaf_catalog.json'),
               a.art(OLD/'assembly_identity.json'), a.art(OLD/'old/candidate.parquet'),
               a.art(OLD/'old/audit.json'), a.art(OUT/'v3/identity.json'),
               a.art(OUT/'v3/result.json'), a.art(OUT/'v3/final.json'), a.art(OUT/'v3/old.json')]
    sources += a.read(OUT/'v3/identity.json')['sources']
    for rec in leaves['components'].values():
        sources.append(rec['final'])
        for key in ['model','classifier','audit','result','receipt']:
            if key in rec:
                sources.append(rec[key])
    weights = model_records(a, leaves)
    for rec in weights:
        sources.append(rec['artifact'])
        if 'receipt' in rec:
            sources.append(rec['receipt'])
    for rec in sources:
        a.verify(rec)
    identity['sources'] += sources
    old = a.read(OLD/'assembly_identity.json')
    for key in ['geometry71','geometry73','versions','weights']:
        if identity[key] != old[key]:
            raise ValueError(f'Selected v84 recipe changed: {key}')
    identity.update(schema='selected_v84_fixed_final_assembly_v1',
                    old_selected_v84=leaves['old_v84'], final_leaf_catalog=a.art(OUT/'leaf_catalog.json'),
                    source=a.art(Path(__file__)), original_assembler=a.art(ASSEMBLER),
                    recipe='Owner-selected v84: same fixed affine/correction recipe as measured old-v84; recovered originals and corrected-source v3',
                    no_fit=True, no_upload=True, no_old_id_splicing=True,
                    scope='Fresh final-row fixed-recipe assembly from authenticated final components',
                    all_model_weights_available='Pending separate explicit weights audit')
    a.write(target, identity)
    a.write(OUT/'dependency_inventory.json', dict(
        schema='v84_final_selected_dependency_inventory_v1', identity=a.art(target),
        source_and_input_artifacts=identity['sources'], explicit_saved_recovery_and_refit_weights=weights,
        retained_component_provenance=leaves['components'],
        all_model_weights_available='Not asserted; see separate complete weight availability audit',
        no_fit=True, no_upload=True))
    return identity, leaves


def final_basis(a, leaves, ids):
    import numpy as np
    import pandas as pd
    frame = pd.DataFrame(dict(mvt_id=ids))
    cached = {}
    for version, rec in leaves['components'].items():
        path = a.verify(rec['final'])
        key = (str(path), rec.get('id_column','mvt_id'))
        if key not in cached:
            cached[key] = a.align(pd.read_parquet(path), ids, key[1])
        values = cached[key][rec.get('value_column','submitted')].to_numpy(float)
        np.testing.assert_array_equal(values, a.serial(values), err_msg=f'Final leaf v{version}')
        frame[f'v{version}'] = values.astype(np.int32)
    return frame


def run():
    from threadpoolctl import threadpool_limits
    started = time.time()
    a = original_assembler()
    identity, leaves = freeze(a)
    mods = a.modules()
    def selected_final_basis(view, out, frozen, corrections):
        if view != 'final' or out != OUT:
            raise ValueError('Only this selected final assembly is authorized')
        return final_basis(a, leaves, corrections.template('final'))
    a.basis = selected_final_basis
    with threadpool_limits(1):
        a.build('final', OUT, identity, mods)
        audit = a.audit('final', OUT, identity, mods)
    a.write(OUT/'assembly_complete.json', dict(status='pass', identity=a.art(OUT/'assembly_identity.json'),
         source=a.art(Path(__file__)), candidate=audit['candidate'], audit=a.art(OUT/'final/audit.json'),
         old_selected_v84=leaves['old_v84'], rows=670790, elapsed_seconds=time.time()-started,
         no_fit=True, no_upload=True, independent_audit_pending=True,
         full_weights_availability_pending_separate_audit=True))
    print({'v84_final':'assembled','rows':670790,'elapsed_seconds':time.time()-started}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()
