"""Source-only legacy134 PLE reconstruction plan; no encoding or fitting.

    .venv/bin/python VALIDATION/tools/plan_final_missing_ple_20261012.py

Uses all original stopping decisions. A shared fixed17-epoch trajectory may
save checkpoints14/15/17; this is recipe reconstruction, not proof that the
original nondeterministic GPU executions produced identical intermediate states.
"""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261012_final_missing_ple_plan'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def main():
    assert not (OUT / 'plan.json').exists()
    records = {v: RUNS / f'v3_20261003_nn_improve_submit/submission_v{v}/submission_v{v}.json' for v in [48, 49]}
    records[50] = RUNS / 'v3_20261003_ideas_submit/submission_v50/submission_v50.json'
    manifests = {v: read(p) for v, p in records.items()}
    part_path = RUNS / 'v3_20261003_ideas_submit/work/parts/nn_v46.json'
    parts = read(part_path)
    assert parts == manifests[50]['identity']['parts']['nn_v46']
    sources = {Path(__file__), part_path, *records.values()}
    for version, manifest in manifests.items():
        for name, digest in manifest['identity']['signature'].items():
            assert sha(ROOT / name) == digest, (version, name)
            sources.add(ROOT / name)
    for name, digest in manifests[48]['identity']['packages'].items():
        source = Path(importlib.util.find_spec(name).origin)
        assert sha(source) == digest, name
    torch_version_source = Path(importlib.util.find_spec('torch').origin).parent/'version.py'
    current_torch = next(ast.literal_eval(n.value) for n in ast.parse(torch_version_source.read_text()).body
                         if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in n.targets))
    assert manifests[48]['identity']['torch'] == manifests[49]['identity']['torch'] == current_torch
    assert current_torch.split('+')[0] == importlib.metadata.version('torch')
    assert manifests[48]['network']['inputs'] == manifests[49]['network']['inputs'] == parts['inputs']
    assert manifests[48]['network']['label_mean'] == manifests[49]['network']['label_mean']
    assert manifests[48]['network']['label_sd'] == manifests[49]['network']['label_sd']
    five = [42+1000*k for k in range(5)]
    assert [r['seed'] for r in manifests[48]['network']['networks']] == five
    assert [r['seed'] for r in manifests[49]['network']['networks']] == five
    assert manifests[48]['network']['stop_epoch'] == 15
    assert manifests[49]['network']['stop_epoch'] == 17
    stops = {int(g): r['stop_epoch'] for g, r in parts['groups'].items()}
    assert stops == {42:14, 142:17, 242:17}
    assert manifests[48]['weight'] == manifests[50]['weight'] == .6 and manifests[49]['weight'] == .3
    original_train = next(n for n in ast.parse((ROOT/'VALIDATION/nn_improve_jan_jul.py').read_text()).body
                          if isinstance(n,ast.FunctionDef) and n.name=='train')
    stop_loads = [n for n in ast.walk(original_train) if isinstance(n,ast.Name) and n.id=='stop' and isinstance(n.ctx,ast.Load)]
    assert len(stop_loads) == 1
    stop_loop = next(n for n in ast.walk(original_train) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
                     and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range')
    assert isinstance(stop_loop.iter.args[0],ast.Name) and stop_loop.iter.args[0].id=='stop'
    train_source = ast.get_source_segment((ROOT/'VALIDATION/nn_improve_jan_jul.py').read_text(),original_train)
    assert "total_steps=NEW['epochs'] * math.ceil(n / batch)" in train_source
    assert manifests[48]['identity']['recipe']['new']['epochs'] == 24
    old_members = part_path.with_suffix('.parquet')
    assert sha(old_members) == parts['artifacts']['parts/nn_v46.parquet']
    sources.add(old_members)
    comparisons = {}
    with duckdb.connect(config={'threads':1,'memory_limit':'128MB'}) as con:
        con.read_parquet(str(old_members)).create_view('members')
        mean = '(' + '+'.join(f'cast(n42_{k} AS DOUBLE)' for k in range(5)) + ')/5'
        for version in [48,49]:
            paths = [records[version].parent/f'ranking_{m}.parquet' for m in ['2026-01','2026-07']]
            for month,path in zip(['2026-01','2026-07'],paths):
                expected = manifests[version]['network']['ranking'][month]['artifacts'][path.name]
                assert sha(path) == expected
                sources.add(path)
            con.read_parquet([str(p) for p in paths]).create_view('original',replace=True)
            row = con.execute(f'''SELECT count(*),sqrt(avg(pow(({mean})-o.nn_base,2))),max(abs(({mean})-o.nn_base))
                FROM members m JOIN original o USING(mvt_id)''').fetchone()
            assert row[0] == 344841
            comparisons[str(version)] = dict(rows=row[0],v50_group42_mean_vs_original_mean_rms=row[1],max_abs=row[2],
                interchangeable=False,reason=f'Original v50 first five stop14; v{version} stops'+str(manifests[version]['network']['stop_epoch']))
    prepared_path = RUNS/'v3_20261008_open_meteo_submit/work/prepared.json'
    prepared = read(prepared_path);sources.add(prepared_path)
    trajectories = []
    for group in [42,142,242]:
        for member in range(5):
            epochs = [14,15,17] if group==42 else [17]
            trajectories.append(dict(group=group,member=member,seed=group+1000*member,stop=17,
                save_epochs=epochs,original_v50_column=f'n{group}_{member}'))
    seconds = sum(n['seconds'] for n in manifests[49]['network']['networks'])
    plan = dict(purpose='Recover absent original legacy134 PLE weights without repeating epoch selection',
        versions=[48,49,50],input_features=134,ordinary_rows=2051327,
        original_device='NVIDIA GeForce RTX 3080',torch=current_torch,
        original_stops={'48':15,'49':17,'50':stops},blend_weights={'48':.6,'49':.3,'50':.6},
        shared_training_trajectory_proof=dict(stop_referenced_only_by_loop_bound=True,
            one_cycle_total_epochs=24,trainer_ast_sha256=hashlib.sha256(ast.dump(original_train).encode()).hexdigest(),
            inference_monitor_does_not_change_optimizer_or_select_epochs=True,
            prefix_checkpoint_CPU_RNG_weight_parity_required_before_fit=True),
        trajectories=trajectories,unique_fits=15,checkpoint_states=25,no_inner_stop_fit=True,
        first_five_reuse_is_seed_and_recipe_only=True,old_mean_comparisons=comparisons,
        old_input_summary=parts['inputs'],label_mean=manifests[48]['network']['label_mean'],
        label_sd=manifests[48]['network']['label_sd'],
        input_source=dict(path='VALIDATION/runs/v3_20261003_ideas_submit/work/inputs.pkl',
            sha256=prepared['original_inputs_sha256'],verification='Previously verified by weather preparation; rehash before recovery preparation',
            reuse='Select exact original134 columns, original category metadata, ordered membership and label vector'),
        gate_sequence=['Independent full input/membership/category/encoder/bin/card/scale checks, all preserved original summaries exact',
            'CPU original trainer versus checkpoint-observation wrapper prefix/RNG/parameter equivalence',
            'First fixed seed42 trajectory saves14/15/17, model+encoder states before diagnostics',
            'Compare checkpoint14 old float32 vector to retained v50 n42_0; report all-row exactness and float/rounded fidelity',
            'If nonexact, STOP before fourteen further fits; parent decides explicit approximation route using Jan/Jul validation',
            'If exact, finish remaining14 fixed trajectories and compare all15 v50 old vectors plus v48/v49 means and all3 integer vectors',
            'Final inference/independent audit only for exact old integer recipes, unless explicit approximation is separately authorized'],
        estimated_full15_fit_minutes_from_original_v49=seconds*3/60,
        no_fit_started=True,no_encoding_started=True,no_upload=True,
        caveats=['Original scripts did not save these model weights; existing complete old prediction vectors remain untouched.',
            'GPU training explicitly documented non-repeatability; same hardware/source/seeds do not prove original state identity.',
            'v48/v49/v50 used distinct historical executions and stop15/17/14; first-five old means are not reused interchangeably.',
            'Stopping receipts, seeds and24epoch learning-rate schedule fixed; no seed/thread/epoch/device sweep.'],
        sources=[art(p) for p in sorted(sources)])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='source_plan_only',unique_fits=15,checkpoint_states=25,first_fixed_pilot_seed=42,
        stops=plan['original_stops'],estimated_full_fit_minutes=plan['estimated_full15_fit_minutes_from_original_v49'],
        comparisons=comparisons)))


if __name__=='__main__':
    main()
