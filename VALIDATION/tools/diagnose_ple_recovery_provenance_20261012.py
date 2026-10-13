"""Read-only source/log provenance diagnosis of the closed legacy PLE pilot.

No models, encoded arrays, raw input cache or labels are loaded. Historical
four-decimal training losses establish chronology, not bitwise trajectory
identity. Unrecorded CUDA state remains unknown, rather than a proven cause.
"""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_ple_pilot'
OUT = RUN / 'provenance_diagnosis.json'


def read(path):
    return json.loads(path.read_text())


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def function(path, name):
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def main():
    assert not OUT.exists()
    original = ROOT / 'VALIDATION/ideas_submit.py'
    trainer = ROOT / 'VALIDATION/nn_improve_jan_jul.py'
    encoder = ROOT / 'VALIDATION/nn_jan_jul.py'
    pilot = ROOT / 'VALIDATION/final_missing_ple_pilot.py'
    meta_path = ROOT / 'VALIDATION/runs/v3_20261003_ideas_submit/submission_v50/submission_v50.json'
    part_path = ROOT / 'VALIDATION/runs/v3_20261003_ideas_submit/work/parts/nn_v46.json'
    manifest, parts = read(meta_path), read(part_path)
    identity, prepared, trained, result = [read(RUN / f'{name}.json')
                                         for name in ['identity', 'prepared', 'trained', 'result']]
    assert manifest['identity']['signature']['VALIDATION/ideas_submit.py'] == art(original)['sha256']
    assert manifest['identity']['signature']['VALIDATION/nn_improve_jan_jul.py'] == art(trainer)['sha256']
    assert parts == manifest['identity']['parts']['nn_v46']
    assert parts['groups']['42']['stop_epoch'] == 14
    network = function(original, 'network_part')
    calls = [n for n in ast.walk(network) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == 'nni.train']
    assert len(calls) == 2
    inner_call, full_call = sorted(calls, key=lambda node: node.lineno)
    assert ast.unparse(inner_call.args[3]) == 'g'
    assert ast.unparse(full_call.args[2]) == 'np.arange(len(d.y))'
    assert ast.unparse(full_call.args[3]) == 'g + 1000 * k'
    assert ast.unparse(full_call.args[4]) == 'stop'
    assert "out[f'n{g}_{k}'] = base.astype(np.float32)" in ast.unparse(network)
    train_node = function(trainer, 'train')
    train_text = ast.unparse(train_node)
    for phrase in ['torch.manual_seed(seed)', 'np.random.seed(seed)',
                   'torch.Generator(device=d.device).manual_seed(seed)',
                   'torch.randperm(n, device=d.device, generator=gen)',
                   "NEW['epochs'] * math.ceil(n / batch)"]:
        assert phrase in train_text, phrase
    observed = ast.unparse(function(pilot, 'observed_train'))
    assert "nni.train('ple', d, np.arange(len(d.y)), 42, stop, (d.held, d.yv))" in observed
    successful = ROOT / 'tmp/ideas_submit/chain_detail.log'
    chronology = ROOT / 'tmp/ideas_submit/chain.log'
    abandoned = ROOT / 'tmp/ideas_submit/fit_nn_v46.log'
    raw = successful.read_text()
    completed = raw.split('nn_v46 done in ', 1)[0]
    assert 'nn_v46 done in 21.5 min' in raw
    assert '22:27:11' in raw and parts['finished_at'] == '2026-10-03T20:27:11+00:00'
    pattern = re.compile(r'\[(\d\d:\d\d:\d\d)\]\s+epoch (\d+): train ([0-9.]+), monitored RMSE ([a-zA-Z0-9.]+) \((\d+)s\)')
    segments = []
    for match in pattern.finditer(completed):
        clock, epoch, loss, rmse, seconds = match.groups()
        if int(epoch) == 1:
            segments.append([])
        segments[-1].append(dict(clock=clock, epoch=int(epoch), train_mse_std_printed=float(loss),
                                 rmse_printed=rmse, seconds_printed=int(seconds)))
    assert len(segments) == 18
    assert [len(x) for x in segments] == [24] + [14]*5 + [24] + [17]*5 + [24] + [17]*5
    groups = re.findall(r'nn_v46 group (\d+): stop (\d+), five networks fitted', completed)
    assert groups == [('42', '14'), ('142', '17'), ('242', '17')]
    teacher = segments[1]
    assert all(x['rmse_printed'] == 'nan' for x in teacher)
    comparison = []
    for old, new in zip(teacher, trained['curve']):
        current_printed = float(f"{new['train_mse_std']:.4f}")
        comparison.append(dict(epoch=old['epoch'], original=old['train_mse_std_printed'],
                               pilot=current_printed, equal_printed=old['train_mse_std_printed']==current_printed))
    first = next(item['epoch'] for item in comparison if not item['equal_printed'])
    assert first == 11
    assert prepared['inputs'] == parts['inputs'] == identity['expected_inputs']
    assert len(prepared['columns']) == 134 and prepared['rows'] == 2051327 and prepared['old_rows'] == 344841
    assert result['state'] == 'closed_nonexact_pilot' and result['acceptance_tolerance_fixed'] == 0
    report = dict(status='provenance_trace_complete_cause_not_proven',
        sources=[art(p) for p in [Path(__file__), original, trainer, encoder, pilot, meta_path,
                  part_path, successful, chronology, abandoned, RUN/'identity.json',
                  RUN/'prepared.json', RUN/'trained.json', RUN/'result.json']],
        seed_and_member=dict(original_group=42, member=0, expression='g + 1000*k',
            original_seed=42, pilot_seed=42, teacher_column='n42_0', original_stop=14, pilot_checkpoint=14,
            seed42000_excluded_by_exact_original_source=True),
        source_identity=dict(original_network_part_exact=True, original_train_exact=True,
            teacher_part_artifact_sha256=parts['artifacts']['parts/nn_v46.parquet']),
        original_execution=dict(successful_log=str(successful.relative_to(ROOT)),
            original_18_segments_epoch_counts=[len(x) for x in segments],
            successful_stop_groups=groups, original_part_finished_at=parts['finished_at'],
            ignored_earlier_log_reason='fit_nn_v46.log ends during third member; its first inner stop15 is an incomplete earlier attempt.',
            all_original_members_full_rows='np.arange(len(d.y))',
            reseeding='torch.manual_seed and np.random.seed before model initialization; fresh CUDA generator with same seed before each full fit',
            inner_selection='Original inner fit precedes full fit; pilot reuses saved stop and omits inner fit. Explicit training RNG resets remain unchanged.'),
        training_curve=dict(precision='Only historical printed four-decimal losses survive; equal printed values are not exact trajectory equality',
            all_14_comparisons=comparison, first_visible_divergence_epoch=first,
            prefix_equal_printed_epochs=10, original_epoch14_clock=teacher[-1]['clock']),
        preprocessing=dict(training_cache='Same retained original inputs.pkl selected by original data[names], with names order unchanged',
            source_dtype_contract='Original Encoder emits int64 categories and float32 numeric values; chunked adapter enforces exactly those dtypes',
            same_original_encoder_source=True, same_feature_order_source=True,
            same_original_134_column_count=True, same_original_cards_bin_counts_scales=True,
            original_summary=parts['inputs'], original_encoder_and_encoded_tensor_hashes_absent=True,
            current_original_vs_adapter_CPU_tensors_bins_and_wrapper_RNG_parity=prepared['controls'],
            full_historical_encoded_tensor_equality_unprovable_without_original_tensors=True),
        runtime=dict(original_gpu=parts['device'], original_torch=manifest['identity']['torch'],
            pilot_gpu=identity['design']['device'], pilot_torch=identity['torch'],
            pilot_flags=identity['execution'], original_thread_count_TF32_CUDA_allocator_and_atomic_order='unrecorded',
            original_source_TF32='ideas_submit.py performs no TF32 override for network_part; inherited library defaults were not logged',
            checkpoint_observer='Copies state only before original monitor; CPU control confirms no RNG or model update changes'),
        conclusion='Seed, member, stop, recorded dimensions and source chronology match. Losses match to printed precision through epoch10 then diverge. This supports accumulated numerical trajectory divergence, but its GPU/kernel/allocator cause cannot be proven from retained artifacts. Do not label the large error unavoidable or exact reconstruction.',
        no_refit=True, no_expansion=True, no_inference=True, no_labels_loaded=True)
    OUT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], first_visible_divergence_epoch=first,
                         seed=42, original_stop=14, report=art(OUT))))


if __name__ == '__main__':
    main()
