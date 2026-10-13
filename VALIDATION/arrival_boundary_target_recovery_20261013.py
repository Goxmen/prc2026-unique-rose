"""Manual, separately reviewed recovery of ARR targets after SQL parse failure.

Frozen producer/input packets stay unchanged. The only numerical-tail code
change is an explicit AS on the raw target query's day alias. No input packet
builder or sampler is called. Root must allocate the single target-stage run.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'VALIDATION/runs/v3_20261013_arrival_boundary_transfer'
REC = OUT/'target_sql_recovery'
ORIGINAL = ROOT/'VALIDATION/arrival_boundary_transfer_pilot_20261013.py'
AUDITOR = ROOT/'VALIDATION/tools/audit_arrival_boundary_transfer_20261013.py'
OLD = "strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') day"
NEW = "strftime(MVT_TIME_UTC_mvt,'%Y-%m-%d') AS day"


def read(p): return json.loads(Path(p).read_text())


def art(p):
    p = Path(p).resolve(); h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return dict(path=str(p.relative_to(ROOT)), sha256=h.hexdigest())


def verify(r):
    p = ROOT/r['path']; assert art(p) == r, p
    return p


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(v, f, indent=2, allow_nan=False); f.write('\n')


def tail_tree():
    tree = ast.parse(ORIGINAL.read_text())
    function = next(v for v in tree.body if isinstance(v, ast.FunctionDef) and v.name == 'prepare')
    split = [i for i, v in enumerate(function.body) if isinstance(v, ast.Assign)
        and isinstance(v.value, ast.Call) and isinstance(v.value.func, ast.Name)
        and v.value.func.id == 'input_packets']
    assert len(split) == 1
    old_body = copy.deepcopy(function.body[split[0]+1:])
    body = copy.deepcopy(old_body); changes = 0
    for statement in body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and OLD in node.value:
                assert node.value.count(OLD) == 1
                node.value = node.value.replace(OLD, NEW); changes += 1
    assert changes == 1
    reverted = copy.deepcopy(body)
    for statement in reverted:
        for node in ast.walk(statement):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and NEW in node.value:
                node.value = node.value.replace(NEW, OLD)
    assert ast.dump(ast.Module(body=reverted, type_ignores=[]), include_attributes=False) == ast.dump(
        ast.Module(body=old_body, type_ignores=[]), include_attributes=False)
    arguments = ast.arguments(posonlyargs=[], args=[ast.arg(arg=s) for s in
        ['ident', 'meta', 'packets', 'started', 'deadline']], kwonlyargs=[], kw_defaults=[], defaults=[])
    recovered = ast.FunctionDef(name='recovered_target_tail', args=arguments, body=body,
        decorator_list=[], returns=None, type_comment=None, type_params=[])
    result = ast.fix_missing_locations(ast.Module(body=[recovered], type_ignores=[]))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in
        ['input_packets', 'hash_id', 'point_features'] for n in ast.walk(result))
    return result


def absent_targets():
    for p in ['labels.parquet', 'prepared.json', 'fit_started.json', 'fit.json', 'result.json']:
        assert not (OUT/p).exists(), p
    assert not list((OUT/'inputs').rglob('targets.npy'))
    assert not (OUT/'models').exists()


def audit_tree():
    original = ast.parse(AUDITOR.read_text()); changed = copy.deepcopy(original); count = 0
    for node in ast.walk(changed):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and OLD in node.value:
            assert node.value.count(OLD) == 1
            node.value = node.value.replace(OLD, NEW); count += 1
    assert count == 1
    reverted = copy.deepcopy(changed)
    for node in ast.walk(reverted):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and NEW in node.value:
            node.value = node.value.replace(NEW, OLD)
    assert ast.dump(reverted, include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(changed)


def extend_draft():
    """Prospective source-only extension, before any recovery identity/run."""
    absent_targets(); assert not (REC/'identity.json').exists()
    plan = read(REC/'plan.json')
    assert art(AUDITOR) == read(OUT/'identity.json')['auditor']
    target = REC/'before'/AUDITOR.name; shutil.copy2(AUDITOR, target)
    assert art(AUDITOR)['sha256'] == art(target)['sha256']
    plan['archived'].append(dict(original=art(AUDITOR), preserved=art(target)))
    snapshot = REC/'auditor_snapshot.py'
    snapshot.write_text('# Frozen independent auditor AST; exactly one explicit SQL AS alias repair.\n'+ast.unparse(audit_tree())+'\n')
    plan['source'] = art(__file__)
    plan['auditor_original'] = art(AUDITOR); plan['auditor_snapshot'] = art(snapshot)
    plan['auditor_change'] = 'Exactly the same single raw target SQL alias; all input, target, objective, held loss and bootstrap checks unchanged.'
    plan['audit_receipt_binding'] = 'Legacy auditor field identifies frozen audited method; executed_auditor_snapshot and recovery_identity explicitly identify actual alias-repaired execution.'
    plan['stages'] = ['targets', 'audit_prepare', 'audit_result']
    plan['before_auditor_extension_plan'] = art(REC/'before_auditor_extension/plan.json')
    temporary = REC/'plan.new.json'; write(temporary, plan); temporary.replace(REC/'plan.json')
    print(json.dumps(dict(plan=art(REC/'plan.json'), auditor_snapshot=art(snapshot), source=art(__file__))))


def draft():
    assert not REC.exists(); absent_targets(); REC.mkdir()
    identity = read(OUT/'identity.json'); assert identity['source'] == art(ORIGINAL)
    inputs = read(OUT/'inputs.json'); assert inputs['identity'] == art(OUT/'identity.json')
    assert inputs['features_persisted_before_label_read'] and len(inputs['packets']) == 24
    failure = (OUT/'prepare.log').read_text()
    assert 'Parser Error: syntax error at or near "day"' in failure
    original_files = [ORIGINAL, OUT/'identity.json', OUT/'prepare_started.json', OUT/'prepare.log',
        OUT/'inputs.json', OUT/'source_review.json', OUT/'allocation.json']
    archived = []
    for path in original_files:
        target = REC/'before'/path.name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert art(path)['sha256'] == art(target)['sha256']
        archived.append(dict(original=art(path), preserved=art(target)))
    tree = tail_tree(); snapshot = REC/'target_tail.py'
    snapshot.write_text('# AST-derived original prepare target tail; exactly one SQL AS alias repair.\n'+ast.unparse(tree)+'\n')
    inputs_to_keep = [inputs['flights']]+[p[k] for p in inputs['packets'] for k in ['features', 'times', 'flights']]
    for a in inputs_to_keep: verify(a)
    write(REC/'plan.json', dict(state='source_only_manual_target_recovery_no_execution',
        original_identity=art(OUT/'identity.json'), original_source=art(ORIGINAL), source=art(__file__),
        snapshot=art(snapshot), inputs_receipt=art(OUT/'inputs.json'), retained_inputs=inputs_to_keep,
        archived=archived, only_target_tail_AST_change=dict(old=OLD, new=NEW),
        prefix_replacement='Verified existing input_flights.parquet and inputs.json packets replace completed input_packets call; no resampling/rebuild.',
        targets_coverage_and_write_order_unchanged=True, target_labels_never_decoded_by_failed_query=True,
        elapsed_seconds_scope='Recovered target stage only; prior input stage duration not reclassified.',
        resource_scope='Existing fixed300second target-stage deadline and2GB high-water guard; no model fit.',
        no_repeat_input_preparation=True, no_frozen_source_edit=True, no_arrays_decoded_in_draft=True))
    print(json.dumps(dict(plan=art(REC/'plan.json'), snapshot=art(snapshot))))


def freeze():
    absent_targets(); plan = read(REC/'plan.json'); review = read(REC/'source_review.json')
    assert review['status'] == 'pass' and review['source'] == art(__file__)
    assert review['plan'] == art(REC/'plan.json') and review['snapshot'] == art(REC/'target_tail.py')
    assert review['auditor_snapshot'] == art(REC/'auditor_snapshot.py')
    assert plan['source'] == art(__file__) and plan['snapshot'] == art(REC/'target_tail.py')
    expected = ast.unparse(tail_tree())
    assert ast.dump(ast.parse((REC/'target_tail.py').read_text()), include_attributes=False) == ast.dump(
        ast.parse(expected), include_attributes=False)
    assert ast.dump(ast.parse((REC/'auditor_snapshot.py').read_text()), include_attributes=False) == ast.dump(
        audit_tree(), include_attributes=False)
    for a in plan['retained_inputs']: verify(a)
    for pair in plan['archived']: verify(pair['original']); verify(pair['preserved'])
    write(REC/'identity.json', dict(source=art(__file__), original_identity=plan['original_identity'],
        plan=art(REC/'plan.json'), snapshot=art(REC/'target_tail.py'), source_review=art(REC/'source_review.json'),
        auditor_original=art(AUDITOR), auditor_snapshot=art(REC/'auditor_snapshot.py')))


def release(stage):
    identity = read(REC/'identity.json'); plan = read(verify(identity['plan']))
    for key in ['source', 'original_identity', 'snapshot', 'source_review', 'auditor_original', 'auditor_snapshot']:
        verify(identity[key])
    allocation = read(REC/'allocation.json')
    assert allocation['authority'] == '/root' and allocation['identity'] == art(REC/'identity.json')
    assert allocation['state'] == 'released_for_single_cached_input_target_recovery' and stage in allocation['stages']
    write(REC/(stage+'_attempt.json'), dict(identity=art(REC/'identity.json'), allocation=art(REC/'allocation.json'),
        started=time.time(), no_automatic_retry=True))
    for pair in plan['archived']: verify(pair['original']); verify(pair['preserved'])
    for a in plan['retained_inputs']: verify(a)
    return identity, plan


def targets():
    absent_targets(); identity, plan = release('targets')
    spec = importlib.util.spec_from_file_location('frozen_arrival_original', ORIGINAL)
    parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
    ident = parent.guard('prepare')
    inputs = read(verify(plan['inputs_receipt']))
    meta = parent.pd.read_parquet(verify(inputs['flights']))
    packets = inputs['packets']
    namespace = dict(parent.__dict__)
    original_write = parent.write
    def recovered_write(path, value):
        if Path(path) == OUT/'prepared.json':
            value = dict(value, target_recovery=art(REC/'identity.json'),
                target_recovery_allocation=art(REC/'allocation.json'), retained_inputs=plan['inputs_receipt'],
                elapsed_seconds_scope=plan['elapsed_seconds_scope'])
        original_write(path, value)
    namespace['write'] = recovered_write
    exec(compile((REC/'target_tail.py').read_text(), str(REC/'target_tail.py'), 'exec'), namespace)
    started = time.monotonic()
    with parent.threadpool_limits(1):
        namespace['recovered_target_tail'](ident, meta, packets, started, started+300)
    for a in plan['retained_inputs']: verify(a)
    prepared = read(OUT/'prepared.json')
    assert prepared['target_recovery'] == art(REC/'identity.json')
    write(REC/'result.json', dict(status='pass', identity=art(REC/'identity.json'),
        prepared=art(OUT/'prepared.json'), original_inputs_unchanged=True,
        coverage_gate=prepared['coverage_gate'], elapsed_seconds=time.monotonic()-started,
        no_input_builder_or_sampler_called=True, no_fit=True))


def recovered_audit(stage):
    identity, plan = release('audit_'+stage)
    recovered = read(verify(art(REC/'result.json')))
    assert recovered['status'] == 'pass' and recovered['identity'] == art(REC/'identity.json')
    assert recovered['prepared'] == art(OUT/'prepared.json')
    prepared = read(OUT/'prepared.json'); assert prepared['target_recovery'] == art(REC/'identity.json')
    namespace = {'__file__': str(AUDITOR), '__name__': 'explicit_alias_repaired_arrival_auditor'}
    exec(compile((REC/'auditor_snapshot.py').read_text(), str(REC/'auditor_snapshot.py'), 'exec'), namespace)
    original_save = namespace['save']
    def audited_save(name, value):
        value = dict(value, recovery_identity=art(REC/'identity.json'),
            executed_auditor_snapshot=art(REC/'auditor_snapshot.py'),
            frozen_auditor_semantics=plan['auditor_original'],
            sole_execution_change='Explicit AS for SQL day alias; audited calculations unchanged')
        original_save(name, value)
    namespace['save'] = audited_save
    with namespace['threadpool_limits'](1): namespace[stage]()
    path = OUT/('prepare_audit.json' if stage == 'prepare' else 'independent_audit.json')
    for a in plan['retained_inputs']: verify(a)
    write(REC/('audit_'+stage+'_result.json'), dict(status='pass', identity=art(REC/'identity.json'),
        audit=art(path), executed_auditor_snapshot=art(REC/'auditor_snapshot.py'),
        original_inputs_unchanged=True, no_fit=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=[
        'draft', 'extend_draft', 'freeze', 'targets', 'audit_prepare', 'audit_result'])
    args = parser.parse_args()
    if args.stage.startswith('audit_'): recovered_audit(args.stage.removeprefix('audit_'))
    else: globals()[args.stage]()
