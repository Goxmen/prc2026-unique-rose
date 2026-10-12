"""Synthetic-only repeated CUDA control; no fits, labels or mutable run outputs."""
from pathlib import Path
import contextlib
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import torch
import nn_public_tabm_daily_cohort_model as candidate


def artifact(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def difference(a, b):
    a, b = a.detach().double(), b.detach().double()
    delta = b - a
    norm = a.norm().item()
    peak = a.abs().max().item()
    return dict(max_abs=delta.abs().max().item(), mismatches=int((a != b).sum()),
        reference_max_abs=peak, reference_l2=norm, difference_l2=delta.norm().item(),
        max_abs_over_reference_max=delta.abs().max().item() / max(peak, 1e-300),
        relative_l2=delta.norm().item() / max(norm, 1e-300), finite=bool(torch.isfinite(delta).all()))


def once(training):
    old = candidate.original
    cards, bins, plain = [3, 5], [torch.tensor([-1., 0., 1., 2.])]*3, 5
    models = []
    constructor_rng = []
    for kind in [old.PublicTabM, old.PublicTabM, candidate.DailyTabM]:
        torch.manual_seed(42)
        models.append(kind(cards, bins, plain+12).cuda().train(training))
        constructor_rng.append((torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()))
    constructor_exact = all(torch.equal(constructor_rng[0][j], x[j])
        for x in constructor_rng[1:] for j in [0, 1])
    initial_exact = all(torch.equal(value, model.state_dict()[name])
        for model in models[1:] for name, value in models[0].state_dict().items())
    tab, track = old.prototype.synthetic(cards, bins, plain+12, 12, 'cuda')
    packet = candidate.synthetic_packet(12, 'cuda')
    target = torch.linspace(-1, 1, 12, device='cuda')
    outputs, rngs = [], []
    for i, model in enumerate(models):
        torch.manual_seed(913)
        output = model(*tab, *track, *([packet] if i == 2 else []))
        old.loss(output, target).backward()
        outputs.append(output.detach())
        rngs.append((torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()))
    names = list(dict(models[0].named_parameters()))
    gradients = [dict(m.named_parameters()) for m in models]
    comparisons = {}
    for i, label in [(1, 'original_repeat'), (2, 'candidate_zero_branch')]:
        comparisons[label] = dict(output=difference(outputs[0], outputs[i]),
            rng_exact=all(torch.equal(rngs[0][j], rngs[i][j]) for j in [0, 1]),
            gradients={n:difference(gradients[0][n].grad, gradients[i][n].grad) for n in names})
    original_before = {n:p.detach().clone() for n,p in models[0].named_parameters()}
    for model in models:
        torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0003).step()
    for i, label in [(1, 'original_repeat'), (2, 'candidate_zero_branch')]:
        comparisons[label]['parameters_after_step'] = {
            n:difference(gradients[0][n], gradients[i][n]) for n in names}
        comparisons[label]['optimizer_updates'] = {
            n:difference(gradients[0][n]-original_before[n], gradients[i][n]-original_before[n]) for n in names}
    models[2].context_disabled = True
    post = []
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():post.append(model(*tab, *track, *([packet] if i == 2 else [])))
    for i, label in [(1, 'original_repeat'), (2, 'candidate_zero_branch')]:
        comparisons[label]['original_path_output_after_step'] = difference(post[0], post[i])
    return dict(training=training, constructor_rng_exact=constructor_exact,
        original_parameters_initial_exact=initial_exact, comparisons=comparisons)


def compact(rows):
    out = {}
    for label in ['original_repeat', 'candidate_zero_branch']:
        cases = [r['comparisons'][label] for r in rows]
        item = dict(initial_outputs_exact=all(c['output']['mismatches']==0 for c in cases),
            rng_exact=all(c['rng_exact'] for c in cases),
            post_step_original_outputs_max_abs=max(c['original_path_output_after_step']['max_abs'] for c in cases))
        for field in ['gradients', 'parameters_after_step', 'optimizer_updates']:
            values = [x for c in cases for x in c[field].values()]
            item[field] = dict(max_abs=max(x['max_abs'] for x in values),
                max_relative_l2=max(x['relative_l2'] for x in values),
                max_normalized_inf=max(x['max_abs_over_reference_max'] for x in values),
                total_mismatches=sum(x['mismatches'] for x in values))
        out[label] = item
    return out


def main():
    out = Path(sys.argv[1])
    assert not out.exists(), out
    torch.set_num_threads(2)
    candidate.original.prototype.shared.recipe()
    flags = lambda: dict(cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    before = flags()
    result = dict(status='diagnostic_only', no_labels=True, no_fit=True, repeats=4,
        source=artifact(__file__), model=artifact(candidate.__file__),
        old_model=artifact(candidate.original.__file__), torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), original_backend_flags=before, modes={})
    for mode in ['original_recipe', 'scoped_deterministic_cudnn']:
        cm = contextlib.nullcontext() if mode == 'original_recipe' else torch.backends.cudnn.flags(deterministic=True, benchmark=False)
        with cm:
            rows = [dict(repeat=i, **once(training)) for training in [False, True] for i in range(4)]
            result['modes'][mode] = dict(flags=flags(), cases=rows, summary=compact(rows))
    result['backend_flags_restored'] = flags() == before
    assert result['backend_flags_restored']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(out), modes={k:v['summary'] for k,v in result['modes'].items()},
        backend_flags_restored=result['backend_flags_restored']), indent=2), flush=True)


if __name__ == '__main__':main()
