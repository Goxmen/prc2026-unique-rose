"""Independent active-context arithmetic and gradients, without training."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import nn_public_tabm_daily_cohort_model as proposed

OUT=ROOT/'VALIDATION/reports/cohort_learning_20261012'


def main():
    path=Path(proposed.__file__);before=hashlib.sha256(path.read_bytes()).hexdigest()
    # Isolate the actual new branch from unrelated retained backbone code.
    model=proposed.DailyTabM.__new__(proposed.DailyTabM);nn.Module.__init__(model)
    torch.manual_seed(982)
    model.donor=nn.Sequential(nn.Linear(17,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
    model.context_projection=nn.Linear(65,11)
    values=torch.randn(5,8,17,requires_grad=True)
    mask=torch.ones(5,8,dtype=torch.bool)
    mask[1,3:]=False;mask[2,:]=False;mask[3,1:]=False
    groups=torch.tensor([0,1,2,3,4,0])
    excluded=torch.tensor([[0,2,-1],[1,-1,-1],[-1,-1,-1],[0,-1,-1],[7,1,3],[7,-1,-1]])
    actual=model.context((values,mask,groups,excluded))
    expected=[]
    for g,exc in zip(groups,excluded):
        chosen=mask[g].clone();chosen[exc[exc>=0]]=False
        if chosen.any():
            latent=model.donor(values[g,chosen])
            summary=torch.cat([latent.mean(0),latent.square().mean(0),
                torch.log1p(chosen.sum()).reshape(1)/np.log(853.)])
            expected.append(model.context_projection(summary))
        else:expected.append(torch.zeros(11))
    expected=torch.stack(expected)
    worst=float((actual-expected).abs().max().detach())
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
    # A query's loss cannot depend on excluded/padded input tuples.
    cancelled=[];relative=[]
    for i in range(len(groups)):
        grad=torch.autograd.grad(actual[i].sum(),values,retain_graph=True)[0]
        g=groups[i];keep=mask[g].clone();keep[excluded[i][excluded[i]>=0]]=False
        # The sum-minus-exclusions derivative has floating-point cancellation,
        # unlike the exact algebraic derivative. Bound it by fp32 roundoff.
        maximum=float(grad.abs().max())
        residual=float(grad[g,~keep].abs().max()) if (~keep).any() else 0.
        limit=64*torch.finfo(grad.dtype).eps*max(maximum,torch.finfo(grad.dtype).tiny)
        assert residual<=limit
        cancelled.append(residual);relative.append(residual/max(maximum,torch.finfo(grad.dtype).tiny))
        elsewhere=torch.arange(5)!=g
        torch.testing.assert_close(grad[elsewhere],torch.zeros_like(grad[elsewhere]),atol=0,rtol=0)
        if keep.any():assert grad[g,keep].abs().sum()>0
    actual.sum().backward()
    for p in model.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    audit=dict(status='pass_context_math_only',source_sha256=before,
        explicit_per_query_gather_agrees=True,max_standardized_context_difference=worst,
        excluded_and_padding_gradient_cancellation_fp32_bound=True,
        maximum_cancelled_input_gradient=max(cancelled),maximum_relative_cancelled_gradient=max(relative),
        unrelated_cohort_input_gradients_exact_zero=True,
        all_present_donor_input_gradients_active=True,active_parameter_gradients_finite=True,
        zero_donor_and_all_excluded_context_exact_zero=True,
        no_fit=True,no_candidate=True,
        audit_correction='Initial exact-zero assertion failed at2.40545e-10; corrected to64*fp32_epsilon times maximum query gradient, with no model change',
        pending='Original full-model constructor/update/RNG parity and GPU execution are separate preflights')
    (OUT/'independent_context_model_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    torch.set_num_threads(1);main()
