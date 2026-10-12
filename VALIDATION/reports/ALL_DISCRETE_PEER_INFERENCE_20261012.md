# All-discrete public-peer inference: numerical feasibility

This is a source-only numerical study. It reads no departure labels or CV
scores, fits no parameters, and creates no candidate or upload. All original
priors, source memberships, t5 likelihoods, ordinary-flight covariance and
15-second common error remain fixed. Inference is local to one taxi/runway
cohort and one departure-delay cohort, not a globally conditioned factor graph.

## Prior Sobol sampling rejected

`joint_public_all_discrete_pilot.py` selects 12 cohorts per validation month by
equally spaced ranks of long-peer count, then the smallest active movement ID.
The 24 targets cover 7–475 long peers. Every peer keeps its complete discrete
support; one sampled value drives all source components in which it participates.
The target is integrated exactly. Every sampled peer's projected mean and
covariance are removed from the remaining Gaussian approximation.

Two fixed scrambles use 1,024, 4,096 and 16,384 nested draws. At 16,384,
increment differences between scrambles have median **1.90 seconds**, 90th
percentile **47.98**, and maximum **1,068.58**. The largest cohort has joint
ESS **3.53**; its increment changes by as much as **8,198.70 seconds** between
4,096 and 16,384 draws. Numerical convergence is inadequate; no CV scoring follows.

Independent reconstruction with explicit residual arrays and SciPy's
multivariate-t density reproduces posterior means within 5.5e-12 seconds and
ESS within 5.4e-14. One draw carries 49.8% of posterior weight and the ten
largest carry 91.9%. The covariance is positive definite. This is importance
weight collapse, not a quadratic-expansion error. Source-only preparation took
5.93 seconds and inference 4.97 seconds on one core.

Retained local artifacts are under
`tmp/research_20261012/error_mechanisms/all_discrete_peers/`; the independent
audit is `tmp/research_20261012/model_diversity/all_peer_worst_audit.json`.
The tracked [receipt](all_discrete_peer_20261012/qmc_receipt.json) preserves
the decisions, source/output hashes and independent numerical evidence.

## Bounded Gibbs rescue rejected

`joint_public_gibbs_pilot.py` and `joint_public_gibbs_kernel.c` implement exact
single-site conditionals over the unchanged supports. Deterministic sweeps
visit peers in sorted movement-ID order. Joint and departure-only chains stay
separate; target summaries use its full conditional distribution.

The fixed pilot uses the smallest, median-rank and largest cohorts from the
original 24-target pilot. Four chains start from two independent prior draws
and two overdispersed minimum/maximum-delay configurations. The predeclared
budgets are 128 burn-in plus 256 retained sweeps, then 256 plus 512. Diagnostics
include rank/folded split Rhat, bulk and mean ESS, cross-budget agreement,
target copy probability, all source totals, aggregate copy count and likelihood.
Failure closes this sampler at this budget; no extra sweeps or outcome-selected
gate is authorized by the design.

Finite three-peer tests prove single-site detailed balance, random-scan
stationarity and deterministic full-sweep invariance (maximum error 1.67e-16).
Exact posterior and Rao–Blackwell moments agree. The compiled kernel matches
independent Python explicit-residual transitions and conditional means.
The systematic full-sweep kernel is not claimed to be reversible.
An independent final-state reconstruction also checks all 24 chain/view/cohort
outputs against explicit residual vectors and SciPy multivariate-t densities;
the largest absolute discrepancy is 6.37e-12. This supports a mixing failure,
not a conditional-likelihood implementation error.

The fixed source-only pilot is complete: three cohorts (7, 69 and 475 peers),
two likelihood views and four chains took **1.23 seconds** on one core. The
smallest cohort passes the predeclared diagnostics. The median-rank cohort
fails the shorter budget's taxi-count ESS threshold (354 versus 400), although
its longer-budget diagnostics pass. No easier-cohort gate is selected.

The largest cohort fails substantially at 256 burn-in plus 512 retained sweeps:
joint target rank/folded split Rhat is **1.0582**, and copy-probability Rhat is
**1.0620**. Joint target bulk ESS is **101.68**, while raw-mean ESS is **17.39**.
The paired joint-minus-departure increment has Rhat **1.0555**, raw-mean ESS
**20.13** and estimated Monte Carlo error **1,263 seconds**. Its mean shifts
from **−253.29 to +525.06 seconds** between the fixed budgets. Rare-mode mixing
remains unresolved; no additional sweeps, alternate gate, CV score or candidate
follows this rescue.

The compiled C99 kernel uses `-O2 -std=c99 -fPIC -shared`, without fast-math.
Source, compiled-library, input and deterministic-initializer receipts are under
`tmp/research_20261012/error_mechanisms/gibbs_peers/`. The tracked
[Gibbs receipt](all_discrete_peer_20261012/gibbs_receipt.json) preserves the
fixed design, transition proof, diagnostic results and artifact identities.

These outcomes reject the two tested numerical approaches at their fixed
budgets. They do not measure a taxi-out gain or show that full discrete
conditioning would improve predictions if it could be computed accurately.
