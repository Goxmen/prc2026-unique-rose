# Source-only scout exclusions

The proposed one-dimensional characteristic-function/FFT treatment overlaps
existing deterministic all-long-peer messages in `daily_delay_posterior.py`.
The selected daily branch already uses forward/backward convolution on a
60-second grid. `daily_grid_interpolation.py` preserves continuous-state first
moments by allocating mass to adjacent nodes; its completed v76 screen changed
CV by +0.0011 seconds, interval [−0.0032,+0.0058], and was closed.
The only FFT already in the joint-peer code computes Gibbs autocorrelation;
there is no continuous Fourier posterior implementation. Nevertheless a
departure-only Fourier pilot would supply numerical infrastructure, not new
information or a currently justified score hypothesis. Root declined it.
No pilot, fit, tuning, candidate or upload was run.

Any future continuous check of the retained all-discrete joint pilot must
preserve its actual residual distribution: one t5 with scale covariance
0.6×remainder_cov, which already incorporates ordinary, common-day and source
uncertainty. Convolving an additional Gaussian with a t5 would change this
model and double-count some uncertainty. The old daily posterior uses a
different separated ordinary-offset/t3 likelihood. These are not equivalent
controls, despite sharing finite long-flight supports.

Cached old/final jitter inspection also gives no strong new lead. All 344,841
old departures overlap the final view; 316,651 movement clocks differ, by at
most12 seconds. Supplied schedule and all six flight-plan clocks are identical.
Among344,693 overlapping arrivals,330,721 supplied taxi labels differ, by up to
24 seconds; movement-clock and block-clock changes correlate −0.9896965721.
Thus naïve averaging across these views is not a label-preserving augmentation.
Only supplied arrival labels were inspected; hidden departure labels remain
null. The timing scale alone does not establish a multi-second opportunity.
No augmentation, model execution or prediction modification followed.

The distinct spatial-peer motion scout uses other aircraft's observed raw
ground positions/speeds, absent from the retained movement-count queue and
static OSM route inputs. Its frozen source-only pilot and negative result are
in `spatial_peer_motion_20261012/`. No closed clock/weight/covariance/model-loss
screen was reopened.

The proposed source-only hierarchical Rome copy-odds calibration overlaps
`VALIDATION/rome_public_monthly_shift.py`: that completed screen already fits
regularized airline/month logit shifts from other dates' public aggregate
likelihood with fixed source bias/noise and no individual departure labels.
Its exact-v76 change was −0.0507 seconds, interval [−0.3466,+0.2002], with July
worse (+0.0900); see `TARGET215_RESEARCH.md` lines 1374–1405 and 1505–1535 and
`VALIDATION/runs/v3_20261011_rome_public_monthly_shift/`. Leaving a query's own
date out improves likelihood separation but adds no new information. Source
population bias can imitate an odds shift; the selected support's subsequent
mean-preserving tilt also prevents interpreting a classifier logit adjustment
as an isolated latent copy-prior change. This proposal is closed as overlap;
no new fit, taxi-CV read, threshold scan or submission followed.
