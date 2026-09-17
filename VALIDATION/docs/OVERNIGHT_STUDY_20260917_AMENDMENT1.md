# Pre-unblinding amendment: transfer changes in squared error

Recorded 2026-09-17 while local validation is still running, **before any of the
five new submissions has been uploaded or any new official outcome observed**.
The original preregistration remains unchanged and its hash is preserved. No
model recipe, candidate role, submission order, fitting setting or quota changes.

## Reason

RMSE is nonlinear. Transferring a local RMSE difference directly to an official
score with a different starting level assumes that RMSE changes transfer. A
different, equally simple hypothesis is that **mean squared error changes**
transfer, while the baseline error floor differs between the populations.

Let `L0` be the observed Q0-4h LB RMSE, `R0` the matching control CV RMSE, and
`Rc` a candidate's CV RMSE. The original fixed forecast is:

```text
RMSE-change transport: forecast = L0 + Rc - R0
```

Add the following fixed alternative:

```text
MSE-change transport: forecast = sqrt(L0² + Rc² - R0²)
```

The radicand must be nonnegative. An impossible negative value is rejected,
not silently clipped or relabelled a valid forecast. For small changes, MSE
transport rescales the CV RMSE change approximately by `R0 / L0`.

This is a calibration hypothesis, not a change to local CV scoring or to flight
predictions. It cannot repair a wrong candidate ordering within a CV readout:
the transformation is monotone. It may address improvement-size errors arising
from a different overall error level. Its success is not assumed.

## Selection and confirmation

The ten already specified CV readouts/blends each receive an MSE-transport
counterpart, denoted `<readout>__mse`, for **twenty fixed forecast policies**.
The original policies retain their names and fixed order; MSE counterparts follow
in the same order. No slopes, per-fold weights or new free parameters are fitted.

The primary criterion is unchanged: mean absolute anchored change-prediction error
over four historical alternatives plus the three discovery candidates. Preserve
the original `R_combined` RMSE-change policy when it is within one second of the
best calibration error. The selected **readout** and **transport form** are
recorded separately. Raw CV changes and forecasted LB changes must be labelled
separately; transformed forecasts are not reported as raw CV RMSE.

All twenty policies' predictions for the same two confirmation models are frozen
before their official outcomes are read. Only the selected policy versus the
unchanged original policy is confirmatory. Other results remain descriptive.
The family-level sensitivity check is retained. With twenty candidates and only
two confirmation cases, even a positive result remains limited evidence.

The final report will explicitly disclose this amendment's timing and rationale.
Original model training source and already-completed fits are unaffected.
