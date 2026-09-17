# Stage 1 laptop execution settings

This session runs on an **Intel Core i7-13650HX**, under WSL with 20 exposed
logical CPUs and approximately 11.5 GiB RAM. The previous Stage 0 measurements
came from a Ryzen 7 7800X3D desktop and were not assumed to transfer.

The local benchmark used the full January–June training matrix (1,005,290
retained rows), seed 42, a 20-round warm-up, and repeated 300-round fits in
forward/reverse thread order. The two fastest settings then ran 800 rounds.

| Threads | 300-round measurements (seconds) | 800-round fit (seconds) |
|---:|---|---:|
| 4 | 14.87, 14.94, 14.79 | — |
| 6 | 14.25, 13.38 | 34.80 |
| 8 | 14.14, 14.07 | 34.99 |
| 12 | 15.29, 14.28 | — |
| 20 | Excluded: initial warm-up rounds took roughly 10–14 seconds each | — |

The extra four-thread measurement re-established the reference after stopping
and restarting the probe to exclude 20 threads. The observed full-fit timings
at six/eight threads are essentially tied. Select **`--fit-workers 1 --threads
6`**, reserving resources without a measured throughput penalty.

Probe predictions differed by at most `9.1e-13` seconds at 300 rounds and
`4.6e-13` seconds at 800 rounds. Round counts, seeds, data, label filters and
model parameters remain fixed between the recovery variants and their control.
The largest measured process-lifetime RSS was 6,616 MiB; this includes loading
and matrix preparation and is not an isolated per-fit memory measurement.

During the full baseline, overlapping a full-data recovery audit with matrix
construction caused approximately 553 MiB swap use. The audit was stopped with
completed monthly results preserved. **Run full-data audits and experiments
sequentially on this WSL memory allocation**, as well as keeping seed fits
sequential. Completed fits remain valid; contention affects throughput. Later
fit times and actual thread settings remain in each run's model records.

## Prediction has a separate thread setting

Inspection of the installed LightGBM code showed that `Booster.predict()`
receives a fresh parameter dictionary and does not inherit the fit's thread
budget. On a saved M1/42 model and sampled July feature matrix, predictions were
**bit-for-bit equal**: automatic threading took 0.179/0.156 seconds, explicit
six-thread prediction took 0.265/0.283 seconds, and explicit 20-thread prediction
took 0.158/0.153 seconds. Retain **20 prediction threads explicitly**, while
keeping training at six. The severe 20-thread training slowdown does not imply
the same behavior for inference.

The harness now accepts `--prediction-threads`; omission uses the training
budget. Candidate commands use `--threads 6 --prediction-threads 20`. The
completed control used the previous automatic prediction behavior.
[Prediction equivalence measurements](benchmarks/stage1_prediction_threads.json).

CPU and Python/LightGBM/NumPy/pandas/DuckDB identities are now saved in run
records. These measurements do not establish a universal default for another
machine or power/thermal state.

## Completed full benchmarks

All four runs completed 18 fits and 16 validation folds. Run time includes
loading, feature construction, prediction and paired reporting; standalone
audits and plots are separate.

| Run | Total minutes | Recorded median fit seconds |
|---|---:|---:|
| B0-800 | 18.775 | 36.30 |
| R1 exact | 19.062 | 34.80 |
| R2 relaxed | 25.343 | 38.55 |
| R3 history | 25.569 | 34.85 |

R3's recorded fit times range from 32.3 to 35.9 seconds. The later recovery
layers add preprocessing work; their total run time is not evidence of a
training-thread slowdown.

[Raw benchmark](benchmarks/stage1_i7_13650hx.json). The reproducible tool is
`VALIDATION/tools/benchmark_cpu.py`; it does not fit a full-year model or upload.
