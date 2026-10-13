# Third-party software and research acknowledgements

Team unique-rose's source code is distributed under [GNU GPLv3](../LICENSE).
That license does not replace dependencies' own notices or data-source terms.
Dependencies are installed from their upstream distributions; preserve their
packaged license/NOTICE files if redistributing an environment or binary bundle.
Pinned package versions are in [requirements.txt](../requirements.txt).

## Software

| Project | Use here | Upstream license/reference |
|---|---|---|
| [LightGBM](https://github.com/lightgbm-org/LightGBM) | Global and per-airport boosted regression; historical classifier variants. | [MIT](https://github.com/lightgbm-org/LightGBM/blob/master/LICENSE) |
| [CatBoost](https://github.com/catboost/catboost) | Regression and Rome recording-mode classification. | [Apache 2.0](https://github.com/catboost/catboost/blob/master/LICENSE) |
| [PyTorch](https://github.com/pytorch/pytorch) | Neural training, tensor operations, track encoders and checkpointing. | [Upstream license and bundled third-party notices](https://github.com/pytorch/pytorch/blob/main/LICENSE); binary distributions contain multiple licenses. |
| [TabM, Yandex Research](https://github.com/yandex-research/tabm) | `EnsembleView`, TabM backbone and linear ensemble layers; our model adds competition-specific tabular/physical-track inputs and training/inference integration. | [Apache 2.0](https://github.com/yandex-research/tabm/blob/main/LICENSE) |
| [RTDL numerical embeddings, Yandex Research](https://github.com/yandex-research/rtdl-num-embeddings) | Piecewise-linear numerical embeddings and numerical feature representation. | [Apache 2.0](https://github.com/yandex-research/rtdl-num-embeddings/blob/main/LICENSE) |
| [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [SciPy](https://scipy.org/) | Arrays, tabular feature construction, optimization, numerical integration and statistical checks. | BSD-family core licenses; installed distributions preserve additional bundled notices. |
| [Apache Arrow / PyArrow](https://arrow.apache.org/) | Parquet schemas, columnar reads/writes and input/output checks. | Apache 2.0. |
| [DuckDB](https://duckdb.org/) | Independent SQL feature/aggregation audits and joins. | [MIT](https://github.com/duckdb/duckdb/blob/main/LICENSE). |
| [XGBoost](https://github.com/dmlc/xgboost) | Retained exploratory comparisons; not one of the three new tree components. | Apache 2.0. |
| Matplotlib, tqdm, threadpoolctl | Figures, progress reporting and runtime thread limits. | Matplotlib's project license; tqdm's MPL-2.0/MIT terms; threadpoolctl's BSD-3-Clause. Consult each installed distribution. |
| ipykernel, nbclient, nbconvert | Notebook execution and reports. | Jupyter BSD-family project licenses; consult installed distributions. |
| [MinIO client](https://github.com/minio/mc), [Git](https://git-scm.com/), [GitHub CLI](https://cli.github.com/), curl | External command-line acquisition/versioning tools. | Separately installed tools retain their own licenses and are not vendored into this repository. |

This is an attribution inventory for the pinned direct dependencies, not a
replacement for the full license metadata of every transitive package or CUDA
runtime. The release environment capture records the exact installed versions.

## Data and external ideas

Data-provider attribution and transformation notices are in
[DATA_SOURCES.md](DATA_SOURCES.md), including EUROCONTROL, OpenSky Network,
ADSB.lol contributors, OpenStreetMap contributors, X-Plane Scenery Gateway
artists, Iowa Environmental Mesonet, Open-Meteo and ERA5 providers.

The idea of using EUROCONTROL aggregate pre-departure-delay totals was informed
by the public documentation of
[team gracious-tornado](https://github.com/il-miscusi/prc-2026-gracious-tornado),
specifically commit `a0344440b497c38bc80390a61b3e7890904b3634`. Our retained
research record states that no public source code was copied for that method.
The implementation here uses its own flight-population checks, discrete
recording-mode prior, robust measurement likelihood, calibration and posterior
inference. Attribution to the idea does not imply that the other team endorsed
this implementation or that their reported permissions cover our source uses.

The neural numerical-embedding and TabM architectures are upstream research
contributions. We use their published implementations as dependencies rather
than claiming to have invented those architectures. Original work in this
repository includes the competition-specific clock/recording-mode analysis,
fold-scoped feature construction, aircraft/track matching and quality features,
physical-track integration, Rome mixture, public-source uncertainty corrections,
portable ensemble reconstruction/distillation, artifact identity checks and
independent SQL/numerical audits. The method is a combination and adaptation of
these contributions, not merely renamed inputs and outputs of another team's
submitted solution.

Historical exploratory scripts are retained for transparency and reproducibility.
They are not all part of the accepted model. Refer to [METHOD.md](METHOD.md) and
the selected recipe's manifests to determine the actual execution path.

## Unexecuted TabICL source review

The separate source-only feasibility review examined
[Soda/Inria TabICL](https://github.com/soda-inria/tabicl/tree/c91f00df184a5097e584cb376f7e699e9f3444c4)
at commit `c91f00df184a5097e584cb376f7e699e9f3444c4`. It is not an input to the
accepted final or the ten-candidate selection suite. Its
[exact upstream license](https://github.com/soda-inria/tabicl/blob/c91f00df184a5097e584cb376f7e699e9f3444c4/LICENSE)
places the core under BSD-3-Clause, copyright 2025 Soda team @ Inria, with a
separate Apache-2.0 notice for the forecast directory's derived code.

Downloaded upstream inspection copies are a local cache, not unique-rose's
original source. The retained `source_inventory.json` and
`independent_source_inventory.json` under
`VALIDATION/reports/tabicl_frontier_20261013/` provide the pinned original URLs,
relative paths and SHA-256 values for restoration. Fetch those exact URLs when
reconstructing the review cache and verify the recorded hashes; the files are
not necessary for accepted-final replay. If redistributing these upstream files,
include the complete license and preserve their attribution. The local wrapper
`VALIDATION/tabicl_frontier_20261013.py` and its prospective protocol do not
establish that model installation, checkpoint loading, inference or evaluation
was executed.
