"""Plot the 10,000 largest observed taxi-out times, without a target ceiling.

Run: .venv/bin/python eda/plot_top_taxi_out.py
Figures: eda/figures/top_10000_taxi_out.{png,pdf}
Selected rows, including rank and source: tmp/eda/top_10000_taxi_out.parquet
"""

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOP_N = 10_000
TARGET = "TAXITIME_SEC_mvt"


def main():
    files = sorted((ROOT / "data").glob("training_2025-*.parquet"))
    if len(files) != 12:
        raise ValueError(f"Expected all 12 training months; found {len(files)}")

    with duckdb.connect() as con:
        con.read_parquet([str(path) for path in files], filename=True).create_view(
            "training"
        )
        departures, labeled, maximum, above_200k = con.sql(f"""
            SELECT count(*), count({TARGET}), max({TARGET}),
                   count(*) FILTER (WHERE {TARGET} > 200000)
            FROM training
            WHERE PHASE_mvt = 'DEP'
        """).fetchone()
        rows = con.sql(f"""
            SELECT * FROM training
            WHERE PHASE_mvt = 'DEP' AND {TARGET} IS NOT NULL
            ORDER BY {TARGET} DESC, MVT_ID_mvt, filename
            LIMIT {TOP_N}
        """).df()

    values = rows[TARGET].to_numpy()
    if len(rows) != TOP_N or not np.all(values > 0):
        raise ValueError("Expected 10,000 positive taxi-out times for both scales")
    if not np.all(values[:-1] >= values[1:]) or values[0] != maximum:
        raise ValueError("Selected rows must be sorted and include the global maximum")

    ranks = np.arange(1, len(rows) + 1)
    rows.insert(0, "taxi_out_rank", ranks)
    rows = rows.rename(columns={"filename": "source_file"})
    rows["source_file"] = rows["source_file"].map(lambda name: Path(name).name)
    table_path = ROOT / "tmp" / "eda" / "top_10000_taxi_out.parquet"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(table_path, index=False)

    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "font.family": "sans-serif",
        "text.color": "#0b0b0b",
        "axes.labelcolor": "#52514e",
        "axes.edgecolor": "#c3c2b7",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "font.size": 11,
    })
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.13, hspace=0.30)
    fig.suptitle("The 10,000 longest taxi-outs", x=0.10, y=0.97,
                 ha="left", fontsize=23, weight="bold")
    fig.text(0.10, 0.915,
             f"All 2025 training departures · one bar per row · sorted longest first\n"
             f"Largest {TOP_N / labeled:.2%} of {labeled:,} labeled departures · "
             f"range: {values[-1]:,}–{values[0]:,} seconds",
             color="#52514e", fontsize=11, linespacing=1.6, va="top")

    comma = FuncFormatter(lambda value, _: f"{value:,.0f}")
    for ax, title in zip(axes, ["Linear scale", "Same 10,000 rows · logarithmic scale"]):
        ax.bar(ranks, values, width=1.0, color="#2a78d6", linewidth=0,
               rasterized=True)
        # Keep subpixel-width bars visible in the exported image.
        ax.step(ranks, values, where="mid", color="#2a78d6", linewidth=0.5)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=12)
        ax.set_xlim(-60, TOP_N + 20)
        ax.set_ylabel("Taxi-out time (seconds)")
        ax.yaxis.set_major_formatter(comma)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.tick_params(length=0, pad=7)

    axes[0].set_ylim(0, values[0] * 1.10)
    axes[0].yaxis.set_major_locator(MultipleLocator(40_000))
    axes[0].annotate(
        f"Maximum: {values[0]:,} s ({values[0] / 3600:.2f} h)",
        xy=(1, values[0]), xytext=(1300, values[0] * 0.94),
        fontsize=11, color="#52514e",
        arrowprops={"arrowstyle": "-", "color": "#898781", "lw": 0.8},
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(2_000, values[0] * 1.20)
    axes[1].set_yticks([3_000, 5_000, 10_000, 30_000, 100_000])
    axes[1].yaxis.set_major_formatter(comma)
    axes[1].minorticks_off()
    axes[1].set_xticks([1, 2_000, 4_000, 6_000, 8_000, 10_000])
    axes[1].xaxis.set_major_formatter(comma)
    axes[1].set_xlabel("Departure rank by taxi-out time (1 = longest)", labelpad=10)
    fig.text(0.10, 0.035,
             "Source: 12 original training parquets · PHASE_mvt = DEP · "
             "no upper target filter or clipping\n"
             "Bars touch at this density. Both panels contain the same individual rows.",
             fontsize=10, color="#52514e", linespacing=1.6)

    figures = ROOT / "eda" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = figures / f"top_10000_taxi_out.{extension}"
        fig.savefig(path, dpi=180)
        print(f"Wrote {path.relative_to(ROOT)}")
    plt.close(fig)
    print(f"Wrote {table_path.relative_to(ROOT)}")
    print(f"Training departures: {departures:,}; labeled: {labeled:,}")
    print(f"Selected rows: {len(rows):,}; range: {values[-1]:,}–{values[0]:,} s")
    print(f"Training departures above 200,000 seconds: {above_200k:,}")


if __name__ == "__main__":
    main()
