"""Plot the 1,000 smallest training taxi-outs, retaining zero/negative labels.

Run: .venv/bin/python eda/plot_bottom_taxi_out.py
Figures: eda/figures/bottom_1000_taxi_out.{png,pdf}
Selected original rows: tmp/eda/bottom_1000_taxi_out.parquet
"""

from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOTTOM_N = 1_000
TARGET = "TAXITIME_SEC_mvt"
NEGATIVE, ZERO, POSITIVE = "#c75445", "#bf880d", "#2a78d6"


def main():
    files = sorted((ROOT / "data").glob("training_2025-*.parquet"))
    if len(files) != 12:
        raise ValueError(f"Expected all 12 training months; found {len(files)}")
    with duckdb.connect() as con:
        con.sql("SET TimeZone='UTC'")
        con.read_parquet([str(path) for path in files], filename=True).create_view("training")
        total, missing, negative, zero = con.sql(f"""
            SELECT count(*), count(*) FILTER (WHERE {TARGET} IS NULL),
                   count(*) FILTER (WHERE {TARGET} < 0),
                   count(*) FILTER (WHERE {TARGET} = 0)
            FROM training WHERE PHASE_mvt = 'DEP'
        """).fetchone()
        rows = con.sql(f"""
            SELECT * FROM training
            WHERE PHASE_mvt = 'DEP' AND {TARGET} IS NOT NULL
            ORDER BY {TARGET}, MVT_ID_mvt, filename LIMIT {BOTTOM_N}
        """).df()
    values = rows[TARGET].to_numpy()
    if (len(rows) != BOTTOM_N or not rows.MVT_ID_mvt.is_unique
            or not np.all(values[:-1] <= values[1:])):
        raise ValueError("Expected 1,000 distinct departures sorted by increasing taxi-out time")
    ranks = np.arange(1, len(rows) + 1)
    rows.insert(0, "taxi_out_rank_ascending", ranks)
    rows = rows.rename(columns={"filename": "source_file"})
    rows["source_file"] = rows["source_file"].map(lambda name: Path(name).name)
    table = ROOT / "tmp" / "eda" / "bottom_1000_taxi_out.parquet"
    table.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(table, index=False)

    counts = [int((values < 0).sum()), int((values == 0).sum()), int((values > 0).sum())]
    central = int((np.abs(values) <= 12).sum())
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb", "font.family": "sans-serif",
        "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
        "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True,
        "xtick.color": "#52514e", "ytick.color": "#52514e", "font.size": 11,
    })
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.13, hspace=0.52)
    fig.suptitle("The 1,000 shortest taxi-outs", x=0.09, y=0.97,
                 ha="left", fontsize=23, weight="bold")
    fig.text(0.09, 0.912,
             f"All 2025 training departures · zeros and negatives retained\n"
             f"Range: {values[0]} to {values[-1]} seconds · {central:,} of 1,000 rows "
             "are between −12 and +12 seconds",
             color="#52514e", fontsize=11, linespacing=1.6, va="top")

    ax = axes[0]
    colors = np.where(values < 0, NEGATIVE, np.where(values == 0, ZERO, POSITIVE))
    ax.bar(ranks, values, width=1.0, color=colors, linewidth=0, rasterized=True)
    ax.scatter(ranks[values == 0], values[values == 0], color=ZERO, marker="|", s=35, zorder=4)
    ax.axhline(0, color="#898781", linewidth=0.7, zorder=0)
    ax.set(title="Sorted rows · one bar per departure", xlabel="Rank (1 = smallest taxi-out time)",
           ylabel="Taxi-out time (seconds)", xlim=(-5, BOTTOM_N + 5),
           ylim=(min(0, values[0]) - 5, values[-1] + 8))
    ax.set_xticks([1, 200, 400, 600, 800, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.legend(handles=[Patch(color=color, label=f"{label}: {count:,} rows")
                       for color, label, count in zip([NEGATIVE, ZERO, POSITIVE],
                                                     ["Negative", "Exactly zero", "Positive"], counts)],
              loc="upper left", frameon=False)

    ax = axes[1]
    seconds, frequency = np.unique(values, return_counts=True)
    colors = np.where(seconds < 0, NEGATIVE, np.where(seconds == 0, ZERO, POSITIVE))
    ax.axvspan(-12.5, 12.5, color="#ebe7de", alpha=0.55, zorder=0)
    ax.bar(seconds, frequency, width=0.85, color=colors, linewidth=0)
    ax.set(title="Frequency at each exact second · same 1,000 rows",
           xlabel="Taxi-out time (seconds)", ylabel="Departures",
           xlim=(values[0] - 1.5, values[-1] + 1.5), ylim=(0, max(frequency) + 12))
    ax.set_xticks([-12, -6, 0, 6, 12, 20, 30, 40, 50, 53])
    ax.text(0, max(frequency) + 7, f"{central} rows within ±12 s", ha="center", color="#52514e")
    for ax in axes:
        ax.set_title(ax.get_title(), loc="left", fontsize=12, weight="bold", pad=12)
        ax.set_title("", loc="center")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.tick_params(length=0, pad=6)
    fig.text(0.09, 0.032,
             "Source: 12 original training parquets · PHASE_mvt = DEP · no target-value filter\n"
             "Zero-height rows are marked on the upper baseline. Ties are ordered by movement ID; "
             "the last second may be only partly included.",
             fontsize=9.5, color="#52514e", linespacing=1.6)
    output = ROOT / "eda" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = output / f"bottom_1000_taxi_out.{extension}"
        fig.savefig(path, dpi=180)
        print(f"Wrote {path.relative_to(ROOT)}")
    plt.close(fig)
    print(f"Wrote {table.relative_to(ROOT)}")
    print(f"All {total:,} departures: {missing} missing, {negative} negative, {zero} exactly zero")
    print(f"Bottom {len(rows):,}: {dict(zip(['negative', 'zero', 'positive'], counts))}")
    print(f"Median: {np.median(values):g} s; within ±12 s: {central}")
    print(rows.groupby("ADEP_mvt").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
