"""Compare feature distributions: 2025 training vs Jan/Jul-2026 ranking departures.

Writes figures to eda/figures/. Run: .venv/bin/python eda/eda_train_vs_ranking.py
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "eda" / "figures"
T = f"read_parquet('{ROOT}/data/training_2025-*.parquet') WHERE PHASE_mvt='DEP'"
R = f"read_parquet('{ROOT}/data/ranking.parquet') WHERE PHASE_mvt='DEP'"

con = duckdb.connect()
con.sql("SET TimeZone='UTC'")

# dataviz reference palette, light mode; 2 series: blue = train, orange = ranking
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9.5,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "figure.dpi": 150,
})


def style(ax, *, x_grid=False, y_grid=True):
    ax.grid(axis="x", visible=x_grid)
    ax.grid(axis="y", visible=y_grid)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote eda/figures/{name}")


def share_by(expr, where_extra=""):
    q = lambda src: con.sql(
        f"SELECT {expr} k, 100.0*count(*)/sum(count(*)) OVER () pct "
        f"FROM {src} {where_extra} GROUP BY 1"
    ).df()
    t, r = q(T), q(R)
    return t.merge(r, on="k", how="outer", suffixes=("_t", "_r")).fillna(0)

# ------------------------------------------------- 1. airport & aircraft mix
ap = share_by("ADEP_mvt").sort_values("pct_t")
act = share_by("coalesce(AIRCRAFT_TYPE_mvt,'(missing)')")
top = act.sort_values("pct_t", ascending=False).head(15).iloc[::-1]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5))
for ax, df, title in [(a1, ap, "Share of departures by airport"),
                      (a2, top, "— by aircraft type (top 15)")]:
    y = np.arange(len(df))
    ax.barh(y + 0.19, df.pct_t, height=0.34, color=BLUE, label="training 2025")
    ax.barh(y - 0.19, df.pct_r, height=0.34, color=ORANGE, label="ranking 2026")
    ax.set_yticks(y, df.k, color=INK2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("% of departures")
    style(ax, x_grid=True, y_grid=False)
a1.legend(loc="lower right")
fig.tight_layout(w_pad=2.5)
save(fig, "07_mix_train_vs_ranking.png")

# --------------------------------------------------------- 2. hour-of-day mix
hr = share_by("hour(MVT_TIME_UTC_mvt)").sort_values("k")
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hr.k, hr.pct_t, color=BLUE, linewidth=2, label="training 2025")
ax.plot(hr.k, hr.pct_r, color=ORANGE, linewidth=2, label="ranking 2026")
ax.set_title("Share of departures by takeoff hour (UTC)")
ax.set_xlabel("hour of day (UTC)")
ax.set_ylabel("% of departures")
ax.set_xticks(range(0, 24, 3))
ax.set_ylim(0, None)
ax.legend()
style(ax)
save(fig, "08_hour_mix_train_vs_ranking.png")

# ------------------------- 3. implied taxi-out (MVT_TIME − AOBT_3) vs actual
q_hist = lambda src, expr, extra: con.sql(f"""
    SELECT least(floor(({expr})*2)/2.0, 60.0) bin_min,
           100.0*count(*)/sum(count(*)) OVER () pct
    FROM {src} AND {extra} GROUP BY 1 ORDER BY 1""").df()
tr = q_hist(T, "TAXITIME_SEC_mvt/60.0", "TAXITIME_SEC_mvt > 0")
tr_i = q_hist(T, "epoch(MVT_TIME_UTC_mvt - AOBT_3_flt)/60.0",
              "AOBT_3_flt IS NOT NULL AND MVT_TIME_UTC_mvt > AOBT_3_flt")
rk_i = q_hist(R, "epoch(MVT_TIME_UTC_mvt - AOBT_3_flt)/60.0",
              "AOBT_3_flt IS NOT NULL AND MVT_TIME_UTC_mvt > AOBT_3_flt")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(tr.bin_min, tr.pct, color=BLUE, linewidth=2, label="actual taxi-out (training)")
ax.plot(tr_i.bin_min, tr_i.pct, color=BLUE, linewidth=1.4,
        linestyle=(0, (4, 3)), label="implied MVT−AOBT₃ (training)")
ax.plot(rk_i.bin_min, rk_i.pct, color=ORANGE, linewidth=2,
        label="implied MVT−AOBT₃ (ranking)")
ax.set_title("Taxi-out distribution: actual vs implied from flight plan")
ax.set_xlabel("minutes (30 s bins, right-clipped at 60)")
ax.set_ylabel("% of departures")
ax.set_xlim(0, 61)
ax.set_ylim(0, None)
ax.legend()
style(ax)
save(fig, "09_implied_taxi_train_vs_ranking.png")

print("done")
