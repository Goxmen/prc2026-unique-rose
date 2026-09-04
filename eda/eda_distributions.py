"""Light EDA: feature & target distributions for the PRC 2026 taxi-out challenge.

Reads the mirrored training parquets in data/ and writes figures to eda/figures/.
Run: .venv/bin/python eda/eda_distributions.py
"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "eda" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
T = f"read_parquet('{ROOT}/data/training_2025-*.parquet')"
DEP = f"{T} WHERE PHASE_mvt = 'DEP'"

con = duckdb.connect()

# palette / chrome (dataviz reference palette, light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
BLUE_LIGHT = "#9ec5f4"  # sequential step 200

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


# ---------------------------------------------------------------- 1. target
mins = con.sql(
    f"SELECT TAXITIME_SEC_mvt/60.0 AS m FROM {DEP} AND TAXITIME_SEC_mvt > 0"
).df()["m"].to_numpy()
med, p99 = np.median(mins), np.percentile(mins, 99)
clipped = (mins > 60).mean()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(np.clip(mins, 0, 60), bins=120, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
ax.axvline(med, color=INK, linewidth=1, linestyle=(0, (4, 3)))
ax.text(med + 0.7, ax.get_ylim()[1] * 0.95, f"median {med:.1f} min",
        color=INK, fontsize=9, va="top")
ax.axvline(p99, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
ax.text(p99 + 0.7, ax.get_ylim()[1] * 0.95, f"p99 {p99:.0f} min",
        color=INK2, fontsize=9, va="top")
ax.set_title("Taxi-out time, all 2025 departures")
ax.set_xlabel(f"minutes (right-clipped at 60; {clipped:.2%} of flights beyond)")
ax.set_ylabel("flights")
ax.set_xlim(0, 61)
style(ax)
save(fig, "01_taxiout_hist.png")

# ------------------------------------------------------- 2. target x airport
ap = con.sql(f"""
    SELECT ADEP_mvt AS airport,
           quantile_cont(TAXITIME_SEC_mvt, 0.10)/60 p10,
           quantile_cont(TAXITIME_SEC_mvt, 0.25)/60 p25,
           quantile_cont(TAXITIME_SEC_mvt, 0.50)/60 med,
           quantile_cont(TAXITIME_SEC_mvt, 0.75)/60 p75,
           quantile_cont(TAXITIME_SEC_mvt, 0.90)/60 p90,
           count(*) n
    FROM {DEP} AND TAXITIME_SEC_mvt > 0
    GROUP BY 1 ORDER BY med
""").df()

fig, ax = plt.subplots(figsize=(8, 4.5))
y = np.arange(len(ap))
ax.barh(y, ap.p90 - ap.p10, left=ap.p10, height=0.32, color=BLUE_LIGHT)
ax.barh(y, ap.p75 - ap.p25, left=ap.p25, height=0.32, color=BLUE)
ax.scatter(ap.med, y, color=INK, s=42, zorder=3, marker="|", linewidths=1.6)
for i, r in ap.iterrows():
    ax.text(r.p90 + 0.4, i, f"{r.med:.0f}", color=INK2, fontsize=9, va="center")
ax.set_yticks(y, ap.airport, color=INK2)
ax.set_title("Taxi-out by airport — p10–p90 band, p25–p75 box, median tick (labelled, min)")
ax.set_xlabel("minutes")
ax.set_xlim(0, None)
style(ax, x_grid=True, y_grid=False)
save(fig, "02_taxiout_by_airport.png")

# ------------------------------------------------- 3. hour-of-day seasonality
hr = con.sql(f"""
    SELECT hour(BLOCK_TIME_UTC_mvt) h,
           quantile_cont(TAXITIME_SEC_mvt, 0.25)/60 p25,
           quantile_cont(TAXITIME_SEC_mvt, 0.50)/60 med,
           quantile_cont(TAXITIME_SEC_mvt, 0.75)/60 p75,
           count(*) n
    FROM {DEP} AND TAXITIME_SEC_mvt > 0
    GROUP BY 1 ORDER BY 1
""").df()

fig, (ax, axn) = plt.subplots(
    2, 1, figsize=(8, 5), sharex=True, height_ratios=[2.2, 1])
ax.fill_between(hr.h, hr.p25, hr.p75, color=BLUE_LIGHT, alpha=0.6, linewidth=0)
ax.plot(hr.h, hr.med, color=BLUE, linewidth=2)
ax.set_title("Taxi-out by off-block hour (UTC) — median line, IQR band")
ax.set_ylabel("minutes")
ax.set_ylim(0, None)
style(ax)
axn.bar(hr.h, hr.n / 1000, color=BLUE, width=0.8)
axn.set_title("Departures per hour", fontsize=10)
axn.set_ylabel("thousand")
axn.set_xlabel("hour of day (UTC)")
axn.set_xticks(range(0, 24, 3))
style(axn)
save(fig, "03_taxiout_by_hour.png")

# ---------------------------------------------------------- 4. month-of-year
mo = con.sql(f"""
    SELECT month(BLOCK_TIME_UTC_mvt) mth,
           quantile_cont(TAXITIME_SEC_mvt, 0.25)/60 p25,
           quantile_cont(TAXITIME_SEC_mvt, 0.50)/60 med,
           quantile_cont(TAXITIME_SEC_mvt, 0.75)/60 p75,
           count(*) n
    FROM {DEP} AND TAXITIME_SEC_mvt > 0
    GROUP BY 1 ORDER BY 1
""").df()
labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, (ax, axn) = plt.subplots(
    2, 1, figsize=(8, 5), sharex=True, height_ratios=[2.2, 1])
ax.fill_between(mo.mth, mo.p25, mo.p75, color=BLUE_LIGHT, alpha=0.6, linewidth=0)
ax.plot(mo.mth, mo.med, color=BLUE, linewidth=2)
ax.set_title("Taxi-out by month (2025) — median line, IQR band")
ax.set_ylabel("minutes")
ax.set_ylim(0, None)
style(ax)
axn.bar(mo.mth, mo.n / 1000, color=BLUE, width=0.7)
axn.set_title("Departures per month", fontsize=10)
axn.set_ylabel("thousand")
axn.set_xticks(mo.mth, labels)
style(axn)
save(fig, "04_taxiout_by_month.png")

# ------------------------------------------------------ 5. categorical mix
seg = con.sql(f"""
    SELECT coalesce(MARKET_SEGMENT_flt, '(no flight plan)') k, count(*) n
    FROM {DEP} GROUP BY 1 ORDER BY n
""").df()
wk = con.sql(f"""
    SELECT coalesce(WK_TBL_CAT_flt, '(missing)') k, count(*) n
    FROM {DEP} GROUP BY 1 ORDER BY n
""").df()
act = con.sql(f"""
    SELECT AIRCRAFT_TYPE_mvt k, count(*) n
    FROM {DEP} AND AIRCRAFT_TYPE_mvt IS NOT NULL
    GROUP BY 1 ORDER BY n DESC LIMIT 15
""").df().iloc[::-1]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
for a, (df, title) in zip(axes, [
        (seg, "Market segment"),
        (wk, "Wake category"),
        (act, "Aircraft type (top 15 of 269)")]):
    a.barh(df.k, df.n / 1000, color=BLUE, height=0.55)
    a.set_title(title, fontsize=11)
    a.set_xlabel("thousand departures")
    a.tick_params(axis="y", labelcolor=INK2)
    style(a, x_grid=True, y_grid=False)
fig.tight_layout(w_pad=2.5)
save(fig, "05_categorical_mix.png")

# ---------------------------------- 6. taxi-out by categorical value (signal)
op = con.sql(f"""
    SELECT ADEP_mvt || ' · ' || RUNWAY_mvt k, count(*) n,
           quantile_cont(TAXITIME_SEC_mvt, 0.5)/60 med
    FROM {DEP} AND TAXITIME_SEC_mvt > 0
    GROUP BY 1 HAVING n > 20000 ORDER BY med DESC LIMIT 18
""").df().iloc[::-1]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(op.k, op.med, color=BLUE, height=0.55)
for i, r in op.reset_index().iterrows():
    ax.text(r.med + 0.15, i, f"{r.med:.1f}", color=INK2, fontsize=8.5, va="center")
ax.set_title("Median taxi-out by airport · runway (busy runways, >20k departures)")
ax.set_xlabel("minutes")
ax.tick_params(axis="y", labelcolor=INK2)
style(ax, x_grid=True, y_grid=False)
save(fig, "06_taxiout_by_runway.png")

print("done")
