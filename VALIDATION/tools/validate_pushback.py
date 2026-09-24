"""Score the ADSB.lol pushback extractor against recorded off-block times.

Zurich is the calibration airport: labels are clean and OPDI reaches roughly 56%
within two minutes there, which is the bar a new extractor has to clear before
any bulk download is justified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adsb_fetch import cache_path
from adsb_geometry import AirportGround
from adsb_pushback import extract, load_trace


def run(resolved: Path, icao: str, days: list[str], **kw) -> pd.DataFrame:
    df = pd.read_parquet(resolved)
    df["dep_date"] = df.dep_date.astype(str)
    if days:
        df = df[df.dep_date.isin(days)]
    df = df[df.ADEP_mvt == icao].copy()
    ground = AirportGround(icao)

    rows = []
    for (icao24, day), grp in df.groupby(["icao24", "dep_date"], sort=False):
        path = cache_path(icao24, day)
        if not path.exists():
            continue
        try:
            trace = load_trace(path)
        except (OSError, ValueError):
            continue
        if len(trace[0]) == 0:
            continue
        for r in grp.itertuples():
            takeoff = r.takeoff.timestamp()
            p = extract(*trace, takeoff, ground, stand_ref=r.STAND_mvt, **kw)
            rows.append({
                "MVT_ID_mvt": r.MVT_ID_mvt, "icao24": icao24, "dep_date": day,
                "stand": r.STAND_mvt, "taxi_true": r.taxi_sec,
                "offblock_true": r.offblock.timestamp(), "takeoff": takeoff,
                "offblock_est": p.offblock, "method": p.method,
                "park_points": p.park_points, "on_apron": p.on_apron,
                "own_stand": p.own_stand, "ground_points": p.ground_points,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["err"] = out.offblock_est - out.offblock_true
    out["taxi_est"] = out.takeoff - out.offblock_est
    return out


def report(out: pd.DataFrame, label: str = "") -> dict:
    ok = out[out.offblock_est.notna()]
    n, m = len(out), len(ok)
    if m == 0:
        print(f"{label} no estimates"); return {}
    e = ok.err.to_numpy()
    stats = {
        "flights": n, "estimated": m, "coverage_pct": 100 * m / n,
        "within_2min_pct": 100 * (np.abs(e) <= 120).mean(),
        "within_5min_pct": 100 * (np.abs(e) <= 300).mean(),
        "median_err_s": float(np.median(e)),
        "mae_s": float(np.abs(e).mean()),
        "corr_taxi": float(np.corrcoef(ok.taxi_est, ok.taxi_true)[0, 1]),
    }
    if label:
        print(f"--- {label} ---")
    for k, v in stats.items():
        print(f"  {k:18} {v:10.2f}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resolved", type=Path, required=True)
    ap.add_argument("--airport", default="LSZH")
    ap.add_argument("--days", nargs="*", default=[])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    if args.sweep:
        for min_park in (300, 600, 1200):
            for stationary in (40.0, 60.0, 100.0):
                out = run(args.resolved, args.airport, args.days,
                          min_park_s=min_park, stationary_m=stationary)
                report(out, f"min_park={min_park}s stationary={stationary:.0f}m")
        return

    out = run(args.resolved, args.airport, args.days)
    report(out, args.airport)
    print("\nby method:")
    print(out.groupby("method").agg(
        n=("err", "size"),
        within_2min=("err", lambda s: 100 * (s.abs() <= 120).mean()),
        median_err=("err", "median"),
    ).round(1).to_string())
    if args.out:
        out.to_parquet(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
