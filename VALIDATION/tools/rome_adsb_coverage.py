"""Is ADSB.lol's thin coverage of Rome's apron real, or an artefact of our pipeline?

Checks that read no block times or labels (Rome's off-block times are themselves suspect):
  1. Airport box: the heatmap kept positions within 20 km of each aerodrome reference point;
     LIRF's point is compared with the stand map's centre.
  2. Ground flag: the features use only positions flagged on-ground (alt == -123). Positions
     near the airport that are slow (< 40 kt) and low (<= 300 ft) but not flagged would be ground
     traffic the pipeline drops; they are counted per airport.
  3. Where ground positions are: share within 100 m of a mapped stand, and share of mapped
     stands with any ground position within 60 m, on one day per month (the 15th of each 2025
     month, 2026-01-15, 2026-07-15).
  4. Per-departure coverage from the full feature tables (all 2025 and ranking departures):
     matched by callsign, seen on the ground, distance of the first ground point to its own
     stand, speed at that first point, any point within 100 m of its own stand; and LIRF by month.
Writes VALIDATION/reports/rome_adsb_coverage.png and .json.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import adsblol
from download_adsblol_heatmap import AIRPORTS, GROUND
from records import write_json

OUT = ROOT / 'VALIDATION' / 'reports' / 'rome_adsb_coverage'
DAYS = [f'2025-{m:02d}-15' for m in range(1, 13)] + ['2026-01-15', '2026-07-15']
SURFACE_M, STAND_M, SEEN_M = 4000, 100, 60
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def xy(lat, lon, ref):
    """Metres east/north of `ref` (equirectangular; fine within a few km)."""
    k = np.cos(np.radians(ref[0]))
    return (lon - ref[1]) * 111_320 * k, (lat - ref[0]) * 110_540


def nearest(px, py, sx, sy, chunk=200_000):
    out = np.empty(len(px))
    for i in range(0, len(px), chunk):
        d = (px[i:i + chunk, None] - sx[None]) ** 2 + (py[i:i + chunk, None] - sy[None]) ** 2
        out[i:i + chunk] = np.sqrt(d.min(1))
    return out


def main():
    stands = pd.read_parquet(adsblol.sg.OUT_DIR / 'stand_coords.parquet').dropna(subset=['lat'])
    out = dict(days=DAYS, box={}, positions={}, departures={}, lirf_by_month={})
    for ap, ref in AIRPORTS.items():
        s = stands[stands.airport == ap]
        if len(s):
            cx, cy = xy(s.lat.mean(), s.lon.mean(), ref)
            out['box'][ap] = dict(stand_centre_from_reference_m=float(np.hypot(cx, cy)), stands=len(s))
    # 2-3: positions on the sample days
    acc, keep = {ap: dict(ground=0, unflagged_slow_low=0, ground_near_stand=0, ground_still=0) for ap in AIRPORTS}, {}
    seen = {ap: np.zeros((stands.airport == ap).sum(), bool) for ap in AIRPORTS}
    for day in DAYS:
        p = pd.read_parquet(adsblol.HEATMAP / 'positions' / f'{day}.parquet')
        for ap, ref in AIRPORTS.items():
            q = p[p.airport == ap]
            if not len(q):
                continue
            x, y = xy(q.lat.to_numpy() / 1e6, q.lon.to_numpy() / 1e6, ref)
            near = np.hypot(x, y) <= SURFACE_M
            alt, gs = q.alt.to_numpy(), q.gs.to_numpy()
            ground = near & (alt == GROUND)
            slow_low = near & (alt != GROUND) & (alt * 25 <= 300) & (gs >= 0) & (gs < 400)
            s = stands[stands.airport == ap]
            sx, sy = xy(s.lat.to_numpy(), s.lon.to_numpy(), ref)
            gd = nearest(x[ground], y[ground], sx, sy) if ground.any() else np.zeros(0)
            a = acc[ap]
            a['ground'] += int(ground.sum())
            a['unflagged_slow_low'] += int(slow_low.sum())
            a['ground_near_stand'] += int((gd <= STAND_M).sum())
            a['ground_still'] += int((ground & (gs >= 0) & (gs < 10)).sum())
            if ground.any():
                d2 = (sx[:, None] - x[ground][None]) ** 2 + (sy[:, None] - y[ground][None]) ** 2 if ground.sum() < 400_000 else None
                if d2 is not None:
                    seen[ap] |= np.sqrt(d2.min(1)) <= SEEN_M
            if ap in ('LIRF', 'LSZH'):
                keep.setdefault(ap, []).append(pd.DataFrame(dict(x=x[ground | slow_low], y=y[ground | slow_low],
                                                                 ground=ground[ground | slow_low])))
    for ap, a in acc.items():
        if a['ground']:
            out['positions'][ap] = dict(
                ground_per_day=a['ground'] / len(DAYS), unflagged_slow_low_vs_ground=a['unflagged_slow_low'] / a['ground'],
                share_ground_within_100m_of_stand=a['ground_near_stand'] / a['ground'],
                share_ground_stationary=a['ground_still'] / a['ground'],
                share_stands_seen_within_60m=float(seen[ap].mean()))
    # 4: per-departure coverage, all departures
    f = adsblol.load_features().join(adsblol.load_features(adsblol.STAND_PATH), how='left')
    f['month'] = pd.to_datetime(f.to_ms, unit='ms').dt.strftime('%Y-%m')
    seg = f.ab_seg_sec.notna()
    for ap, g in f.groupby('airport'):
        k = seg[g.index]
        out['departures'][ap] = dict(
            departures=len(g), matched=float(g.ab_matched.mean()), seen_on_ground=float(k.mean()),
            first_point_to_own_stand_median_m=float(g.ab_seg_stand_m[k].median()),
            first_point_speed_median_kt=float(g.ab_first_gs[k].median()),
            point_within_100m_of_own_stand=float((g.ab_stand_n[k] > 0).mean()))
    g = f[f.airport == 'LIRF']
    for m, h in g.groupby('month'):
        k = h.ab_seg_sec.notna()
        out['lirf_by_month'][m] = dict(departures=len(h), seen_on_ground=float(k.mean()),
                                       first_point_to_own_stand_median_m=float(h.ab_seg_stand_m[k].median()),
                                       point_within_100m_of_own_stand=float((h.ab_stand_n[k] > 0).mean()))
    write_json(OUT.with_suffix('.json'), out)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GRID,
                         'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'axes.titlecolor': INK, 'axes.titlesize': 10, 'axes.titleweight': 'bold'})
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), facecolor=BG)
    for a, (ap, title) in zip(ax[:2], (('LIRF', 'A. Rome: ground positions'), ('LSZH', 'B. Zurich: ground positions'))):
        d = pd.concat(keep[ap])
        d = d[d.ground]
        s = stands[stands.airport == ap]
        sx, sy = xy(s.lat.to_numpy(), s.lon.to_numpy(), AIRPORTS[ap])
        cx, cy = sx.mean(), sy.mean()
        hb = a.hexbin(d.x - cx, d.y - cy, gridsize=110, bins='log', cmap='Blues', mincnt=1,
                      extent=(-2500, 2500, -2500, 2500), linewidths=0)
        a.scatter(sx - cx, sy - cy, s=9, color=ORANGE, edgecolor=BG, linewidth=0.6, label='mapped stands', zorder=3)
        a.set(xlim=(-2500, 2500), ylim=(-2500, 2500), aspect='equal', xlabel='m east of stand centre',
              ylabel='m north of stand centre')
        a.set_title(f'{title} ({len(d):,} on {len(DAYS)} days)', loc='left')
        a.legend(frameon=False, loc='lower left', labelcolor=INK2)
        fig.colorbar(hb, ax=a, shrink=0.75, label='positions per cell (log)')
    a = ax[2]
    aps = [ap for ap in out['positions']]
    vals = [out['positions'][ap]['share_stands_seen_within_60m'] * 100 for ap in aps]
    order = np.argsort(vals)
    yy = np.arange(len(aps))
    a.barh(yy, np.array(vals)[order], height=0.6,
           color=[ORANGE if aps[i] == 'LIRF' else BLUE for i in order])
    for yi, i in zip(yy, order):
        a.text(vals[i] + 1, yi, f'{vals[i]:.0f}%', va='center', color=INK)
    a.set_yticks(yy, [aps[i] for i in order])
    a.set_xlim(0, 110)
    a.grid(axis='x', color=GRID, lw=0.6)
    a.set_axisbelow(True)
    for side in ('top', 'right'):
        a.spines[side].set_visible(False)
    a.set_xlabel(f'% of mapped stands with a ground position within {SEEN_M} m')
    a.set_title('C. Stands ever heard (14 sample days)', loc='left')
    for a in ax:
        a.set_facecolor(BG)
    fig.suptitle('ADSB.lol heatmap ground coverage: Rome vs Zurich', x=0.01, ha='left', fontsize=12,
                 fontweight='bold', color=INK)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix('.png'), dpi=150, facecolor=BG)
    print('box', {k: round(v['stand_centre_from_reference_m']) for k, v in out['box'].items()})
    print(pd.DataFrame(out['positions']).T.round(3).to_string())
    print(pd.DataFrame(out['departures']).T.round(3).to_string())
    print(pd.DataFrame(out['lirf_by_month']).T.round(3).to_string())


if __name__ == '__main__':
    main()
