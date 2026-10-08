"""Cached Open-Meteo ERA5 weather and Airport Corner event feasibility study.

Weather: CC BY 4.0, https://open-meteo.com/; ERA5 via the archive API.
Airport Corner: publicly readable; open reuse licence not established. Experimental only.
All joins use airport and UTC movement time, never departure block time or target.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlencode

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/external/conditions_20261008'
AIRPORTS = {'EDDF': (50.0333, 8.5706), 'EDDM': (48.3538, 11.7861),
            'EGLL': (51.4700, -.4543), 'EHAM': (52.3086, 4.7639),
            'LEBL': (41.2971, 2.0785), 'LEMD': (40.4719, -3.5626),
            'LFPG': (49.0097, 2.5479), 'LIRF': (41.8003, 12.2389),
            'LSZH': (47.4647, 8.5492), 'LTFM': (41.2753, 28.7519)}
VARIABLES = ['temperature_2m', 'precipitation', 'rain', 'snowfall', 'wind_speed_10m']
WEATHER = ['om_' + x for x in VARIABLES] + [f'om_{v}_{h}h' for v in ['precipitation', 'snowfall'] for h in [6, 24, 48]]
EVENTS = ['evt_runway', 'evt_taxiway', 'evt_industrial', 'evt_ops_change']


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + '.part')
        subprocess.run(['curl', '-fLsS', '--retry', '2', '--max-time', '90', url, '-o', str(temp)], check=True)
        temp.replace(path)
    return path.read_bytes()


def download():
    manifest = []
    # Continuous periods provide full preceding 48h histories for each scored month.
    for airport, (lat, lon) in AIRPORTS.items():
        for name, start, end in [('2025', '2024-12-29', '2025-12-31'),
                                 ('2026-01', '2025-12-29', '2026-01-31'),
                                 ('2026-07', '2026-06-29', '2026-07-31')]:
            params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                          hourly=','.join(VARIABLES), models='era5', timezone='GMT', wind_speed_unit='ms')
            url = 'https://archive-api.open-meteo.com/v1/archive?' + urlencode(params)
            path = CACHE / 'weather' / f'{airport}_{name}.json'
            data = fetch(url, path)
            obj = json.loads(data)
            assert 'hourly' in obj and all(v in obj['hourly'] for v in VARIABLES)
            manifest.append(dict(path=str(path.relative_to(ROOT)), url=url,
                                 sha256=hashlib.sha256(data).hexdigest(), units=obj['hourly_units'],
                                 grid_latitude=obj['latitude'], grid_longitude=obj['longitude']))
            print(airport, name, len(obj['hourly']['time']), flush=True)
    (CACHE / 'weather_manifest.json').write_text(json.dumps(manifest, indent=2))


def hourly_features(frame):
    f = frame.sort_values('time').copy().set_index('time')
    assert f.index.is_unique and (f.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    out = f[VARIABLES].add_prefix('om_')
    for v in ['precipitation', 'snowfall']:
        for h in [6, 24, 48]:
            out[f'om_{v}_{h}h'] = f[v].rolling(h, min_periods=h).sum()
    return out.reset_index()


def weather_table():
    tables = []
    for airport in AIRPORTS:
        frames = []
        for path in sorted((CACHE / 'weather').glob(f'{airport}_*.json')):
            f = pd.DataFrame(json.loads(path.read_text())['hourly'])
            f['time'] = pd.to_datetime(f.time, utc=True)
            frames.append(hourly_features(f))
        if not frames:
            raise FileNotFoundError(airport)
        # Prefer complete history when overlapping files begin with a partial window.
        f = pd.concat(frames).sort_values('om_precipitation_48h', na_position='last')
        f = f.drop_duplicates('time').assign(airport=airport)
        tables.append(f)
    return pd.concat(tables).sort_values('time')


def weather_join(raw, table):
    left = pd.DataFrame({'airport': raw.ADEP_mvt.to_numpy(),
                         'time': pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True).to_numpy(),
                         'row': np.arange(len(raw))})
    known = left.time.notna()
    joined = pd.merge_asof(left[known].sort_values('time'), table.sort_values('time'),
                           on='time', by='airport', direction='backward', tolerance=pd.Timedelta(hours=1))
    out = pd.DataFrame(np.nan, index=np.arange(len(raw)), columns=WEATHER)
    out.loc[joined.row, WEATHER] = joined[WEATHER].to_numpy()
    out.index = raw.index
    return out.astype('float32')


def event_archive():
    from bs4 import BeautifulSoup
    params = dict(filters=1, f_evt_completed=1, f_evt_ongoing=1, f_evt_notstarted=0,
                  f_evt_custom_start_end='true', f_evt_start_date='01/01/2025', f_evt_end_date='31/07/2026')
    url = 'https://ext.eurocontrol.int/airport_corner_public/events/airports_events?' + urlencode(params)
    source = fetch(url, CACHE / 'events.html')
    soup = BeautifulSoup(source, 'html.parser')
    rows = []
    for card in soup.select('.airport-label'):
        airport = card.select_one('.card-header').get_text(' ', strip=True)[:4]
        if airport not in AIRPORTS:
            continue
        for tr in card.select('tr.commentable'):
            t = tr.find_all('td')
            title = t[0].get_text(' ', strip=True)
            desc = t[1].get_text(' ', strip=True)
            # Title-based classification avoids turning weather capacity notices into closures.
            title_l = title.lower()
            works = bool(re.search(r'clos|maintenance|resurfac|not in use', title_l))
            flags = dict(evt_runway=int(works and bool(re.search(r'\brwy\b|runway', title_l))),
                         evt_taxiway=int(works and bool(re.search(r'\btwy\b|taxiway', title_l))),
                         evt_industrial=int(bool(re.search(r'strike|industrial action', title_l)) and
                                            not bool(re.search(r'rail|train|cancel', title_l))),
                         evt_ops_change=int(bool(re.search(r'tower|sequencer|triple runway', title_l))))
            rows.append(dict(airport=airport, id=t[0].find('a').get('data-id'), title=title,
                             description=desc, start=t[2].get('data-sort'), end=t[3].get('data-sort'), **flags))
    (CACHE / 'events.json').write_text(json.dumps(rows, indent=2))
    (CACHE / 'events_manifest.json').write_text(json.dumps(dict(url=url, sha256=hashlib.sha256(source).hexdigest(),
        licence='Open reuse licence not established; exploratory use only', rows=len(rows)), indent=2))
    return rows


def event_join(raw, events):
    times = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)
    out = pd.DataFrame(0, index=raw.index, columns=EVENTS, dtype='float32')
    for e in events:
        if not any(e[c] for c in EVENTS):
            continue
        start = pd.to_datetime(e['start'], format='%Y%m%d-%H%M', utc=True, errors='coerce')
        end = pd.to_datetime(e['end'], format='%Y%m%d-%H%M', utc=True, errors='coerce')
        # Unknown or coarse dates are not imputed into definitive event flags.
        if pd.isna(start) or pd.isna(end):
            continue
        active = (raw.ADEP_mvt == e['airport']) & times.between(start, end)
        for c in EVENTS:
            if e[c]:
                out.loc[active, c] = 1
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['weather', 'events'])
    a = p.parse_args()
    download() if a.stage == 'weather' else event_archive()
