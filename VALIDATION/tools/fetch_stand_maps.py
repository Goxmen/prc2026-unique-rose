"""Download the two stand maps for idea 34: X-Plane Gateway apt.dat and OSM.

X-Plane Scenery Gateway (https://gateway.x-plane.com, GPL v2 or later): the
recommended scenery of each airport, via its public API. Only `<ICAO>.dat` is
kept from the scenery zip. OpenStreetMap (ODbL) via Overpass, using the ADS-B
probe's query and folder, so its LIRF/LSZH files are reused. Both are external
datasets to declare to the organisers.

Files already present are skipped. A manifest records scenery ids, upload dates,
byte counts and SHA-256 hashes.

Run: .venv/bin/python VALIDATION/tools/fetch_stand_maps.py
"""

import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import adsb_geometry
from stand_geometry import AIRPORTS, OSM_DIR, OUT_DIR, XPLANE_DIR

API = 'https://gateway.x-plane.com/apiv1'
MANIFEST = OUT_DIR / 'manifest.json'


def get_json(url):
    req = Request(url, headers={'User-Agent': adsb_geometry.AGENT})
    with urlopen(req, timeout=300) as resp:
        return json.load(resp)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_xplane(icao):
    """Save the recommended scenery's apt.dat; return its provenance."""
    airport = get_json(f'{API}/airport/{icao}')['airport']
    sid = airport['recommendedSceneryId']
    meta = next(s for s in airport['scenery'] if s['sceneryId'] == sid)
    dest = XPLANE_DIR / f'{icao}.dat'
    if not dest.exists():
        blob = get_json(f'{API}/scenery/{sid}')['scenery']['masterZipBlob']
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(blob))) as z:
            dest.write_bytes(z.read(f'{icao}.dat'))
    return {'scenery_id': sid, 'uploaded': meta['dateUploaded'], 'approved': meta['dateApproved'],
            'artist': meta['userName'], 'lat': airport['latitude'], 'lon': airport['longitude']}


def fetch_osm(icao, lat, lon, tries=6):
    """The public Overpass server answers 429/504 when busy; wait and retry."""
    for attempt in range(tries):
        time.sleep(20 if attempt == 0 else 90 * attempt)
        try:
            return adsb_geometry.download(icao, lat, lon)
        except HTTPError as err:
            if err.code not in (429, 504) or attempt == tries - 1:
                raise
            print(f'  {icao}: Overpass {err.code}, retrying', flush=True)


def main():
    XPLANE_DIR.mkdir(parents=True, exist_ok=True)
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest = {}
    for icao in AIRPORTS:
        xp = fetch_xplane(icao)
        osm = OSM_DIR / f'osm_{icao}.json'
        if not osm.exists():
            fetch_osm(icao, xp['lat'], xp['lon'])
            fetched = datetime.now(timezone.utc).isoformat(timespec='seconds')
        else:
            fetched = old.get(icao, {}).get('osm', {}).get('fetched') or datetime.fromtimestamp(
                osm.stat().st_mtime, timezone.utc).isoformat(timespec='seconds')
        dat = XPLANE_DIR / f'{icao}.dat'
        manifest[icao] = {
            'xplane': {**xp, 'file': str(dat.relative_to(adsb_geometry.REPO)),
                       'bytes': dat.stat().st_size, 'sha256': sha256(dat)},
            'osm': {'file': str(osm.relative_to(adsb_geometry.REPO)), 'fetched': fetched,
                    'bytes': osm.stat().st_size, 'sha256': sha256(osm)},
        }
        print(icao, 'x-plane scenery', xp['scenery_id'], xp['uploaded'][:10],
              '| osm', manifest[icao]['osm']['bytes'], 'bytes', flush=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main()
