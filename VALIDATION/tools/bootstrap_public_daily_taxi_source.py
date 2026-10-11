"""Cache metadata from EUROCONTROL's public taxi-out iframe before data export.

Run once, then fetch_public_daily_taxi.py fetch. Existing downloads are reused.
The published anonymous resource key is not an account credential.
"""
import json
from pathlib import Path
import urllib.request
import uuid

import fetch_public_daily_taxi as source

URLS={
    'taxiout.html':source.PAGE,
    'powerbi.html':'https://app.powerbi.com/view?r=eyJrIjoiOWNlMjRlOTctZDIxYi00ZGU3LTg0YWYtMDMxNDEyOGExZjZmIiwidCI6Ijc2ZjMzYzIwLTU5NzktNDQwOC1hZGY3LThiM2M0YmU5NWU1MiIsImMiOjl9',
    'models.json':source.API+source.KEY+'/modelsAndExploration?preferReadOnlySession=true',
    'schema.json':source.API+source.KEY+'/conceptualschema',
    'disclaimer.html':'https://ansperformance.eu/about/disclaimer/',
}


def main():
    source.RESEARCH.mkdir(parents=True,exist_ok=True)
    for name,url in URLS.items():
        path=source.RESEARCH/name
        if not path.exists():
            headers={'User-Agent':'Mozilla/5.0'}
            if name.endswith('.json'):
                headers.update({'Accept':'application/json','X-PowerBI-ResourceKey':source.KEY,
                    'ActivityId':str(uuid.uuid4()),'RequestId':str(uuid.uuid4())})
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=30) as response:
                data=response.read()
            if name.endswith('.json'):json.loads(data)
            temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_bytes(data);temporary.replace(path)
        print(name,source.digest(path),flush=True)
    models=json.loads((source.RESEARCH/'models.json').read_text())
    assert models['models'][0]['id']==3679962


if __name__=='__main__':main()
