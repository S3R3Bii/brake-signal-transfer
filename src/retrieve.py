"""Download a fixed, public NGSIM convenience cohort with provenance."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
BASE='https://data.transportation.gov/resource/8ect-6jqj'
COLUMNS='vehicle_id,frame_id,global_time,v_vel,v_acc,local_y,v_class,location'

def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'S3R3Bii public research / GitHub portfolio'})
    with urllib.request.urlopen(request,timeout=120) as response:
        return response.read()

def main():
    DATA.mkdir(exist_ok=True)
    meta=json.loads(fetch('https://data.transportation.gov/api/views/8ect-6jqj.json'))
    keep={'name':meta['name'],'description':meta['description'],'license':meta['license'],
          'columns':[{k:c.get(k) for k in ['fieldName','dataTypeName','description']}
                     for c in meta['columns']]}
    (DATA/'source_metadata.json').write_text(json.dumps(keep,indent=2),encoding='utf-8')
    manifest=[]; counts=[]
    for location in ['i-80','lankershim','us-101']:
        where=f"location='{location}' AND vehicle_id BETWEEN 1 AND 80"
        count_url=BASE+'.json?'+urllib.parse.urlencode({'$select':'count(*) AS n','$where':where})
        expected=int(json.loads(fetch(count_url))[0]['n'])
        downloaded=0
        for offset in range(0,expected,50000):
            query={'$select':COLUMNS,'$where':where,'$order':'vehicle_id,global_time,frame_id,:id',
                   '$limit':50000,'$offset':offset}
            url=BASE+'.csv?'+urllib.parse.urlencode(query)
            path=DATA/f'{location}_{offset:06d}.csv'
            if not path.exists(): path.write_bytes(fetch(url))
            content=path.read_bytes()
            row_count=len(list(csv.DictReader(io.StringIO(content.decode('utf-8-sig')))))
            downloaded+=row_count
            manifest.append({'file':path.relative_to(ROOT).as_posix(),'url':url,
                'retrieved_utc':datetime.now(timezone.utc).isoformat(),'rows':row_count,
                'sha256':hashlib.sha256(content).hexdigest()})
            (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(location,offset,row_count,flush=True)
            time.sleep(.3)
        assert downloaded==expected,(location,downloaded,expected)
        counts.append({'location':location,'source_count':expected,'downloaded_rows':downloaded,'count_url':count_url})
    (DATA/'source_reconciliation.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')

if __name__=='__main__': main()
