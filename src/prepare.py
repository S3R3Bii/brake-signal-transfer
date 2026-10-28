"""Reconstruct anonymous contiguous tracks, then execute causal SQL features."""
from pathlib import Path
import hashlib
import json
import sqlite3
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def build():
    parts=[]
    for item in json.loads((ROOT/'data/manifest.json').read_text()):
        path=ROOT/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        part=pd.read_csv(path)
        assert len(part)==item['rows']
        parts.append(part)
    raw=pd.concat(parts,ignore_index=True)
    counts=[]; cleaned=[]
    for loc,group in raw.groupby('location'):
        original=len(group)
        unique=group.drop_duplicates()
        exact_removed=original-len(unique)
        required=['vehicle_id','frame_id','global_time','v_vel','v_acc','local_y','v_class']
        numeric=unique[required].apply(pd.to_numeric,errors='coerce')
        valid=np.isfinite(numeric).all(axis=1)&(numeric.v_vel>=0)
        clean=unique.loc[valid].copy()
        conflicts=clean.duplicated(['vehicle_id','global_time'],keep=False)
        conflict_count=int(conflicts.sum()); clean=clean.loc[~conflicts].copy()
        clean=clean.sort_values(['vehicle_id','global_time','frame_id'])
        new=(clean.vehicle_id.ne(clean.vehicle_id.shift()) |
             clean.global_time.diff().ne(100) | clean.frame_id.diff().ne(1))
        clean['segment_number']=new.cumsum()
        clean['track']=loc+'-'+clean.segment_number.astype(str)
        clean['speed']=clean.v_vel*.3048
        clean['reference']=clean.v_acc*.3048
        counts.append({'location':loc,'raw':original,'exact_duplicate_rows_removed':exact_removed,
             'invalid_rows_removed':int((~valid).sum()),'conflicting_rows_removed':conflict_count,
             'clean_rows':len(clean),'segments_before_history_filter':clean.track.nunique()})
        cleaned.append(clean)
    clean=pd.concat(cleaned,ignore_index=True)
    # Center each short window before squaring to avoid cancellation at constant speed.
    clean['stable_sd']=np.nan
    for _,segment in clean.groupby('track',sort=False):
        if len(segment)>10:
            windows=np.lib.stride_tricks.sliding_window_view(segment.speed.to_numpy(),10)[:-1]
            clean.loc[segment.index[10:],'stable_sd']=windows.std(axis=1)
    db=sqlite3.connect(':memory:')
    clean[['location','vehicle_id','frame_id','global_time','v_class','track','speed','reference']].to_sql('frames',db,index=False)
    db.execute('CREATE UNIQUE INDEX unique_frame ON frames(location,vehicle_id,global_time)')
    lag_fields=',\n'.join(f'LAG(speed,{k}) OVER w AS speed_lag{k}, LAG(global_time,{k}) OVER w AS time_lag{k}' for k in [1,3,5,10])
    query=f'''WITH causal AS (
      SELECT *, {lag_fields},
        AVG(speed) OVER (PARTITION BY track ORDER BY global_time ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS past_mean,
        AVG(speed*speed) OVER (PARTITION BY track ORDER BY global_time ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING) AS past_mean_square
      FROM frames WINDOW w AS (PARTITION BY track ORDER BY global_time)
    ) SELECT * FROM causal WHERE speed_lag10 IS NOT NULL ORDER BY location,track,global_time;'''
    (ROOT/'sql').mkdir(exist_ok=True)
    (ROOT/'sql/features.sql').write_text(query,encoding='utf-8')
    features=pd.read_sql_query(query,db)
    features=features.merge(clean[['location','vehicle_id','global_time','stable_sd']],
        on=['location','vehicle_id','global_time'],how='left',validate='one_to_one')
    features['past_sd']=features.pop('stable_sd')
    for k in [1,3,5,10]:
        assert (features.global_time-features[f'time_lag{k}']==100*k).all()
        features[f'diff{k}']=(features.speed-features[f'speed_lag{k}'])/(.1*k)
    assert features.notna().all().all()
    for c in counts:
        subset=features[features.location==c['location']]
        c['history_rows_removed']=c['clean_rows']-len(subset)
        c['analysis_rows']=len(subset); c['analysis_tracks']=subset.track.nunique()
    return features,counts

if __name__=='__main__':
    f,c=build(); print(json.dumps(c,indent=2))
