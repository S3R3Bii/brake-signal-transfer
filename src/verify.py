"""Recompute exported evidence independently of pandas and fitting helpers."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import math
import numpy as np

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'

def rows(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig',newline='') as stream:
        yield from csv.DictReader(stream)

def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8),(a,b)

def evaluate(path):
    by_model=defaultdict(lambda:defaultdict(lambda:[0,0.,0.]))
    count=defaultdict(int); identities=defaultdict(set)
    for row in rows(path):
        error=float(row['prediction'])-float(row['reference'])
        assert math.isfinite(error)
        model=row['model']; track=row['track']; v=by_model[model][track]
        v[0]+=1;v[1]+=error*error;v[2]+=abs(error);count[model]+=1
        identity=(row['location'],row['vehicle_id'],row['global_time'])
        assert identity not in identities[model]
        identities[model].add(identity)
    answer={}
    for model,groups in by_model.items():
        group_mse=[v[1]/v[0] for v in groups.values()]
        balanced_mse=math.fsum(group_mse)/len(groups)
        frame_mse=math.fsum(v[1] for v in groups.values())/count[model]
        answer[model]={'rows':count[model],'tracks':len(groups),'balanced_mse':balanced_mse,
            'balanced_rmse':math.sqrt(balanced_mse),
            'balanced_mae':math.fsum(v[2]/v[0] for v in groups.values())/len(groups),
            'frame_mse':frame_mse,'frame_rmse':math.sqrt(frame_mse),
            'frame_mae':math.fsum(v[2] for v in groups.values())/count[model]}
    return answer,by_model,identities

def confusion(export,thresholds,cached=None):
    counts={name:dict(tp=0,fp=0,fn=0,tn=0) for name in thresholds}
    for row in rows(export) if cached is None else cached:
        name=row['model']
        if name not in thresholds: continue
        actual=float(row['reference'])<=-2.4384
        limit=thresholds[name]
        flag=False if limit is None else float(row['prediction'])<=limit
        label='tp' if flag and actual else 'fp' if flag else 'fn' if actual else 'tn'
        counts[name][label]+=1
    for value in counts.values():
        tp,fp,fn,tn=[value[k] for k in ['tp','fp','fn','tn']]
        value.update(precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None,fpr=fp/(fp+tn) if fp+tn else None,
            prevalence=(tp+fn)/(tp+fp+fn+tn))
    return counts

def main(phase):
    source={}; source_counts=defaultdict(int)
    for item in json.loads((ROOT/'data/manifest.json').read_text()):
        path=ROOT/item['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        n=0
        for row in rows(path):
            n+=1;key=(row['location'],int(row['vehicle_id']),int(row['global_time']))
            assert key not in source,'Source conflict needs explicit review'
            source[key]=(float(row['v_vel'])*.3048,float(row['v_acc'])*.3048,int(row['frame_id']))
            source_counts[row['location']]+=1
        assert n==item['rows']
    for item in json.loads((ROOT/'data/source_reconciliation.json').read_text()):
        assert source_counts[item['location']]==item['source_count']==item['downloaded_rows']
    feature_counts=defaultdict(int);tracks=defaultdict(set)
    for row in rows(R/'feature_lineage.csv.gz'):
        loc=row['location'];vehicle=int(row['vehicle_id']);timestamp=int(row['global_time'])
        speed,reference,frame=source[(loc,vehicle,timestamp)]
        close(row['speed'],speed);close(row['reference'],reference)
        for lag in [1,3,5,10]:
            previous=source[(loc,vehicle,timestamp-lag*100)]
            assert previous[2]==frame-lag
            close(row[f'time_lag{lag}'],timestamp-lag*100)
            close(row[f'speed_lag{lag}'],previous[0])
            close(row[f'diff{lag}'],(speed-previous[0])/(lag/10))
        previous=[source[(loc,vehicle,timestamp-k*100)][0] for k in range(1,11)]
        mean=math.fsum(previous)/10
        close(row['past_mean'],mean)
        close(row['past_sd'],math.sqrt(math.fsum((v-mean)**2 for v in previous)/10))
        feature_counts[loc]+=1;tracks[loc].add(row['track'])
    assert all(tracks[a].isdisjoint(tracks[b]) for a,b in [('i-80','lankershim'),('i-80','us-101'),('lankershim','us-101')])
    for row in json.loads((R/'data_quality.json').read_text()):
        assert feature_counts[row['location']]==row['analysis_rows']
        assert len(tracks[row['location']])==row['analysis_tracks']
        assert row['raw']==row['analysis_rows']+sum(row[k] for k in
            ['exact_duplicate_rows_removed','invalid_rows_removed','conflicting_rows_removed','history_rows_removed'])
    metrics,groups,valid_ids=evaluate(R/'validation_predictions.csv.gz')
    table=list(rows(R/'validation_scores.csv'));assert len(table)==len(metrics)==18
    for reported in table:
        assert reported['status']=='completed'
        for k,v in metrics[reported['name']].items():close(reported[k],v)
    selected=min(metrics,key=lambda name:(metrics[name]['balanced_mse'],name))
    baseline=min([r['name'] for r in table if r['kind']=='baseline'],key=lambda name:(metrics[name]['balanced_mse'],name))
    selection=json.loads((R/'selection.json').read_text())
    assert selected==selection['selected_model'] and baseline==selection['selected_baseline']
    frozen=selection['alert_thresholds']
    # Recompute every threshold, then independently apply the selection ordering.
    threshold_rows=list(rows(R/'threshold_search.csv'))
    cached=[r for r in rows(R/'validation_predictions.csv.gz') if r['model'] in frozen]
    for name in frozen:
        subset=[r for r in threshold_rows if r['model']==name];expected=[]
        # One pass per threshold is deliberate independence from the vectorized implementation.
        for row in subset:
            threshold=float(row['threshold_mps2']) if row['threshold_mps2'] else None
            recalculated=confusion(R/'validation_predictions.csv.gz',{name:threshold},cached)[name]
            for k,v in recalculated.items():
                if v is not None: close(row[k],v)
                else: assert row[k]==''
            if recalculated['fpr']<=.01:
                expected.append((threshold,recalculated))
        best=min(expected,key=lambda x:(-(x[1]['recall'] or 0),x[1]['fpr'],
                                       x[0] if x[0] is not None else -math.inf))
        assert best[0]==frozen[name]
    checks=['Six source-page hashes and 246144 rows reconcile',
        'Every causal feature independently joined to source timestamps, frames and units',
        '18 candidate metrics, model selection and all alert thresholds independently reproduced',
        'Training, validation and test road/trajectory identifiers are disjoint']
    if phase=='test':
        test_metrics,test_groups,test_ids=evaluate(R/'test_predictions.csv.gz')
        for name in test_ids:
            assert test_ids[name].isdisjoint(valid_ids[name])
        alerts=confusion(R/'test_predictions.csv.gz',frozen)
        for row in rows(R/'test_metrics.csv'):
            for k,v in {**test_metrics[row['model']],**alerts[row['model']]}.items():
                if v is not None:close(row[k],v)
                else:assert row[k]==''
        summary=json.loads((R/'test_summary.json').read_text())
        assert summary['evaluated_utc']>selection['frozen_utc']
        track_ids=sorted(test_groups[selected]);deltas=[]
        for key in track_ids:
            a=test_groups[selected][key];b=test_groups[baseline][key]
            assert a[0]==b[0]
            deltas.append(a[1]/a[0]-b[1]/b[0])
        close(summary['paired_balanced_mse_difference'],math.fsum(deltas)/len(deltas))
        rng=np.random.default_rng(20261028);means=[]
        for _ in range(2000):
            indices=rng.integers(0,len(deltas),size=len(deltas))
            means.append(math.fsum(deltas[int(i)] for i in indices)/len(deltas))
        ordered=sorted(means)
        for p,reported in zip([.025,.975],summary['trajectory_bootstrap_interval_95']):
            ix=p*(len(ordered)-1);low=math.floor(ix);fraction=ix-low
            close(reported,ordered[low]*(1-fraction)+ordered[low+1]*fraction)
        checks.append('Frozen test metrics, confusion counts and paired trajectory interval independently reproduced')
    receipt={'status':'PASS','phase':phase,'checks':checks,
        'limits':'Numerical and lineage checks do not establish independent sensor ground truth or road-population generalization.'}
    (R/f'{phase}_verification.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['validation','test'])
    main(parser.parse_args().phase)
