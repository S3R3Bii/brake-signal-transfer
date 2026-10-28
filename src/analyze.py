"""Separate validation/freeze and test commands for a fixed road-transfer study."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
from datetime import datetime,timezone
import argparse
import gzip
import json
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from prepare import build

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'
FEATURES=['speed','diff1','diff3','diff5','diff10','past_mean','past_sd']
KEYS=['location','track','vehicle_id','frame_id','global_time','v_class','speed','reference']
SEED=20261028
REFERENCE=-8*.3048

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def write_csv(frame,name):
    frame.to_csv(OUT/name,index=False,compression={'method':'gzip','mtime':0})

def candidates():
    items=[{'name':'zero','kind':'baseline'}]+[{'name':f'diff{k}','kind':'baseline'} for k in (1,3,5,10)]
    items += [{'name':f'ridge_{a}','kind':'ridge','alpha':a} for a in (.01,.1,1,10,100)]
    for depth in (2,3):
        for rate in (.05,.1):
            for l2 in (0,1):
                items.append({'name':f'boost_d{depth}_lr{rate}_l2{l2}','kind':'boost',
                    'max_depth':depth,'learning_rate':rate,'l2_regularization':l2})
    return items

def predict(spec,train,target):
    if spec['kind']=='baseline':
        return np.zeros(len(target)) if spec['name']=='zero' else target[spec['name']].to_numpy()
    counts=train.groupby('track').track.transform('size')
    weights=1/counts.to_numpy(); weights*=len(weights)/weights.sum()
    if spec['kind']=='ridge':
        model=make_pipeline(StandardScaler(),Ridge(alpha=spec['alpha']))
        model.fit(train[FEATURES],train.reference,ridge__sample_weight=weights)
    else:
        model=HistGradientBoostingRegressor(max_iter=100,min_samples_leaf=30,
            early_stopping=False,random_state=SEED,**{k:spec[k] for k in
                ['max_depth','learning_rate','l2_regularization']})
        model.fit(train[FEATURES],train.reference,sample_weight=weights)
    return model.predict(target[FEATURES])

def metrics(frame):
    error=frame.prediction-frame.reference
    per=frame[['track']].assign(square=error**2,absolute=np.abs(error)).groupby('track').mean()
    return {'rows':len(frame),'tracks':len(per),'balanced_mse':float(per.square.mean()),
        'balanced_rmse':float(np.sqrt(per.square.mean())),
        'balanced_mae':float(per.absolute.mean()),'frame_mse':float(np.mean(error**2)),
        'frame_rmse':float(np.sqrt(np.mean(error**2))),'frame_mae':float(np.mean(np.abs(error)))}

def confusion(frame,threshold):
    positive=frame.reference.to_numpy()<=REFERENCE
    flag=np.zeros(len(frame),dtype=bool) if threshold is None else frame.prediction.to_numpy()<=threshold
    tp=int((positive&flag).sum());fp=int((~positive&flag).sum())
    fn=int((positive&~flag).sum());tn=int((~positive&~flag).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,
        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
        'fpr':fp/(fp+tn) if fp+tn else None,'prevalence':float(positive.mean())}

def threshold_search(frame):
    rows=[]
    for ft in np.arange(-12,-3.99,.5):
        rows.append({'threshold_mps2':float(ft*.3048),**confusion(frame,ft*.3048)})
    feasible=[r for r in rows if r['fpr'] is not None and r['fpr']<=.01]
    if feasible:
        best=min(feasible,key=lambda r:(-(r['recall'] or 0),r['fpr'],r['threshold_mps2']))
    else:
        best={'threshold_mps2':None,**confusion(frame,None)}
        rows.append(best)
    return best,rows

def validation():
    OUT.mkdir(exist_ok=True)
    frame,quality=build()
    train=frame[frame.location=='i-80']; valid=frame[frame.location=='lankershim']
    assert set(train.track).isdisjoint(valid.track)
    save('data_quality.json',quality)
    write_csv(frame,'feature_lineage.csv.gz')
    results=[]; predictions=[]
    for spec in candidates():
        start=time.perf_counter(); pred=valid[KEYS].copy()
        pred['prediction']=predict(spec,train,valid); pred['model']=spec['name']
        results.append({**spec,**metrics(pred),'elapsed_seconds':time.perf_counter()-start,'status':'completed'})
        predictions.append(pred)
        print(spec['name'],results[-1]['balanced_mse'],flush=True)
    table=pd.DataFrame(results).sort_values(['balanced_mse','name'])
    table.to_csv(OUT/'validation_scores.csv',index=False)
    combined=pd.concat(predictions,ignore_index=True)
    write_csv(combined,'validation_predictions.csv.gz')
    winner=table.iloc[0]['name'];baseline=table[table.kind=='baseline'].iloc[0]['name']
    thresholds={}; searches=[]
    for name in dict.fromkeys([winner,baseline]):
        best,trials=threshold_search(combined[combined.model==name])
        thresholds[name]=best['threshold_mps2']
        searches.extend([{'model':name,**r} for r in trials])
    pd.DataFrame(searches).to_csv(OUT/'threshold_search.csv',index=False)
    save('selection.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'selected_model':winner,'selected_baseline':baseline,'alert_thresholds':thresholds,
        'objective':'Validation mean of per-trajectory MSE; lower is better; model name tie break',
        'train_site':'i-80','validation_site':'lankershim','test_site':'us-101',
        'test_outcomes_used_for_selection':False,'candidate_count':len(table),'seed':SEED,
        'candidates':candidates(),'boost_fixed':{'max_iter':100,'min_samples_leaf':30,'early_stopping':False},
        'features':FEATURES,'target':'Recorded acceleration converted to m/s²; processed proxy, not sensor ground truth',
        'scope':'Finite grid, not a global optimum; one validation road only'})

def test():
    selection=json.loads((OUT/'selection.json').read_text())
    frame,_=build();train=frame[frame.location=='i-80']; test=frame[frame.location=='us-101']
    assert set(train.track).isdisjoint(test.track)
    specs={s['name']:s for s in candidates()}
    results=[];predictions=[]
    for name in dict.fromkeys([selection['selected_model'],selection['selected_baseline']]):
        pred=test[KEYS].copy();pred['prediction']=predict(specs[name],train,test);pred['model']=name
        results.append({'model':name,**metrics(pred),**confusion(pred,selection['alert_thresholds'][name])})
        predictions.append(pred)
    combined=pd.concat(predictions,ignore_index=True)
    write_csv(combined,'test_predictions.csv.gz')
    pd.DataFrame(results).to_csv(OUT/'test_metrics.csv',index=False)
    combined['square']=(combined.prediction-combined.reference)**2
    per=combined.groupby(['track','model']).square.mean().unstack()
    delta=per[selection['selected_model']]-per[selection['selected_baseline']]
    rng=np.random.default_rng(SEED)
    draws=np.array([rng.choice(delta.to_numpy(),size=len(delta),replace=True).mean() for _ in range(2000)])
    bounds=np.quantile(draws,[.025,.975]).tolist()
    per.to_csv(OUT/'paired_track_mse.csv')
    combined['speed_group']=pd.cut(combined.speed,[-np.inf,5,15,np.inf],labels=['below_5_mps','5_to_15_mps','above_15_mps'])
    subgroup=combined.groupby(['model','speed_group'],observed=True).agg(n=('square','size'),mse=('square','mean'))
    subgroup.to_csv(OUT/'speed_subgroups.csv')
    combined.groupby(['model','v_class']).agg(n=('square','size'),mse=('square','mean')).to_csv(OUT/'vehicle_subgroups.csv')
    save('test_summary.json',{'evaluated_utc':datetime.now(timezone.utc).isoformat(),
        'selected_model':selection['selected_model'],'selected_baseline':selection['selected_baseline'],
        'metrics':results,'paired_balanced_mse_difference':float(delta.mean()),
        'trajectory_bootstrap_interval_95':bounds,'bootstrap_draws':2000,'seed':SEED,
        'test_rows_per_model':len(test),'test_tracks':test.track.nunique(),
        'limitations':['Processed reference shares provenance with speed; no independent braking labels',
           'Convenience cohort; one test road; trajectories may share traffic conditions',
           'Bootstrap conditional on this road and sample, not population-wide uncertainty',
           'Validation FPR constraint is not guaranteed on the test road']})
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['validate','test'])
    args=parser.parse_args(); validation() if args.phase=='validate' else test()
