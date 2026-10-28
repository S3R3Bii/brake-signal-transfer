"""Render decision-focused figures from retained prediction evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
 'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
 'axes.edgecolor':'#bac4d1','text.color':'#17283d','axes.labelcolor':'#17283d',
 'xtick.color':'#526273','ytick.color':'#526273','figure.facecolor':'white',
 'savefig.facecolor':'white','svg.fonttype':'none'})
TEAL='#087f8c';ORANGE='#d18337';INK='#23354d';GRAY='#9aa8b9'
summary=json.loads((R/'test_summary.json').read_text());selection=json.loads((R/'selection.json').read_text())
scores=pd.read_csv(R/'validation_scores.csv');test=pd.read_csv(R/'test_metrics.csv').set_index('model')
winner=selection['selected_model'];base=selection['selected_baseline'];row=test.loc[winner]

def save(fig,name):
    fig.savefig(F/f'{name}.png',dpi=190,bbox_inches='tight')
    fig.savefig(F/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1,1.2]})
x=np.arange(2);width=.32
for offset,(name,label,color) in zip([-.16,.16],[(base,'Selected baseline: zero',ORANGE),(winner,'Selected boosting model',TEAL)]):
    values=[float(scores.set_index('name').loc[name,'balanced_rmse']),float(test.loc[name,'balanced_rmse'])]
    axes[0].bar(x+offset,values,width,label=label,color=color)
    for xx,val in zip(x+offset,values):axes[0].text(xx,val+.05,f'{val:.3f}',ha='center',fontsize=10)
axes[0].set_xticks(x,['Lankershim\nvalidation','US-101\nheld-out test'])
axes[0].set_ylim(0,2.35);axes[0].set_ylabel('Trajectory-balanced RMSE (m/s²)')
axes[0].set_title('Reference reconstruction',loc='left',fontweight='bold');axes[0].legend(frameon=False,fontsize=9,loc='upper right')
values=[row['recall']*100,row['precision']*100]
axes[1].barh([0,1],values,color=[ORANGE,TEAL],height=.48)
axes[1].set_yticks([0,1],['Recall','Precision']);axes[1].invert_yaxis()
axes[1].set_xlim(0,112);axes[1].set_xticks([0,25,50,75,100]);axes[1].set_xlabel('Percent of relevant frames')
for y,v in enumerate(values):axes[1].text(v+1.8,y,f'{v:.1f}%',va='center',fontweight='bold')
axes[1].set_title('A conservative alert misses most positives',loc='left',fontweight='bold')
axes[1].text(0,1.8,f"{int(row.tp):,} true-positive frames · {int(row.fn):,} missed\n{int(row.fp):,} false positives / {int(row.fp+row.tn):,} reference negatives",fontsize=10,color=INK)
fig.suptitle('Low reconstruction error does not guarantee useful frame coverage',x=.01,ha='left',fontsize=17,fontweight='bold',y=1.06)
fig.text(.01,-.10,'Test: 121,981 frames from 214 tracks on US-101. Alert threshold fixed on Lankershim before test scoring.\nReference = processed NGSIM acceleration, not independent braking ground truth. Historical convenience sample; no safety-effect claim.',fontsize=9,color='#526273')
fig.tight_layout(w_pad=3);save(fig,'decision-summary')

def label(name):
    if name=='zero':return 'Zero acceleration'
    if name.startswith('diff'):return f'Backward difference, {int(name[4:])/10:.1f} s'
    if name.startswith('ridge'):return 'Ridge, alpha '+name[6:]
    return name.replace('boost_d','Boost depth ').replace('_lr',' · rate ').replace('_l2',' · L2 ')

fig,ax=plt.subplots(figsize=(11,8))
sorted_scores=scores.sort_values(['balanced_mse','name'])
colors=[TEAL if n==winner else ORANGE if k=='baseline' else GRAY for n,k in zip(sorted_scores.name,sorted_scores.kind)]
ax.barh(np.arange(len(scores)),sorted_scores.balanced_mse,color=colors,height=.66)
ax.set_yticks(np.arange(len(scores)),[label(n) for n in sorted_scores.name]);ax.invert_yaxis()
for i,v in enumerate(sorted_scores.balanced_mse):ax.text(v+.13,i,f'{v:.3f}',va='center',fontsize=9)
ax.set_xlim(0,float(scores.balanced_mse.max())*1.14)
ax.set_xlabel('Validation mean trajectory MSE (m²/s⁴), lower is better')
ax.set_title('Every prespecified candidate, including the unsuccessful ones',loc='left',fontweight='bold',pad=18)
fig.text(.01,-.01,'Training: I-80 · Selection: Lankershim · 18 configurations · No test-based model or metric selection\nTeal: selected model · Orange: simple baselines · Gray: other fitted candidates. A finite grid does not establish global optimality.',fontsize=9,color='#526273')
fig.tight_layout(rect=[0,.07,1,1]);save(fig,'optimization-evidence')

pred=pd.read_csv(R/'test_predictions.csv.gz');chosen=pred[pred.model==winner].copy()
chosen['residual']=chosen.prediction-chosen.reference
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
sub=pd.read_csv(R/'speed_subgroups.csv');sub=sub[sub.model==winner]
names=['below_5_mps','5_to_15_mps','above_15_mps'];sub=sub.set_index('speed_group').reindex(names)
axes[0].bar(np.arange(3),np.sqrt(sub.mse),color=TEAL,width=.55)
axes[0].set_xticks(np.arange(3),['<5 m/s','5–15 m/s','>15 m/s'])
axes[0].set_ylabel('Frame-weighted RMSE (m/s²)');axes[0].set_title('Error by speed group',loc='left',fontweight='bold')
for i,(_,r) in enumerate(sub.iterrows()):
    if pd.notna(r.mse):axes[0].text(i,np.sqrt(r.mse)+.01,f'n={int(r.n):,}',ha='center',fontsize=9)
axes[0].set_ylim(0,max(np.sqrt(sub.mse))+.10)
axes[1].hist(chosen.residual,bins=np.linspace(-2,2,81),color=TEAL,edgecolor='white',linewidth=.3)
outside=int((chosen.residual.abs()>2).sum())
axes[1].axvline(0,color=INK,lw=1);axes[1].set_xlabel('Prediction − recorded acceleration (m/s²)')
axes[1].set_ylabel('Frames');axes[1].set_title('Residual distribution',loc='left',fontweight='bold')
axes[1].text(.98,.95,f'{outside:,} frames outside displayed ±2 m/s²',transform=axes[1].transAxes,va='top',ha='right',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.94,pad=4))
fig.suptitle('Inspect the error distribution before proposing deployment',x=.01,ha='left',fontsize=17,fontweight='bold',y=1.06)
fig.text(.01,-.08,'Frozen model on US-101 only. Subgroups are descriptive and were not used to change model selection.\nFrames are temporally correlated; these plots do not establish independent sensor accuracy.',fontsize=9,color='#526273')
fig.tight_layout(w_pad=3);save(fig,'error-diagnostics')
print('Three quantitative figures exported to PNG and PDF.')
