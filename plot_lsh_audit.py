# 日本語説明：監査結果を図にする補助スクリプトです。現在の主検証ではノートブックの比較図を使用します。
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
r=json.loads((root/'lsh_audit_results.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
keys=list(r['baselines'])
labels=['Direct j0','Cached roll*','Cached FFT','Current proposal','Spatial-basis variant']
values=[r['baselines'][k]['median_s']*1000 for k in keys]
bars=axs[0].barh(labels,values,color=['0.65','0.65','0.65','#386cb0','#339966'])
axs[0].set_xscale('log'); axs[0].invert_yaxis()
axs[0].bar_label(bars,fmt='%.1f',padding=3)
axs[0].set(xlabel='Median online time [ms, logarithmic axis]',title='16 modes, 1024 x 1024, rank 80; 5 repeats')
axs[0].set_xlim(20,2500)
for rank in [20,40,80]:
    shifts=sorted({x['maxshift'] for x in r['stress']})
    groups=[[x['relative_field_error'] for x in r['stress'] if x['rank']==rank and x['maxshift']==s] for s in shifts]
    axs[1].plot(shifts,[100*np.median(g) for g in groups],'-o',label=f'rank {rank}')
    axs[1].fill_between(shifts,[100*min(g) for g in groups],[100*max(g) for g in groups],alpha=.15)
axs[1].set_yscale('log'); axs[1].set(xlabel='Random x/y shift bound [W0]',ylabel='Relative complex-field error [%]',title='3 seeds; continuous shifts and complex random weights')
axs[1].legend();axs[1].grid(alpha=.3)
fig.suptitle('LSH synthesis audit: current toy problem, not a full LSH reproduction\n*Cached roll applies only to integer-pixel shifts. Stress reference uses periodic Fourier translation.',fontsize=12)
fig.savefig(root/'lsh_audit_summary.png',dpi=160)
print('Saved lsh_audit_summary.png')
