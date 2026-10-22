"""Static publication-style figures from verified descriptive summaries."""
from common import *
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
pairs=json.loads((OUT/'analysis/reporting-supplement.json').read_text())['pair_interactions'];out=OUT/'figures';out.mkdir(exist_ok=True)
labels=['Mi1/Tm3','Mi1/Mi4','Tm3/Mi4','Mi4/Mi9','Tm1/Tm2','Tm1/Tm9','Tm2/Tm4','Tm4/Tm9'];cols=[(m,s) for m in ['000','004','001','005'] for s in [.5,1.]]
data=np.array([[max(r['max_abs_dsi'] for r in pairs if '/'.join(r['pair'])==pair and r['model']==m and r['strength']==s and r['max_abs_dsi'] is not None) for m,s in cols] for pair in labels])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9.84,4.2));im=ax.imshow(data,cmap='YlGnBu',aspect='auto',vmin=0);ax.set_yticks(range(8),labels);ax.set_xticks(range(8),[f'{m}\ns={s:g}' for m,s in cols]);ax.set_xlabel('Development 000,004; held-out 001,005');ax.set_title('Largest eligible absolute DSI interaction by pair and model');ax.axvline(3.5,color='white',lw=2)
for i in range(8):
 for j in range(8):ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',color='white' if data[i,j]>.55*data.max() else '#162831',fontsize=10.5)
fig.colorbar(im,ax=ax,label='Absolute DSI units');fig.tight_layout();fig.savefig(out/'pairs.png',dpi=180);fig.savefig(out/'pairs.svg');plt.close(fig)
