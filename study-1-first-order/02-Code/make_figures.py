from pathlib import Path
import numpy as np,json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];O=P/'04-Results/figures';O.mkdir(exist_ok=True)
a=np.load(P/'04-Results/qualification/native-responses.npz');r=json.load(open(P/'04-Results/qualification/qualification.json'));v=json.load(open(P/'04-Results/qualification/numerical-verification.json'))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
fig,axs=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
rows=r['rows'];colors=['#24738a']*4+['#b87135']*4
axs[0].bar([x['cell_type'] for x in rows],[x['dsi'] for x in rows],color=colors);axs[0].set(ylim=(0,1),ylabel='Native tutorial DSI',title='One released model | central cells');axs[0].tick_params(axis='x',rotation=45)
idx=int(np.flatnonzero(a['cell_type']=='T4c')[0]);mask=a['intensity']==1;peaks=a['peaks'][0,mask,idx];angles=a['angle'][mask];axs[1].plot(angles,peaks,'o-',color='#24738a');axs[1].set(xlabel='Motion direction (degrees)',ylabel='Peak rectified voltage (model units)',title='T4c ON-edge tuning | 110.2 degrees/s');axs[1].set_xticks([0,90,180,270,330]);fig.savefig(O/'tuning.png',dpi=200);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
x=np.load(P/'04-Results/qualification/streaming-baseline.npy');y=np.load(P/'04-Results/qualification/half-step.npy')[1::2];z=np.load(P/'04-Results/qualification/float64.npy');axs[0].plot(a['time'],x[:,idx],label='float32, dt=5 ms',color='#24738a');axs[0].plot(a['time'],y[:,idx],'--',label='float32, dt=2.5 ms',color='#b87135');axs[0].set(xlabel='Time from stimulus onset (s)',ylabel='T4c voltage (model units)',title='Numerical diagnostic | ON, 60 degrees');axs[0].legend(frameon=False,fontsize=8)
mem=[7645360/1024**2,987876/1024**2,2118616/1024**2];axs[1].bar(['Native\ninterrupted','Dense chunks\nstopped early','Sparse renderer\ncompleted'],mem,color=['#ac554e','#a8b2b6','#24738a']);axs[1].axhline(4,color='#ac554e',linestyle='--',label='Adopted worker limit');axs[1].set(ylabel='Peak worker RSS (GiB)',title='Different attempt stages | not speed benchmarks');axs[1].legend(frameon=False,fontsize=8);fig.savefig(O/'verification-and-memory.png',dpi=200);plt.close(fig)
