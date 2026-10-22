"""Native published-tutorial qualification; no intervention ranking."""
import json,time,resource,os
from pathlib import Path
import numpy as np
import torch
import flyvis
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.analysis.moving_bar_responses import direction_selectivity_index,preferred_direction,peak_responses
P=Path(__file__).resolve().parents[1];out=P/'04-Results'/'qualification';out.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
t=time.monotonic()
dataset=MovingEdge(offsets=[-10,11],intensities=[0,1],speeds=[19],height=80,post_pad_mode='continue',t_pre=1.,t_post=1.,dt=1/200,angles=list(np.arange(0,360,30)))
print('Dataset ready',dataset.arg_df.to_dict('records'),flush=True)
v=flyvis.NetworkView(flyvis.results_dir/'flow/0000/000')
x=v.moving_edge_responses(dataset,batch_size=1)
d=direction_selectivity_index(x);p=preferred_direction(x);peaks=peak_responses(x)
np.savez_compressed(out/'native-responses.npz',responses=x.responses.values,stimulus=x.stimulus.values,time=x.time.values,cell_type=x.cell_type.values,angle=x.angle.values,intensity=x.intensity.values,peaks=peaks.values)
rows=[]
for typ in ['T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d']:
 i=1 if typ.startswith('T4') else 0
 rows.append({'cell_type':typ,'intensity':i,'dsi':float(d.custom.where(cell_type=typ,intensity=i).item()),'pd_degrees':float(p.custom.where(cell_type=typ,intensity=i).item()*180/np.pi)})
r=next(r for r in rows if r['cell_type']=='T4c');de=abs(r['dsi']-.63);ae=abs((r['pd_degrees']-56.24+180)%360-180)
report={'model':'flow/0000/000','status':'pass' if de<=.02 and ae<=2 and np.isfinite(x.responses.values).all() and str(flyvis.device)=='cpu' else 'fail','elapsed_seconds':time.monotonic()-t,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'torch':torch.__version__,'flyvis':flyvis.__version__,'finite':bool(np.isfinite(x.responses.values).all()),'target_dsi':.63,'target_pd_degrees':56.24,'dsi_error':de,'pd_error_degrees':ae,'rows':rows,'config':dataset.config.to_dict()}
(out/'qualification.json').write_text(json.dumps(report,indent=2,default=str));print(json.dumps(report,indent=2,default=str),flush=True)

if report['status'] != 'pass':
    raise SystemExit(2)
