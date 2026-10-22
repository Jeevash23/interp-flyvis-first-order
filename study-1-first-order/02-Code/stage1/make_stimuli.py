"""Render native panels, reusing qualified spatial cache. No model responses."""
from common import *
import sys,numpy as np,torch
sys.path.insert(0,str(P/'02-Code'))
from safe_render_sparse import safe_sparse_renderer
torch.set_num_threads(2);admit()
import flyvis
torch.set_default_device('cpu');flyvis.device=torch.device('cpu')
from flyvis.datasets.moving_bar import MovingEdge
manifest=[]
for key,speed in [('dev_slow',9.5),('dev_fast',19.),('final_slow',4.75),('final_fast',38.),('bio_slow',12.5/5.8),('bio_fast',300/5.8)]:
 path=DATA/f'stimulus-{key}.npz';t=time.perf_counter()
 if not path.exists():
  with safe_sparse_renderer():
   d=MovingEdge(offsets=[-10,11],intensities=[0,1],speeds=[speed],height=80,post_pad_mode='continue',t_pre=1.,t_post=1.,dt=.005,angles=list(np.arange(0,360,30)))
   inputs=np.stack([np.asarray(d[i]) for i in range(len(d))]).astype('float32')
   np.savez_compressed(path,stimulus=inputs,angle=d.arg_df.angle.to_numpy(),intensity=d.arg_df.intensity.to_numpy(),speed=speed)
   save_json(DATA/f'stimulus-{key}-config.json',d.config.to_dict())
  del d,inputs
 z=np.load(path);n=len(z['stimulus']);T=z['stimulus'].shape[1]
 manifest.append({'key':key,'native_speed':speed,'degrees_per_second':speed*5.8,'trajectories':n,'frames':T,'stimulus_onset_frame':200,'motion_window_end_s':21*2.25/(speed*5.8),'path':str(path.relative_to(P)),'sha256':sha(path),'render_or_load_s':time.perf_counter()-t})
 print(manifest[-1],flush=True);health()
save_json(DATA/'stimulus-manifest.json',manifest)
