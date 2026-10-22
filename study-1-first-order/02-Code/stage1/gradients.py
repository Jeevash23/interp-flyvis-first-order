"""First-order derivative of the identical DSI at baseline attenuation zero."""
from core import *
def gradient_bank(a,frames,state,angles,pol,speed,block=4,announce=print):
 # Only the causal prefix through the last measured frame can influence DSI.
 end=21*2.25/(speed*5.8);K=int(np.floor(end/.005+1e-10))+1
 f=frames[:,:K];atten=torch.zeros(a.c,device=a.device,requires_grad=True)
 t=time.perf_counter();trace,_,_=a.run(f,atten,state=state.detach(),start=0,grad=True,block=block);m=metrics(trace,angles,pol,speed,time_start=0);sync(a.device);forward=time.perf_counter()-t
 derivatives=[];times=[]
 for r in range(8):
  t=time.perf_counter();d=torch.autograd.grad(m['dsi'][r],atten,retain_graph=r<7)[0];sync(a.device);times.append(time.perf_counter()-t);derivatives.append(d.detach().cpu().numpy());announce(f'gradient {TARGETS[r]} {times[-1]:.2f}s')
 # Positive temporal maxima with multiple exact maximizers are flagged. Flat
 # negative trajectories are identically zero rectified peaks, not tie failures.
 v=trace.detach().cpu().numpy();pk=np.maximum(v,0).max(1);ties=((v==pk[:,None,:]).sum(1)>1)&(pk>0)
 peak=m['peaks'].detach().cpu().numpy();sums=np.stack([peak[pol==i].sum(0) for i in [0,1]])
 den_tie=np.isclose(sums[0],sums[1],atol=1e-7,rtol=1e-6)
 top=np.partition(v,-2,axis=1)[:,-2:];near=(pk>0)&(abs(top[:,1]-top[:,0])<=1e-6)
 return np.stack(derivatives),{'forward_s':forward,'backward_s':times,'causal_frames':K,'peak_tie_counts_per_readout':ties.sum(0).tolist(),'near_peak_counts_per_readout':near.sum(0).tolist(),'denominator_ties':den_tie.tolist(),'baseline_dsi':m['dsi'].detach().cpu().numpy().tolist(),'autograd_tie_rule':'torch amax distributes gradients equally across exact maxima; norm uses native zero subgradient; exact and near (<=1e-6 voltage) maxima flagged; estimates retained as fixed branch/subgradient algorithm'}

if __name__=='__main__':
 import sys,gc
 device=sys.argv[1];admit();torch.set_num_threads(2);torch.set_num_interop_threads(2)
 if device=='cuda':
  if not os.environ.get('AI06_GPU_LOCK_FD'):raise RuntimeError('GPU lock required')
  torch.cuda.set_per_process_memory_fraction(.72);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 z=np.load(P/'04-Results/qualification/native-responses.npz');f=z['stimulus'][:,:,0].astype('float32');a=Core(device=device);g=a.gray();_,st,_=a.run(f[:,:200],state=g.expand(24,-1).clone());grad,record=gradient_bank(a,f[:,200:],st,z['angle'],z['intensity'],19,announce=lambda s:print(s,flush=True))
 checks=[];eps=.001
 for name in ['Mi1','Tm3','Mi11']:
  vals=[]
  for s in [-eps,eps]:
   mask=torch.zeros(65,device=device);mask[a.names.index(name)]=s
   y,_,_=a.run(f[:,200:286],mask,state=st.detach(),start=0);vals.append(metrics(y,z['angle'],z['intensity'],19,time_start=0)['dsi'].detach().cpu().numpy())
  fd=(vals[1]-vals[0])/(2*eps);auto=grad[:,a.names.index(name)]
  checks.append({'coordinate':name,'autograd':auto.tolist(),'central_difference':fd.tolist(),'absolute_error':abs(fd-auto).tolist(),'pass':bool(np.all(abs(fd-auto)<=1e-4+.02*abs(fd)))})
 record.update(finite_difference_checks=checks,status='pass' if all(c['pass'] for c in checks) else 'fail',peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 if device=='cuda':record.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
 save_json(OUT/f'benchmark/{device}/gradient.json',record);np.save(OUT/f'benchmark/{device}/gradients.npy',grad);print(record,flush=True)
 if record['status']!='pass':raise SystemExit('Gradient gate failed')
