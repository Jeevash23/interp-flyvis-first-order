"""Resolve finite-difference branch crossings using an independent tangent recurrence."""
from common import *
from core import *
import sys
device=sys.argv[1];torch.set_num_threads(2);admit()
if device=='cuda':
 if not os.environ.get('AI06_GPU_LOCK_FD'):raise RuntimeError('GPU lock required')
 torch.cuda.set_per_process_memory_fraction(.72)
a=Core(device=device);z=np.load(P/'04-Results/qualification/native-responses.npz');frames=z['stimulus'][:,:,0].astype('float32');angles=z['angle'];pol=z['intensity']
# Float64 diagnostic retains the exact recovered float32 parameter values.
# This reduces cancellation; it is not the production solver or a refit.
a.weight=a.weight.double();a.bias=a.bias.double();a.tau=a.tau.double();g=a.gray();_,st,_=a.run(frames[:,:200],state=g.expand(24,-1).clone());f=torch.tensor(frames[:,200:286],device=device);alpha=torch.zeros((24,65),device=device,dtype=torch.float64)
@torch.no_grad()
def tangent(name):
 v=st.clone();dv=torch.zeros_like(v);out=[];dout=[];indicator=(a.types==a.names.index(name)).double()[None];invtau=1/a.tau
 for k in range(86):
  # Independent analytic directional derivative of output-gated Euler recurrence.
  dr=(v>0)*dv-torch.relu(v)*indicator
  dedge=dr[:,a.source]*a.weight
  dcurrent=torch.zeros_like(v).scatter_add_(1,a.target[None].expand(24,-1),dedge)
  dv=dv+.005*invtau*(-dv+dcurrent)
  v=a.step(v,f[:,k],alpha,.005);out.append(v[:,a.readout]);dout.append(dv[:,a.readout])
 return torch.stack(out,1),torch.stack(dout,1)
records=[]
for name in ['Mi1','Tm3','Mi11']:
 trace,dtrace=tangent(name);leaf=trace.detach().requires_grad_(True);m=metrics(leaf,angles,pol,19,time_start=0);d=[];active=[]
 for r in range(8):
  partial=torch.autograd.grad(m['dsi'][r],leaf,retain_graph=r<7)[0];d.append(float((partial*dtrace).sum()));active.append((partial[:,:,r].abs().sum(1)>0).cpu().numpy())
 top=trace.topk(2,dim=1).values.cpu().numpy();near=(top[:,0]>0)&((top[:,0]-top[:,1])<=1e-6)
 active_near=[np.flatnonzero(near[:,r]&active[r]).tolist() for r in range(8)]
 auto=np.load(OUT/f'benchmark/{device}/gradients.npy')[:,a.names.index(name)];d=np.array(d);fdrows=[];baseidx=trace.argmax(1).cpu().numpy()
 for eps in [.001,.0001,.00001]:
  vals=[];changed=[]
  for sign in [-1,1]:
   mask=torch.zeros(65,device=device,dtype=torch.float64);mask[a.names.index(name)]=sign*eps;y,_,_=a.run(f,mask,state=st.clone(),start=0)
   vals.append(metrics(y,angles,pol,19,time_start=0)['dsi'].cpu().numpy());changed.append((y.argmax(1).cpu().numpy()!=baseidx).sum(0).tolist())
  fd=(vals[1]-vals[0])/(2*eps);fdrows.append({'epsilon':eps,'central_difference':fd.tolist(),'maximizing_frame_changes_minus_plus':changed,'agreement_with_tangent':bool(np.all(abs(fd-d)<=1e-4+.02*abs(fd)))})
 lastfd=np.array(fdrows[-1]['central_difference']);fdok=abs(lastfd-d)<=1e-4+.02*abs(lastfd);flags=[TARGETS[r] for r in range(8) if not fdok[r] and active_near[r]]
 cpu=np.load(OUT/'benchmark/cpu/gradients.npy')[:,a.names.index(name)]
 row={'coordinate':name,'float32_autograd':auto.tolist(),'float64_independent_tangent':d.tolist(),'autograd_tangent_max_error':float(abs(auto-d).max()),'autograd_tangent_pass':bool(np.all(abs(auto-d)<=1e-4+.02*abs(d))),'cpu_autograd_tangent_max_error':float(abs(cpu-d).max()),'cpu_autograd_tangent_pass':bool(np.all(abs(cpu-d)<=1e-4+.02*abs(d))),'finite_differences':fdrows,'fine_difference_pass':fdrows[-1]['agreement_with_tangent'],'readout_relevant_near_peak_sample_indices':active_near,'nonsmooth_sensitive_readouts':flags,'every_fd_disagreement_has_active_near_peak':bool(all(fdok[r] or active_near[r] for r in range(8)))};records.append(row);print(row,flush=True)
record={'status':'pass_with_nonsmooth_flags' if all(r['autograd_tangent_pass'] and r['cpu_autograd_tangent_pass'] and r['every_fd_disagreement_has_active_near_peak'] for r in records) else 'fail','device':device,'records':records,'diagnostic_only_dtype':'float64 with recovered float32 weights','original_epsilon_0001_check_retained':'benchmark/'+device+'/gradient.json','scope':'Independent forward tangent validates the fixed autograd branch/subgradient. Failed finite differences at active near-maxima remain flagged, not treated as smooth derivatives or tuned away.'};save_json(OUT/f'benchmark/{device}/gradient-gate-resolution.json',record)
if record['status']=='fail':raise SystemExit('Derivative remains unqualified')
