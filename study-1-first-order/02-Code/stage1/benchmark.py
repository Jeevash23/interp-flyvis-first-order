"""Development-only numerical gates and measured throughput. No held-out reads."""
from common import *
import sys,gc,subprocess
from core import *
torch.set_num_threads(2);torch.set_num_interop_threads(2)
device=sys.argv[1] if len(sys.argv)>1 else 'cpu';admit()
if device=='cuda':
 if not os.environ.get('AI06_GPU_LOCK_FD'):raise RuntimeError('Use gpu_lock.py')
 os.fstat(int(os.environ['AI06_GPU_LOCK_FD']))
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.72)
torch.use_deterministic_algorithms(False)
z=np.load(P/'04-Results/qualification/native-responses.npz');frames=z['stimulus'][:,:,0].astype('float32');angles=z['angle'];pol=z['intensity'];ix=[z['cell_type'].tolist().index(n) for n in TARGETS]
report={'device':device,'model':'000','torch':torch.__version__,'dtype':'float32','threads':2,'deterministic_algorithms':False,'tf32':False,'timing':[],'checks':{}}
out=OUT/'benchmark'/device;out.mkdir(exist_ok=True)
def log(s):print(s,flush=True)
def save():
 report['peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 if device=='cuda':
  report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
  report['gpu_state']=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,memory.used,memory.free','--format=csv,noheader,nounits'],text=True).strip()
 save_json(out/'benchmark.json',report)
t=time.perf_counter();a=Core(device=device);g=a.gray();sync(device);report['setup_and_gray_s']=time.perf_counter()-t;save_json(DATA/'network-structure.json',a.metadata()) if device=='cpu' else None
sub=[int(np.flatnonzero((pol==p)&(angles==an))[0]) for p,an in [(1,60),(0,240)]]
checks=report['checks'];singles=[];conditions=[(),('Mi1',),('Mi1',),('Mi1','Tm3'),('Mi11',),('Tm30',)];strengths=[0,.5,1,1,1,1]
batchframes=np.stack([frames[i] for i in sub for _ in conditions]);atten=torch.zeros((len(batchframes),65),device=device)
for j in range(len(sub)):
 for k,(names,s) in enumerate(zip(conditions,strengths)):
  for n in names:atten[j*len(conditions)+k,a.names.index(n)]=s
t=time.perf_counter();bat,_,_=a.run(batchframes,atten,state=g.expand(len(batchframes),-1).clone());sync(device);log(f'{device} subset batch {time.perf_counter()-t:.2f}s')
for j,(f,mask) in enumerate(zip(batchframes,atten)):
 y,_,_=a.run(f[None],mask,state=g.clone());singles.append(y.cpu().numpy()[0])
single=np.stack(singles);b=bat.cpu().numpy();checks['batch_serial']=bool(np.allclose(b,single,atol=1e-6 if device=='cpu' else 1e-5,rtol=1e-5));report['batch_serial_max_error']=float(np.max(abs(b-single)))
for j in range(2):
 base=b[j*6];checks[f'clean_preperiod_{j}']=bool(np.allclose(b[j*6:j*6+6,:200],base[None,:200],atol=1e-6,rtol=1e-5))
 checks[f'structural_zeros_{j}']=bool(np.allclose(b[j*6+4:j*6+6],base[None],atol=1e-6,rtol=1e-5))
checks['native_saved_baseline']=bool(np.allclose(b[::6],z['responses'][0,sub][:,:,ix],atol=1e-6 if device=='cpu' else 1e-4,rtol=1e-5))
np.savez_compressed(out/'subset.npz',batched=b,serial=single,stimulus_indices=sub,atten=atten.cpu().numpy())
if device=='cpu':
 sys.path.insert(0,str(P/'02-Code'));from streaming_native import gray_initial_state,streaming_forward,build_edge_gate,synaptic_gate
 ng=gray_initial_state(a.native);native=[];baseweight=a.native._param_api().edges.weight.detach().clone()
 for idx in sub:
  pre=streaming_forward(a.native,frames[idx,:200],.005,state=ng,indices=a.readout.cpu().numpy())
  for names,s in zip(conditions[:4],strengths[:4]):
   gate=build_edge_gate(a.native,source_types=names,multiplier=1-s)
   with synaptic_gate(a.native,gate):
    w=a.native._param_api().edges.weight.detach();checks['mask_locality']=bool(torch.equal(w[~gate.selected],baseweight[~gate.selected])) and checks.get('mask_locality',True)
    post=streaming_forward(a.native,frames[idx,200:],.005,state=pre.final_state,indices=a.readout.cpu().numpy())
   native.append(np.concatenate([pre.responses,post.responses]))
 native=np.stack(native);chosen=b[[0,1,2,3,6,7,8,9]];checks['independent_native_onset_mask']=bool(np.allclose(native,chosen,atol=1e-6,rtol=1e-5));report['native_onset_max_error']=float(np.max(abs(native-chosen)))
 np.savez_compressed(out/'native-onset.npz',native=native)
 del native
else:
 c=np.load(OUT/'benchmark/cpu/subset.npz')['batched'];report['cpu_gpu_normalized_max_error']=float(np.max(abs(c-b))/max(1.,float(np.max(abs(c)))));checks['cpu_gpu_trajectories']=report['cpu_gpu_normalized_max_error']<=1e-4
save();log(f'{device} native/batch gates {checks}')
del bat,single,b,batchframes,atten;gc.collect()
# Fresh full485-frame timing excludes shared grey setup, includes transfers.
for B in ([1,8] if device=='cpu' else [1,8,32,64]):
 health(device);fs=np.repeat(frames[sub[0]][None],B,axis=0);st=g.expand(B,-1).clone()
 try:
  a.run(fs,state=st);sync(device);times=[]
  for rep in range(3):
   t=time.perf_counter();y,_,_=a.run(fs,state=st);sync(device);times.append(time.perf_counter()-t)
  row={'batch':B,'seconds':times,'median_seconds_per_trajectory':float(np.median(times)/B),'frames':485,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
  if device=='cuda':row.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
  report['timing'].append(row);save();log(row);del y
 except torch.OutOfMemoryError:
  report['timing'].append({'batch':B,'failed':'out_of_memory'});save();torch.cuda.empty_cache();break
 del fs,st;gc.collect()
# Full native24 DSIs and Mi1 effects; execution at small stable batch.
responses=[]
for s in [0,.5,1]:
 mask=torch.zeros(65,device=device);mask[a.names.index('Mi1')]=s;bank=[]
 for k in range(0,24,8):
  y,_,_=a.run(frames[k:k+8],mask,state=g.expand(min(8,24-k),-1).clone());bank.append(y.cpu())
 bank=torch.cat(bank);responses.append(bank.numpy());log(f'{device} DSI bank strength{s}');health(device)
responses=np.stack(responses);ms=[metrics(torch.from_numpy(x),angles,pol,19) for x in responses];dsis=np.stack([m['dsi'].numpy() for m in ms]);p=ms[0]['peaks'].numpy();v=np.sum(p[pol==1,2]*np.exp(1j*np.deg2rad(angles[pol==1])));pd=float(np.angle(v,deg=True)%360)
report['tutorial']={'T4c_DSI':float(dsis[0,2]),'T4c_PD':pd};checks['tutorial_dsi']=abs(dsis[0,2]-.63)<=.02;checks['tutorial_pd']=abs(pd-56.24)<=2
np.savez_compressed(out/'dsi-banks.npz',responses=responses,dsi=dsis,angle=angles,intensity=pol)
if device=='cuda':
 cpu=np.load(OUT/'benchmark/cpu/dsi-banks.npz');err=float(np.max(abs((dsis-dsis[0])-(cpu['dsi']-cpu['dsi'][0]))));report['cpu_gpu_dsi_effect_max_error']=err;checks['cpu_gpu_dsi_effects']=err<=.001
# Full onset-to-final raw-mean VJP for a declared ON60/T4c readout.
pre,st,_=a.run(frames[sub[0]:sub[0]+1,:200],state=g.clone(),start=200)
mask=torch.zeros(65,device=device,requires_grad=True);post=frames[sub[0]:sub[0]+1,200:];t=time.perf_counter();tr,_,_=a.run(post,mask,state=st.detach(),start=0,grad=True,block=4);F=metrics(tr,np.array([60]),np.array([1]),19,time_start=0)['raw_mean'][0,2];der=torch.autograd.grad(F,mask)[0];sync(device);report['gradient_raw_single_forward_backward_s']=time.perf_counter()-t
eps=.001;vals=[]
for s in [-eps,eps]:
 ma=torch.zeros(65,device=device);ma[a.names.index('Mi1')]=s;tr,_,_=a.run(post,ma,state=st.detach(),start=0);vals.append(float(metrics(tr,np.array([60]),np.array([1]),19,time_start=0)['raw_mean'][0,2]))
fd=(vals[1]-vals[0])/(2*eps);d=float(der[a.names.index('Mi1')]);report['raw_gradient_check']={'coordinate':'Mi1','readout':'ON60 T4c raw mean','autograd':d,'central_difference':fd,'epsilon':eps};checks['raw_gradient']=abs(d-fd)<=1e-4+.02*abs(fd)
report['status']='pass' if all(checks.values()) else 'fail';save();log(report)
if report['status']!='pass':raise SystemExit('Numerical gate failed')
