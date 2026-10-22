import json,time,sys,gc
from pathlib import Path
import numpy as np,torch,flyvis
from streaming_native import gray_initial_state,streaming_forward,validate_streaming
P=Path(__file__).resolve().parents[1];O=P/'04-Results/qualification';torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
a=np.load(O/'native-responses.npz',allow_pickle=False);idx=int(np.flatnonzero((a['angle']==60)&(a['intensity']==1))[0]);stim=a['stimulus'][idx];ref=a['responses'][0,idx]
v=flyvis.NetworkView(flyvis.results_dir/'flow/0000/000');n=v.init_network();types=n.connectome.nodes.type[:].astype(str);s=types[n._source_indices.numpy()];t=types[n._target_indices.numpy()];pairs=sorted(set(zip(s,t)))
census={'n_nodes':n.n_nodes,'n_edges':n.n_edges,'model_types':len(set(types)),'realized_type_pairs':len(pairs),'missing_source_pairs': sorted(set((e['src'],e['tar']) for e in json.load(open(P/'01-Papers/sources/flyvis/flyvis/connectome/fib25-fib19_v2.2.json'))['edges'])-set(pairs))}
(P/'03-Data/constructed-network-census.json').write_text(json.dumps(census,indent=2))
def trial(dt=.005,frames=stim):
 ts=time.monotonic();state=gray_initial_state(n,1.,dt);adapt=time.monotonic()-ts;ts=time.monotonic();r=streaming_forward(n,frames,dt,state=state);secs=time.monotonic()-ts;return r.responses,{'adaptation_seconds':adapt,'trajectory_seconds':secs,'frames':len(frames),'dt':dt}
if '--replay' in sys.argv:
 x,cost=trial();np.save(O/'clean-replay.npy',x);expected=np.load(O/'streaming-baseline.npy');r={'status':'pass' if np.array_equal(x,expected) else 'fail','exact_equal':bool(np.array_equal(x,expected)),'cost':cost};(O/'clean-replay.json').write_text(json.dumps(r,indent=2));print(r);raise SystemExit()
short=validate_streaming(n,stim);(O/'streaming-validation.json').write_text(json.dumps(short,indent=2));print('short',short['status'],flush=True)
if short['status']!='pass':raise SystemExit('Streaming control failed')
x,cost=trial();np.save(O/'streaming-baseline.npy',x);native_close=bool(np.allclose(x,ref,atol=1e-6,rtol=1e-5));y,repeatcost=trial();det=bool(np.array_equal(x,y));scale=max(float(abs(x).max()),1e-8)
# Float64 factories must match model parameter dtype.
torch.set_default_dtype(torch.float64);n.double();z,doublecost=trial();np.save(O/'float64.npy',z);precision=float(abs(z-x).max()/scale)
torch.set_default_dtype(torch.float32);n.float();fine,finecost=trial(.0025,np.repeat(stim,2,axis=0));np.save(O/'half-step.npy',fine);delta=fine[1::2]-x;stepmax=float(abs(delta).max()/scale);steprms=float(np.sqrt(np.mean(delta**2))/scale)
params=n._param_api();tau=params.nodes.time_const.detach().numpy();r={'status':'pass' if native_close and det and precision<=1e-4 and stepmax<=.05 and np.isfinite(z).all() and np.isfinite(fine).all() else 'fail','native_stream_close':native_close,'native_stream_max_absolute_error':float(abs(x-ref).max()),'deterministic_bitwise':det,'float64_normalized_max_error':precision,'half_step_normalized_max_error':stepmax,'half_step_normalized_rms_error':steprms,'tau_min':float(tau.min()),'nodes_tau_below_dt':int((tau<.005).sum()),'cost':cost,'repeat_cost':repeatcost,'double_cost':doublecost,'fine_cost':finecost,'census':census,'scope':'one development model; one ON60degree trial; all central outputs; no method ranking'}
(O/'numerical-verification.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)

if r['status'] != 'pass':
    raise SystemExit(2)
