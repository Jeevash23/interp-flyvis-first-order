"""Fresh native CPU replay, seeded before study; development ReLU diagnostic."""
from common import *
import sys,numpy as np,torch
from core import *
sys.path.insert(0,str(P/'02-Code'))
from streaming_native import gray_initial_state,streaming_forward,build_edge_gate,synaptic_gate
torch.set_num_threads(2);admit();rng=np.random.default_rng(20261002);names=json.loads((DATA/'network-structure.json').read_text())['model_types'];zeros=['Mi11','Tm30'];eligible=[n for n in names if n not in zeros];rows=[]
for split,models,keys in [('development',['000','004'],['dev_slow','dev_fast']),('final',['001','005'],['final_slow','final_fast'])]:
 for model in models:
  name=str(rng.choice(eligible));s=float(rng.choice([.5,1.]));key=str(rng.choice(keys));idx=int(rng.integers(24));a=Core(model,'cpu');g=gray_initial_state(a.native);z=np.load(DATA/f'stimulus-{key}.npz');f=z['stimulus'];f=f[:,:,0] if f.ndim==4 else f;t=time.perf_counter();pre=streaming_forward(a.native,f[idx,:200],.005,state=g,indices=a.readout.numpy());gate=build_edge_gate(a.native,source_types=[name],multiplier=1-s)
  with synaptic_gate(a.native,gate):post=streaming_forward(a.native,f[idx,200:],.005,state=pre.final_state,indices=a.readout.numpy())
  saved=np.load(OUT/split/model/key/f's{int(s*100):03d}-cell{names.index(name):02d}.npz')['trace'][idx];err=float(np.max(abs(saved-post.responses))/max(1.,float(abs(saved).max())));row={'model':model,'split':split,'cell':name,'strength':s,'stimulus':key,'trajectory_index':idx,'normalized_max_error':err,'status':'pass' if err<=1e-4 else 'fail','wall_s':time.perf_counter()-t};rows.append(row);print(row,flush=True);health()
 # Drop both copies before the next model.
 del a,g,pre,post
save_json(OUT/'verification/fresh-native-replay.json',{'status':'pass' if all(r['status']=='pass' for r in rows) else 'fail','seed':20261002,'rows':rows})
a=Core('000','cpu');z=np.load(DATA/'stimulus-dev_fast.npz');f=z['stimulus'];f=f[:,:,0] if f.ndim==4 else f;idx=int(np.flatnonzero((z['angle']==60)&(z['intensity']==1))[0]);g=a.gray();_,st,_=a.run(f[idx:idx+1,:200],state=g.clone());v=st.expand(4,-1).clone();atten=torch.zeros((4,65));atten[1,a.names.index('Mi1')]=1;atten[2,a.names.index('Tm3')]=1;atten[3,a.names.index('Mi1')]=1;atten[3,a.names.index('Tm3')]=1;changes=[]
with torch.no_grad():
 for k in range(86):
  v=a.step(v,torch.as_tensor(f[idx,200+k]).expand(4,-1),atten,.005);changes.append(((v[1:]>0)!=(v[0]>0)[None]).sum(1).numpy().tolist())
save_json(OUT/'verification/relu-diagnostic.json',{'model':'000','stimulus':'dev_fast','angle':60,'intensity':1,'strength':1,'labels':['Mi1','Tm3','Mi1+Tm3'],'differing_node_count_by_frame':changes,'max_differing_nodes':np.max(changes,axis=0).tolist(),'n_nodes':a.n,'scope':'Positive-state ReLU pattern differences on one declared development trial, not a causal decomposition of interaction.'})
