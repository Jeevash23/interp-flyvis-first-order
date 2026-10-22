"""Resumable cell-type experiment. A frozen manifest is mandatory."""
from common import *
import sys,gc,argparse
from core import *
from gradients import gradient_bank
parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','cuda'],required=True);parser.add_argument('--batch',type=int,required=True);parser.add_argument('--split',choices=['development','final','all'],default='all');args=parser.parse_args()
torch.set_num_threads(2);torch.set_num_interop_threads(2);admit()
if args.device=='cuda':
 if not os.environ.get('AI06_GPU_LOCK_FD'):raise RuntimeError('Launch through gpu_lock.py')
 os.fstat(int(os.environ['AI06_GPU_LOCK_FD']));torch.cuda.set_per_process_memory_fraction(.72);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
freeze_path=HIST/'freeze-manifest.json';freeze=json.loads(freeze_path.read_text());freeze_hash=sha(freeze_path)
for rel,digest in freeze['files'].items():
 if sha(P/rel)!=digest:raise RuntimeError(f'Frozen file changed: {rel}')
selection=json.loads((DATA/'model-selection.json').read_text());stimuli=json.loads((DATA/'stimulus-manifest.json').read_text());t_global=time.perf_counter()
def say(s):print(s,flush=True)
def write_record(path,arrays,metadata):
 tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
 tmp.replace(path);metadata.update(sha256=sha(path),freeze_sha256=freeze_hash,device=args.device,utc_unix=time.time());save_json(path.with_suffix('.json'),metadata)
def valid(path):
 if not path.exists():return False
 meta=json.loads(path.with_suffix('.json').read_text())
 if meta['freeze_sha256']!=freeze_hash or meta['sha256']!=sha(path):raise RuntimeError(f'Bad existing result {path}')
 return True
def arrays(trace,stim):
 m=metrics(torch.as_tensor(trace),stim['angle'],stim['intensity'],float(stim['speed']),time_start=0)
 return {**{k:v.numpy() for k,v in m.items() if k!='window'},'trace':trace}
for split,models,keys in [('development',selection['development'],['dev_slow','dev_fast']),('final',selection['held_out'],['final_slow','final_fast'])]:
 if args.split not in [split,'all']:continue
 for model in models:
  health(args.device);admit();a=Core(model,args.device);gray=a.gray();model_t=time.perf_counter()
  for key in keys+['bio_slow','bio_fast']:
   health(args.device);dest=OUT/split/model/key;dest.mkdir(parents=True,exist_ok=True);info=next(s for s in stimuli if s['key']==key);stim=np.load(P/info['path']);frames=stim['stimulus'];frames=frames[:,:,0] if frames.ndim==4 else frames;f=frames[:,200:];speed=float(stim['speed']);t_panel=time.perf_counter()
   # Clean onset state, unmodified by any attenuation coordinate.
   pre,st,_=a.run(frames[:,:200],state=gray.expand(24,-1).clone());st=st.detach()
   if not valid(dest/'preperiod.npz'):write_record(dest/'preperiod.npz',{'trace':pre.cpu().numpy()},{'kind':'clean_preperiod','model':model,'stimulus':key})
   baseline=dest/'baseline.npz'
   if not valid(baseline):
    t=time.perf_counter();trace,_,cur=a.run(f,state=st.clone(),start=0,currents=True);sync(args.device);arr=arrays(trace.cpu().numpy(),stim);window=metrics(trace,stim['angle'],stim['intensity'],speed,time_start=0)['window'];cs=(cur[:,window].sum(1)*.005).cpu().numpy();score=np.stack([cs[stim['intensity']==(1 if r<4 else 0),r].mean(0) for r in range(8)])
    arr.update(current_score=score,synapse_count=a.counts);write_record(baseline,arr,{'kind':'baseline','model':model,'stimulus':key,'wall_s':time.perf_counter()-t});del trace,cur,arr
   if not key.startswith('bio') and not valid(dest/'gradient.npz'):
    health(args.device);grad,meta=gradient_bank(a,f,st,stim['angle'],stim['intensity'],speed,announce=say);write_record(dest/'gradient.npz',{'gradient':grad},{**meta,'kind':'first_order','model':model,'stimulus':key})
   conditions=[]
   for strength in [.5,1.]:
    if key.startswith('bio'):cells=['Mi1','Tm3'];pairs=[('Mi1','Tm3')]
    else:cells=a.names;pairs=PAIRS
    for kind,items in [('cell',[(x,) for x in cells]),('pair',pairs)]:
     for names in items:
      code=('cell'+str(a.names.index(names[0])).zfill(2)) if kind=='cell' else 'pair'+str(PAIRS.index(tuple(names)))
      path=dest/f's{int(strength*100):03d}-{code}.npz';meta={'kind':kind,'names':list(names),'strength':strength,'model':model,'stimulus':key}
      if valid(path):continue
      if all(n in a.zeros for n in names):
       b=np.load(baseline);write_record(path,{k:b[k] for k in b.files if k not in ['current_score','synapse_count']},{**meta,'structural_zero':True,'simulated_trajectories':0,'wall_s':0});continue
      conditions.append((path,meta))
   queue=[(j,i) for j in range(len(conditions)) for i in range(24)];pending={};batch_times=[]
   for begin in range(0,len(queue),args.batch):
    health(args.device);jobs=queue[begin:begin+args.batch];idx=[i for _,i in jobs];masks=torch.zeros((len(jobs),65),device=args.device)
    for row,(j,i) in enumerate(jobs):
     for name in conditions[j][1]['names']:masks[row,a.names.index(name)]=conditions[j][1]['strength']
    t=time.perf_counter();tr,_,_=a.run(f[idx],masks,state=st[idx].clone(),start=0);sync(args.device);elapsed=time.perf_counter()-t;bank=tr.cpu().numpy();batch_times.append(elapsed);del tr
    for row,(j,i) in enumerate(jobs):
     if j not in pending:pending[j]=[np.empty((24,bank.shape[1],8),dtype=np.float32),set(),0.]
     pending[j][0][i]=bank[row];pending[j][1].add(i);pending[j][2]+=elapsed/len(jobs)
     if len(pending[j][1])==24:
      trace,_,wall=pending.pop(j);path,meta=conditions[j];write_record(path,arrays(trace,stim),{**meta,'structural_zero':False,'simulated_trajectories':24,'wall_s':wall});say(f'{split} {model} {key} {path.stem} complete')
    if args.device=='cuda':health(args.device)
   if pending:raise RuntimeError('Incomplete condition assembly')
   elapsed=time.perf_counter()-t_panel
   save_json(dest/'panel-run.json',{'model':model,'key':key,'wall_s':elapsed,'new_exact_batches':len(batch_times),'exact_batch_wall_s':sum(batch_times),'batch':args.batch,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'peak_cuda_reserved_mib':torch.cuda.max_memory_reserved()/2**20 if args.device=='cuda' else 0,'freeze_sha256':freeze_hash})
   say(f'PANEL DONE {model} {key}: {elapsed:.1f}s');del st,pre;gc.collect()
  say(f'MODEL DONE {model}: {time.perf_counter()-model_t:.1f}s');del a,gray;gc.collect()
  if args.device=='cuda':torch.cuda.empty_cache()
save_json(OUT/f'run-complete-{args.split}.json',{'device':args.device,'batch':args.batch,'wall_s':time.perf_counter()-t_global,'freeze_sha256':freeze_hash,'utc_unix':time.time()});say('RUN COMPLETE')
