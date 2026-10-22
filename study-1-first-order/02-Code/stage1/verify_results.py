"""Independent NumPy recomputation and checksum/completeness audit."""
from common import *
import numpy as np
from core import PAIRS,TARGETS
def independent(trace,angle,pol,speed):
 # Integer frame selection in float64, independently of Torch metrics.
 times=(np.arange(trace.shape[1],dtype=np.float64)+200)*.005-1
 end=(22.5/5.8+2.25/5.8+22.5/5.8)/speed
 v=trace[:,(times>=0)&(times<=end)].astype(np.float64)
 peaks=np.maximum(v,0).max(1);sums=[];vectors=[]
 for p in [0,1]:
  q=peaks[pol==p];z=q*np.exp(1j*np.deg2rad(angle[pol==p]))[:,None];sums.append(q.sum(0));vectors.append(abs(z.sum(0)))
 den=np.maximum(sums[0],sums[1]);both=np.array(vectors)/(den+1e-15);dsi=np.array([both[1 if r<4 else 0,r] for r in range(8)])
 return {'peaks':peaks,'denominator':den,'dsi':dsi,'raw_mean':v.mean(1),'raw_integral':v.sum(1)*.005}
def verify():
 selection=json.loads((DATA/'model-selection.json').read_text());names=json.loads((DATA/'network-structure.json').read_text())['model_types'];records=[];maxerr={k:0. for k in ['peaks','denominator','dsi','raw_mean','raw_integral']};total=0;simulated=0
 for split,models,keys in [('development',selection['development'],['dev_slow','dev_fast']),('final',selection['held_out'],['final_slow','final_fast'])]:
  for model in models:
   for key in keys+['bio_slow','bio_fast']:
    dest=OUT/split/model/key;stim=np.load(DATA/f'stimulus-{key}.npz');files=[dest/'baseline.npz']
    for s in [.5,1.]:
     cells=['Mi1','Tm3'] if key.startswith('bio') else names;ps=[0] if key.startswith('bio') else range(8)
     files += [dest/f's{int(s*100):03d}-cell{names.index(n):02d}.npz' for n in cells]+[dest/f's{int(s*100):03d}-pair{j}.npz' for j in ps]
    for f in files:
     meta=json.loads(f.with_suffix('.json').read_text());assert sha(f)==meta['sha256'],str(f);z=np.load(f);calc=independent(z['trace'],stim['angle'],stim['intensity'],float(stim['speed']))
     for field in maxerr:
      err=float(np.max(abs(calc[field]-z[field])));maxerr[field]=max(maxerr[field],err)
      assert np.allclose(calc[field],z[field],atol=2e-6,rtol=2e-5),(f,field,err)
     assert np.isfinite(z['trace']).all();total+=24;simulated+=0 if meta.get('structural_zero') else 24
    if not key.startswith('bio'):
     grad=dest/'gradient.npz';meta=json.loads(grad.with_suffix('.json').read_text());assert sha(grad)==meta['sha256'];assert np.isfinite(np.load(grad)['gradient']).all()
    records.append({'model':model,'panel':key,'condition_banks':len(files),'status':'pass'})
 # Cross-check first-order MAE without importing analysis implementation.
 summary=json.loads((OUT/'analysis/summary.json').read_text());maes=[]
 for row in summary:
  keys=['dev_slow','dev_fast'] if row['split']=='development' else ['final_slow','final_fast'];errors=[]
  for key in keys:
   d=OUT/row['split']/row['model']/key;b=np.load(d/'baseline.npz');g=np.load(d/'gradient.npz')['gradient'];
   for i in range(65):
    z=np.load(d/f's{int(row["strength"]*100):03d}-cell{i:02d}.npz');valid=(z['denominator']>1e-8)&(b['denominator']>1e-8);errors.extend(abs(row['strength']*g[:,i]-(z['dsi']-b['dsi']))[valid].tolist())
  independent_mae=float(np.mean(errors));reported=row['methods']['first_order']['mae'];assert abs(independent_mae-reported)<1e-7;maes.append({'model':row['model'],'strength':row['strength'],'mae':independent_mae})
 result={'status':'pass','complete_panels':records,'represented_trajectories_including_biology':total,'fresh_reference_trajectories_including_biology':simulated,'independent_max_errors':maxerr,'mae_cross_checks':maes,'scope':'Every saved condition metric and checksum; independent float64 NumPy DSI and MAE. Fresh simulation replay is a separate check.'};save_json(OUT/'verification/independent-audit.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
