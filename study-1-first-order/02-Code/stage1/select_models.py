"""Extract cluster labels without invoking serialized analysis/scikit-learn classes."""
from common import *
import pickle,io,numpy as np,h5py
class Record:pass
class Reader(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('numpy.core.multiarray','_reconstruct'):np.core.multiarray._reconstruct,('numpy.core.multiarray','scalar'):np.core.multiarray.scalar,('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype}
  if (module,name) in allowed:return allowed[(module,name)]
  if (module,name) in [('flyvis.analysis.clustering','GaussianMixtureClustering'),('flyvis.analysis.clustering','Embedding'),('sklearn.preprocessing._data','MinMaxScaler'),('sklearn.mixture._gaussian_mixture','GaussianMixture')]:return Record
  raise ValueError(f'Forbidden serialized class {module}.{name}')
root=P/'03-Data/flyvis-root/results/flow/0000';p=root/'umap_and_clustering/T4c.pickle';obj=Reader(io.BytesIO(p.read_bytes())).load();labels=obj.labels;mask=obj.embedding.mask
assert mask.shape==(50,) and mask.dtype==bool and labels.shape==(int(mask.sum()),)
ids=sorted(x.name for x in root.iterdir() if x.is_dir() and x.name.isdecimal());assert len(ids)==50
labelmap=dict(zip(np.asarray(ids)[mask],map(int,labels)));rows=[]
for m in ids:
 with h5py.File(root/m/'validation_loss.h5') as f:loss=float(f['data'][()])
 rows.append({'model':m,'validation_loss':loss,'T4c_cluster':labelmap.get(m),'checkpoint_sha256':sha(root/m/'best_chkpt')})
rows=sorted(rows,key=lambda r:(r['validation_loss'],r['model']));c0=labelmap['000'];other=next(r for r in rows if r['T4c_cluster'] is not None and r['T4c_cluster']!=c0);dev=['000',other['model']];held=[next(r['model'] for r in rows if r['T4c_cluster']==c and r['model'] not in dev) for c in [c0,other['T4c_cluster']]]
result={'status':'verified_released_metadata','cluster_cell':'T4c','cluster_file_sha256':sha(p),'ordering':'lexical model directory order as native Ensemble.model_paths_from_parent; mask maps50indices','development':dev,'held_out':held,'rule':'tutorial000; lowest validation loss from different cluster; next-lowest unused per represented cluster; path tie-break','models':rows,'serialized_estimators_not_executed':True}
save_json(DATA/'model-selection.json',result);print(json.dumps({k:v for k,v in result.items() if k!='models'},indent=2))
