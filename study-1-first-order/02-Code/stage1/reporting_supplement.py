"""Post-run reporting corrections; preserve the frozen analysis and all source outputs."""
from common import *
import numpy as np
summary_path=OUT/'analysis/summary.json';units_path=OUT/'analysis/readout-speed-units.json';pairs_path=OUT/'analysis/pair-interactions.json'
summary=json.loads(summary_path.read_text());units=json.loads(units_path.read_text());pairs=json.loads(pairs_path.read_text());names=json.loads((DATA/'network-structure.json').read_text())['model_types']
perms=[]
for row in summary:
 for method in ['first_order','synapse_count','native_current']:
  us=[u for u in units if u['model']==row['model'] and u['strength']==row['strength'] and u['method']==method and u['n_valid']==65]
  finite=[u['spearman'] for u in us if u['spearman'] is not None]
  perms.append({'model':row['model'],'split':row['split'],'strength':row['strength'],'method':method,'complete_units':len(us),'estimable_complete_units':len(finite),'observed_mean_spearman_same_units':float(np.mean(finite)) if finite else None,'unrestricted':row['methods'][method]['unrestricted_permutation_mean_rho'],'count_stratified':row['methods'][method]['count_stratified_permutation_mean_rho'],'units':[{'stimulus':u['stimulus'],'readout':u['readout'],'spearman':u['spearman']} for u in us]})
qualified=[]
for row in pairs:
 d=OUT/row['split']/row['model']/row['stimulus'];strength=f"s{int(row['strength']*100):03d}";A,B=row['pair']
 # Read the pair ID from saved metadata instead of reimplementing its enumeration.
 files=[d/'baseline.npz',d/f'{strength}-cell{names.index(A):02d}.npz',d/f'{strength}-cell{names.index(B):02d}.npz']
 joint=next(f for f in d.glob(strength+'-pair*.npz') if json.loads(f.with_suffix('.json').read_text())['names']==[A,B]);files.append(joint)
 banks=[]
 for f in files:
  assert sha(f)==json.loads(f.with_suffix('.json').read_text())['sha256']
  with np.load(f) as z:banks.append({'denominator':z['denominator'],'dsi':z['dsi']})
 valid=np.logical_and.reduce([(z['denominator']>1e-8)&np.isfinite(z['dsi']) for z in banks])
 original=np.asarray(row['dsi_interaction']);q=dict(row)
 q.update(unfiltered_frozen_max_abs_dsi=row['max_abs_dsi'],unfiltered_frozen_dsi_count=row['dsi_at_least_001_count'],dsi_eligible_mask=valid.tolist(),dsi_eligible_count=int(valid.sum()),dsi_interaction_valid=[float(x) if ok else None for x,ok in zip(original,valid)],max_abs_dsi=float(abs(original[valid]).max()) if valid.any() else None,dsi_at_least_001_count=int((abs(original[valid])>=.01).sum()))
 qualified.append(q)
# Report biology values only where baseline and intervention denominators are valid.
bio=json.loads((OUT/'analysis/biology.json').read_text())
for row in bio:
 d=OUT/row['split']/row['model']/row['stimulus'];prefix=f"s{int(row['strength']*100):03d}";b=np.load(d/'baseline.npz')
 for label,code in [('Mi1',f'cell{names.index("Mi1"):02d}'),('Tm3',f'cell{names.index("Tm3"):02d}'),('Mi1+Tm3','pair0')]:
  z=np.load(d/f'{prefix}-{code}.npz');valid=(b['denominator']>1e-8)&(z['denominator']>1e-8)&np.isfinite(z['dsi'])&np.isfinite(b['dsi']);v=z['dsi']-b['dsi']
  row['effects'][label].update(dsi_valid_mask=valid.tolist(),valid_T4_count=int(valid[:4].sum()),mean_valid_T4_delta=float(v[:4][valid[:4]].mean()) if valid[:4].any() else None)
record={'date':'2026-10-03','status':'reporting_corrections_after_frozen_run','reason':'Observed mean ranks and permutation controls had differing coverage; pair/biology DSI displays lacked the predeclared denominator eligibility filter. Original frozen analysis and results remain unchanged. No thresholds, models, stimuli or methods were selected after outcomes.','source_sha256':{str(f.relative_to(P)):sha(f) for f in [summary_path,units_path,pairs_path,OUT/'analysis/biology.json']},'permutation_comparisons':perms,'pair_interactions':qualified,'biology':bio}
save_json(OUT/'analysis/reporting-supplement.json',record)
print(json.dumps({'status':record['status'],'eligible_pair_dsi':sum(q['dsi_eligible_count'] for q in qualified),'nominal_pair_dsi':len(qualified)*8,'qualifying_pairs':sorted(set('/'.join(q['pair']) for q in qualified if q['max_abs_dsi'] is not None and q['max_abs_dsi']>=.01)),'max_biology_invalid_T4':max(4-e['valid_T4_count'] for r in bio for e in r['effects'].values())},indent=2))
