"""Frozen model-level summaries, pair contrasts and 1000-permutation controls."""
from common import *
import numpy as np
from scipy.stats import rankdata
from core import PAIRS,TARGETS
def finite(x):return float(x) if np.isfinite(x) else None
def mean(x):
 x=[v for v in x if v is not None and np.isfinite(v)];return float(np.mean(x)) if x else None
def corr(x,y):
 if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:return None
 return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
def top_k(scores,truth):
 k=int(truth.sum())
 if k==0:return None
 cutoff=np.sort(scores)[-k];above=scores>cutoff;ties=scores==cutoff
 return float((truth[above].sum()+(k-above.sum())*truth[ties].mean())/k)
def shuffled_corr(scores,truth,perms):
 x=rankdata(scores);y=rankdata(truth);x=x-x.mean();y=y-y.mean();den=np.linalg.norm(x)*np.linalg.norm(y)
 return (x[perms]@y)/den if den>0 else np.full(len(perms),np.nan)
def intervals(a):
 if not np.isfinite(a).any():return None
 return {'mean':float(np.nanmean(a)),'q025':float(np.nanquantile(a,.025)),'q975':float(np.nanquantile(a,.975))}
def read(path):
 meta=json.loads(path.with_suffix('.json').read_text())
 if sha(path)!=meta['sha256']:raise RuntimeError(f'Corrupt result {path}')
 with np.load(path) as z:return {k:z[k] for k in z.files}
def analyse():
 sel=json.loads((DATA/'model-selection.json').read_text());structure=json.loads((DATA/'network-structure.json').read_text());names=structure['model_types'];summary=[];pairs=[];biology=[];units=[];diagnostics=[]
 for split,models,keys in [('development',sel['development'],['dev_slow','dev_fast']),('final',sel['held_out'],['final_slow','final_fast'])]:
  for model in models:
   rng=np.random.default_rng(20261002+int(model));unrestricted=np.stack([rng.permutation(65) for _ in range(1000)])
   for strength in [.5,1.]:
    errors=[];exact_flags=[];pred_flags=[];rank={k:[] for k in ['first_order','synapse_count','native_current']};precision={k:[] for k in rank};perm={k:[] for k in rank};stratperm={k:[] for k in rank};baseline=[];invalid=0;tie_units=0;near_units=0
    for key in keys:
     d=OUT/split/model/key;b=read(d/'baseline.npz');g=read(d/'gradient.npz')['gradient'];meta=json.loads((d/'gradient.json').read_text());baseline.extend(b['dsi'].tolist())
     records=[read(d/f's{int(strength*100):03d}-cell{i:02d}.npz') for i in range(65)];exact=np.stack([r['dsi']-b['dsi'] for r in records],1);prediction=strength*g;den=np.stack([r['denominator'] for r in records],1)
     for r in range(8):
      valid=(den[r]>1e-8)&(b['denominator'][r]>1e-8)&np.isfinite(exact[r])&np.isfinite(prediction[r]);invalid+=int((~valid).sum());truth=abs(exact[r,valid])>=.10;pred=abs(prediction[r,valid])>=.10
      errors.extend((prediction[r,valid]-exact[r,valid]).tolist());exact_flags.extend(truth.tolist());pred_flags.extend(pred.tolist());scores={'first_order':abs(prediction[r]),'synapse_count':b['synapse_count'][r],'native_current':b['current_score'][r]}
      tie=meta['peak_tie_counts_per_readout'][r]>0 or meta['denominator_ties'][r];tie_units+=int(tie);near=meta['near_peak_counts_per_readout'][r]>0;near_units+=int(near)
      for method,score in scores.items():
       rho=corr(score[valid],abs(exact[r,valid]));pk=top_k(score[valid],truth);rank[method].append(rho);precision[method].append(pk)
       unit={'split':split,'model':model,'strength':strength,'stimulus':key,'readout':TARGETS[r],'method':method,'spearman':rho,'top_k_precision':pk,'important_count':int(truth.sum()),'n_valid':int(valid.sum()),'gradient_tie_flag':bool(tie)};units.append(unit)
       if valid.all():
        perm[method].append(shuffled_corr(score,abs(exact[r]),unrestricted))
        counts=b['synapse_count'][r];positive=counts[counts>0];cuts=np.quantile(positive,[1/3,2/3]) if len(positive) else np.array([0,0]);bins=np.where(counts==0,0,1+np.digitize(counts,cuts));pr=np.tile(np.arange(65),(1000,1));srng=np.random.default_rng(20261002+int(model)*100+r)
        for binid in np.unique(bins):
         idx=np.flatnonzero(bins==binid)
         if len(idx)>1:pr[:,idx]=np.stack([srng.permutation(idx) for _ in range(1000)])
        stratperm[method].append(shuffled_corr(score,abs(exact[r]),pr))
      diagnostics.append({'model':model,'strength':strength,'stimulus':key,'readout':TARGETS[r],'baseline_dsi':float(b['dsi'][r]),'loss_claim_eligible':bool(b['dsi'][r]>=.10),'denominator':float(b['denominator'][r]),'gradient_tie_flag':bool(tie),'near_peak_flag_conservative':bool(near),'important_count':int(truth.sum())})
     for pidx,(A,B) in enumerate(PAIRS):
      joint=read(d/f's{int(strength*100):03d}-pair{pidx}.npz');aa=records[names.index(A)];bb=records[names.index(B)]
      vals={field:joint[field]-aa[field]-bb[field]+b[field] for field in ['dsi','peaks','raw_mean','raw_integral']};norm={field:vals[field]/np.maximum(abs(b[field]),.10 if field=='dsi' else .01) for field in vals}
      pairs.append({'model':model,'split':split,'strength':strength,'stimulus':key,'pair':[A,B],'dsi_interaction':vals['dsi'].tolist(),'raw_mean_interaction':vals['raw_mean'].tolist(),'raw_integral_interaction':vals['raw_integral'].tolist(),'peak_interaction':vals['peaks'].tolist(),'max_abs_dsi':float(abs(vals['dsi']).max()),'dsi_at_least_001_count':int((abs(vals['dsi'])>=.01).sum()),'max_abs_raw_mean':float(abs(vals['raw_mean']).max()),'max_abs_raw_integral':float(abs(vals['raw_integral']).max()),'max_abs_normalized_raw_mean':float(abs(norm['raw_mean']).max()),'max_abs_normalized_raw_integral':float(abs(norm['raw_integral']).max()),'normalized_dsi':norm['dsi'].tolist(),'normalized_raw_mean':norm['raw_mean'].tolist(),'normalized_raw_integral':norm['raw_integral'].tolist(),'normalized_peaks':norm['peaks'].tolist()})
    er=np.array(errors);ef=np.array(exact_flags);pf=np.array(pred_flags);tp=int((ef&pf).sum());methods={}
    for method in rank:
     methods[method]={'mean_spearman':mean(rank[method]),'valid_rank_units':sum(x is not None for x in rank[method]),'mean_top_k_precision':mean(precision[method]),'unrestricted_permutation_mean_rho':intervals(np.nanmean(perm[method],axis=0)) if perm[method] else None,'count_stratified_permutation_mean_rho':intervals(np.nanmean(stratperm[method],axis=0)) if stratperm[method] else None}
    methods['first_order'].update(mae=float(abs(er).mean()),signed_bias=float(er.mean()),precision_at_010=tp/int(pf.sum()) if pf.sum() else None,recall_at_010=tp/int(ef.sum()) if ef.sum() else None,predicted_important=int(pf.sum()),exact_important=int(ef.sum()),true_positives=tp,n_valid=int(len(er)))
    methods['random_ranking']={'expected_mean_top_k_precision':mean([u['important_count']/u['n_valid'] for u in units if u['model']==model and u['strength']==strength and u['method']=='first_order' and u['important_count']>0]),'expected_spearman':0.0}
    summary.append({'split':split,'model':model,'strength':strength,'methods':methods,'baseline_dsi_range':[float(min(baseline)),float(max(baseline))],'invalid_effects':invalid,'gradient_tie_readout_speed_units':tie_units,'near_peak_readout_speed_units_conservative':near_units})
    for key in ['bio_slow','bio_fast']:
     d=OUT/split/model/key;b=read(d/'baseline.npz');fields={}
     for label,code in [('Mi1',f'cell{names.index("Mi1"):02d}'),('Tm3',f'cell{names.index("Tm3"):02d}'),('Mi1+Tm3','pair0')]:
      q=read(d/f's{int(strength*100):03d}-{code}.npz');fields[label]={'dsi_delta':(q['dsi']-b['dsi']).tolist(),'raw_mean_delta':(q['raw_mean']-b['raw_mean']).tolist(),'baseline_dsi':b['dsi'].tolist()}
     biology.append({'model':model,'split':split,'strength':strength,'stimulus':key,'degrees_per_second':12.5 if key=='bio_slow' else 300,'effects':fields,'eligibility':'not tested under matched conditions','reason':'Speeds matched; T4/T5 voltage versus experimental HS/VS voltage/behaviour, stimulus geometry and acute synaptic-output attenuation versus chronic Kir2.1 remain unmatched.'})
 save_json(OUT/'analysis/summary.json',summary);save_json(OUT/'analysis/readout-speed-units.json',units);save_json(OUT/'analysis/pair-interactions.json',pairs);save_json(OUT/'analysis/biology.json',biology);save_json(OUT/'analysis/diagnostics.json',diagnostics)
 finite_set=[]
 for subset in ['all','development','final']:
  for strength in [.5,1.]:
   rows=[r for r in summary if r['strength']==strength and (subset=='all' or r['split']==subset)]
   record={'subset':subset,'strength':strength,'models':[r['model'] for r in rows],'metrics':{}}
   for method in ['first_order','synapse_count','native_current','random_ranking']:
    for metric in ['mae','signed_bias','mean_spearman','precision_at_010','recall_at_010','mean_top_k_precision','expected_mean_top_k_precision']:
     values=[r['methods'][method].get(metric) for r in rows];values=[x for x in values if x is not None]
     if values:record['metrics'][method+'/'+metric]={'mean':float(np.mean(values)),'range':[float(min(values)),float(max(values))]}
   finite_set.append(record)
 save_json(OUT/'analysis/finite-set.json',finite_set);print(json.dumps(summary,indent=2))
if __name__=='__main__':analyse()
