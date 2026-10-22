"""Aggregate measured compute, preserving the interrupted attempt's timing limits."""
from common import *
from datetime import datetime
cost=json.loads((OUT/'cost-decision.json').read_text())
stim=json.loads((DATA/'stimulus-manifest.json').read_text())
length={r['key']:r['frames']-200 for r in stim}
runs=[json.loads((OUT/f'run-complete-{s}.json').read_text()) for s in ['development','final']]
archive=HIST/'interruption-2026-10-03'
interruption=json.loads((archive/'interruption.json').read_text())
cutoff=datetime.fromisoformat(interruption['inspection_utc']).timestamp()
original_panels=[json.loads(p.read_text()) for p in sorted((archive/'final').glob('*/*/panel-run.json'))]
finished_original={(p['model'],p['key']) for p in original_panels}
partial_original_s=0.; partial_detail=[]
panels=[];frame_trajectories=0;exact_time=0.;reference_count=0;gradtime=0.;baseline_time=0.
for split in ['development','final']:
 for panel in sorted((OUT/split).glob('*/*/panel-run.json')):
  d=json.loads(panel.read_text());panels.append(d)
  baseline=json.loads((panel.parent/'baseline.json').read_text());baseline_time+=baseline['wall_s'];reference_count+=24
  is_partial=split=='final' and (d['model'],d['key']) not in finished_original
  old_partial=baseline['wall_s'] if is_partial and baseline['utc_unix']<cutoff else 0.
  for meta in panel.parent.glob('s*.json'):
   m=json.loads(meta.read_text())
   if not m['structural_zero']:
    reference_count+=m['simulated_trajectories'];frame_trajectories+=m['simulated_trajectories']*length[m['stimulus']]
    exact_time+=m['wall_s']
    if is_partial and m['utc_unix']<cutoff:old_partial+=m['wall_s']
  grad=panel.parent/'gradient.json'
  if grad.exists():
   g=json.loads(grad.read_text());seconds=g['forward_s']+sum(g['backward_s']);gradtime+=seconds
   if is_partial and g['utc_unix']<cutoff:old_partial+=seconds
  if old_partial:partial_detail.append({'model':d['model'],'panel':d['key'],'recorded_component_s':old_partial});partial_original_s+=old_partial
# Per-condition allocations survive a resume, unlike panel-run.json, which records only the latest invocation.
rate=exact_time/frame_trajectories
core_length=sum(length[k] for k in ['dev_slow','dev_fast','final_slow','final_fast'])*2
s2_frames=core_length*24*2*(cost['E']-len(cost['connection_structural_zeros']))
s2_exact=s2_frames*rate;s2_total=(s2_exact+2*gradtime+baseline_time)*1.25+900
lower_s=sum(r['wall_s'] for r in runs)+sum(p['wall_s'] for p in original_panels)+partial_original_s
all_panels=panels+original_panels
record={
 'study_wall_hours':lower_s/3600,
 'study_wall_time_status':'recorded_lower_bound_due_to_interrupted_original_final_attempt',
 'timing_components':{'completed_development_worker_s':runs[0]['wall_s'],'completed_final_resume_worker_s':runs[1]['wall_s'],'preserved_original_final_completed_panel_s':sum(p['wall_s'] for p in original_panels),'original_final_partial_panel_recorded_s':partial_original_s,'partial_panel_detail':partial_detail},
 'timing_limits':'No complete original final-worker timer survived. Original model initialization, partial-panel preperiod/serialization, and unfinished/unrecorded work are excluded. Original completed panels and partial components are disjoint from the resumed worker. Overnight downtime excluded. This is not a complete elapsed or device-energy total.',
 'runs':runs,'panels':panels,'original_final_panels':original_panels,
 'reference_trajectories':reference_count,'exact_intervention_wall_s':exact_time,
 'exact_time_definition':'Sum of recorded per-condition allocations for completed nonzero interventions; baseline, preperiod, gradient, serialization, and incomplete discarded work excluded.',
 'exact_intervention_frame_trajectories':frame_trajectories,'observed_seconds_per_frame_trajectory':rate,
 'observed_equivalent_seconds_per_485frame_trajectory':rate*485,'gradient_total_s':gradtime,'baseline_total_s':baseline_time,
 'peak_rss_mib':max(r['peak_rss_mib'] for r in all_panels),'peak_cuda_reserved_mib':max(r['peak_cuda_reserved_mib'] for r in all_panels),
 'stage2_extrapolation':{'additional_connection_exact_hours':s2_exact/3600,'with_allowances_hours':s2_total/3600,'scope':'Same four-model/48-stimulus/two-strength grid; 2 zero type pairs skipped; measured completed Stage1 throughput. Twice observed cell-gradient time, observed baseline time, 25 percent and 15 minutes allowance; connection gradients not directly benchmarked.'},
 'excluded_from_study_worker_wall_time':'Environment downloads/setup; qualification/benchmarks; analysis/report generation; time between separate worker launches; unrecorded work in interrupted attempt.'}
save_json(OUT/'actual-compute.json',record)
print(json.dumps({k:v for k,v in record.items() if k not in ['runs','panels','original_final_panels']},indent=2))
