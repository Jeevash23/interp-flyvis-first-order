"""Recompute native tutorial metrics from saved numeric records without FlyVis."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parents[1];out=P/'04-Results/qualification'
a=np.load(out/'native-responses.npz',allow_pickle=False);v=a['responses'][0];times=a['time'];angles=a['angle'];ints=a['intensity'];types=a['cell_type']
# Native full-offset peak window: 21 LED steps, 2.25 deg/step, speed19*5.8deg/s.
mask=(times>=0)&(times<=21*2.25/(19*5.8));peaks=np.maximum(v[:,mask,:],0).max(axis=1)
ref=json.loads((out/'qualification.json').read_text());rows=[]
for r in ref['rows']:
 j=int(np.flatnonzero(types==r['cell_type'])[0]);weights=peaks[ints==r['intensity'],j];ang=angles[ints==r['intensity']];z=(weights*np.exp(1j*np.deg2rad(ang))).sum();den=max(peaks[ints==k,j].sum() for k in (0,1));d=float(abs(z)/(den+1e-15));pd=float(np.rad2deg(np.angle(z)));rows.append({'cell_type':r['cell_type'],'dsi':d,'pd_degrees':pd,'dsi_error':abs(d-r['dsi']),'pd_error_degrees':abs((pd-r['pd_degrees']+180)%360-180)})
err=max(x['dsi_error'] for x in rows);angleerr=max(x['pd_error_degrees'] for x in rows)
result={'status':'pass' if err<1e-6 and angleerr<1e-4 and np.isfinite(v).all() and np.max(abs(peaks-a['peaks'][0]))<1e-6 else 'fail','independent_of_flyvis':True,'dtype':str(v.dtype),'finite':bool(np.isfinite(v).all()),'max_dsi_error':err,'max_angle_error_degrees':angleerr,'peak_max_error':float(np.max(abs(peaks-a['peaks'][0]))),'rows':rows}
(out/'independent-metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if result['status'] != 'pass':
    raise SystemExit(2)
