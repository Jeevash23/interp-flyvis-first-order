import json,runpy
from pathlib import Path
from safe_render_sparse import validate_full_sparse_renderer
P=Path(__file__).resolve().parents[1]
r=validate_full_sparse_renderer();(P/'04-Results/sparse-central-edge-validation.json').write_text(json.dumps(r,indent=2));print('central edge',r,flush=True)
if r['status']!='pass':raise SystemExit('Central edge renderer check failed')
runpy.run_path(str(P/'02-Code/recompute_metrics.py'),run_name='__main__')
runpy.run_path(str(P/'02-Code/verify_pilot.py'),run_name='__main__')
