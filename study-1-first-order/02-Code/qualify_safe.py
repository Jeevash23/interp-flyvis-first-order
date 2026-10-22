import json,runpy
from pathlib import Path
from safe_render import validate_small_renderer,safe_renderer,renderer_metadata
P=Path(__file__).resolve().parents[1]
r=validate_small_renderer();(P/'04-Results/renderer-validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if r['status']!='pass':raise SystemExit('Renderer equivalence failed')
(P/'06-Study-History/renderer-adapter.json').write_text(json.dumps(renderer_metadata(),indent=2))
with safe_renderer():runpy.run_path(str(P/'02-Code/qualify.py'),run_name='__main__')
