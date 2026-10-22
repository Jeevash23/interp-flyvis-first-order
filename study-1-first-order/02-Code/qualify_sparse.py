import json,runpy,time
from pathlib import Path
from safe_render_sparse import validate_small_sparse_renderer,validate_full_sparse_renderer,safe_sparse_renderer
P=Path(__file__).resolve().parents[1]
for name,fn in [('sparse-small',validate_small_sparse_renderer),('sparse-full',validate_full_sparse_renderer)]:
 t=time.monotonic();r=fn();r['elapsed_seconds']=time.monotonic()-t;(P/f'04-Results/{name}-validation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
 if r['status']!='pass':raise SystemExit(name+' equivalence failed')
with safe_sparse_renderer():runpy.run_path(str(P/'02-Code/qualify.py'),run_name='__main__')
