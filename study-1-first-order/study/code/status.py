"""Compact, read-only status for a long-running local revision."""
import json
from pathlib import Path
from revision_common import RESULTS, CONFIG, MODELS

for name in ("pipeline.json","progress.json","delivery-check.json"):
    path=RESULTS/name
    if path.exists():
        print(name, json.dumps(json.loads(path.read_text())))
done=[m for m in MODELS if (RESULTS/"runs"/m/"complete.json").exists()]
print(f"Completed model simulations: {len(done)}/50; development {sum(m in CONFIG['development_models'] for m in done)}/4; confirmation {sum(m not in CONFIG['development_models'] for m in done)}/46")
