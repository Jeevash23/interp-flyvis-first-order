"""Post hoc (17 October 2026, after the confirmation results were read): first- and second-order errors in the cases
where the linearised trajectory changes the index's denominator by more than its own size, and in the rest.

Reads analysis/results/cases-silencing-peak-s100.npz (written by posthoc_paper.py) and writes
analysis/results/posthoc-denominator.json: per-model mean absolute errors of AI-06's first-order estimate, the
second-order estimate and predicting no change in the two groups, averaged over the confirmation models.
"""
from pathlib import Path
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT / "analysis/results/cases-silencing-peak-s100.npz")
models = sorted(set(data["model"].tolist()))
large = np.abs(data["relative_denominator_change"]) > 1
records = []
for model in models:
    select = data["model"] == model
    record = {"model": model}
    for group, mask in (("large", select & large), ("rest", select & ~large)):
        if not mask.any():
            continue
        exact = data["exact"][mask]
        record[group] = {"cases": int(mask.sum()),
                         "first": float(np.abs(data["first"][mask] - exact).mean()),
                         "second": float(np.abs(data["second"][mask] - exact).mean()),
                         "zero": float(np.abs(exact).mean())}
    records.append(record)


def ensemble(group, field):
    values = [r[group][field] for r in records if group in r]
    return {"mean": float(np.mean(values)), "models": len(values)}


summary = {"label": "post hoc (written after the confirmation results were read)",
           "threshold": "linearised relative change of the index's denominator at strength 1 above 1 in absolute value",
           **{f"{group}_{field}": ensemble(group, field) for group in ("large", "rest") for field in ("first", "second", "zero")},
           "rest_second_below_first_models": int(sum(r["rest"]["second"] < r["rest"]["first"] for r in records)),
           "models": len(records), "per_model": records}
(ROOT / "analysis/results/posthoc-denominator.json").write_text(json.dumps(summary, indent=1) + "\n")
print({k: v for k, v in summary.items() if k != "per_model"})
