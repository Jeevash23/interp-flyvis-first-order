"""Apply AI-09 Amendment 2 (approved by the author on 16 October 2026) to the project at argv[1].

The baseline and zero-replacement gates move from an absolute 1e-5 to AI-06's normalised replay rule; the freeze check
uses freeze-2.json when it exists; freeze.py gains `freeze2`; the Makefile gains `freeze-2` and
`confirmation-continue`; a unit test covers the new rule; PROTOCOL.md gains Section 14; incomplete stimulus folders of
models that stopped at a gate are moved aside. Refuses to run while an AI-09 GPU job is alive. Edits are exact string
replacements and fail loudly if a target text is not found once.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(sys.argv[1]).resolve()
CODE = ROOT / "study/code"

alive = subprocess.run(["pgrep", "-f", "run_arms.py|continue_confirmation.py"], capture_output=True, text=True).stdout
if alive.strip() and "--test-copy" not in sys.argv:
    raise SystemExit(f"An AI-09 GPU job is still running ({alive.split()}); apply after it ends")


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{path.name}: expected one match for {old[:60]!r}, found {text.count(old)}")
    path.write_text(text.replace(old, new))


# 1. run_arms.py: AI-06's normalised replay tolerance for the baseline and zero-replacement gates.
run_arms = CODE / "run_arms.py"
replace(run_arms, '''def gate(estimate, reference):
    return np.abs(estimate - reference) <= 1e-4 + 0.02 * np.abs(reference)
''', '''def gate(estimate, reference):
    return np.abs(estimate - reference) <= 1e-4 + 0.02 * np.abs(reference)


REPLAY_TOLERANCE = 1e-4  # Amendment 2: AI-06's fresh-replay rule (study/code/verify_revision.py)


def normalised_difference(trace, reference) -> float:
    """Largest absolute difference divided by max(1, largest absolute reference voltage), as AI-06's replay check."""
    return float(np.abs(trace - reference).max()) / max(1.0, float(np.abs(reference).max()))
''')
replace(run_arms, '''        gates["baseline_max_abs_difference"] = float(np.abs(baseline - ai06_record(model, key, "baseline.npz")["trace"]).max())
        if gates["baseline_max_abs_difference"] > 1e-5:
            raise RuntimeError(f"Baseline differs from AI-06: {model} {key}")
''', '''        saved_baseline = ai06_record(model, key, "baseline.npz")["trace"]
        gates["baseline_max_abs_difference"] = float(np.abs(baseline - saved_baseline).max())
        gates["baseline_normalised_difference"] = normalised_difference(baseline, saved_baseline)
        if gates["baseline_normalised_difference"] > REPLAY_TOLERANCE:
            raise RuntimeError(f"Baseline differs from AI-06: {model} {key}")
''')
replace(run_arms, '''            gates[f"zero_replacement_{name}_max_abs_difference"] = float(np.abs(trace[0] - saved).max())
            if gates[f"zero_replacement_{name}_max_abs_difference"] > 1e-5:
                raise RuntimeError(f"Replacement by zero differs from AI-06: {model} {key} {name}")
''', '''            gates[f"zero_replacement_{name}_max_abs_difference"] = float(np.abs(trace[0] - saved).max())
            gates[f"zero_replacement_{name}_normalised_difference"] = normalised_difference(trace[0], saved)
            if gates[f"zero_replacement_{name}_normalised_difference"] > REPLAY_TOLERANCE:
                raise RuntimeError(f"Replacement by zero differs from AI-06: {model} {key} {name}")
''')

# 2. ai09_common.py: the freeze check uses the latest freeze.
replace(CODE / "ai09_common.py", '''    freeze = STUDY / "freeze.json"
    if not freeze.exists():''', '''    freeze = STUDY / "freeze-2.json"  # Amendment 2; the first freeze, freeze.json, is kept unchanged
    if not freeze.exists():
        freeze = STUDY / "freeze.json"
    if not freeze.exists():''')

# 3. freeze.py: a second freeze that records the first and the confirmation records already present.
freeze = CODE / "freeze.py"
replace(freeze, '''  python -B study/code/freeze.py freeze        writes study/freeze.json (hashes of every file the study depends on)
''', '''  python -B study/code/freeze.py freeze        writes study/freeze.json (hashes of every file the study depends on)
  python -B study/code/freeze.py freeze2       writes study/freeze-2.json after Amendment 2 (PROTOCOL.md Section 14)
''')
replace(freeze, '''if __name__ == "__main__":''', '''def freeze2() -> None:
    path = STUDY / "freeze-2.json"
    if path.exists():
        raise SystemExit(f"{path} exists; Amendment 2 is frozen once")
    first = json.loads((STUDY / "freeze.json").read_text())
    if (RESULTS / "analysis/confirmation").exists():
        raise SystemExit("A confirmation analysis exists; Amendment 2 must be frozen before any confirmation outcome")
    files = sorted(set(first["files"]) | {"study/freeze.json", "study/results/GATE-FAILURE-009.md",
                                          "study/amendment-2/apply_amendment2.py"}
                   | {str(p.relative_to(ROOT)) for p in (STUDY / "code").glob("*.py")})
    records = sorted(m for m in CONFIRMATION if (RESULTS / "runs" / m / "dev_fast" / "complete.json").exists()
                     or (RESULTS / "runs" / m / "final_slow" / "complete.json").exists())
    save_json(path, {"utc": utc_now(), "kind": "second local analysis freeze, after Amendment 2 (PROTOCOL.md Section 14)",
                     "previous_freeze_sha256": sha256(STUDY / "freeze.json"),
                     "development_models": DEVELOPMENT, "confirmation_models": CONFIRMATION,
                     "confirmation_outcomes_computed": False,
                     "confirmation_models_with_run_records": records,
                     "files": {rel: sha256(ROOT / rel) for rel in files},
                     "ai06_inputs": {rel: sha256(AI06 / rel) for rel in AI06_INPUTS}})


if __name__ == "__main__":''')
replace(freeze, '''import sys
''', '''import json
import sys
''')
replace(freeze, '''{"predictions": record_predictions, "addendum": record_addendum, "freeze": freeze}[sys.argv[1]]()''',
        '''{"predictions": record_predictions, "addendum": record_addendum, "freeze": freeze,
     "freeze2": freeze2}[sys.argv[1]]()''')

# 4. Makefile targets.
makefile = ROOT / "Makefile"
replace(makefile, ".PHONY: pilot-cpu pilot-gpu test predictions development analysis-development addendum freeze confirmation analysis",
        ".PHONY: pilot-cpu pilot-gpu test predictions development analysis-development addendum freeze confirmation freeze-2 confirmation-continue analysis")
replace(makefile, '''# Confirmation analysis:''', '''# Amendment 2 (PROTOCOL.md Section 14): the second freeze, and the per-model run that logs gate failures.
freeze-2:
	$(PY) study/code/freeze.py freeze2
confirmation-continue:
	$(LOCK) study/code/continue_confirmation.py

# Confirmation analysis:''')

# 5. A unit test of the new rule.
replace(CODE / "test_arms.py", '''

if __name__ == "__main__":''', '''

def test_replay_tolerance():
    import run_arms
    reference = np.zeros((2, 3, 8))
    reference[0, 0, 0] = 20.0
    assert np.isclose(run_arms.normalised_difference(reference + 1e-5, reference), 1e-5 / 20)
    assert np.isclose(run_arms.normalised_difference(np.full_like(reference, 1e-5), np.zeros_like(reference)), 1e-5)
    assert run_arms.normalised_difference(reference + 9e-6, reference) <= run_arms.REPLAY_TOLERANCE


if __name__ == "__main__":''')

# 6. PROTOCOL.md Section 14.
replace(ROOT / "study/PROTOCOL.md", "## References", """## 14. Amendment 2 (16 October 2026, during the confirmation run)

Approved by the author after the confirmation run stopped at model 009 (`results/GATE-FAILURE-009.md`). No AI-09
outcome had been computed for any confirmation model; only the gate values of the models already simulated had been
seen.

**Change.** Two of the per-model gates of Section 10, the baseline and replacement by zero, compared traces with an
absolute tolerance of 1e-5. That is ten times stricter than AI-06's own fresh-replay check, and the GPU's unfixed order
of summation produces absolute differences of that size in models whose voltages are large (model 009 reaches 15–22).
Both gates now use AI-06's replay rule: the largest absolute difference divided by max(1, the largest absolute voltage
of the AI-06 trace) must be at most 1e-4. The absolute differences are still recorded. The derivative gate is
unchanged. No computed quantity depends on a gate.

**Records.** Models 002, 003, 006, 007 and 008, and the models then completed by `code/continue_confirmation.py`
(010–012 and 014–018; it applies the gates per model and logs failures instead of stopping), passed the stricter gates
and are kept. Model 013 stopped at the same zero-replacement gate on its fast stimulus. At [local time] the run was
interrupted partway through model 019's slow stimulus, when the [redacted] session that had started it ended. The
incomplete stimulus folders are kept on the drive with the suffix `-attempt1-gate-stop` (009 and 013) or
`-attempt1-interrupted` (019), and those stimuli are simulated again under this amendment. The first freeze,
`freeze.json`, is kept unchanged; `freeze-2.json` records its hash, the amended files and the confirmation models whose
run records already existed.

## References""")

# 7. Move aside every incomplete stimulus folder: those of models that stopped at a gate, and any left by an
#    interrupted run. They were written by the code before this amendment, whose signature the runner would refuse.
failures_path = ROOT / "study/results/confirmation-gate-failures.json"
failed = sorted({f["model"] for f in json.loads(failures_path.read_text())}) if failures_path.exists() else []
moved = []
for model_folder in sorted((ROOT / "study/results/runs").iterdir()):
    for key in ("dev_fast", "final_slow"):
        folder = model_folder / key
        if folder.exists() and not (folder / "complete.json").exists():
            suffix = "-attempt1-gate-stop" if model_folder.name in failed else "-attempt1-interrupted"
            target = folder.with_name(key + suffix)
            if target.exists():
                raise SystemExit(f"{target} exists already")
            folder.rename(target)
            moved.append(str(target))
print(json.dumps({"applied": True, "models_with_gate_failures_overnight": failed, "moved_aside": moved}, indent=1))
