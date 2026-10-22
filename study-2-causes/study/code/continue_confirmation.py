"""Continue the confirmation run past a model whose gate stopped it (added 16 October 2026, after the freeze).

PROTOCOL.md Section 10 applies the gates per model. The frozen runner (`run_arms.py --phase confirmation`) stops the
whole run at the first gate failure; this driver calls the frozen `run_arms.run_model` unchanged for each remaining
confirmation model, in the same order, skips the models listed with --skip, and records any gate failure in
`results/confirmation-gate-failures.json` instead of stopping. No record, gate or tolerance changes: a gate failure
still stops that model's run before its gate record is written, and that model stays incomplete. Any other error stops
the run. Usage (from the project folder):
  python -B study/code/gpu_lock.py <python> -B study/code/continue_confirmation.py --skip 009
"""
from __future__ import annotations

import argparse
import json
import time

import run_arms
from ai09_common import CONFIRMATION, RESULTS, save_json, utc_now

GATE_MESSAGES = ("differs from AI-06", "disagree with AI-06")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()
    signature = run_arms.code_signature()
    path = RESULTS / "confirmation-gate-failures.json"
    failures = json.loads(path.read_text()) if path.exists() else []
    started = time.perf_counter()
    for model in CONFIRMATION:
        if model in args.skip:
            continue
        try:
            run_arms.run_model(model, "cuda", signature)
        except RuntimeError as error:
            if not any(message in str(error) for message in GATE_MESSAGES):
                raise
            failures.append({"model": model, "error": str(error), "utc": utc_now(), "code_signature": signature})
            save_json(path, failures)
            print(f"gate failure, continuing: {error}", flush=True)
    complete = [m for m in CONFIRMATION if all((RESULTS / "runs" / m / k / "complete.json").exists()
                                               for k in ("dev_fast", "final_slow"))]
    save_json(RESULTS / "confirmation-continue-complete.json", {
        "complete_models": complete, "incomplete_models": [m for m in CONFIRMATION if m not in complete],
        "skipped": args.skip, "code_signature": signature, "elapsed_seconds": time.perf_counter() - started,
        "utc": utc_now()})


if __name__ == "__main__":
    main()
