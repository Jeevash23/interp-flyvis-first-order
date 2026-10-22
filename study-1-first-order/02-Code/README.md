# Qualification and measurement infrastructure

This directory currently supports a bounded CPU reproduction of the official flyvis moving-edge tutorial, renderer memory adaptations, and independent recomputation of saved tutorial metrics. It does **not** implement the full [draft comparative study](../05-Notes/STUDY-PROTOCOL-DRAFT.md). Creating a source file or passing an import check is not evidence that its numerical checks passed.

Use [REPRODUCTION.md](../05-Notes/REPRODUCTION.md) for environment creation, verified archive provenance and the guarded command sequence. Source `runtime.env` from the AI-06 project root before importing flyvis: its paths are relative to the shell's current working directory. The environment hides CUDA, caps numerical threads at two and places `FLYVIS_ROOT_DIR` and caches inside this project. Other research environments, checkpoints and reserved evaluation data remain read-only.

| File | Purpose and limits |
|---|---|
| `requirements.in` | Direct package choices: flyvis 1.2.0, NumPy 1.26.4 and CPU PyTorch/torchvision. |
| `requirements.lock.txt` | Exact installed package versions; install the pinned CPU wheels from the PyTorch CPU index first. This is a version freeze, not a wheel-hash lock or proof of cross-platform equivalence. |
| `run_guarded.py` | Admission check, worker logs and 0.5-second resource samples; time/RAM stop affects only the launched process group. Its RSS sample is the direct worker's, not aggregate descendant RSS. Use one scientific process. |
| `qualify_sparse.py` | Current qualification entry point. Runs small-eye and full-grid renderer equivalence checks, then calls `qualify.py` under a temporary sparse renderer. Stops on renderer failure. |
| `safe_render_sparse.py` | Preserves native geometry, full-resolution rasterization, incoming pixel dtype and receptor membership, while avoiding zero-mask products. Summation order may differ; bitwise identity is not promised. The full-grid reference is the previously checked dense chunk adapter, not a new giant native allocation. |
| `safe_render.py`, `qualify_safe.py` | Earlier bounded dense receptor-chunk repair. Retained for provenance and the layered renderer comparison. |
| `qualify.py` | Fixed tutorial model `flow/0000/000`, 24 moving-edge conditions, native model recurrence and native DSI. Writes response arrays and a qualification JSON. Calling it directly bypasses the renderer protection; use the sparse entry point. |
| `recompute_metrics.py` | Reads saved numeric response arrays and recomputes peak responses, vector-sum DSI and preferred direction with NumPy, without importing flyvis. Tailored to this tutorial configuration; not a general arbitrary-stimulus evaluator. |
| `streaming_native.py` | Detached, single-trial native-step execution retaining central readouts, plus scoped spatial-edge output gates. Includes an explicitly called short validation routine. No simulation runs on import. Not yet a method-panel runner, differentiable gradient engine or independent implementation of model dynamics. |
| `monitor_worker.py` | Historical monitor used during the initial attempt. Subsequent jobs use the guardian from launch. |

Qualification writes `04-Results/qualification/native-responses.npz` and `qualification.json`. Independent recomputation writes `independent-metrics.json` in that same directory. Renderer results are `04-Results/sparse-small-validation.json` and `sparse-full-validation.json`. Guardian outputs are `06-Study-History/<name>.log`, `<name>-resources.jsonl` and `<name>-summary.json`. These paths are fixed; a fresh reproduction copy is needed to preserve an existing evidence set.

Inspect the JSON `status` fields and guardian summary. `qualify.py` and `recompute_metrics.py` can write a scientific `fail` while their Python process exits normally; exit code zero alone is insufficient. A resource-terminated or incomplete run has no valid scientific result. Fresh simulation timing must be distinguished from native cache hits.

The two incomplete renderer caches are quarantined under `06-Study-History/failed-cache/`. They must never be returned to the active cache or reused as completed renderings. The attempt journal preserves the initial memory-limit stop and the later speed-related stop.

`streaming_native.py` output gates multiply expanded physical-edge weights rather than changing shared fitted parameters or writing voltages. The helper defaults are infrastructure, not a frozen scientific intervention: whether grey adaptation and the dataset preperiod occur inside a gate must be explicit. Its `@torch.no_grad()` execution and detached gate tensors cannot supply attribution gradients. The draft's exhaustive reference, gradient approximations, held-out model panel, pair interactions and biological scoring remain separate work.

Numerical completion and current run status are recorded in the result JSON files and [attempt journal](../06-Study-History/ATTEMPT-JOURNAL.md). The documentation was prepared while qualification/recomputation were pending; no complete method-ranking study or complete numerical clean-environment replay is asserted here. See the final report/status record for later updates.

Flyvis source is MIT; preserve its upstream license and adaptation provenance. The pretrained-weight/data license was not separately verified. Dependency licenses are recorded individually in Study History. Do not label every model/data asset MIT or redistribute them on that assumption.


## Verified delivery status (2 October 2026)

The qualification and bounded numerical checks now pass; see `04-Results/qualification/` via the project entry page. Final status supersedes earlier pending wording in this guide. Scientific failure branches now exit nonzero after saving their JSON. Source/weight license and direct-worker RSS caveats still apply. Full comparative methods and evaluation remain unimplemented/unexecuted.
