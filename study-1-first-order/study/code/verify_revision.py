"""Independent saved-trace checks and fresh CPU replay of the declared grid."""
from __future__ import annotations

import argparse
import gc
import json
import sys

import numpy as np

from revision_common import (
    ROOT, STUDY, RESULTS, CONFIG, MODELS, NAMES, causal_frames,
    independent_metrics, legacy_check, load_npz, original_freeze_check,
    save_json, sha256, stimulus, utc_now,
)


def check_record(path, stim, signature, max_errors):
    metadata=json.loads(path.with_suffix(".json").read_text())
    if metadata["sha256"]!=sha256(path) or metadata["simulation_signature"]!=signature:
        raise RuntimeError(f"Integrity mismatch: {path}")
    arrays=load_npz(path)
    if "trace" not in arrays:
        if not np.isfinite(arrays["gradient"]).all():
            raise RuntimeError(f"Nonfinite gradient: {path}")
        return
    if not np.isfinite(arrays["trace"]).all():
        raise RuntimeError(f"Nonfinite trace: {path}")
    recalculated=independent_metrics(arrays["trace"],stim)
    for key in ("dsi","denominator","peaks","raw_mean","raw_integral"):
        error=float(np.max(np.abs(recalculated[key]-arrays[key])))
        max_errors[key]=max(max_errors.get(key,0),error)
        if not np.allclose(recalculated[key],arrays[key],atol=2e-6,rtol=2e-5):
            raise RuntimeError(f"Independent readout mismatch: {path} {key}")
    if metadata.get("reused"):
        old=ROOT/metadata["source"]
        legacy_check(old)
        if sha256(old)!=metadata["source_sha256"]:
            raise RuntimeError(f"Reused input changed: {old}")


def replay(model):
    sys.path.insert(0,str(ROOT/"02-Code/stage1"))
    sys.path.insert(0,str(ROOT/"02-Code"))
    import torch
    from core import Core
    from streaming_native import gray_initial_state, streaming_forward, build_edge_gate, synaptic_gate
    rng=np.random.default_rng(CONFIG["bootstrap_seed"]+int(model))
    key=CONFIG["common_stimuli"][int(rng.integers(0,2))]
    candidates=[i for i,name in enumerate(NAMES) if name not in ("Mi11","Tm30")]
    cell=int(rng.choice(candidates)); strength=CONFIG["main_strengths"][int(rng.integers(0,2))]
    trajectory=int(rng.integers(0,24))
    stim=stimulus(key);frames=stim["stimulus"]
    if frames.ndim==4:
        frames=frames[:,:,0]
    core=Core(model,"cpu")
    gray=gray_initial_state(core.native)
    pre=streaming_forward(core.native,frames[trajectory,:200],.005,state=gray,indices=core.readout.numpy())
    gate=build_edge_gate(core.native,source_types=[NAMES[cell]],multiplier=1-strength)
    with synaptic_gate(core.native,gate):
        post=streaming_forward(core.native,frames[trajectory,200:200+causal_frames(key)],.005,
                               state=pre.final_state,indices=core.readout.numpy())
    actual=post.responses
    reference=load_npz(RESULTS/"runs"/model/key/f"s{round(100*strength):03d}-cell{cell:02d}.npz")["trace"][trajectory]
    absolute=float(np.max(np.abs(actual-reference)))
    relative=absolute/max(1.0,float(np.max(np.abs(reference))))
    if relative>1e-4:
        raise RuntimeError(f"Fresh CPU replay failed: model {model}, normalized error {relative}")
    result={"model":model,"stimulus":key,"cell":NAMES[cell],"strength":strength,"trajectory":trajectory,
            "max_absolute_error":absolute,"normalized_error":relative,"status":"pass",
            "execution":"independent native _next_state on CPU; spatial-edge gate"}
    del core, gray, pre, post, gate
    gc.collect()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("development","confirmation","all"),required=True)
    parser.add_argument("--skip-replay",action="store_true")
    args=parser.parse_args()
    from run_experiments import simulation_signature
    import torch
    torch.set_num_threads(CONFIG["cpu_threads"])
    torch.set_num_interop_threads(CONFIG["cpu_threads"])
    signature=simulation_signature()
    models=CONFIG["development_models"] if args.phase=="development" else [m for m in MODELS if m not in CONFIG["development_models"]]
    if args.phase=="all":
        models=MODELS
    maximum={};files=0;replays=[]
    for model in models:
        for key in CONFIG["common_stimuli"]:
            dest=RESULTS/"runs"/model/key
            completion=json.loads((dest/"complete.json").read_text())
            if completion["simulation_signature"]!=signature:
                raise RuntimeError(f"Panel signature mismatch: {model} {key}")
            strengths=CONFIG["main_strengths"]+(CONFIG["development_extra_strengths"] if model in CONFIG["development_models"] else [])
            paths=[dest/"baseline.npz",dest/"gradient.npz"]
            paths += [dest/f"s{round(100*s):03d}-cell{i:02d}.npz" for s in strengths for i in range(65)]
            if model in CONFIG["development_models"] and key=="dev_fast":
                paths.append(dest/"s100-ct1_joint.npz")
            stim=stimulus(key)
            for path in paths:
                check_record(path,stim,signature,maximum)
                files+=1
        if not args.skip_replay:
            replays.append(replay(model))
        print(f"Verified model {model}",flush=True)
    record={"status":"pass" if not args.skip_replay else "saved_metrics_pass_replay_pending",
        "models":models,"checked_files":files,"independent_max_errors":maximum,
        "fresh_cpu_replays":replays,"original_freeze":original_freeze_check(),
        "simulation_signature":signature,"utc":utc_now()}
    filename="development-gate.json" if args.phase=="development" else f"{args.phase}-verification.json"
    save_json(RESULTS/"verification"/filename,record)
    print(json.dumps({k:v for k,v in record.items() if k!="fresh_cpu_replays"}))


if __name__=="__main__":
    main()
