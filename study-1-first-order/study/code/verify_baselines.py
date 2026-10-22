"""Check checkpoint indexing against the native parameter API, without new responses."""
import json
import sys
import numpy as np
from revision_common import ROOT, RESULTS, MODELS, save_json, sha256, utc_now


def main():
    sys.path.insert(0, str(ROOT / "02-Code/stage1"))
    import torch
    from core import Core
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    core = Core("000", "cpu")
    path = ROOT / "01-Papers/sources/flyvis/flyvis/connectome/fib25-fib19_v2.2.json"
    edges = json.loads(path.read_text())["edges"]
    pairs = [(e["src"], e["tar"]) for e in edges if not (e["src"] == e["tar"] == "Lawf1")]
    native_pairs = [tuple(pair) for pair in core.native.edge_params.syn_strength.keys]
    if pairs != native_pairs:
        raise RuntimeError("Anatomical edge order differs from the native learned-strength API")
    records = []
    for model in MODELS:
        checkpoint = ROOT / f"03-Data/flyvis-root/results/flow/0000/{model}/chkpts/chkpt_00000"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["network"]
        strength = state["edges_syn_strength"].numpy()
        if len(strength) != len(pairs) or not np.isfinite(strength).all() or np.min(strength) < 0:
            raise RuntimeError(f"Invalid learned-strength baseline: model {model}")
        records.append({"model": model, "minimum_strength": float(np.min(strength)),
                        "maximum_strength": float(np.max(strength)), "checkpoint_sha256": sha256(checkpoint)})
    clamped = core.native.edges_syn_strength.detach().numpy()
    raw = torch.load(ROOT / "03-Data/flyvis-root/results/flow/0000/000/chkpts/chkpt_00000",
                     map_location="cpu", weights_only=False)["network"]["edges_syn_strength"].numpy()
    np.testing.assert_array_equal(raw, clamped)
    save_json(RESULTS / "verification/baseline-indexing.json", {
        "status": "pass", "utc": utc_now(), "ordered_type_pairs": len(pairs),
        "matches_native_parameter_keys": True, "raw_equals_native_clamped_model000": True,
        "native_strength_clamp": str(core.native.clamp_config["edges_syn_strength"]),
        "connectome_sha256": sha256(path), "models": records,
        "scope": "Parameter indexing only; no confirmation responses computed or inspected."})
    print(f"PASS: {len(pairs)} ordered type pairs, {len(records)} checkpoints")


if __name__ == "__main__":
    main()
