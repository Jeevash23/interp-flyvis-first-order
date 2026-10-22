"""Record the local analysis freeze before any new confirmation responses exist."""
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from revision_common import ROOT, STUDY, RESULTS, CONFIG, MODELS, STIMULI, save_json, sha256, utc_now


def main():
    from run_experiments import simulation_signature
    freeze=STUDY/"freeze.json"
    if freeze.exists():
        from revision_common import freeze_check
        print("Existing freeze verified:",freeze_check())
        return
    confirmation=[m for m in MODELS if m not in CONFIG["development_models"]]
    if (RESULTS/"confirmation-started.json").exists() or any((RESULTS/"runs"/m).exists() for m in confirmation):
        raise RuntimeError("Cannot create an untouched-confirmation freeze after confirmation work starts")
    baseline=json.loads((RESULTS/"verification/baseline-indexing.json").read_text())
    if baseline["status"]!="pass" or len(baseline["models"])!=50:
        raise RuntimeError("Baseline indexing check is required before freeze")
    files=[STUDY/"PROTOCOL.md",STUDY/"config.json",STUDY/"LITERATURE.md"]
    files += [STUDY/"code"/name for name in ("revision_common.py","run_experiments.py","analyse_revision.py",
        "verify_revision.py","verify_baselines.py","supplement.py","test_analysis.py","freeze_revision.py","check_delivery.py")]
    files += [ROOT/"02-Code/stage1"/name for name in ("core.py","common.py","gradients.py","gpu_lock.py","requirements-cuda.lock.txt")]
    files += [ROOT/"02-Code/streaming_native.py",ROOT/"03-Data/stage1/model-selection.json",
        ROOT/"03-Data/stage1/network-structure.json",ROOT/"03-Data/stage1/stimulus-manifest.json",
        ROOT/"01-Papers/sources/flyvis/flyvis/connectome/fib25-fib19_v2.2.json"]
    files += [ROOT/record["path"] for record in STIMULI.values()]
    for model in MODELS:
        files += [ROOT/f"03-Data/flyvis-root/results/flow/0000/{model}"/name for name in ("best_chkpt","chkpts/chkpt_00000","_meta.yaml")]
    import flyvis
    package=Path(flyvis.__file__).parent
    files += [package/path for path in ("network/network.py","network/dynamics.py","network/initialization.py")]
    environment={"utc":utc_now(),"python":sys.version,"executable":sys.executable,
        "platform":platform.platform(),"packages":sorted({(d.metadata["Name"],d.version) for d in importlib.metadata.distributions()})}
    save_json(ROOT/"provenance/revision-environment.json",environment)
    files.append(ROOT/"provenance/revision-environment.json")
    record={"utc":utc_now(),"kind":"local analysis freeze; not external preregistration",
        "development_models":CONFIG["development_models"],"confirmation_models":confirmation,
        "confirmation_responses_accessed":False,"simulation_signature":simulation_signature(),
        "files":{str(path.relative_to(ROOT)):sha256(path) for path in files},
        "manuscript_builder":"Presentation code may be repaired after the freeze; scientific analysis changes require a preserved amendment.",
        "verification_indexing_sha256":sha256(RESULTS/"verification/baseline-indexing.json")}
    save_json(freeze,record)
    print(f"Froze {len(record['files'])} sources and inputs before 46-model confirmation")


if __name__=="__main__":
    main()
