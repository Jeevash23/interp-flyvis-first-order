"""Audit the complete delivery without launching additional simulations."""
import argparse
import csv
import json
import subprocess
from revision_common import ROOT, STUDY, RESULTS, CONFIG, MODELS, freeze_check, original_freeze_check, save_json, sha256, utc_now


def verify_originals():
    manifest=json.loads((ROOT/"provenance/original-records.json").read_text())
    changed=[rel for rel,digest in manifest["files"].items() if not (ROOT/rel).is_file() or sha256(ROOT/rel)!=digest]
    if changed:
        raise RuntimeError(f"Original evidence changed: {changed[:20]}")
    return {"files":len(manifest["files"]),"status":"pass"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--require-visual",action="store_true")
    args=parser.parse_args()
    frozen=freeze_check()
    original=verify_originals()
    for phase,name in [("development","development-gate.json"),("confirmation","confirmation-verification.json")]:
        record=json.loads((RESULTS/"verification"/name).read_text())
        expected=CONFIG["development_models"] if phase=="development" else [m for m in MODELS if m not in CONFIG["development_models"]]
        if record["status"]!="pass" or set(record["models"])!=set(expected) or len(record["fresh_cpu_replays"])!=len(expected):
            raise RuntimeError(f"Incomplete verification: {phase}")
        if any(r["status"]!="pass" for r in record["fresh_cpu_replays"]):
            raise RuntimeError(f"Failed replay: {phase}")
    evidence=json.loads((ROOT/"paper/evidence.json").read_text())
    if evidence["status"]!="confirmation" or len(evidence["models"])!=46:
        raise RuntimeError("Development previews cannot be released as the completed manuscript")
    if sha256(ROOT/"paper/manuscript.tex")!=evidence["manuscript_sha256"]:
        raise RuntimeError("Manuscript differs from its generation record")
    for rel,digest in evidence["sources"].items():
        if sha256(ROOT/rel)!=digest:
            raise RuntimeError(f"Manuscript analysis source changed: {rel}")
    with (RESULTS/"analysis/confirmation/models.csv").open() as stream:
        rows=list(csv.DictReader(stream))
    selected=[r for r in rows if r["method"]=="gradient" and float(r["strength"])==1
              and float(r["dsi_threshold"])==CONFIG["primary_direction_selectivity_threshold"]
              and r["amplification_limit"] and float(r["amplification_limit"])==CONFIG["primary_amplification_threshold"]]
    for field,key in [("mae","full_gradient_mae"),("zero_mae","full_zero_mae"),("bounded_mae","full_bounded_mae")]:
        observed=sum(float(r[field]) for r in selected)/len(selected)
        if abs(observed-evidence["key_numbers"][key])>1e-12:
            raise RuntimeError(f"Manuscript key number differs from model table: {field}")
    supplement=json.loads((RESULTS/"analysis/supplement/status.json").read_text())
    if set(supplement["development_models_in_common_panel"])!=set(CONFIG["development_models"]):
        raise RuntimeError("Joint-CT1 supplementary panel is incomplete")
    pdf=ROOT/"output/pdf/AI-06-manuscript.pdf"
    if not pdf.exists() or pdf.stat().st_size<10000:
        raise RuntimeError("Missing compiled final PDF")
    pdf_hash=sha256(pdf)
    if sha256(ROOT/"paper/manuscript.pdf")!=pdf_hash:
        raise RuntimeError("Convenience manuscript PDF is not identical to the output PDF")
    extracted=subprocess.run(["pdftotext",str(pdf),"-"],capture_output=True,text=True,check=True).stdout
    if "@@" in extracted or "Development manuscript:" in extracted or "confirmation is in progress" in extracted:
        raise RuntimeError("A preview or unresolved field remains in the final PDF")
    log=(ROOT/"paper/build/compile-console.log").read_text()
    if any(marker in log for marker in ("Overfull", "undefined references", "undefined on input", "Fatal error", "LaTeX Error")):
        raise RuntimeError("Unresolved PDF compilation/layout diagnostic")
    visual_status="pending"
    visual_path=ROOT/"paper/visual-review.json"
    if visual_path.exists():
        visual=json.loads(visual_path.read_text())
        if visual.get("status")=="pass" and visual.get("pdf_sha256")==pdf_hash and visual.get("manuscript_sha256")==evidence["manuscript_sha256"]:
            visual_status="pass"
    if args.require_visual and visual_status!="pass":
        raise RuntimeError("Final rendered pages require visual review of this exact PDF")
    record={"status":"complete" if visual_status=="pass" else "computational_checks_pass_visual_review_pending",
        "utc":utc_now(),"freeze_sha256":frozen,"original_evidence":original,"original_freeze":original_freeze_check(),
        "models":50,"confirmation_models":46,"pdf_sha256":pdf_hash,"visual_review":visual_status,
        "key_numbers":evidence["key_numbers"]}
    save_json(RESULTS/"delivery-check.json",record)
    print(json.dumps(record))


if __name__=="__main__":
    main()
