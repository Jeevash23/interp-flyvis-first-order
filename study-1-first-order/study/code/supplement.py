"""Exploratory pair and biological-context analyses; no biological accuracy score."""
import json
import numpy as np
from analyse_revision import checked_measurement, save_csv
from revision_common import ROOT, RESULTS, CONFIG, NAMES, READOUTS, legacy_panel, stimulus, save_json, utc_now

PAIRS=[("Mi1","Tm3"),("Mi1","Mi4"),("Tm3","Mi4"),("Mi4","Mi9"),
       ("Tm1","Tm2"),("Tm1","Tm9"),("Tm2","Tm4"),("Tm4","Tm9")]


def pairs():
    rows=[]
    for model in CONFIG["development_models"]:
        keys=["dev_slow","dev_fast"] if model in ("000","004") else ["final_slow","final_fast"]
        for key in keys:
            folder=legacy_panel(model,key);stim=stimulus(key)
            base=checked_measurement(folder/"baseline.npz",stim)
            for strength in CONFIG["main_strengths"]:
                for number,(a,b) in enumerate(PAIRS):
                    A=checked_measurement(folder/f"s{round(strength*100):03d}-cell{NAMES.index(a):02d}.npz",stim)
                    B=checked_measurement(folder/f"s{round(strength*100):03d}-cell{NAMES.index(b):02d}.npz",stim)
                    J=checked_measurement(folder/f"s{round(strength*100):03d}-pair{number}.npz",stim)
                    amp=max(float(v["max_abs_voltage"]) for v in (A,B,J))/max(float(base["max_abs_voltage"]),1e-8)
                    for r,readout in enumerate(READOUTS):
                        select=stim["intensity"]==(1 if r<4 else 0)
                        valid=all(v["denominator"][r]>1e-8 for v in (base,A,B,J))
                        ia=A["dsi"][r]-base["dsi"][r];ib=B["dsi"][r]-base["dsi"][r]
                        ij=J["dsi"][r]-base["dsi"][r]
                        raw=J["raw_mean"][select,r]-A["raw_mean"][select,r]-B["raw_mean"][select,r]+base["raw_mean"][select,r]
                        raw_a=A["raw_mean"][select,r]-base["raw_mean"][select,r]
                        raw_b=B["raw_mean"][select,r]-base["raw_mean"][select,r]
                        rms=lambda x:float(np.sqrt(np.mean(x*x)))
                        rows.append({"model":model,"stimulus":key,"strength":strength,"pair":a+"+"+b,
                            "readout":readout,"dsi_defined":valid,"baseline_dsi":float(base["dsi"][r]),
                            "dsi_interaction":float(ij-ia-ib) if valid else None,
                            "voltage_interaction_rms":rms(raw),
                            "relative_voltage_interaction":rms(raw)/(rms(raw_a)+rms(raw_b)+1e-6),
                            "amplification_ratio":amp,
                            "primary_eligible":bool(valid and base["dsi"][r]>=0.2 and amp<=100),
                            "same_direction_singles":bool(ia*ib>0),
                            "subadditive_dsi":bool(ia*ib>0 and abs(ij)<abs(ia+ib))})
    save_csv(RESULTS/"analysis/supplement/pairs.csv",rows)
    summaries=[]
    for a,b in PAIRS:
        selected=[row for row in rows if row["pair"]==a+"+"+b and row["primary_eligible"] and row["strength"]==1]
        summaries.append({"pair":a+"+"+b,"eligible_units":len(selected),
            "median_absolute_dsi_interaction":float(np.median([abs(r["dsi_interaction"]) for r in selected])) if selected else None,
            "median_voltage_interaction_rms":float(np.median([r["voltage_interaction_rms"] for r in selected])) if selected else None,
            "median_relative_voltage_interaction":float(np.median([r["relative_voltage_interaction"] for r in selected])) if selected else None})
    save_json(RESULTS/"analysis/supplement/pairs-summary.json",summaries)


def biology():
    rows=[]
    for model in CONFIG["development_models"]:
        for key in ("bio_slow","bio_fast"):
            folder=legacy_panel(model,key);stim=stimulus(key)
            base=checked_measurement(folder/"baseline.npz",stim)
            for cell in ("Mi1","Tm3"):
                condition=checked_measurement(folder/f"s100-cell{NAMES.index(cell):02d}.npz",stim)
                for r in range(4):
                    indices=np.flatnonzero(stim["intensity"]==1)
                    preferred=indices[np.argmax(base["peaks"][indices,r])]
                    peak=base["peaks"][preferred,r]
                    rows.append({"model":model,"stimulus":key,"degrees_per_second":float(stim["speed"])*5.8,
                        "cell":cell,"readout":READOUTS[r],"baseline_preferred_angle":float(stim["angle"][preferred]),
                        "amplitude_ratio_at_baseline_preferred_angle":float(condition["peaks"][preferred,r]/peak) if peak>1e-8 else None,
                        "comparison_status":"qualitative context; different biological readout and silencing mechanism"})
    save_csv(RESULTS/"analysis/supplement/biology-speed-context.csv",rows)
    common=[]
    for model in CONFIG["development_models"]:
        folder=RESULTS/"runs"/model/"dev_fast"
        if not (folder/"complete.json").exists():
            continue
        stim=stimulus("dev_fast");base=checked_measurement(folder/"baseline.npz",stim)
        for cell in ("Mi1","Tm3","Mi4","Mi9","C2","C3","CT1_joint"):
            filename="s100-ct1_joint.npz" if cell=="CT1_joint" else f"s100-cell{NAMES.index(cell):02d}.npz"
            condition=checked_measurement(folder/filename,stim)
            for r,name in enumerate(READOUTS):
                select=stim["intensity"]==(1 if r<4 else 0)
                peak=float(base["peaks"][select,r].sum())
                valid=base["denominator"][r]>1e-8 and condition["denominator"][r]>1e-8
                common.append({"model":model,"cell":cell,"readout":name,"baseline_dsi":float(base["dsi"][r]),
                    "dsi_change":float(condition["dsi"][r]-base["dsi"][r]) if valid else None,
                    "preferred_polarity_amplitude_ratio":float(condition["peaks"][select,r].sum()/peak) if peak>1e-8 else None,
                    "comparison_status":"model voltage, acute output attenuation; biological assay not matched"})
    save_csv(RESULTS/"analysis/supplement/common-stimulus-biology-context.csv",common)
    save_json(RESULTS/"analysis/supplement/status.json",{"created":utc_now(),"speed_context_rows":len(rows),
        "common_stimulus_rows":len(common),"development_models_in_common_panel":sorted({r["model"] for r in common}),
        "biological_validation_claim":False})


if __name__=="__main__":
    pairs();biology()
