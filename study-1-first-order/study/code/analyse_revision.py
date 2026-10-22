"""Declared model-level evaluation, with explicit failure and eligibility counts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from revision_common import (
    ROOT, STUDY, RESULTS, CONFIG, MODELS, NAMES, READOUTS, CLUSTERS,
    independent_metrics, legacy_panel, load_npz, save_json, stimulus, utc_now,
)


def average(values):
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(valid)) if valid else None


def correlation(predicted, exact):
    if len(exact) < 3 or np.ptp(predicted) == 0 or np.ptp(exact) == 0:
        return None
    return float(np.corrcoef(rankdata(predicted), rankdata(exact))[0, 1])


def top_budget(scores, truth, budget=10):
    """Expected hit count at a fixed experiment budget, including tied predictions."""
    scores, truth = np.asarray(scores), np.asarray(truth, dtype=bool)
    n = len(scores)
    if not n:
        return {"precision": None, "recall": None, "random_precision": None,
                "random_recall": None, "selected": 0, "important": 0}
    k = min(n, budget)
    cutoff = np.partition(scores, n-k)[n-k]
    above, tied = scores > cutoff, scores == cutoff
    hits = truth[above].sum() + (k-above.sum()) * truth[tied].mean()
    count = int(truth.sum())
    return {"precision": float(hits/k), "recall": float(hits/count) if count else None,
            "random_precision": count/n, "random_recall": k/n if count else None,
            "selected": k, "important": count}


def bounded_prediction(prediction, baseline):
    return np.clip(baseline + prediction, 0, 1) - baseline


def path_sums(matrix, max_hops=4):
    normalized = np.divide(matrix, matrix.sum(axis=1, keepdims=True),
                           out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True)>0)
    power = np.eye(len(matrix)); total = np.zeros_like(matrix); out = {}
    for hop in range(1, max_hops+1):
        power = power @ normalized
        total = total + power
        out[hop] = total.copy()
    return out


def structural_scores():
    connectome = json.loads((ROOT / "01-Papers/sources/flyvis/flyvis/connectome/fib25-fib19_v2.2.json").read_text())
    matrix = np.zeros((65, 65))
    for edge in connectome["edges"]:
        matrix[NAMES.index(edge["tar"]), NAMES.index(edge["src"])]+=sum(count for _,count in edge["offsets"])
    indices = [NAMES.index(name) for name in READOUTS]
    scores = {f"anatomy_hop{k}": value[indices] for k,value in path_sums(matrix).items()}
    distances = np.full_like(matrix, np.inf)
    for target in range(65):
        distances[target,target] = 0
        frontier=[target]
        while frontier:
            following=[]
            for node in frontier:
                for source in np.flatnonzero(matrix[node]>0):
                    if not np.isfinite(distances[target,source]):
                        distances[target,source]=distances[target,node]+1
                        following.append(source)
            frontier=following
    scores["graph_distance"] = np.where(np.isfinite(distances[indices]),-distances[indices],-1e6)
    return scores, connectome


def weighted_scores(model, connectome):
    import torch
    checkpoint=ROOT/f"03-Data/flyvis-root/results/flow/0000/{model}/chkpts/chkpt_00000"
    network=torch.load(checkpoint,map_location="cpu",weights_only=False)["network"]
    weights=network["edges_syn_strength"].detach().numpy()
    edges=[edge for edge in connectome["edges"] if not(edge["src"]=="Lawf1" and edge["tar"]=="Lawf1")]
    if len(edges)!=len(weights):
        raise RuntimeError("Checkpoint strength order does not match pinned connectome")
    matrix=np.zeros((65,65))
    for edge,weight in zip(edges,weights):
        matrix[NAMES.index(edge["tar"]),NAMES.index(edge["src"])]+=abs(float(weight))*sum(n for _,n in edge["offsets"])
    return path_sums(matrix,3)[3][[NAMES.index(name) for name in READOUTS]]


def checked_measurement(path, stim):
    archive=load_npz(path)
    measured=independent_metrics(archive["trace"],stim)
    for field in ("dsi","denominator","raw_mean","raw_integral","peaks"):
        if not np.allclose(archive[field],measured[field],atol=2e-6,rtol=2e-5):
            raise RuntimeError(f"Metric mismatch: {path}, {field}")
    return {**archive,**measured}


def panels(source, allow_incomplete=False):
    if source=="legacy":
        for model in CONFIG["development_models"]:
            keys=["dev_slow","dev_fast"] if model in ("000","004") else ["final_slow","final_fast"]
            for key in keys:
                yield model,key,legacy_panel(model,key),CONFIG["main_strengths"]
        return
    models=CONFIG["development_models"] if source=="development" else [m for m in MODELS if m not in CONFIG["development_models"]]
    if source=="all":
        models=MODELS
    for model in models:
        for key in CONFIG["common_stimuli"]:
            folder=RESULTS/"runs"/model/key
            if not (folder/"complete.json").exists():
                if allow_incomplete:
                    continue
                raise RuntimeError(f"Missing completed panel: {model} {key}")
            strengths=CONFIG["main_strengths"] + (CONFIG["development_extra_strengths"] if model in CONFIG["development_models"] else [])
            yield model,key,folder,sorted(strengths)


def evaluate(source, allow_incomplete=False):
    anatomy,connectome=structural_scores()
    weights={};units=[];effects=[];failures=[];paired=[]
    thresholds=CONFIG["direction_selectivity_thresholds"]
    cutoff=CONFIG["important_effect_threshold"]
    denominator_cutoff=CONFIG["denominator_threshold"]
    for model,key,folder,strengths in panels(source,allow_incomplete):
        stim=stimulus(key)
        baseline=checked_measurement(folder/"baseline.npz",stim)
        gradient=load_npz(folder/"gradient.npz")["gradient"].astype(np.float64)
        if model not in weights:
            weights[model]=weighted_scores(model,connectome)
        base_max=max(float(baseline["max_abs_voltage"]),denominator_cutoff)
        for strength in strengths:
            conditions=[checked_measurement(folder/f"s{round(strength*100):03d}-cell{i:02d}.npz",stim) for i in range(65)]
            exact=np.stack([item["dsi"]-baseline["dsi"] for item in conditions],axis=1)
            denominator=np.stack([item["denominator"] for item in conditions],axis=1)
            amplitude=np.array([float(item["max_abs_voltage"])/base_max for item in conditions])
            prediction=strength*gradient
            for r,name in enumerate(READOUTS):
                responsive=bool(baseline["denominator"][r]>denominator_cutoff)
                defined=(denominator[r]>denominator_cutoff)&np.isfinite(exact[r])&np.isfinite(prediction[r])
                abolished=(denominator[r]<=denominator_cutoff)&responsive
                valid=defined & responsive
                preferred=stim["intensity"]==(1 if r<4 else 0)
                base_peaks=baseline["peaks"][preferred,r]
                base_peak_sum=float(base_peaks.sum())
                scores={"gradient":np.abs(prediction[r]), "direct_count":baseline["synapse_count"][r],
                        "input_current":baseline["current_score"][r],
                        **{method:value[r] for method,value in anatomy.items()},
                        "trained_hop3":weights[model][r]}
                prefix={"source":source,"model":model,"cluster":CLUSTERS[model],"stimulus":key,
                        "strength":strength,"readout":name,"baseline_dsi":float(baseline["dsi"][r])}
                failures.append({**prefix,"baseline_responsive":responsive,"defined_effects":int(valid.sum()),
                    "response_abolitions":int(abolished.sum()),"gradient_constant":bool(np.ptp(prediction[r])==0),
                    "gradient_zero":bool(np.all(prediction[r]==0)),
                    **{f"amplified_{int(t)}":int(np.sum(amplitude>t)) for t in CONFIG["amplification_thresholds"]}})
                for i,condition in enumerate(conditions):
                    after=condition["peaks"][preferred,r]
                    switched=bool(np.any((after==0)!=(base_peaks==0)))
                    effects.append({**prefix,"cell":NAMES[i],"defined":bool(valid[i]),"abolished":bool(abolished[i]),
                        "exact":float(exact[r,i]) if valid[i] else None,"prediction":float(prediction[r,i]),
                        "bounded_prediction":float(bounded_prediction(prediction[r,i],baseline["dsi"][r])),
                        "amplification_ratio":float(amplitude[i]),"zero_response_directions_changed":switched,
                        "direct_input":bool(baseline["synapse_count"][r,i]>0),
                        "preferred_amplitude_ratio":float(after.sum()/base_peak_sum) if base_peak_sum>denominator_cutoff else None})
                for threshold in thresholds:
                    if not responsive or baseline["dsi"][r]<threshold:
                        continue
                    # The primary screen is reported together with inclusive and threshold sensitivity strata.
                    for amp_limit in [None]+CONFIG["amplification_thresholds"]:
                        stable=np.ones(65,bool) if amp_limit is None else amplitude<=amp_limit
                        eligible=valid&stable
                        n=int(eligible.sum())
                        if not n:
                            continue
                        truth=np.abs(exact[r,eligible])>=cutoff
                        method_rows={}
                        for method,score in scores.items():
                            x=score[eligible];y=np.abs(exact[r,eligible])
                            screen=top_budget(x,truth,CONFIG["screening_budget"])
                            record={**prefix,"dsi_threshold":threshold,"amplification_limit":amp_limit,"method":method,
                                "n":n,"n_important":int(truth.sum()),"rho":correlation(x,y),
                                "rho_important":correlation(x[truth],y[truth]),"constant_prediction":bool(np.ptp(x)==0),
                                **{f"top10_{metric}":value for metric,value in screen.items()}}
                            # Composite endpoint is explicitly separate from conditional tuning screening.
                            composite_eligible=(valid|abolished)&stable
                            composite_truth=(valid&(np.abs(exact[r])>=cutoff))|abolished
                            composite=top_budget(score[composite_eligible],composite_truth[composite_eligible],CONFIG["screening_budget"])
                            record.update({f"composite_top10_{metric}":value for metric,value in composite.items()})
                            if method=="gradient":
                                ex=exact[r,eligible];pred=prediction[r,eligible]
                                bounded=bounded_prediction(pred,baseline["dsi"][r])
                                record.update(mae=float(np.abs(pred-ex).mean()),zero_mae=float(np.abs(ex).mean()),
                                    bounded_mae=float(np.abs(bounded-ex).mean()),bias=float((pred-ex).mean()),
                                    important_mae=float(np.abs(pred[truth]-ex[truth]).mean()) if truth.any() else None,
                                    important_zero_mae=float(np.abs(ex[truth]).mean()) if truth.any() else None,
                                    opposite_sign_count=int(np.sum(pred[truth]*ex[truth]<0)),
                                    zero_prediction_count=int(np.sum(pred[truth]==0)))
                            units.append(record);method_rows[method]=record
                        for method,other in method_rows.items():
                            if method=="gradient":
                                continue
                            g=method_rows["gradient"]
                            paired.append({**prefix,"dsi_threshold":threshold,"amplification_limit":amp_limit,"baseline":method,
                                "rho_difference":g["rho"]-other["rho"] if g["rho"] is not None and other["rho"] is not None else None,
                                "rho_important_difference":g["rho_important"]-other["rho_important"] if g["rho_important"] is not None and other["rho_important"] is not None else None,
                                "top10_precision_difference":g["top10_precision"]-other["top10_precision"],
                                "top10_recall_difference":g["top10_recall"]-other["top10_recall"] if g["top10_recall"] is not None else None})
    return units,effects,failures,paired


def summarize(units,paired):
    grouped={}
    for row in units:
        key=(row["model"],row["strength"],row["dsi_threshold"],row["amplification_limit"],row["method"])
        grouped.setdefault(key,[]).append(row)
    model_rows=[]
    for (model,strength,threshold,amp,method),rows in grouped.items():
        item={"model":model,"cluster":CLUSTERS[model],"strength":strength,"dsi_threshold":threshold,
              "amplification_limit":amp,"method":method,"units":len(rows),"effects":sum(r["n"] for r in rows),
              "rho_units":sum(r["rho"] is not None for r in rows),
              "rho_important_units":sum(r["rho_important"] is not None for r in rows),
              "constant_prediction_units":sum(r["constant_prediction"] for r in rows)}
        for field in ("rho","rho_important","top10_precision","top10_recall","top10_random_precision","top10_random_recall",
                      "composite_top10_precision","composite_top10_recall"):
            item[field]=average(r[field] for r in rows)
        if method=="gradient":
            # Pool effects within model; only model summaries are averaged across the ensemble.
            for field in ("mae","zero_mae","bounded_mae","bias"):
                item[field]=float(sum(r[field]*r["n"] for r in rows)/item["effects"])
            nimp=sum(r["n_important"] for r in rows)
            item["important_effects"]=nimp
            item["opposite_sign_rate"]=sum(r["opposite_sign_count"] for r in rows)/nimp if nimp else None
            item["zero_prediction_rate"]=sum(r["zero_prediction_count"] for r in rows)/nimp if nimp else None
            for field in ("important_mae","important_zero_mae"):
                item[field]=sum(r[field]*r["n_important"] for r in rows if r[field] is not None)/nimp if nimp else None
        model_rows.append(item)
    group_pairs={}
    for row in paired:
        key=(row["model"],row["strength"],row["dsi_threshold"],row["amplification_limit"],row["baseline"])
        group_pairs.setdefault(key,[]).append(row)
    pair_rows=[]
    for (model,strength,threshold,amp,baseline),rows in group_pairs.items():
        record={"model":model,"strength":strength,"dsi_threshold":threshold,"amplification_limit":amp,"baseline":baseline}
        for field in ("rho_difference","rho_important_difference","top10_precision_difference","top10_recall_difference"):
            record[field]=average(r[field] for r in rows)
            record[field+"_units"]=sum(r[field] is not None for r in rows)
        pair_rows.append(record)
    return model_rows,pair_rows


def estimate(values, confirmation):
    values=np.array([value for value in values if value is not None and np.isfinite(value)])
    if not len(values):
        return {"mean":None,"models":0}
    result={"mean":float(values.mean()),"min":float(values.min()),"max":float(values.max()),"models":len(values)}
    if confirmation and len(values)>=10:
        rng=np.random.default_rng(CONFIG["bootstrap_seed"])
        means=values[rng.integers(0,len(values),(CONFIG["bootstrap_replicates"],len(values)))].mean(axis=1)
        result["ensemble_bootstrap_95"]=np.quantile(means,[0.025,0.975]).tolist()
    return result


def save_csv(path,rows):
    if not rows:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields)
        writer.writeheader();writer.writerows(rows)


def expanded_summaries(model_rows, pair_rows, effects, confirmation):
    """Report prespecified sensitivity, matched uncertainty and descriptive strata."""
    groups={}
    for row in model_rows:
        key=(row["strength"],row["dsi_threshold"],row["amplification_limit"],row["method"])
        groups.setdefault(key,[]).append(row)
    sensitivities=[]
    for (strength,threshold,amp,method),group in groups.items():
        record={"strength":strength,"dsi_threshold":threshold,"amplification_limit":amp,"method":method,
                "models":len(group),"units":sum(r["units"] for r in group),"effects":sum(r["effects"] for r in group)}
        for field in ("top10_precision","top10_recall","composite_top10_precision","composite_top10_recall",
                      "top10_random_precision","top10_random_recall","mae","zero_mae","bounded_mae"):
            if any(field in r for r in group):
                record[field]=estimate([r.get(field) for r in group],confirmation)
        sensitivities.append(record)
    primary=lambda r:r["dsi_threshold"]==CONFIG["primary_direction_selectivity_threshold"] and r["amplification_limit"]==CONFIG["primary_amplification_threshold"]
    paired_summary=[]
    for strength in sorted({r["strength"] for r in pair_rows}):
        for baseline in sorted({r["baseline"] for r in pair_rows}):
            group=[r for r in pair_rows if primary(r) and r["strength"]==strength and r["baseline"]==baseline]
            record={"strength":strength,"baseline":baseline}
            for field in ("rho_difference","rho_important_difference","top10_precision_difference","top10_recall_difference"):
                record[field]=estimate([r[field] for r in group],confirmation)
                record[field+"_units"]=sum(r[field+"_units"] for r in group)
            paired_summary.append(record)
    cluster_summary=[]
    for strength in sorted({r["strength"] for r in model_rows}):
        for cluster in sorted({r["cluster"] for r in model_rows}):
            group=[r for r in model_rows if primary(r) and r["strength"]==strength and r["cluster"]==cluster and r["method"]=="gradient"]
            record={"strength":strength,"cluster":cluster,"models":[r["model"] for r in group]}
            for field in ("top10_precision","top10_recall","mae","zero_mae","bounded_mae"):
                record[field]=estimate([r[field] for r in group],False)
            cluster_summary.append(record)
    diagnostic_groups={}
    for row in effects:
        if not row["defined"]:
            continue
        strata=["all_responsive"]
        if row["baseline_dsi"]>=CONFIG["primary_direction_selectivity_threshold"] and row["amplification_ratio"]<=CONFIG["primary_amplification_threshold"]:
            strata.append("primary")
        categories=[("effect_size","important" if abs(row["exact"])>=CONFIG["important_effect_threshold"] else "smaller"),
                    ("direct_input","direct" if row["direct_input"] else "indirect"),
                    ("zero_peak_pattern","changed" if row["zero_response_directions_changed"] else "unchanged"),
                    ("baseline_selectivity","below_0.2" if row["baseline_dsi"]<0.2 else "at_least_0.2")]
        for stratum in strata:
            for diagnostic,category in categories:
                key=(row["model"],row["strength"],stratum,diagnostic,category)
                diagnostic_groups.setdefault(key,[]).append(row)
    diagnostics=[]
    for (model,strength,stratum,diagnostic,category),group in diagnostic_groups.items():
        errors=np.array([abs(r["prediction"]-r["exact"]) for r in group])
        diagnostics.append({"model":model,"strength":strength,"stratum":stratum,"diagnostic":diagnostic,"category":category,
            "effects":len(group),"mae":float(errors.mean()),"median_absolute_error":float(np.median(errors)),
            "zero_mae":float(np.mean([abs(r["exact"]) for r in group])),
            "bounded_mae":float(np.mean([abs(r["bounded_prediction"]-r["exact"]) for r in group])),
            "infeasible_prediction_fraction":float(np.mean([not 0<=r["baseline_dsi"]+r["prediction"]<=1 for r in group]))})
    return sensitivities,paired_summary,cluster_summary,diagnostics


def timing_summary(source,allow_incomplete):
    records=[]
    if source=="legacy":
        return records
    for model,key,folder,_ in panels(source,allow_incomplete):
        gradient=json.loads((folder/"gradient.json").read_text())
        if gradient.get("reused"):
            continue
        costs={}
        for strength in CONFIG["main_strengths"]:
            details=[json.loads((folder/f"s{round(strength*100):03d}-cell{i:02d}.json").read_text()) for i in range(65)]
            if any(r.get("reused") or (not r.get("structural_zero") and "wall_seconds" not in r) for r in details):
                break
            costs[str(strength)]=sum(r.get("wall_seconds",0) for r in details)
        if len(costs)!=2:
            continue
        gradient_seconds=gradient["forward_s"]+sum(gradient["backward_s"])
        records.append({"model":model,"stimulus":key,"eight_readout_gradient_seconds":gradient_seconds,
            "half_exact_seconds":costs["0.5"],"full_exact_seconds":costs["1.0"],
            "both_exact_seconds":sum(costs.values()),
            "full_exact_over_gradient":costs["1.0"]/gradient_seconds,
            "both_exact_over_gradient":sum(costs.values())/gradient_seconds})
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",choices=("legacy","development","confirmation","all"),required=True)
    parser.add_argument("--allow-incomplete",action="store_true")
    args=parser.parse_args()
    units,effects,failures,paired=evaluate(args.source,args.allow_incomplete)
    model_rows,pair_rows=summarize(units,paired)
    sensitivities,paired_summary,cluster_summary,diagnostics=expanded_summaries(model_rows,pair_rows,effects,args.source=="confirmation")
    dest=RESULTS/"analysis"/args.source
    for filename,rows in [("units.csv",units),("effects.csv",effects),("failures.csv",failures),
                          ("models.csv",model_rows),("paired-models.csv",pair_rows),("diagnostics.csv",diagnostics)]:
        save_csv(dest/filename,rows)
    save_json(dest/"sensitivity.json",sensitivities)
    save_json(dest/"paired-summary.json",paired_summary)
    save_json(dest/"cluster-summary.json",cluster_summary)
    error_differences=[]
    for strength in sorted({r["strength"] for r in model_rows}):
        selected=[r for r in model_rows if r["method"]=="gradient" and r["strength"]==strength
                  and r["dsi_threshold"]==CONFIG["primary_direction_selectivity_threshold"]
                  and r["amplification_limit"]==CONFIG["primary_amplification_threshold"]]
        error_differences.append({"strength":strength,
            "gradient_minus_zero":estimate([r["mae"]-r["zero_mae"] for r in selected],args.source=="confirmation"),
            "bounded_minus_zero":estimate([r["bounded_mae"]-r["zero_mae"] for r in selected],args.source=="confirmation")})
    save_json(dest/"magnitude-differences.json",error_differences)
    timing=timing_summary(args.source,args.allow_incomplete)
    save_csv(dest/"timing-panels.csv",timing)
    timing_models=[]
    for model in sorted({r["model"] for r in timing}):
        selected=[r for r in timing if r["model"]==model]
        gradient_seconds=sum(r["eight_readout_gradient_seconds"] for r in selected)
        exact_full=sum(r["full_exact_seconds"] for r in selected)
        exact_both=sum(r["both_exact_seconds"] for r in selected)
        timing_models.append({"model":model,"panels":len(selected),"gradient_seconds":gradient_seconds,
            "full_exact_seconds":exact_full,"both_exact_seconds":exact_both,
            "full_exact_over_gradient":exact_full/gradient_seconds,"both_exact_over_gradient":exact_both/gradient_seconds})
    save_json(dest/"timing.json",{"models":timing_models,
        "summary":{field:estimate([r[field] for r in timing_models],args.source=="confirmation")
                   for field in ("gradient_seconds","full_exact_seconds","both_exact_seconds","full_exact_over_gradient","both_exact_over_gradient")},
        "scope":"Recorded GPU simulation/gradient sections only; excludes loading, shared adaptation, baseline/current computation, file I/O and verification. Gradients produce all eight readouts; exact runs also produce all eight. Reused panels are excluded. These are observed implementation timings, not algorithmic lower bounds."})
    primary=[r for r in model_rows if r["dsi_threshold"]==CONFIG["primary_direction_selectivity_threshold"] and r["amplification_limit"]==CONFIG["primary_amplification_threshold"]]
    summary=[]
    for strength in sorted({r["strength"] for r in primary}):
        for method in sorted({r["method"] for r in primary}):
            rows=[r for r in primary if r["strength"]==strength and r["method"]==method]
            record={"strength":strength,"method":method}
            for field in ("rho","rho_important","top10_precision","top10_recall","top10_random_precision","top10_random_recall",
                          "composite_top10_precision","composite_top10_recall","mae","zero_mae","bounded_mae",
                          "important_mae","important_zero_mae","opposite_sign_rate","zero_prediction_rate"):
                if any(field in r for r in rows):
                    record[field]=estimate([r.get(field) for r in rows],args.source=="confirmation")
            summary.append(record)
    save_json(dest/"summary.json",{"source":args.source,"created":utc_now(),"partial":args.allow_incomplete,
        "models_present":sorted({r["model"] for r in model_rows}),"primary_dsi_threshold":CONFIG["primary_direction_selectivity_threshold"],
        "primary_amplification_limit":CONFIG["primary_amplification_threshold"],"summary":summary,
        "interpretation":"Model-level descriptive estimates; ensemble bootstrap is conditional on the released ensemble. Undefined correlations remain missing, with matched paired comparisons in paired-models.csv."})
    print(json.dumps({"source":args.source,"models":len({r['model'] for r in model_rows}),"effect_rows":len(effects),"unit_rows":len(units)}))


if __name__=="__main__":
    main()
