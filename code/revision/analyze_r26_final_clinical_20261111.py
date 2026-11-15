"""Internal R2-6 read-only analysis of locked patient predictions.

No model training, deployment-calibrator fitting, threshold selection, or
mutation of protected manuscript assets. Test-label calibration fits below are
diagnostic estimates only and never used to change predictions or policies.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit


ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / "private_predictions"
LOCKED = ROOT / "unified_experiment"
OUT = ROOT / "analyses" / "r26_final_20261111"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = {"VisionFM": [20267001, 20267002, 20267003], "RETFound": [20260906]}
METHODS = ["source", "head_only", "lora"]
DOMAINS = {"BRSET": "SOURCE_TEST_LOCKED", "mBRSET": "TARGET_TEST_LOCKED"}
REPS = 5000
RNG = np.random.default_rng(20261111)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty output")
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def prefix(backbone: str, seed: int) -> str:
    x = "B3" if backbone == "VisionFM" else "RTF"
    return f"UNIFIED_R20261109_{x}_seed{seed}"


def load(backbone: str, seed: int, method: str, domain: str):
    role = DOMAINS[domain]
    path = PRIVATE / f"{prefix(backbone, seed)}_{method}_{domain}_{role}_patients_PRIVATE.csv"
    rows = read_csv(path)
    rows.sort(key=lambda r: r["patient_id"])
    ids = [r["patient_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"duplicate patient IDs: {path.name}")
    y = np.array([int(r["y"]) for r in rows], dtype=int)
    logit_field = "logit_float32" if "logit_float32" in rows[0] else "logit_margin_float32"
    z = np.array([float(r[logit_field]) for r in rows])
    p = np.array([float(r["p_float32"]) for r in rows])
    q = np.array([float(r["p_independently_calibrated"]) for r in rows])
    if not np.all(np.isfinite(z)) or not np.all((0 <= p) & (p <= 1)) or not np.all((0 <= q) & (q <= 1)):
        raise AssertionError(f"invalid scores: {path.name}")
    return {"ids": ids, "y": y, "logit": z, "raw": p, "cal": q, "path": path}


def policy(backbone: str, seed: int, method: str, domain: str):
    path = LOCKED / f"{prefix(backbone, seed)}_locked_dev_maps_thresholds.json"
    root = json.loads(path.read_text(encoding="utf-8"))
    if root["status"] != "FITTED_BEFORE_TEST_OPENED":
        raise AssertionError(f"not locked before test: {path.name}")
    m = root["maps_thresholds"][f"{method}:{domain}"]
    return m, path


def nll(z: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0, z) - y * z))


def diagnostic(z: np.ndarray, y: np.ndarray):
    # Fixed slope=1 offset fit for diagnostic intercept; free intercept/slope
    # fit separately. These test-label fits NEVER alter the deployed score.
    f = lambda a: np.sum(np.logaddexp(0, a + z) - y * (a + z))
    offset = minimize_scalar(f, bounds=(-40, 40), method="bounded")
    g = lambda v: np.sum(np.logaddexp(0, v[0] + v[1] * z) - y * (v[0] + v[1] * z))
    def grad(v):
        err = expit(v[0] + v[1] * z) - y
        return np.array([np.sum(err), np.dot(err, z)])
    free = minimize(g, x0=np.array([0.0, 1.0]), jac=grad,
                    method="L-BFGS-B", options={"ftol": 1e-12, "maxiter": 2000})
    # Save optimizer status; no diagnostic fit is treated as exact by default.
    return float(offset.x), float(free.x[1]), bool(offset.success), bool(free.success)


def confusion(y: np.ndarray, pred: np.ndarray):
    return {"tp": int(np.sum((y == 1) & pred)), "fn": int(np.sum((y == 1) & ~pred)),
            "fp": int(np.sum((y == 0) & pred)), "tn": int(np.sum((y == 0) & ~pred))}


def curve(p: np.ndarray, y: np.ndarray, bins: int = 8):
    # Rank/equal-count bins keep >=25 patients per BRSET bin and >=50 per
    # mBRSET bin; ties may straddle adjacent bins and are only visualized.
    order = np.argsort(p, kind="stable")
    return [(float(np.mean(p[ix])), float(np.mean(y[ix])), len(ix), int(np.sum(y[ix])))
            for ix in np.array_split(order, bins) if len(ix)]


def bootstrap_indices(y: np.ndarray):
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    for _ in range(REPS):
        yield np.concatenate((RNG.choice(pos, len(pos), replace=True),
                              RNG.choice(neg, len(neg), replace=True)))


summary = []
curves = []
clinical = []
metric_ci = []
all_data = {}
max_map_residual = 0.0
for backbone, seeds in SEEDS.items():
    for seed in seeds:
        for domain in DOMAINS:
            for method in METHODS:
                d = load(backbone, seed, method, domain)
                m, map_path = policy(backbone, seed, method, domain)
                zz = m["intercept"] + m["slope"] * d["logit"]
                residual = float(np.max(np.abs(expit(zz) - d["cal"])))
                max_map_residual = max(max_map_residual, residual)
                if residual > 1e-5:
                    raise AssertionError(f"mapping mismatch {backbone}/{seed}/{method}/{domain}: {residual}")
                y = d["y"]
                pred = d["cal"] >= float(m["threshold"])
                c = confusion(y, pred)
                n, pos, neg = len(y), int(y.sum()), int((1-y).sum())
                old_i, old_s, old_i_ok, old_s_ok = diagnostic(d["logit"], y)
                new_i, new_s, new_i_ok, new_s_ok = diagnostic(zz, y)
                per_patient_brier_delta = (d["cal"]-y)**2-(d["raw"]-y)**2
                per_patient_nll_delta = (np.logaddexp(0, zz)-y*zz -
                                         np.logaddexp(0, d["logit"])+y*d["logit"])
                boot_metric = np.empty((REPS, 2), dtype=float)
                for bi, bix in enumerate(bootstrap_indices(y)):
                    boot_metric[bi] = (np.mean(per_patient_brier_delta[bix]),
                                       np.mean(per_patient_nll_delta[bix]))
                metric_lo, metric_hi = np.quantile(boot_metric, [0.025, 0.975], axis=0)
                row = {"backbone": backbone, "seed": seed, "method": method, "domain": domain,
                       "n_patients": n, "n_positive": pos, "n_negative": neg,
                       "development_n": m["dev_patients"], "development_positive": m["dev_positive"],
                       "platt_fit_intercept": m["intercept"], "platt_fit_slope": m["slope"],
                       "platt_fit_C": m["calibration_C"], "locked_threshold": m["threshold"],
                       "test_diagnostic_intercept_raw": old_i, "test_diagnostic_slope_raw": old_s,
                       "test_diagnostic_intercept_cal": new_i, "test_diagnostic_slope_cal": new_s,
                       "diagnostic_optimizer_success_raw": old_i_ok and old_s_ok,
                       "diagnostic_optimizer_success_cal": new_i_ok and new_s_ok,
                       "brier_raw": float(np.mean((d["raw"]-y)**2)),
                       "brier_cal": float(np.mean((d["cal"]-y)**2)),
                       "brier_delta_cal_minus_raw": float(np.mean((d["cal"]-y)**2-(d["raw"]-y)**2)),
                       "brier_delta_ci_low": metric_lo[0], "brier_delta_ci_high": metric_hi[0],
                       "nll_raw_stable_logit": nll(d["logit"], y), "nll_cal": nll(zz, y),
                       "nll_delta_cal_minus_raw": nll(zz, y)-nll(d["logit"], y),
                       "nll_delta_ci_low": metric_lo[1], "nll_delta_ci_high": metric_hi[1],
                       "metric_bootstrap_reps": REPS,
                       "map_max_probability_residual": residual}
                summary.append(row)
                clinical.append({"backbone": backbone, "seed": seed, "domain": domain,
                                 "method": method, "policy_dev_domain": domain,
                                 "n_patients": n, "n_positive": pos, "n_negative": neg,
                                 "threshold": m["threshold"], **c,
                                 "sensitivity": c["tp"]/pos, "specificity": c["tn"]/neg,
                                 "missed_per_1000": 1000*c["fn"]/n,
                                 "false_referrals_per_1000": 1000*c["fp"]/n,
                                 "total_referrals_per_1000": 1000*(c["tp"]+c["fp"])/n})
                for state, probs in (("raw", d["raw"]), ("independent_platt", d["cal"])):
                    for i, (mean_p, rate, count, plus) in enumerate(curve(probs, y), start=1):
                        curves.append({"backbone": backbone, "seed": seed, "domain": domain,
                                       "method": method, "state": state, "bin": i,
                                       "mean_predicted": mean_p, "observed_positive_rate": rate,
                                       "patients": count, "positive": plus})
                all_data[(backbone, seed, domain, method)] = (d, pred)
            src, src_pred = all_data[(backbone, seed, domain, "source")]
            lora, lora_pred = all_data[(backbone, seed, domain, "lora")]
            if src["ids"] != lora["ids"] or not np.array_equal(src["y"], lora["y"]):
                raise AssertionError(f"unpaired patients {backbone}/{seed}/{domain}")
            y = src["y"]
            pos, neg, n = int(y.sum()), int((1-y).sum()), len(y)
            delta_sens = float(np.mean(lora_pred[y==1])-np.mean(src_pred[y==1]))
            delta_spec = float(np.mean(~lora_pred[y==0])-np.mean(~src_pred[y==0]))
            delta_fp = float(1000*(np.sum(lora_pred[y==0])-np.sum(src_pred[y==0]))/n)
            delta_total = float(1000*(np.sum(lora_pred)-np.sum(src_pred))/n)
            vals = np.empty((REPS, 4), dtype=float)
            for j, ix in enumerate(bootstrap_indices(y)):
                yy, s, l = y[ix], src_pred[ix], lora_pred[ix]
                vals[j] = (np.mean(l[yy==1])-np.mean(s[yy==1]),
                           np.mean(~l[yy==0])-np.mean(~s[yy==0]),
                           1000*(np.sum(l[yy==0])-np.sum(s[yy==0]))/n,
                           1000*(np.sum(l)-np.sum(s))/n)
            qlo, qhi = np.quantile(vals, [0.025, 0.975], axis=0)
            metric_ci.append({"backbone": backbone, "seed": seed, "domain": domain,
                              "contrast": "lora_minus_source_under_each_locked_domain_policy",
                              "patients": n, "positive": pos, "negative": neg,
                              "source_positive_to_negative": int(np.sum((y==1)&src_pred&~lora_pred)),
                              "source_negative_to_positive": int(np.sum((y==1)&~src_pred&lora_pred)),
                              "source_nonreferral_to_referral": int(np.sum((y==0)&~src_pred&lora_pred)),
                              "source_referral_to_nonreferral": int(np.sum((y==0)&src_pred&~lora_pred)),
                              "delta_sensitivity": delta_sens, "delta_sensitivity_ci_low": qlo[0], "delta_sensitivity_ci_high": qhi[0],
                              "delta_specificity": delta_spec, "delta_specificity_ci_low": qlo[1], "delta_specificity_ci_high": qhi[1],
                              "delta_false_referrals_per_1000": delta_fp,
                              "delta_false_referrals_per_1000_ci_low": qlo[2],
                              "delta_false_referrals_per_1000_ci_high": qhi[2],
                              "delta_total_referrals_per_1000": delta_total,
                              "delta_total_referrals_per_1000_ci_low": qlo[3],
                              "delta_total_referrals_per_1000_ci_high": qhi[3],
                              "bootstrap_reps": REPS,
                              "bootstrap_scope": "fixed trained models, fixed dev-fitted maps and thresholds; paired patient stratified; no refitting"})

write_csv(OUT / "r26_calibration_metrics_and_diagnostics.csv", summary)
write_csv(OUT / "r26_calibration_curve_bins.csv", curves)
write_csv(OUT / "r26_locked_clinical_working_points.csv", clinical)
write_csv(OUT / "r26_paired_clinical_deltas.csv", metric_ci)

for backbone, seeds in SEEDS.items():
    for domain in DOMAINS:
        fig, axes = plt.subplots(len(seeds), len(METHODS), figsize=(12, 3.6*len(seeds)), squeeze=False,
                                 sharex=True, sharey=True)
        for ri, seed in enumerate(seeds):
            for ci, method in enumerate(METHODS):
                ax = axes[ri, ci]
                ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", linewidth=1.0, label="Ideal")
                for state, color, marker in (("raw", "#bb4a36", "o"), ("independent_platt", "#176784", "s")):
                    rows = [r for r in curves if r["backbone"]==backbone and r["seed"]==seed and
                            r["domain"]==domain and r["method"]==method and r["state"]==state]
                    ax.plot([r["mean_predicted"] for r in rows],
                            [r["observed_positive_rate"] for r in rows],
                            marker=marker, color=color, linewidth=1.3, markersize=4,
                            label=("Before" if state=="raw" else "After Platt"))
                ax.set_title(f"{method.replace('_',' ')} | seed {seed}\nN={len(all_data[(backbone, seed, domain, method)][0]['y'])}, positives={int(all_data[(backbone, seed, domain, method)][0]['y'].sum())}", fontsize=9)
                ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
                ax.grid(alpha=0.2)
                if ci == 0: ax.set_ylabel("Observed rDR fraction")
                if ri == len(seeds)-1: ax.set_xlabel("Mean predicted probability")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{backbone} | {domain} test | 8 equal-count patient bins\nInternal diagnostic only; test labels did not refit deployment Platt/threshold", fontsize=12)
        fig.tight_layout()
        base = OUT / f"r26_{backbone}_{domain}_calibration_before_after"
        fig.savefig(base.with_suffix(".png"), dpi=240)
        fig.savefig(base.with_suffix(".pdf"))
        plt.close(fig)

qc = {"status": "PASS_INTERNAL_R26_FINAL_ANALYSIS", "models": len(summary),
      "clinical_contrasts": len(metric_ci), "calibration_bin_rows": len(curves),
      "map_max_probability_residual": max_map_residual,
      "bootstrap_reps_per_contrast": REPS,
      "deployment_calibrator_refit_on_test": False,
      "threshold_selected_on_test": False,
      "test_calibration_diagnostics_only": True,
      "figures": 4,
      "source_prediction_files_private": True}
(OUT / "r26_final_analysis_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
print(json.dumps(qc, ensure_ascii=False))
