
# ============================================================
# PHASE 3B
# Clinical consequences per 1,000 screened patients
#
# VisionFM:
#   BRSET-derived deployment threshold
#   versus
#   local recalibration + local threshold retuning
#
# Domains:
#   mBRSET
#   DeepDRiD
#
# IMPORTANT:
#   This is a derived clinical-interpretability analysis.
#   No model is trained.
#   No frozen test threshold is optimized on the test set.
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# 0. LOCATE PROJECT
# ============================================================

PROJECT = "Retinal_Retraining_Phase1A1"

candidates = [
    Path("/content/vfm_head_drive/MyDrive") / PROJECT,
    Path("/content/vfm_cpu_drive/MyDrive") / PROJECT,
    Path("/content/vfm_rescue_drive/MyDrive") / PROJECT,
    Path("/content/drive_c3/MyDrive") / PROJECT,
    Path("/content/drive/MyDrive") / PROJECT,
]

ROOT = next(
    (p for p in candidates if p.exists()),
    None
)

assert ROOT is not None, "Project root not found."

CACHE = (
    ROOT
    / "results"
    / "visionfm_feature_cache"
)

HEAD_DIR = (
    ROOT
    / "results"
    / "phase2a1_visionfm"
    / "decoder_checkpoints"
)

LOCKED_METRICS = (
    ROOT
    / "results"
    / "phase2a1_visionfm"
    / "01_seed_level_visionfm_metrics.csv"
)

OUT = (
    ROOT
    / "results"
    / "phase3b_clinical_consequences"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

SEEDS = [
    20267001,
    20267002,
    20267003,
]

N_BOOT = 5000
EPS = 1e-7

print("=" * 80)
print("PHASE 3B — CLINICAL CONSEQUENCES PER 1,000")
print("=" * 80)
print("ROOT:", ROOT)
print("Device: CPU")


# ============================================================
# 1. LOAD FEATURE CACHE
# ============================================================

def load_cache(name):

    p = (
        CACHE
        / f"{name}_features.npz"
    )

    assert p.exists(), p

    d = np.load(p)

    X = d["X"].astype(
        np.float32
    )

    pid = d[
        "patient_id"
    ].astype(str)

    yp = d[
        "y_patient"
    ].astype(int)

    assert X.shape[1] == 3072
    assert len(X) == len(pid)

    return {
        "X": X,
        "patient_id": pid,
        "y_patient": yp,
    }


data = {

    name: load_cache(name)

    for name in [
        "BR_VAL",
        "MB_LOCAL100",
        "MB_TEST",
        "DD_VAL",
        "DD_TEST",
    ]
}

print("\nCOHORTS")

for name,d in data.items():

    print(
        name,
        "images=",
        len(d["X"]),
        "patients=",
        len(
            np.unique(
                d["patient_id"]
            )
        )
    )


# ============================================================
# 2. EXACT SOURCE CLASSIFICATION HEAD
# ============================================================

class VisionFMClsHead(nn.Module):

    def __init__(
        self,
        embed_dim=3072,
    ):

        super().__init__()

        self.channel_bn = nn.BatchNorm1d(
            embed_dim,
            eps=1e-6,
            momentum=0.99,
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                embed_dim,
                embed_dim // 2
            ),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(
                embed_dim // 2,
                embed_dim // 4
            ),

            nn.GELU(),

            nn.Dropout(0.1),

            nn.Linear(
                embed_dim // 4,
                1
            ),
        )


    def forward(self,x):

        x = self.channel_bn(x)

        return (
            self.classifier(x)
            .squeeze(-1)
        )


def load_head(seed):

    model = VisionFMClsHead()

    p = (
        HEAD_DIR
        /
        f"VisionFM_BRSET_decoder_seed{seed}.pth"
    )

    assert p.exists(), p

    ckpt = torch.load(
        p,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["state_dict"],
        strict=True,
    )

    model.eval()

    return model


# ============================================================
# 3. PATIENT-LEVEL PREDICTIONS
# ============================================================

@torch.inference_mode()
def patient_predictions(
    model,
    cohort,
):

    X = torch.from_numpy(
        cohort["X"]
    ).float()

    loader = DataLoader(
        TensorDataset(X),
        batch_size=512,
        shuffle=False,
    )

    probs = []

    for (xb,) in loader:

        p = (
            torch.sigmoid(
                model(xb)
            )
            .numpy()
        )

        probs.append(p)

    probs = np.concatenate(
        probs
    )

    df = pd.DataFrame({

        "patient_id":
            cohort[
                "patient_id"
            ].astype(str),

        "y":
            cohort[
                "y_patient"
            ].astype(int),

        "p_raw":
            probs,
    })

    # patient labels must agree across images
    assert (
        df
        .groupby(
            "patient_id"
        )["y"]
        .nunique()
        .max()
        == 1
    )

    # Locked patient-level endpoint:
    # maximum image probability
    return (
        df
        .groupby(
            "patient_id",
            as_index=False,
        )
        .agg(
            y=("y","first"),
            p_raw=("p_raw","max"),
        )
    )


# ============================================================
# 4. PLATT CALIBRATION
# ============================================================

def logit_np(p):

    p = np.clip(
        np.asarray(p),
        EPS,
        1-EPS,
    )

    return np.log(
        p/(1-p)
    )


def fit_platt(
    y,
    p,
):

    y = np.asarray(
        y
    ).astype(int)

    assert len(
        np.unique(y)
    ) == 2

    m = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
    )

    m.fit(
        logit_np(p)
        .reshape(-1,1),
        y,
    )

    return m


def apply_platt(
    model,
    p,
):

    return (
        model
        .predict_proba(
            logit_np(p)
            .reshape(-1,1)
        )[:,1]
    )


# ============================================================
# 5. LOCKED 90% SENSITIVITY THRESHOLD
# ============================================================

def threshold_90(
    y,
    p,
):

    y = np.asarray(
        y
    ).astype(int)

    p = np.asarray(p)

    best = None

    for th in np.unique(p):

        pred = (
            p >= th
        ).astype(int)

        tn,fp,fn,tp = (
            confusion_matrix(
                y,
                pred,
                labels=[0,1],
            )
            .ravel()
        )

        sens = (
            tp/(tp+fn)
            if tp+fn
            else np.nan
        )

        spec = (
            tn/(tn+fp)
            if tn+fp
            else np.nan
        )

        if sens >= 0.90:

            candidate = (
                spec,
                th,
            )

            if (
                best is None
                or
                candidate[0]
                >
                best[0]
            ):

                best = candidate

    assert best is not None

    return float(
        best[1]
    )


# ============================================================
# 6. CLINICAL CONSEQUENCES
# ============================================================

def clinical_counts(
    y,
    p,
    threshold,
):

    y = np.asarray(
        y
    ).astype(int)

    pred = (
        np.asarray(p)
        >= threshold
    ).astype(int)

    tn,fp,fn,tp = (
        confusion_matrix(
            y,
            pred,
            labels=[0,1],
        )
        .ravel()
    )

    n = len(y)

    sens = (
        tp/(tp+fn)
        if tp+fn
        else np.nan
    )

    spec = (
        tn/(tn+fp)
        if tn+fp
        else np.nan
    )

    prevalence = (
        y.mean()
    )

    scale = (
        1000.0/n
    )

    return {

        "n_test":
            int(n),

        "prevalence":
            float(prevalence),

        "sensitivity":
            float(sens),

        "specificity":
            float(spec),

        "true_positive_per1000":
            float(
                tp * scale
            ),

        "missed_rDR_per1000":
            float(
                fn * scale
            ),

        "false_referrals_per1000":
            float(
                fp * scale
            ),

        "true_negative_per1000":
            float(
                tn * scale
            ),

        "total_referrals_per1000":
            float(
                (tp+fp)
                * scale
            ),

        "correct_classifications_per1000":
            float(
                (tp+tn)
                * scale
            ),
    }


# ============================================================
# 7. PATIENT BOOTSTRAP CIs
# ============================================================

def bootstrap_clinical(
    y,
    p,
    threshold,
    seed,
    n_boot=N_BOOT,
):

    y = np.asarray(
        y
    ).astype(int)

    p = np.asarray(p)

    rng = np.random.default_rng(
        seed
    )

    n = len(y)

    records = []

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            n,
            n,
        )

        yy = y[idx]
        pp = p[idx]

        # reject pathological resamples
        # containing only one class
        if len(
            np.unique(yy)
        ) < 2:
            continue

        x = clinical_counts(
            yy,
            pp,
            threshold,
        )

        records.append([
            x[
                "missed_rDR_per1000"
            ],

            x[
                "false_referrals_per1000"
            ],

            x[
                "total_referrals_per1000"
            ],

            x[
                "sensitivity"
            ],

            x[
                "specificity"
            ],
        ])

    a = np.asarray(
        records
    )

    assert len(a) > 0

    names = [
        "missed_rDR_per1000",
        "false_referrals_per1000",
        "total_referrals_per1000",
        "sensitivity",
        "specificity",
    ]

    out = {}

    for j,name in enumerate(names):

        out[
            f"{name}_CI_low"
        ] = float(
            np.quantile(
                a[:,j],
                0.025
            )
        )

        out[
            f"{name}_CI_high"
        ] = float(
            np.quantile(
                a[:,j],
                0.975
            )
        )

    return out


# ============================================================
# 8. LOAD LOCKED PHASE 2A1 METRICS FOR QC
# ============================================================

locked = pd.read_csv(
    LOCKED_METRICS
)

locked[
    "seed"
] = locked[
    "seed"
].astype(int)


# ============================================================
# 9. MAIN
# ============================================================

rows = []
difference_rows = []

for seed in SEEDS:

    print(
        "\n" + "="*80
    )

    print(
        "SOURCE SEED",
        seed
    )

    print(
        "="*80
    )

    model = load_head(
        seed
    )

    br_val = patient_predictions(
        model,
        data[
            "BR_VAL"
        ]
    )

    mb_local = patient_predictions(
        model,
        data[
            "MB_LOCAL100"
        ]
    )

    mb_test = patient_predictions(
        model,
        data[
            "MB_TEST"
        ]
    )

    dd_val = patient_predictions(
        model,
        data[
            "DD_VAL"
        ]
    )

    dd_test = patient_predictions(
        model,
        data[
            "DD_TEST"
        ]
    )

    # ========================================================
    # SOURCE calibration and source deployment threshold
    # ========================================================

    source_platt = fit_platt(
        br_val["y"],
        br_val["p_raw"],
    )

    br_val[
        "p_cal"
    ] = apply_platt(
        source_platt,
        br_val[
            "p_raw"
        ],
    )

    source_threshold = threshold_90(
        br_val["y"],
        br_val[
            "p_cal"
        ],
    )

    # ========================================================
    # DOMAIN ANALYSES
    # ========================================================

    for (
        domain,
        local_df,
        test_df,
        boot_offset,
    ) in [

        (
            "mBRSET",
            mb_local,
            mb_test,
            10000,
        ),

        (
            "DeepDRiD",
            dd_val,
            dd_test,
            20000,
        ),
    ]:

        # ----------------------------------------------------
        # No local maintenance:
        # source Platt + source threshold
        # ----------------------------------------------------

        p_source = apply_platt(
            source_platt,
            test_df[
                "p_raw"
            ],
        )

        no_maint = clinical_counts(
            test_df["y"],
            p_source,
            source_threshold,
        )

        no_maint_ci = (
            bootstrap_clinical(
                test_df["y"],
                p_source,
                source_threshold,
                seed=
                    seed
                    + boot_offset,
            )
        )

        # ----------------------------------------------------
        # Local recalibration + locally locked threshold
        # ----------------------------------------------------

        local_platt = fit_platt(
            local_df["y"],
            local_df[
                "p_raw"
            ],
        )

        local_df = local_df.copy()

        local_df[
            "p_local"
        ] = apply_platt(
            local_platt,
            local_df[
                "p_raw"
            ],
        )

        local_threshold = threshold_90(
            local_df["y"],
            local_df[
                "p_local"
            ],
        )

        p_local_test = apply_platt(
            local_platt,
            test_df[
                "p_raw"
            ],
        )

        local_maint = clinical_counts(
            test_df["y"],
            p_local_test,
            local_threshold,
        )

        local_maint_ci = (
            bootstrap_clinical(
                test_df["y"],
                p_local_test,
                local_threshold,
                seed=
                    seed
                    + boot_offset
                    + 1000,
            )
        )

        # ----------------------------------------------------
        # Store strategy rows
        # ----------------------------------------------------

        for (
            strategy,
            threshold,
            values,
            cis,
        ) in [

            (
                "Source calibration + source threshold",
                source_threshold,
                no_maint,
                no_maint_ci,
            ),

            (
                "Local recalibration + local threshold",
                local_threshold,
                local_maint,
                local_maint_ci,
            ),
        ]:

            row = {

                "seed":
                    seed,

                "domain":
                    domain,

                "strategy":
                    strategy,

                "threshold":
                    threshold,
            }

            row.update(
                values
            )

            row.update(
                cis
            )

            rows.append(
                row
            )

        # ----------------------------------------------------
        # Paired clinical difference
        #
        # Negative missed change = fewer missed rDR cases
        # Positive false-referral change = extra referrals
        # ----------------------------------------------------

        difference_rows.append({

            "seed":
                seed,

            "domain":
                domain,

            "missed_rDR_change_per1000":
                (
                    local_maint[
                        "missed_rDR_per1000"
                    ]
                    -
                    no_maint[
                        "missed_rDR_per1000"
                    ]
                ),

            "false_referral_change_per1000":
                (
                    local_maint[
                        "false_referrals_per1000"
                    ]
                    -
                    no_maint[
                        "false_referrals_per1000"
                    ]
                ),

            "total_referral_change_per1000":
                (
                    local_maint[
                        "total_referrals_per1000"
                    ]
                    -
                    no_maint[
                        "total_referrals_per1000"
                    ]
                ),

            "sensitivity_change":
                (
                    local_maint[
                        "sensitivity"
                    ]
                    -
                    no_maint[
                        "sensitivity"
                    ]
                ),

            "specificity_change":
                (
                    local_maint[
                        "specificity"
                    ]
                    -
                    no_maint[
                        "specificity"
                    ]
                ),
        })

        print(
            domain,
            "| prevalence=",
            round(
                no_maint[
                    "prevalence"
                ],
                3
            )
        )

        print(
            "  source threshold:"
            f" missed={no_maint['missed_rDR_per1000']:.1f}/1000,"
            f" false referrals={no_maint['false_referrals_per1000']:.1f}/1000,"
            f" sens={no_maint['sensitivity']:.3f},"
            f" spec={no_maint['specificity']:.3f}"
        )

        print(
            "  local maintenance:"
            f" missed={local_maint['missed_rDR_per1000']:.1f}/1000,"
            f" false referrals={local_maint['false_referrals_per1000']:.1f}/1000,"
            f" sens={local_maint['sensitivity']:.3f},"
            f" spec={local_maint['specificity']:.3f}"
        )


# ============================================================
# 10. SAVE SEED-LEVEL RESULTS
# ============================================================

clinical = pd.DataFrame(
    rows
)

differences = pd.DataFrame(
    difference_rows
)

clinical.to_csv(
    OUT
    / "01_seed_level_clinical_consequences.csv",
    index=False,
)

differences.to_csv(
    OUT
    / "02_seed_level_strategy_differences.csv",
    index=False,
)


# ============================================================
# 11. GROUP SUMMARY
# ============================================================

summary = (
    clinical
    .groupby(
        [
            "domain",
            "strategy",
        ]
    )
    .agg(

        mean_prevalence=
            ("prevalence","mean"),

        mean_sensitivity=
            ("sensitivity","mean"),

        mean_specificity=
            ("specificity","mean"),

        mean_missed_rDR_per1000=
            (
                "missed_rDR_per1000",
                "mean",
            ),

        sd_missed_rDR_per1000=
            (
                "missed_rDR_per1000",
                "std",
            ),

        mean_false_referrals_per1000=
            (
                "false_referrals_per1000",
                "mean",
            ),

        sd_false_referrals_per1000=
            (
                "false_referrals_per1000",
                "std",
            ),

        mean_total_referrals_per1000=
            (
                "total_referrals_per1000",
                "mean",
            ),

        mean_correct_classifications_per1000=
            (
                "correct_classifications_per1000",
                "mean",
            ),
    )
    .reset_index()
)

summary.to_csv(
    OUT
    / "03_clinical_consequence_summary.csv",
    index=False,
)


difference_summary = (
    differences
    .groupby(
        "domain"
    )
    .agg(

        mean_missed_rDR_change_per1000=
            (
                "missed_rDR_change_per1000",
                "mean",
            ),

        min_missed_rDR_change_per1000=
            (
                "missed_rDR_change_per1000",
                "min",
            ),

        max_missed_rDR_change_per1000=
            (
                "missed_rDR_change_per1000",
                "max",
            ),

        mean_false_referral_change_per1000=
            (
                "false_referral_change_per1000",
                "mean",
            ),

        mean_total_referral_change_per1000=
            (
                "total_referral_change_per1000",
                "mean",
            ),

        mean_sensitivity_change=
            (
                "sensitivity_change",
                "mean",
            ),

        mean_specificity_change=
            (
                "specificity_change",
                "mean",
            ),
    )
    .reset_index()
)

difference_summary.to_csv(
    OUT
    / "04_clinical_difference_summary.csv",
    index=False,
)


# ============================================================
# 12. REPORT
# ============================================================

report = f"""# PHASE 3B — CLINICAL CONSEQUENCES PER 1,000 SCREENED

## Purpose

To translate the frozen VisionFM external-deployment results into
clinically interpretable consequences for diabetic retinopathy
screening.

This analysis does not change model weights or optimize thresholds on
the frozen test cohorts.

## Strategies

1. Source calibration + source threshold:
   probability calibration and the >=90% sensitivity operating
   threshold were defined exclusively using BRSET validation data.

2. Local recalibration + local threshold:
   Platt calibration and the >=90% sensitivity operating threshold
   were estimated using the prespecified local calibration cohort
   (100 mBRSET patients or 100 DeepDRiD validation patients).

## Outcomes per 1,000 screened

- detected referable DR cases
- missed referable DR cases
- false referrals
- total referrals

Observed target-cohort prevalence was retained.

## Strategy summary

{summary.to_string(index=False)}

## Change after local maintenance

{difference_summary.to_string(index=False)}

## Interpretation boundary

The local-maintenance comparison combines local probability
recalibration with local operating-threshold retuning.

Therefore changes in missed cases and referral burden should not be
attributed to probability recalibration alone.

For mBRSET, discrimination was already shown to deteriorate
substantially. Accordingly, threshold retuning may trade fewer missed
cases for increased referral burden but cannot restore lost ranking
performance.

For DeepDRiD, discrimination was largely preserved, making local
recalibration and threshold retuning a more appropriate low-cost
maintenance strategy.

LoRA, head-only adaptation, and full fine-tuning are not included in
this per-1,000 threshold analysis because no frozen independently
selected clinical operating threshold was stored for those adapted
models. No threshold was retrospectively chosen on the frozen target
test set.
"""

REPORT = (
    OUT
    / "PHASE_3B_CLINICAL_CONSEQUENCES_REPORT.md"
)

REPORT.write_text(
    report,
    encoding="utf-8",
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print(
    "\n" + "="*80
)

print(
    "PHASE 3B FINAL"
)

print(
    "="*80
)

print(
    "\nCLINICAL CONSEQUENCE SUMMARY:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nLOCAL MAINTENANCE DIFFERENCE:"
)

print(
    difference_summary.to_string(
        index=False
    )
)

print(
    "\nREPORT:",
    REPORT
)

print(
    "\n✅ PHASE 3B COMPLETE"
)