
import os
import json
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

ROOT = Path(
    os.environ["PROJECT_ROOT"]
)

CACHE = (
    ROOT
    / "results"
    / "visionfm_feature_cache"
)

OUT = (
    ROOT
    / "results"
    / "phase2a1_visionfm"
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

MAX_EPOCHS = 100
PATIENCE = 15

BATCH_SIZE = 256

# Official VisionFM decoder default base LR
LR = 1e-3

WEIGHT_DECAY = 0.05

N_BOOT = 5000

# ============================================================
# HELPERS
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cache(name):

    path = (
        CACHE
        / f"{name}_features.npz"
    )

    d = np.load(path)

    X = (
        d["X"]
        .astype(np.float32)
    )

    pid = (
        d["patient_id"]
        .astype(str)
    )

    y_image = (
        d["y_image"]
        .astype(int)
    )

    y_patient = (
        d["y_patient"]
        .astype(int)
    )

    assert len(X) == len(pid)
    assert X.shape[1] == 3072

    return {
        "X": X,
        "patient_id": pid,
        "y_image": y_image,
        "y_patient": y_patient,
    }


data = {
    name: load_cache(name)
    for name in [
        "BR_TRAIN",
        "BR_VAL",
        "BR_TEST",
        "MB_LOCAL100",
        "MB_TEST",
        "DD_VAL",
        "DD_TEST",
    ]
}

print("="*80)
print("FEATURE COHORTS")
print("="*80)

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
# OFFICIAL-STYLE VISIONFM 3-LAYER CLASSIFICATION HEAD
# ============================================================

class VisionFMClsHead(nn.Module):

    def __init__(
        self,
        embed_dim=3072
    ):

        super().__init__()

        # mirrors official ClsHead(layers=3)
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

            nn.Dropout(
                p=0.1
            ),

            nn.Linear(
                embed_dim // 2,
                embed_dim // 4
            ),

            nn.GELU(),

            nn.Dropout(
                p=0.1
            ),

            nn.Linear(
                embed_dim // 4,
                1
            ),
        )

        self.init_weights()


    def init_weights(self):

        for m in self.classifier:

            if isinstance(
                m,
                nn.Linear
            ):

                nn.init.constant_(
                    m.bias.data,
                    0.0
                )

                nn.init.normal_(
                    m.weight.data,
                    mean=0.0,
                    std=0.01
                )


    def forward(self,x):

        x = self.channel_bn(x)

        return (
            self.classifier(x)
            .squeeze(-1)
        )


# ============================================================
# PATIENT AGGREGATION
# ============================================================

def patient_from_image_predictions(
    patient_id,
    y_patient,
    p_image
):

    df = pd.DataFrame({
        "patient_id":
            patient_id.astype(str),

        "y_patient":
            y_patient.astype(int),

        "p_image":
            np.asarray(p_image),
    })

    # Locked primary clinical rule:
    # max image probability within patient
    pt = (
        df
        .groupby(
            "patient_id",
            as_index=False
        )
        .agg(
            y=(
                "y_patient",
                "first"
            ),

            p=(
                "p_image",
                "max"
            )
        )
    )

    return pt


# ============================================================
# MODEL PREDICTION
# ============================================================

@torch.inference_mode()
def predict_images(
    model,
    X
):

    model.eval()

    X_tensor = torch.from_numpy(
        X
    ).float()

    loader = DataLoader(
        TensorDataset(
            X_tensor
        ),
        batch_size=512,
        shuffle=False,
    )

    probs = []

    for (xb,) in loader:

        logits = model(xb)

        probs.append(
            torch.sigmoid(
                logits
            ).cpu().numpy()
        )

    return np.concatenate(
        probs
    )


def predict_patient(
    model,
    cohort
):

    p_image = predict_images(
        model,
        cohort["X"]
    )

    return patient_from_image_predictions(
        cohort["patient_id"],
        cohort["y_patient"],
        p_image,
    )


# ============================================================
# CALIBRATION HELPERS
# ============================================================

EPS = 1e-7


def logit_np(p):

    p = np.clip(
        np.asarray(p),
        EPS,
        1-EPS
    )

    return np.log(
        p/(1-p)
    )


def fit_platt(
    y,
    p
):

    X = (
        logit_np(p)
        .reshape(-1,1)
    )

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
    )

    model.fit(
        X,
        np.asarray(y).astype(int)
    )

    return model


def apply_platt(
    model,
    p
):

    return (
        model.predict_proba(
            logit_np(p)
            .reshape(-1,1)
        )[:,1]
    )


def cal_intercept_slope(
    y,
    p
):

    model = fit_platt(
        y,
        p
    )

    return (
        float(
            model.intercept_[0]
        ),

        float(
            model.coef_[0][0]
        )
    )


def ece_score(
    y,
    p,
    bins=10
):

    y = np.asarray(y)
    p = np.asarray(p)

    edges = np.linspace(
        0,
        1,
        bins+1
    )

    total = 0.0

    for i in range(bins):

        if i < bins-1:

            mask = (
                (p >= edges[i])
                &
                (p < edges[i+1])
            )

        else:

            mask = (
                (p >= edges[i])
                &
                (p <= edges[i+1])
            )

        if not mask.any():
            continue

        total += (
            mask.mean()
            *
            abs(
                y[mask].mean()
                -
                p[mask].mean()
            )
        )

    return float(total)


# ============================================================
# CLINICAL THRESHOLD
# ============================================================

def threshold_90(
    y,
    p
):

    y = np.asarray(y).astype(int)
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
                labels=[0,1]
            ).ravel()
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
                th
            )

            if (
                best is None
                or
                candidate[0]
                >
                best[0]
            ):

                best = candidate

    return (
        float(best[1])
        if best is not None
        else 0.0
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    y,
    p,
    threshold=None
):

    y = np.asarray(y).astype(int)

    p = np.clip(
        np.asarray(p),
        EPS,
        1-EPS
    )

    ci,cs = (
        cal_intercept_slope(
            y,
            p
        )
    )

    out = {

        "auroc":
            roc_auc_score(
                y,p
            ),

        "auprc":
            average_precision_score(
                y,p
            ),

        "brier":
            brier_score_loss(
                y,p
            ),

        "nll":
            log_loss(
                y,p,
                labels=[0,1]
            ),

        "ece":
            ece_score(
                y,p
            ),

        "calibration_intercept":
            ci,

        "calibration_slope":
            cs,
    }

    if threshold is not None:

        pred = (
            p >= threshold
        ).astype(int)

        tn,fp,fn,tp = (
            confusion_matrix(
                y,
                pred,
                labels=[0,1]
            ).ravel()
        )

        sens = tp/(tp+fn)
        spec = tn/(tn+fp)

        out.update({

            "sensitivity":
                sens,

            "specificity":
                spec,

            "balanced_accuracy":
                (sens+spec)/2,

            "ppv":
                tp/(tp+fp)
                if tp+fp
                else np.nan,

            "npv":
                tn/(tn+fn)
                if tn+fn
                else np.nan,
        })

    return out


# ============================================================
# PAIRED BOOTSTRAP FOR RECALIBRATION GAIN
# ============================================================

def bootstrap_gain(
    y,
    p_source,
    p_local,
    seed,
    n=N_BOOT
):

    y = np.asarray(y)

    ps = np.asarray(
        p_source
    )

    pl = np.asarray(
        p_local
    )

    rng = np.random.default_rng(
        seed
    )

    brier_gain = []
    nll_gain = []

    N = len(y)

    for _ in range(n):

        ii = rng.integers(
            0,
            N,
            N
        )

        yy = y[ii]

        pps = ps[ii]
        ppl = pl[ii]

        brier_gain.append(
            brier_score_loss(
                yy,
                pps
            )
            -
            brier_score_loss(
                yy,
                ppl
            )
        )

        nll_gain.append(
            log_loss(
                yy,
                pps,
                labels=[0,1]
            )
            -
            log_loss(
                yy,
                ppl,
                labels=[0,1]
            )
        )

    return {

        "brier_gain_CI_low":
            float(
                np.quantile(
                    brier_gain,
                    .025
                )
            ),

        "brier_gain_CI_high":
            float(
                np.quantile(
                    brier_gain,
                    .975
                )
            ),

        "nll_gain_CI_low":
            float(
                np.quantile(
                    nll_gain,
                    .025
                )
            ),

        "nll_gain_CI_high":
            float(
                np.quantile(
                    nll_gain,
                    .975
                )
            ),
    }


# ============================================================
# TRAIN SOURCE DECODER
# ============================================================

def train_decoder(
    seed
):

    set_seed(seed)

    train = data[
        "BR_TRAIN"
    ]

    val = data[
        "BR_VAL"
    ]

    Xtr = torch.from_numpy(
        train["X"]
    ).float()

    ytr = torch.from_numpy(
        train["y_image"]
    ).float()

    loader = DataLoader(
        TensorDataset(
            Xtr,
            ytr
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = VisionFMClsHead(
        embed_dim=3072
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9,0.999),
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=0
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    best_auc = -np.inf
    best_state = None
    best_epoch = None

    no_improve = 0

    history = []

    for epoch in range(
        MAX_EPOCHS
    ):

        model.train()

        losses = []

        for xb,yb in loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            optimizer.step()

            losses.append(
                float(
                    loss.detach()
                )
            )

        scheduler.step()

        # patient-level BRSET validation
        val_pt = predict_patient(
            model,
            val
        )

        val_auc = roc_auc_score(
            val_pt["y"],
            val_pt["p"]
        )

        history.append({

            "seed":
                seed,

            "epoch":
                epoch+1,

            "loss":
                np.mean(losses),

            "val_patient_auroc":
                val_auc,

            "lr":
                optimizer
                .param_groups[0]["lr"],
        })

        if (
            val_auc
            >
            best_auc + 1e-4
        ):

            best_auc = val_auc

            best_epoch = (
                epoch+1
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            no_improve = 0

        else:

            no_improve += 1

        if (
            epoch == 0
            or
            (epoch+1) % 10 == 0
        ):

            print(
                f"seed={seed} "
                f"epoch={epoch+1} "
                f"loss={np.mean(losses):.4f} "
                f"val_patient_AUC={val_auc:.4f}"
            )

        if (
            no_improve
            >= PATIENCE
        ):
            break

    assert best_state is not None

    model.load_state_dict(
        best_state
    )

    return (
        model,
        best_epoch,
        best_auc,
        pd.DataFrame(history)
    )


# ============================================================
# EVALUATE THREE SOURCE DECODER SEEDS
# ============================================================

results = []
prediction_rows = []

checkpoint_dir = (
    OUT /
    "decoder_checkpoints"
)

checkpoint_dir.mkdir(
    exist_ok=True
)

for seed in SEEDS:

    print(
        "\n" + "="*80
    )

    print(
        "VISIONFM DECODER SEED",
        seed
    )

    print(
        "="*80
    )

    model,best_epoch,best_val_auc,history = (
        train_decoder(
            seed
        )
    )

    history.to_csv(
        OUT /
        f"history_{seed}.csv",
        index=False
    )

    torch.save(
        {
            "seed":
                seed,

            "best_epoch":
                best_epoch,

            "state_dict":
                model.state_dict(),
        },
        checkpoint_dir /
        f"VisionFM_BRSET_decoder_seed{seed}.pth"
    )

    # --------------------------------------------------------
    # Source validation calibration
    # --------------------------------------------------------

    br_val = predict_patient(
        model,
        data["BR_VAL"]
    )

    source_platt = fit_platt(
        br_val["y"],
        br_val["p"]
    )

    br_val[
        "p_sourcecal"
    ] = apply_platt(
        source_platt,
        br_val["p"]
    )

    source_threshold = threshold_90(
        br_val["y"],
        br_val["p_sourcecal"]
    )

    # --------------------------------------------------------
    # Evaluation cohorts
    # --------------------------------------------------------

    br_test = predict_patient(
        model,
        data["BR_TEST"]
    )

    mb_local = predict_patient(
        model,
        data["MB_LOCAL100"]
    )

    mb_test = predict_patient(
        model,
        data["MB_TEST"]
    )

    dd_val = predict_patient(
        model,
        data["DD_VAL"]
    )

    dd_test = predict_patient(
        model,
        data["DD_TEST"]
    )

    for df in [
        br_test,
        mb_test,
        dd_test,
    ]:

        df[
            "p_sourcecal"
        ] = apply_platt(
            source_platt,
            df["p"]
        )

    # --------------------------------------------------------
    # Local 100-label recalibration — mBRSET
    # --------------------------------------------------------

    mb_platt = fit_platt(
        mb_local["y"],
        mb_local["p"]
    )

    mb_local[
        "p_localcal"
    ] = apply_platt(
        mb_platt,
        mb_local["p"]
    )

    mb_test[
        "p_localcal"
    ] = apply_platt(
        mb_platt,
        mb_test["p"]
    )

    mb_threshold = threshold_90(
        mb_local["y"],
        mb_local["p_localcal"]
    )

    # --------------------------------------------------------
    # Local 100-label recalibration — DeepDRiD
    # --------------------------------------------------------

    dd_platt = fit_platt(
        dd_val["y"],
        dd_val["p"]
    )

    dd_val[
        "p_localcal"
    ] = apply_platt(
        dd_platt,
        dd_val["p"]
    )

    dd_test[
        "p_localcal"
    ] = apply_platt(
        dd_platt,
        dd_test["p"]
    )

    dd_threshold = threshold_90(
        dd_val["y"],
        dd_val["p_localcal"]
    )

    # --------------------------------------------------------
    # Source performance
    # --------------------------------------------------------

    source_metrics = metrics(
        br_test["y"],
        br_test["p_sourcecal"],
        source_threshold
    )

    # --------------------------------------------------------
    # mBRSET
    # --------------------------------------------------------

    mb_source_metrics = metrics(
        mb_test["y"],
        mb_test["p_sourcecal"],
        source_threshold
    )

    mb_local_metrics = metrics(
        mb_test["y"],
        mb_test["p_localcal"],
        mb_threshold
    )

    mb_boot = bootstrap_gain(
        mb_test["y"],
        mb_test["p_sourcecal"],
        mb_test["p_localcal"],
        seed=seed+100
    )

    # --------------------------------------------------------
    # DeepDRiD
    # --------------------------------------------------------

    dd_source_metrics = metrics(
        dd_test["y"],
        dd_test["p_sourcecal"],
        source_threshold
    )

    dd_local_metrics = metrics(
        dd_test["y"],
        dd_test["p_localcal"],
        dd_threshold
    )

    dd_boot = bootstrap_gain(
        dd_test["y"],
        dd_test["p_sourcecal"],
        dd_test["p_localcal"],
        seed=seed+200
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    for domain,df in [
        ("BRSET",br_test),
        ("mBRSET",mb_test),
        ("DeepDRiD",dd_test),
    ]:

        temp = df.copy()

        temp["seed"] = seed
        temp["domain"] = domain

        if (
            "p_localcal"
            not in temp.columns
        ):

            temp[
                "p_localcal"
            ] = np.nan

        prediction_rows.append(
            temp[
                [
                    "seed",
                    "domain",
                    "patient_id",
                    "y",
                    "p",
                    "p_sourcecal",
                    "p_localcal",
                ]
            ]
        )

    # --------------------------------------------------------
    # Calibration-success rule
    # --------------------------------------------------------

    def recal_success(
        source_m,
        local_m,
        boot
    ):

        brier_gain = (
            source_m["brier"]
            -
            local_m["brier"]
        )

        nll_gain = (
            source_m["nll"]
            -
            local_m["nll"]
        )

        slope_better = (
            abs(
                local_m[
                    "calibration_slope"
                ]
                - 1
            )
            <
            abs(
                source_m[
                    "calibration_slope"
                ]
                - 1
            )
        )

        threshold_safe = (
            local_m[
                "balanced_accuracy"
            ]
            >=
            source_m[
                "balanced_accuracy"
            ]
            - 0.05
        )

        success = (
            brier_gain > 0
            and
            nll_gain > 0
            and
            boot[
                "brier_gain_CI_low"
            ] > 0
            and
            slope_better
            and
            threshold_safe
        )

        return (
            success,
            brier_gain,
            nll_gain
        )

    (
        mb_success,
        mb_brier_gain,
        mb_nll_gain
    ) = recal_success(
        mb_source_metrics,
        mb_local_metrics,
        mb_boot
    )

    (
        dd_success,
        dd_brier_gain,
        dd_nll_gain
    ) = recal_success(
        dd_source_metrics,
        dd_local_metrics,
        dd_boot
    )

    # --------------------------------------------------------
    # Result row
    # --------------------------------------------------------

    result = {

        "seed":
            seed,

        "best_epoch":
            best_epoch,

        "BR_VAL_AUROC":
            best_val_auc,

        # Source
        "BR_TEST_AUROC":
            source_metrics["auroc"],

        "BR_TEST_Brier":
            source_metrics["brier"],

        # mBRSET zero-label
        "MB_AUROC":
            mb_source_metrics["auroc"],

        "MB_delta_AUROC_vs_BR":
            mb_source_metrics["auroc"]
            -
            source_metrics["auroc"],

        "MB_sourcecal_Brier":
            mb_source_metrics["brier"],

        "MB_sourcecal_slope":
            mb_source_metrics[
                "calibration_slope"
            ],

        # mBRSET local calibration
        "MB_localcal_Brier":
            mb_local_metrics["brier"],

        "MB_localcal_slope":
            mb_local_metrics[
                "calibration_slope"
            ],

        "MB_Brier_gain":
            mb_brier_gain,

        "MB_NLL_gain":
            mb_nll_gain,

        "MB_brier_gain_CI_low":
            mb_boot[
                "brier_gain_CI_low"
            ],

        "MB_brier_gain_CI_high":
            mb_boot[
                "brier_gain_CI_high"
            ],

        "MB_recalibration_success":
            mb_success,

        # DeepDRiD zero-label
        "DD_AUROC":
            dd_source_metrics["auroc"],

        "DD_delta_AUROC_vs_BR":
            dd_source_metrics["auroc"]
            -
            source_metrics["auroc"],

        "DD_sourcecal_Brier":
            dd_source_metrics["brier"],

        "DD_sourcecal_slope":
            dd_source_metrics[
                "calibration_slope"
            ],

        # DeepDRiD recalibration
        "DD_localcal_Brier":
            dd_local_metrics["brier"],

        "DD_localcal_slope":
            dd_local_metrics[
                "calibration_slope"
            ],

        "DD_Brier_gain":
            dd_brier_gain,

        "DD_NLL_gain":
            dd_nll_gain,

        "DD_brier_gain_CI_low":
            dd_boot[
                "brier_gain_CI_low"
            ],

        "DD_brier_gain_CI_high":
            dd_boot[
                "brier_gain_CI_high"
            ],

        "DD_recalibration_success":
            dd_success,
    }

    results.append(
        result
    )

    print(
        "BR source AUROC:",
        round(
            result[
                "BR_TEST_AUROC"
            ],
            4
        )
    )

    print(
        "mBRSET AUROC:",
        round(
            result[
                "MB_AUROC"
            ],
            4
        ),
        "| recal Brier:",
        round(
            result[
                "MB_sourcecal_Brier"
            ],
            4
        ),
        "→",
        round(
            result[
                "MB_localcal_Brier"
            ],
            4
        ),
        "| success:",
        mb_success
    )

    print(
        "DeepDRiD AUROC:",
        round(
            result[
                "DD_AUROC"
            ],
            4
        ),
        "| recal Brier:",
        round(
            result[
                "DD_sourcecal_Brier"
            ],
            4
        ),
        "→",
        round(
            result[
                "DD_localcal_Brier"
            ],
            4
        ),
        "| success:",
        dd_success
    )


# ============================================================
# SAVE
# ============================================================

res = pd.DataFrame(
    results
)

pred = pd.concat(
    prediction_rows,
    ignore_index=True
)

res.to_csv(
    OUT /
    "01_seed_level_visionfm_metrics.csv",
    index=False
)

pred.to_csv(
    OUT /
    "02_patient_predictions.csv",
    index=False
)

# ============================================================
# FINAL ROBUSTNESS GATE
# ============================================================

source_viable = int(
    (
        res[
            "BR_TEST_AUROC"
        ]
        >= 0.80
    ).sum()
)

mb_success = int(
    res[
        "MB_recalibration_success"
    ].sum()
)

dd_success = int(
    res[
        "DD_recalibration_success"
    ].sum()
)

if (
    source_viable >= 2
    and
    mb_success >= 2
    and
    dd_success >= 2
):

    VERDICT = (
        "STRONG SECOND-BACKBONE "
        "REPLICATION — PASS"
    )

    DATA_FREEZE = (
        "YES — CORE DATA PACKAGE "
        "CAN BE FROZEN"
    )

elif (
    source_viable >= 2
    and
    (
        mb_success >= 2
        or
        dd_success >= 2
    )
):

    VERDICT = (
        "PARTIAL SECOND-BACKBONE "
        "REPLICATION — BORDERLINE"
    )

    DATA_FREEZE = (
        "NO — HUMAN REVIEW"
    )

else:

    VERDICT = (
        "SECOND-BACKBONE "
        "REPLICATION FAILED"
    )

    DATA_FREEZE = (
        "NO"
    )

# ============================================================
# REPORT
# ============================================================

report = f"""# PHASE 2A1 — VISIONFM SECOND-BACKBONE ROBUSTNESS

## Backbone

VisionFM Fundus ViT-Base

Frozen VisionFM encoder.

Official-style 3-layer binary classification decoder trained
on frozen last-four-block CLS features.

## Source development

BRSET TRAIN:
939 patients.

BRSET VALIDATION:
202 patients.

Three independent decoder training seeds.

Checkpoint selection used BRSET validation patient-level AUROC only.

## External domains

mBRSET:
400-patient frozen target test.

Local recalibration:
the prespecified 100-patient local cohort selected before
VisionFM analysis.

DeepDRiD:
100-patient frozen confirmatory evaluation set.

Local recalibration:
100 official validation patients.

## Seed-level results

{res.to_string(index=False)}

## Recalibration replication

mBRSET successful seeds:

**{mb_success}/3**

DeepDRiD successful seeds:

**{dd_success}/3**

Source-viable decoder seeds:

**{source_viable}/3**

## Phase 2A1 verdict

**{VERDICT}**

## Core data freeze

**{DATA_FREEZE}**

## Interpretation

This stage tests whether the maintenance pattern observed with
RETFound generalizes to a second ophthalmic foundation-model
backbone.

The primary robustness claim is not that VisionFM must exhibit
the identical AUROC shift magnitude as RETFound.

The key claim is that calibration learned in the BRSET source
environment may fail to transport reliably, while a small local
calibration cohort can restore probability reliability without
updating the foundation-model representation.

Aggressive full fine-tuning was not repeated for VisionFM because
that question had already been established as a harmful comparator
in the RETFound development experiments.

No target-domain image was used for encoder or decoder weight
training.
"""

report_path = (
    OUT /
    "PHASE_2A1_VISIONFM_ROBUSTNESS_REPORT.md"
)

report_path.write_text(
    report,
    encoding="utf-8"
)

print(
    "\n" + "="*80
)

print(
    "PHASE 2A1 FINAL"
)

print(
    "="*80
)

print(
    res.to_string(
        index=False
    )
)

print(
    "\nSOURCE-VIABLE SEEDS:",
    source_viable,
    "/3"
)

print(
    "mBRSET RECALIBRATION SUCCESS:",
    mb_success,
    "/3"
)

print(
    "DeepDRiD RECALIBRATION SUCCESS:",
    dd_success,
    "/3"
)

print(
    "\nVERDICT:",
    VERDICT
)

print(
    "\nDATA FREEZE:",
    DATA_FREEZE
)

print(
    "\nREPORT:",
    report_path
)