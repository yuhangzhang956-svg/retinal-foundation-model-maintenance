
# ============================================================
# PHASE 2A3
# VisionFM × mBRSET 100-label HEAD-ONLY comparator
#
# Scientific question:
# Does LoRA provide incremental discrimination benefit beyond
# retraining the downstream classification head alone?
#
# Prespecified before observing head-only target-test results.
# ============================================================

import json
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# 0. CONFIG
# ============================================================

ROOT = Path(
    "/content/vfm_head_drive/MyDrive/"
    "Retinal_Retraining_Phase1A1"
)

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

PHASE2A1 = (
    ROOT
    / "results"
    / "phase2a1_visionfm"
    / "01_seed_level_visionfm_metrics.csv"
)

PHASE2A2_METRICS = (
    ROOT
    / "results"
    / "phase2a2_visionfm_mbrset_rescue"
    / "01_seed_method_metrics.csv"
)

PHASE2A2_PRED = (
    ROOT
    / "results"
    / "phase2a2_visionfm_mbrset_rescue"
    / "02_patient_predictions.csv"
)

OUT = (
    ROOT
    / "results"
    / "phase2a3_visionfm_head_only"
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

# Match LoRA head-training conditions
EPOCHS = 10
BATCH_SIZE = 8

LR = 3e-4
WEIGHT_DECAY = 0.01

N_BOOT = 5000

print("=" * 80)
print("PHASE 2A3")
print("VisionFM × mBRSET 100-label HEAD-ONLY comparator")
print("=" * 80)

print("Device: CPU")

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# 2. LOAD FROZEN FEATURES
# ============================================================

def load_cache(name):

    p = (
        CACHE
        / f"{name}_features.npz"
    )

    d = np.load(p)

    X = d["X"].astype(
        np.float32
    )

    pid = d[
        "patient_id"
    ].astype(str)

    yi = d[
        "y_image"
    ].astype(int)

    yp = d[
        "y_patient"
    ].astype(int)

    assert len(X) == len(pid)
    assert X.shape[1] == 3072

    return {
        "X": X,
        "patient_id": pid,
        "y_image": yi,
        "y_patient": yp,
    }


data = {
    x: load_cache(x)
    for x in [
        "BR_TEST",
        "MB_LOCAL100",
        "MB_TEST",
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

assert (
    len(
        np.unique(
            data[
                "MB_LOCAL100"
            ]["patient_id"]
        )
    )
    == 100
)

assert (
    len(
        np.unique(
            data[
                "MB_TEST"
            ]["patient_id"]
        )
    )
    == 400
)

assert (
    len(
        np.unique(
            data[
                "BR_TEST"
            ]["patient_id"]
        )
    )
    == 202
)

# ============================================================
# 3. EXACT SOURCE HEAD ARCHITECTURE
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
                embed_dim // 2,
            ),

            nn.GELU(),

            nn.Dropout(
                p=0.1
            ),

            nn.Linear(
                embed_dim // 2,
                embed_dim // 4,
            ),

            nn.GELU(),

            nn.Dropout(
                p=0.1
            ),

            nn.Linear(
                embed_dim // 4,
                1,
            ),
        )


    def forward(
        self,
        x,
    ):

        x = self.channel_bn(
            x
        )

        return (
            self.classifier(
                x
            )
            .squeeze(-1)
        )


def load_source_head(seed):

    model = VisionFMClsHead(
        3072
    )

    p = (
        HEAD_DIR
        /
        f"VisionFM_BRSET_decoder_seed{seed}.pth"
    )

    ckpt = torch.load(
        p,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["state_dict"],
        strict=True,
    )

    return model


# ============================================================
# 4. PATIENT AGGREGATION
# ============================================================

def aggregate_patient(
    cohort,
    probs,
):

    df = pd.DataFrame({

        "patient_id":
            cohort[
                "patient_id"
            ].astype(str),

        "y":
            cohort[
                "y_patient"
            ].astype(int),

        "p":
            np.asarray(
                probs
            ),
    })

    out = (
        df
        .groupby(
            "patient_id",
            as_index=False,
        )
        .agg(
            y=(
                "y",
                "first"
            ),

            p=(
                "p",
                "max"
            ),
        )
    )

    # patient label must be internally consistent
    n_unique = (
        df
        .groupby(
            "patient_id"
        )["y"]
        .nunique()
    )

    assert (
        n_unique.max()
        == 1
    )

    return out


# ============================================================
# 5. PREDICTION
# ============================================================

@torch.inference_mode()
def predict_patient(
    model,
    cohort,
):

    model.eval()

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

        logits = model(
            xb
        )

        probs.append(
            torch.sigmoid(
                logits
            )
            .numpy()
        )

    probs = np.concatenate(
        probs
    )

    return aggregate_patient(
        cohort,
        probs,
    )


# ============================================================
# 6. METRICS
# ============================================================

def calc_metrics(df):

    y = df[
        "y"
    ].to_numpy()

    p = df[
        "p"
    ].to_numpy()

    return {

        "auroc":
            float(
                roc_auc_score(
                    y,p
                )
            ),

        "auprc":
            float(
                average_precision_score(
                    y,p
                )
            ),
    }


# ============================================================
# 7. STRATIFIED PAIRED BOOTSTRAP
# ============================================================

def paired_auc_delta_ci(
    reference_df,
    new_df,
    seed,
    n_boot=N_BOOT,
):

    a = (
        reference_df
        .sort_values(
            "patient_id"
        )
        .reset_index(
            drop=True
        )
    )

    b = (
        new_df
        .sort_values(
            "patient_id"
        )
        .reset_index(
            drop=True
        )
    )

    assert np.array_equal(
        a["patient_id"],
        b["patient_id"],
    )

    assert np.array_equal(
        a["y"],
        b["y"],
    )

    y = a[
        "y"
    ].to_numpy()

    p0 = a[
        "p"
    ].to_numpy()

    p1 = b[
        "p"
    ].to_numpy()

    pos = np.where(
        y == 1
    )[0]

    neg = np.where(
        y == 0
    )[0]

    rng = (
        np.random.default_rng(
            seed
        )
    )

    delta = []

    for _ in range(
        n_boot
    ):

        ip = rng.choice(
            pos,
            len(pos),
            replace=True,
        )

        ineg = rng.choice(
            neg,
            len(neg),
            replace=True,
        )

        idx = np.concatenate(
            [ip,ineg]
        )

        yy = y[idx]

        d = (
            roc_auc_score(
                yy,
                p1[idx]
            )
            -
            roc_auc_score(
                yy,
                p0[idx]
            )
        )

        delta.append(
            d
        )

    return (
        float(
            np.quantile(
                delta,
                0.025
            )
        ),

        float(
            np.quantile(
                delta,
                0.975
            )
        ),
    )


# ============================================================
# 8. HEAD-ONLY TRAINING
# ============================================================

def train_head_only(
    seed,
):

    set_seed(seed)

    model = load_source_head(
        seed
    )

    train = data[
        "MB_LOCAL100"
    ]

    X = torch.from_numpy(
        train["X"]
    ).float()

    y = torch.from_numpy(
        train["y_image"]
    ).float()

    loader_generator = (
        torch.Generator()
    )

    loader_generator.manual_seed(
        seed
    )

    loader = DataLoader(
        TensorDataset(
            X,y
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=
            loader_generator,
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=
                WEIGHT_DECAY,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=0,
        )
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    history = []

    # IMPORTANT:
    # fixed 10 epochs.
    # No frozen BRSET/mBRSET test evaluation during training.
    # No target-test model selection.
    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        losses = []

        for xb,yb in loader:

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                xb
            )

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.detach()
                )
            )

        scheduler.step()

        row = {
            "seed":
                seed,

            "epoch":
                epoch,

            "loss":
                float(
                    np.mean(
                        losses
                    )
                ),

            "lr":
                float(
                    optimizer
                    .param_groups[0]["lr"]
                ),
        }

        history.append(
            row
        )

        print(
            f"head_only "
            f"seed={seed} "
            f"epoch={epoch:02d}/{EPOCHS} "
            f"loss={row['loss']:.5f} "
            f"lr={row['lr']:.2e}"
        )

    return (
        model,
        pd.DataFrame(
            history
        )
    )


# ============================================================
# 9. EXISTING RESULTS
# ============================================================

phase2a1 = pd.read_csv(
    PHASE2A1
)

phase2a1[
    "seed"
] = phase2a1[
    "seed"
].astype(int)

phase2a2_metrics = pd.read_csv(
    PHASE2A2_METRICS
)

phase2a2_metrics[
    "seed"
] = phase2a2_metrics[
    "seed"
].astype(int)

phase2a2_pred = pd.read_csv(
    PHASE2A2_PRED
)

phase2a2_pred[
    "seed"
] = phase2a2_pred[
    "seed"
].astype(int)


# ============================================================
# 10. MAIN
# ============================================================

results = []
histories = []
predictions = []

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

    # --------------------------------------------------------
    # Baseline from exact frozen source decoder
    # --------------------------------------------------------

    base_model = load_source_head(
        seed
    )

    br_base = predict_patient(
        base_model,
        data[
            "BR_TEST"
        ]
    )

    mb_base = predict_patient(
        base_model,
        data[
            "MB_TEST"
        ]
    )

    br_base_m = calc_metrics(
        br_base
    )

    mb_base_m = calc_metrics(
        mb_base
    )

    ref = (
        phase2a1[
            phase2a1[
                "seed"
            ]
            == seed
        ]
        .iloc[0]
    )

    print(
        "BR baseline:",
        round(
            br_base_m[
                "auroc"
            ],
            6
        ),
        "| Phase2A1:",
        round(
            ref[
                "BR_TEST_AUROC"
            ],
            6
        )
    )

    print(
        "MB baseline:",
        round(
            mb_base_m[
                "auroc"
            ],
            6
        ),
        "| Phase2A1:",
        round(
            ref[
                "MB_AUROC"
            ],
            6
        )
    )

    assert abs(
        br_base_m[
            "auroc"
        ]
        -
        ref[
            "BR_TEST_AUROC"
        ]
    ) < 1e-6

    assert abs(
        mb_base_m[
            "auroc"
        ]
        -
        ref[
            "MB_AUROC"
        ]
    ) < 1e-6

    # --------------------------------------------------------
    # Head-only adaptation
    # --------------------------------------------------------

    (
        head,
        history
    ) = train_head_only(
        seed
    )

    histories.append(
        history
    )

    # first frozen evaluation after training
    br_head = predict_patient(
        head,
        data[
            "BR_TEST"
        ]
    )

    mb_head = predict_patient(
        head,
        data[
            "MB_TEST"
        ]
    )

    br_head_m = calc_metrics(
        br_head
    )

    mb_head_m = calc_metrics(
        mb_head
    )

    source_change = (
        br_head_m[
            "auroc"
        ]
        -
        br_base_m[
            "auroc"
        ]
    )

    target_gain = (
        mb_head_m[
            "auroc"
        ]
        -
        mb_base_m[
            "auroc"
        ]
    )

    br_ci = paired_auc_delta_ci(
        br_base,
        br_head,
        seed=
            seed + 1000,
    )

    mb_ci = paired_auc_delta_ci(
        mb_base,
        mb_head,
        seed=
            seed + 2000,
    )

    # --------------------------------------------------------
    # Existing LoRA prediction from Phase 2A2
    # --------------------------------------------------------

    lora_mb = (
        phase2a2_pred[
            (
                phase2a2_pred[
                    "seed"
                ]
                == seed
            )
            &
            (
                phase2a2_pred[
                    "method"
                ]
                == "lora"
            )
            &
            (
                phase2a2_pred[
                    "domain"
                ]
                == "mBRSET"
            )
        ][
            [
                "patient_id",
                "y",
                "p",
            ]
        ]
        .copy()
    )

    assert (
        len(
            lora_mb
        )
        == 400
    )

    lora_mb_m = calc_metrics(
        lora_mb
    )

    # LoRA minus head-only AUROC
    lora_minus_head = (
        lora_mb_m[
            "auroc"
        ]
        -
        mb_head_m[
            "auroc"
        ]
    )

    lora_head_ci = (
        paired_auc_delta_ci(
            mb_head,
            lora_mb,
            seed=
                seed + 3000,
        )
    )

    print(
        "\nHEAD-ONLY RESULT"
    )

    print(
        "BR:",
        f"{br_base_m['auroc']:.4f}",
        "→",
        f"{br_head_m['auroc']:.4f}",
        "| Δ",
        f"{source_change:+.4f}",
        "| 95% CI",
        f"[{br_ci[0]:+.4f}, {br_ci[1]:+.4f}]"
    )

    print(
        "MB:",
        f"{mb_base_m['auroc']:.4f}",
        "→",
        f"{mb_head_m['auroc']:.4f}",
        "| Δ",
        f"{target_gain:+.4f}",
        "| 95% CI",
        f"[{mb_ci[0]:+.4f}, {mb_ci[1]:+.4f}]"
    )

    print(
        "LoRA vs head-only:",
        f"{lora_mb_m['auroc']:.4f}",
        "vs",
        f"{mb_head_m['auroc']:.4f}",
        "| LoRA-head Δ",
        f"{lora_minus_head:+.4f}",
        "| 95% CI",
        f"[{lora_head_ci[0]:+.4f}, "
        f"{lora_head_ci[1]:+.4f}]"
    )

    results.append({

        "seed":
            seed,

        "BR_baseline_AUROC":
            br_base_m[
                "auroc"
            ],

        "BR_head_AUROC":
            br_head_m[
                "auroc"
            ],

        "BR_head_change":
            source_change,

        "BR_head_change_CI_low":
            br_ci[0],

        "BR_head_change_CI_high":
            br_ci[1],

        "MB_baseline_AUROC":
            mb_base_m[
                "auroc"
            ],

        "MB_head_AUROC":
            mb_head_m[
                "auroc"
            ],

        "MB_head_gain":
            target_gain,

        "MB_head_gain_CI_low":
            mb_ci[0],

        "MB_head_gain_CI_high":
            mb_ci[1],

        "MB_LoRA_AUROC":
            lora_mb_m[
                "auroc"
            ],

        "MB_LoRA_minus_head":
            lora_minus_head,

        "MB_LoRA_minus_head_CI_low":
            lora_head_ci[0],

        "MB_LoRA_minus_head_CI_high":
            lora_head_ci[1],

        "MB_recalibration_AUROC":
            float(
                ref[
                    "MB_AUROC"
                ]
            ),
    })

    # save head predictions
    for domain,df in [
        ("BRSET",br_head),
        ("mBRSET",mb_head),
    ]:

        x = df.copy()

        x["seed"] = seed
        x["domain"] = domain
        x["method"] = "head_only"

        predictions.append(
            x
        )


# ============================================================
# 11. SAVE
# ============================================================

res = pd.DataFrame(
    results
)

history = pd.concat(
    histories,
    ignore_index=True,
)

pred = pd.concat(
    predictions,
    ignore_index=True,
)

res.to_csv(
    OUT
    / "01_seed_level_head_only_metrics.csv",
    index=False,
)

history.to_csv(
    OUT
    / "02_training_history.csv",
    index=False,
)

pred.to_csv(
    OUT
    / "03_patient_predictions.csv",
    index=False,
)

# ============================================================
# 12. PRESPECIFIED HEAD-ONLY GATE
# ============================================================

median_target_gain = float(
    res[
        "MB_head_gain"
    ].median()
)

mean_target_gain = float(
    res[
        "MB_head_gain"
    ].mean()
)

head_gain_count = int(
    (
        res[
            "MB_head_gain"
        ]
        >= 0.01
    ).sum()
)

median_source_change = float(
    res[
        "BR_head_change"
    ].median()
)

worst_source_change = float(
    res[
        "BR_head_change"
    ].min()
)

HEAD_TARGET_EFFECTIVE = bool(
    median_target_gain >= 0.01
    and
    head_gain_count >= 2
)

HEAD_SOURCE_SAFE = bool(
    median_source_change >= -0.02
    and
    worst_source_change >= -0.03
)

HEAD_STRATEGY_PASS = bool(
    HEAD_TARGET_EFFECTIVE
    and
    HEAD_SOURCE_SAFE
)

# ============================================================
# 13. INCREMENTAL REPRESENTATION GATE
#
# Question:
# Does LoRA provide material incremental AUROC above head-only?
#
# Frozen threshold:
# median LoRA-head >= +0.01
# AND >=2/3 seeds >= +0.01
# ============================================================

median_lora_advantage = float(
    res[
        "MB_LoRA_minus_head"
    ].median()
)

mean_lora_advantage = float(
    res[
        "MB_LoRA_minus_head"
    ].mean()
)

lora_advantage_count = int(
    (
        res[
            "MB_LoRA_minus_head"
        ]
        >= 0.01
    ).sum()
)

ci_positive_count = int(
    (
        res[
            "MB_LoRA_minus_head_CI_low"
        ]
        > 0
    ).sum()
)

REPRESENTATION_INCREMENT_PASS = bool(
    median_lora_advantage >= 0.01
    and
    lora_advantage_count >= 2
)

REPRESENTATION_INCREMENT_STRONG = bool(
    REPRESENTATION_INCREMENT_PASS
    and
    ci_positive_count >= 2
)

# ============================================================
# 14. INTERPRETATION
# ============================================================

if (
    not HEAD_STRATEGY_PASS
    and
    REPRESENTATION_INCREMENT_PASS
):

    VERDICT = (
        "ENCODER-LEVEL REPRESENTATION "
        "ADAPTATION SUPPORTED — HEAD-ONLY INSUFFICIENT"
    )

elif (
    HEAD_STRATEGY_PASS
    and
    REPRESENTATION_INCREMENT_STRONG
):

    VERDICT = (
        "BOTH LIGHTWEIGHT STRATEGIES EFFECTIVE, "
        "WITH STRONG INCREMENTAL LoRA BENEFIT"
    )

elif (
    HEAD_STRATEGY_PASS
    and
    REPRESENTATION_INCREMENT_PASS
):

    VERDICT = (
        "BOTH LIGHTWEIGHT STRATEGIES EFFECTIVE, "
        "WITH INCREMENTAL LoRA SIGNAL"
    )

elif HEAD_STRATEGY_PASS:

    VERDICT = (
        "HEAD-ONLY ADAPTATION SUFFICIENT; "
        "ENCODER-SPECIFIC LoRA ADVANTAGE NOT ESTABLISHED"
    )

else:

    VERDICT = (
        "HEAD-ONLY NOT RELIABLY EFFECTIVE; "
        "INCREMENTAL LoRA SUPERIORITY NOT FORMALLY ESTABLISHED"
    )


# ============================================================
# 15. REPORT
# ============================================================

summary = {
    "median_head_target_gain":
        median_target_gain,

    "mean_head_target_gain":
        mean_target_gain,

    "head_gain_ge_001_count":
        head_gain_count,

    "median_head_source_change":
        median_source_change,

    "worst_head_source_change":
        worst_source_change,

    "head_target_effective":
        HEAD_TARGET_EFFECTIVE,

    "head_source_safe":
        HEAD_SOURCE_SAFE,

    "head_strategy_pass":
        HEAD_STRATEGY_PASS,

    "median_LoRA_minus_head":
        median_lora_advantage,

    "mean_LoRA_minus_head":
        mean_lora_advantage,

    "LoRA_advantage_ge_001_count":
        lora_advantage_count,

    "LoRA_vs_head_CI_positive_count":
        ci_positive_count,

    "representation_increment_pass":
        REPRESENTATION_INCREMENT_PASS,

    "representation_increment_strong":
        REPRESENTATION_INCREMENT_STRONG,
}

report = f"""# PHASE 2A3 — VISIONFM HEAD-ONLY ABLATION

## Purpose

To determine whether the VisionFM mBRSET rescue observed with
LoRA requires encoder-level representation adaptation, or whether
retraining the downstream classification head alone is sufficient.

## Fixed design

Local cohort:
100 mBRSET patients.

Frozen target test:
400 mBRSET patients.

Frozen source-retention test:
202 BRSET patients.

Backbone:
completely frozen.

Trainable component:
classification head only.

Training:
10 fixed epochs.

Optimizer:
AdamW.

Learning rate:
{LR}.

Weight decay:
{WEIGHT_DECAY}.

Batch size:
{BATCH_SIZE}.

These head-training conditions match the trainable-head component
of the Phase 2A2 LoRA configuration.

No frozen test set was evaluated during adaptation or used for
checkpoint selection.

## Seed-level results

{res.to_string(index=False)}

## Prespecified summary

{json.dumps(summary, indent=2)}

## Verdict

**{VERDICT}**

## Interpretation boundary

If head-only adaptation fails while LoRA succeeds, this supports
an incremental role for encoder-level representation adaptation.

If head-only and LoRA perform similarly, the manuscript should
describe the successful intervention as lightweight local
adaptation rather than attributing the effect specifically to
encoder representation updating.

If both improve but LoRA remains materially superior under the
prespecified comparison, an incremental encoder-level adaptation
effect is supported.

This ablation does not justify claims that LoRA is universally
optimal across retinal foundation models or deployment settings.
"""

REPORT = (
    OUT
    / "PHASE_2A3_VISIONFM_HEAD_ONLY_REPORT.md"
)

REPORT.write_text(
    report,
    encoding="utf-8",
)

# ============================================================
# 16. FINAL PRINT
# ============================================================

print(
    "\n" + "="*80
)

print(
    "PHASE 2A3 FINAL"
)

print(
    "="*80
)

print(
    "\nSEED-LEVEL RESULTS:"
)

print(
    res[
        [
            "seed",
            "BR_baseline_AUROC",
            "BR_head_AUROC",
            "BR_head_change",
            "MB_baseline_AUROC",
            "MB_head_AUROC",
            "MB_head_gain",
            "MB_LoRA_AUROC",
            "MB_LoRA_minus_head",
        ]
    ].to_string(
        index=False
    )
)

print(
    "\nHEAD-ONLY GATE:"
)

print(
    {
        "target_effective":
            HEAD_TARGET_EFFECTIVE,

        "source_safe":
            HEAD_SOURCE_SAFE,

        "strategy_pass":
            HEAD_STRATEGY_PASS,
    }
)

print(
    "\nREPRESENTATION INCREMENT GATE:"
)

print(
    {
        "median_LoRA_minus_head":
            median_lora_advantage,

        "LoRA_advantage_ge_001_count":
            lora_advantage_count,

        "CI_positive_count":
            ci_positive_count,

        "representation_increment_pass":
            REPRESENTATION_INCREMENT_PASS,

        "representation_increment_strong":
            REPRESENTATION_INCREMENT_STRONG,
    }
)

print(
    "\nVERDICT:",
    VERDICT
)

print(
    "\nREPORT:",
    REPORT
)

print(
    "\n✅ PHASE 2A3 COMPLETE"
)