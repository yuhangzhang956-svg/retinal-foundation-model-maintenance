
# ============================================================
# PHASE 2A4
# VisionFM × DeepDRiD local recalibration label-dose replication
#
# Budgets:
#   10 / 25 / 50 / 100 local patients
#
# 50 nested trajectories for 10/25/50
# N=100 is evaluated ONCE per source seed to avoid pseudoreplication.
#
# No encoder or decoder weights are updated.
# CPU only.
# ============================================================

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
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

PHASE2A1 = (
    ROOT
    / "results"
    / "phase2a1_visionfm"
    / "01_seed_level_visionfm_metrics.csv"
)

OUT = (
    ROOT
    / "results"
    / "phase2a4_visionfm_deepdrid_dose"
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

BUDGETS = [
    10,
    25,
    50,
    100,
]

N_TRAJECTORIES = 50
N_BOOT = 1000

TRAJECTORY_SEEDS = [
    20269001 + i
    for i in range(N_TRAJECTORIES)
]

EPS = 1e-7

print("=" * 80)
print("PHASE 2A4")
print("VisionFM × DeepDRiD recalibration label-dose")
print("=" * 80)
print("ROOT:", ROOT)
print("Device: CPU")


# ============================================================
# 1. FEATURE CACHE
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

    yi = d[
        "y_image"
    ].astype(int)

    yp = d[
        "y_patient"
    ].astype(int)

    assert X.shape[1] == 3072

    return {
        "X": X,
        "patient_id": pid,
        "y_image": yi,
        "y_patient": yp,
    }


data = {
    name: load_cache(name)

    for name in [
        "BR_VAL",
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

assert len(
    np.unique(
        data["DD_VAL"]["patient_id"]
    )
) == 100

assert len(
    np.unique(
        data["DD_TEST"]["patient_id"]
    )
) == 100


# ============================================================
# 2. EXACT SOURCE HEAD
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
# 3. RAW PREDICTIONS
# ============================================================

@torch.inference_mode()
def image_probs(
    model,
    X,
):

    X = torch.from_numpy(
        X
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

    return np.concatenate(
        probs
    )


def patient_predictions(
    model,
    cohort,
):

    p = image_probs(
        model,
        cohort["X"],
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
            p,
    })

    # Patient labels must be internally consistent
    consistency = (
        df
        .groupby(
            "patient_id"
        )["y"]
        .nunique()
    )

    assert consistency.max() == 1

    # Locked patient endpoint:
    # max probability across available images
    pt = (
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

    return pt


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

    p = np.asarray(p)

    assert len(
        np.unique(y)
    ) == 2

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
    )

    model.fit(
        logit_np(p)
        .reshape(-1,1),
        y,
    )

    return model


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
# 5. CALIBRATION METRICS
# ============================================================

def calibration_slope(
    y,
    p,
):

    m = fit_platt(
        y,
        p,
    )

    return float(
        m.coef_[0][0]
    )


def ece(
    y,
    p,
    n_bins=10,
):

    y = np.asarray(y)
    p = np.asarray(p)

    edges = np.linspace(
        0,
        1,
        n_bins+1,
    )

    total = 0.0

    for i in range(n_bins):

        if i < n_bins-1:

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


def threshold_90(
    y,
    p,
):

    y = np.asarray(y).astype(int)
    p = np.asarray(p)

    best = None

    for th in np.unique(p):

        pred = (
            p >= th
        ).astype(int)

        tn,fp,fn,tp = confusion_matrix(
            y,
            pred,
            labels=[0,1],
        ).ravel()

        sens = (
            tp/(tp+fn)
            if tp+fn > 0
            else np.nan
        )

        spec = (
            tn/(tn+fp)
            if tn+fp > 0
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

    if best is None:
        return 0.0

    return float(
        best[1]
    )


def threshold_metrics(
    y,
    p,
    threshold,
):

    y = np.asarray(y).astype(int)

    pred = (
        np.asarray(p)
        >= threshold
    ).astype(int)

    tn,fp,fn,tp = confusion_matrix(
        y,
        pred,
        labels=[0,1],
    ).ravel()

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

    return {
        "sensitivity":
            float(sens),

        "specificity":
            float(spec),

        "balanced_accuracy":
            float(
                (sens+spec)/2
            ),
    }


def evaluate(
    y,
    p,
    threshold=None,
):

    y = np.asarray(y).astype(int)

    p = np.clip(
        np.asarray(p),
        EPS,
        1-EPS,
    )

    out = {

        "auroc":
            float(
                roc_auc_score(
                    y,p
                )
            ),

        "brier":
            float(
                brier_score_loss(
                    y,p
                )
            ),

        "nll":
            float(
                log_loss(
                    y,p,
                    labels=[0,1],
                )
            ),

        "ece":
            ece(
                y,p
            ),

        "slope":
            calibration_slope(
                y,p
            ),
    }

    if threshold is not None:

        out.update(
            threshold_metrics(
                y,
                p,
                threshold,
            )
        )

    return out


# ============================================================
# 6. STRATIFIED PAIRED BOOTSTRAP FOR BRIER GAIN
# ============================================================

def brier_gain_ci(
    y,
    p_source,
    p_local,
    seed,
    n_boot=N_BOOT,
):

    y = np.asarray(y).astype(int)

    p0 = np.asarray(
        p_source
    )

    p1 = np.asarray(
        p_local
    )

    pos = np.where(
        y == 1
    )[0]

    neg = np.where(
        y == 0
    )[0]

    rng = np.random.default_rng(
        seed
    )

    # vectorised bootstrap indices
    ip = rng.choice(
        pos,
        size=(
            n_boot,
            len(pos)
        ),
        replace=True,
    )

    ineg = rng.choice(
        neg,
        size=(
            n_boot,
            len(neg)
        ),
        replace=True,
    )

    idx = np.concatenate(
        [ip,ineg],
        axis=1,
    )

    yy = y[idx]
    pp0 = p0[idx]
    pp1 = p1[idx]

    b0 = np.mean(
        (yy-pp0)**2,
        axis=1,
    )

    b1 = np.mean(
        (yy-pp1)**2,
        axis=1,
    )

    gain = (
        b0-b1
    )

    return (
        float(
            np.quantile(
                gain,
                0.025
            )
        ),

        float(
            np.quantile(
                gain,
                0.975
            )
        ),
    )


# ============================================================
# 7. CREATE NESTED PATIENT TRAJECTORIES
# ============================================================

def build_nested_trajectories(
    patient_df,
):

    base = (
        patient_df[
            [
                "patient_id",
                "y",
            ]
        ]
        .copy()
    )

    base[
        "patient_id"
    ] = (
        base[
            "patient_id"
        ]
        .astype(str)
    )

    pos_ids = (
        base[
            base["y"] == 1
        ]["patient_id"]
        .tolist()
    )

    neg_ids = (
        base[
            base["y"] == 0
        ]["patient_id"]
        .tolist()
    )

    n_total = len(base)
    n_pos_total = len(pos_ids)

    assert n_total == 100

    trajectories = {}

    for trajectory_seed in TRAJECTORY_SEEDS:

        rng = np.random.default_rng(
            trajectory_seed
        )

        pos_order = list(
            rng.permutation(
                pos_ids
            )
        )

        neg_order = list(
            rng.permutation(
                neg_ids
            )
        )

        trajectories[
            trajectory_seed
        ] = {}

        previous = set()

        for budget in [
            10,
            25,
            50,
        ]:

            n_pos = int(
                round(
                    budget
                    *
                    n_pos_total
                    /
                    n_total
                )
            )

            n_pos = max(
                1,
                min(
                    n_pos,
                    len(pos_ids)-1,
                )
            )

            n_neg = (
                budget
                -
                n_pos
            )

            selected = (
                pos_order[
                    :n_pos
                ]
                +
                neg_order[
                    :n_neg
                ]
            )

            selected = set(
                selected
            )

            assert len(
                selected
            ) == budget

            # Nested within a trajectory
            assert previous.issubset(
                selected
            )

            previous = selected

            trajectories[
                trajectory_seed
            ][budget] = selected

    return trajectories


# ============================================================
# 8. LOAD LOCKED PHASE 2A1 RESULTS
# ============================================================

locked = pd.read_csv(
    PHASE2A1
)

locked[
    "seed"
] = locked[
    "seed"
].astype(int)


# ============================================================
# 9. MAIN
# ============================================================

all_runs = []
full100_rows = []
prediction_qc = []

trajectory_template = None

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

    head = load_head(
        seed
    )

    # raw predictions
    br_val = patient_predictions(
        head,
        data[
            "BR_VAL"
        ]
    )

    dd_val = patient_predictions(
        head,
        data[
            "DD_VAL"
        ]
    )

    dd_test = patient_predictions(
        head,
        data[
            "DD_TEST"
        ]
    )

    # Same patient labels/order across model seeds
    if trajectory_template is None:

        trajectory_template = (
            build_nested_trajectories(
                dd_val
            )
        )

    # --------------------------------------------------------
    # Source-domain calibration
    # --------------------------------------------------------

    source_platt = fit_platt(
        br_val["y"],
        br_val["p_raw"],
    )

    br_val[
        "p_sourcecal"
    ] = apply_platt(
        source_platt,
        br_val[
            "p_raw"
        ],
    )

    dd_test[
        "p_sourcecal"
    ] = apply_platt(
        source_platt,
        dd_test[
            "p_raw"
        ],
    )

    source_threshold = threshold_90(
        br_val["y"],
        br_val[
            "p_sourcecal"
        ],
    )

    source_m = evaluate(
        dd_test["y"],
        dd_test[
            "p_sourcecal"
        ],
        source_threshold,
    )

    # --------------------------------------------------------
    # FULL 100-PATIENT LOCAL CALIBRATION
    # --------------------------------------------------------

    full_platt = fit_platt(
        dd_val["y"],
        dd_val["p_raw"],
    )

    dd_val[
        "p_full100"
    ] = apply_platt(
        full_platt,
        dd_val[
            "p_raw"
        ],
    )

    dd_test[
        "p_full100"
    ] = apply_platt(
        full_platt,
        dd_test[
            "p_raw"
        ],
    )

    full_threshold = threshold_90(
        dd_val["y"],
        dd_val[
            "p_full100"
        ],
    )

    full_m = evaluate(
        dd_test["y"],
        dd_test[
            "p_full100"
        ],
        full_threshold,
    )

    full_brier_gain = (
        source_m["brier"]
        -
        full_m["brier"]
    )

    full_nll_gain = (
        source_m["nll"]
        -
        full_m["nll"]
    )

    assert (
        full_brier_gain > 0
    ), (
        "Full 100-patient calibration "
        "must have positive Brier gain."
    )

    # --------------------------------------------------------
    # QC against Phase 2A1
    # --------------------------------------------------------

    ref = (
        locked[
            locked[
                "seed"
            ]
            == seed
        ]
        .iloc[0]
    )

    print(
        "DD sourcecal Brier:",
        round(
            source_m[
                "brier"
            ],
            6
        ),
        "| locked:",
        round(
            ref[
                "DD_sourcecal_Brier"
            ],
            6
        )
    )

    print(
        "DD full100 Brier:",
        round(
            full_m[
                "brier"
            ],
            6
        ),
        "| locked:",
        round(
            ref[
                "DD_localcal_Brier"
            ],
            6
        )
    )

    assert abs(
        source_m[
            "brier"
        ]
        -
        ref[
            "DD_sourcecal_Brier"
        ]
    ) < 1e-6

    assert abs(
        full_m[
            "brier"
        ]
        -
        ref[
            "DD_localcal_Brier"
        ]
    ) < 1e-6

    # --------------------------------------------------------
    # Full100 is ONE unique run per source seed
    # --------------------------------------------------------

    ci100 = brier_gain_ci(
        dd_test["y"],
        dd_test[
            "p_sourcecal"
        ],
        dd_test[
            "p_full100"
        ],
        seed=
            seed + 900000,
    )

    slope_better100 = (
        abs(
            full_m[
                "slope"
            ]
            - 1
        )
        <
        abs(
            source_m[
                "slope"
            ]
            - 1
        )
    )

    threshold_safe100 = (
        full_m[
            "balanced_accuracy"
        ]
        >=
        source_m[
            "balanced_accuracy"
        ]
        - 0.05
    )

    basic_success100 = bool(
        full_brier_gain > 0
        and
        full_nll_gain > 0
        and
        slope_better100
        and
        threshold_safe100
    )

    strict_success100 = bool(
        basic_success100
        and
        ci100[0] > 0
    )

    row100 = {

        "seed":
            seed,

        "trajectory_seed":
            np.nan,

        "budget":
            100,

        "unique_run":
            True,

        "n_local_patients":
            100,

        "source_brier":
            source_m["brier"],

        "local_brier":
            full_m["brier"],

        "brier_gain":
            full_brier_gain,

        "brier_gain_CI_low":
            ci100[0],

        "brier_gain_CI_high":
            ci100[1],

        "source_nll":
            source_m["nll"],

        "local_nll":
            full_m["nll"],

        "nll_gain":
            full_nll_gain,

        "source_slope":
            source_m["slope"],

        "local_slope":
            full_m["slope"],

        "slope_better":
            slope_better100,

        "source_balanced_accuracy":
            source_m[
                "balanced_accuracy"
            ],

        "local_balanced_accuracy":
            full_m[
                "balanced_accuracy"
            ],

        "threshold_safe":
            threshold_safe100,

        "local_sensitivity":
            full_m[
                "sensitivity"
            ],

        "local_specificity":
            full_m[
                "specificity"
            ],

        "brier_recovery":
            1.0,

        "nll_recovery":
            1.0,

        "basic_success":
            basic_success100,

        "strict_success":
            strict_success100,
    }

    all_runs.append(
        row100
    )

    full100_rows.append(
        row100
    )

    # --------------------------------------------------------
    # 10 / 25 / 50 trajectories
    # --------------------------------------------------------

    for trajectory_seed in TRAJECTORY_SEEDS:

        for budget in [
            10,
            25,
            50,
        ]:

            selected = (
                trajectory_template[
                    trajectory_seed
                ][budget]
            )

            local = (
                dd_val[
                    dd_val[
                        "patient_id"
                    ]
                    .astype(str)
                    .isin(
                        selected
                    )
                ]
                .copy()
            )

            assert len(
                local
            ) == budget

            assert local[
                "y"
            ].nunique() == 2

            local_platt = fit_platt(
                local["y"],
                local["p_raw"],
            )

            local[
                "p_localcal"
            ] = apply_platt(
                local_platt,
                local[
                    "p_raw"
                ],
            )

            test_p = apply_platt(
                local_platt,
                dd_test[
                    "p_raw"
                ],
            )

            local_threshold = threshold_90(
                local["y"],
                local[
                    "p_localcal"
                ],
            )

            m = evaluate(
                dd_test["y"],
                test_p,
                local_threshold,
            )

            brier_gain = (
                source_m[
                    "brier"
                ]
                -
                m[
                    "brier"
                ]
            )

            nll_gain = (
                source_m[
                    "nll"
                ]
                -
                m[
                    "nll"
                ]
            )

            brier_recovery = (
                brier_gain
                /
                full_brier_gain
            )

            nll_recovery = (
                nll_gain
                /
                full_nll_gain
                if abs(
                    full_nll_gain
                ) > 1e-12
                else np.nan
            )

            ci = brier_gain_ci(
                dd_test["y"],
                dd_test[
                    "p_sourcecal"
                ],
                test_p,
                seed=(
                    seed
                    +
                    trajectory_seed
                    +
                    budget
                ),
            )

            slope_better = (
                abs(
                    m[
                        "slope"
                    ]
                    - 1
                )
                <
                abs(
                    source_m[
                        "slope"
                    ]
                    - 1
                )
            )

            threshold_safe = (
                m[
                    "balanced_accuracy"
                ]
                >=
                source_m[
                    "balanced_accuracy"
                ]
                - 0.05
            )

            basic_success = bool(
                brier_gain > 0
                and
                nll_gain > 0
                and
                slope_better
                and
                threshold_safe
            )

            strict_success = bool(
                basic_success
                and
                ci[0] > 0
            )

            all_runs.append({

                "seed":
                    seed,

                "trajectory_seed":
                    trajectory_seed,

                "budget":
                    budget,

                "unique_run":
                    True,

                "n_local_patients":
                    budget,

                "source_brier":
                    source_m[
                        "brier"
                    ],

                "local_brier":
                    m[
                        "brier"
                    ],

                "brier_gain":
                    brier_gain,

                "brier_gain_CI_low":
                    ci[0],

                "brier_gain_CI_high":
                    ci[1],

                "source_nll":
                    source_m[
                        "nll"
                    ],

                "local_nll":
                    m[
                        "nll"
                    ],

                "nll_gain":
                    nll_gain,

                "source_slope":
                    source_m[
                        "slope"
                    ],

                "local_slope":
                    m[
                        "slope"
                    ],

                "slope_better":
                    slope_better,

                "source_balanced_accuracy":
                    source_m[
                        "balanced_accuracy"
                    ],

                "local_balanced_accuracy":
                    m[
                        "balanced_accuracy"
                    ],

                "threshold_safe":
                    threshold_safe,

                "local_sensitivity":
                    m[
                        "sensitivity"
                    ],

                "local_specificity":
                    m[
                        "specificity"
                    ],

                "brier_recovery":
                    brier_recovery,

                "nll_recovery":
                    nll_recovery,

                "basic_success":
                    basic_success,

                "strict_success":
                    strict_success,
            })

    print(
        "✅ completed seed",
        seed
    )


# ============================================================
# 10. SAVE RUN-LEVEL RESULTS
# ============================================================

runs = pd.DataFrame(
    all_runs
)

runs.to_csv(
    OUT
    / "01_run_level_dose_results.csv",
    index=False,
)


# ============================================================
# 11. CORRECT SUMMARY
#
# IMPORTANT:
# N100 has only 3 rows total.
# ============================================================

def q25(x):
    return float(
        np.quantile(
            x,
            0.25
        )
    )


def q75(x):
    return float(
        np.quantile(
            x,
            0.75
        )
    )


summary = (
    runs
    .groupby(
        "budget"
    )
    .agg(

        n_unique_runs=
            (
                "unique_run",
                "sum",
            ),

        basic_success_rate=
            (
                "basic_success",
                "mean",
            ),

        strict_success_rate=
            (
                "strict_success",
                "mean",
            ),

        median_brier_recovery=
            (
                "brier_recovery",
                "median",
            ),

        q25_brier_recovery=
            (
                "brier_recovery",
                q25,
            ),

        q75_brier_recovery=
            (
                "brier_recovery",
                q75,
            ),

        median_nll_recovery=
            (
                "nll_recovery",
                "median",
            ),

        q25_nll_recovery=
            (
                "nll_recovery",
                q25,
            ),

        q75_nll_recovery=
            (
                "nll_recovery",
                q75,
            ),

        median_brier_gain=
            (
                "brier_gain",
                "median",
            ),

        median_nll_gain=
            (
                "nll_gain",
                "median",
            ),

        median_sensitivity=
            (
                "local_sensitivity",
                "median",
            ),

        median_specificity=
            (
                "local_specificity",
                "median",
            ),

        median_balanced_accuracy=
            (
                "local_balanced_accuracy",
                "median",
            ),
    )
    .reset_index()
    .sort_values(
        "budget"
    )
)

summary.to_csv(
    OUT
    / "02_corrected_budget_summary.csv",
    index=False,
)


# ============================================================
# 12. LABEL-DOSE THRESHOLDS
# ============================================================

def first_budget(
    condition,
):

    x = summary[
        condition
    ]

    if len(x) == 0:
        return None

    return int(
        x.iloc[0][
            "budget"
        ]
    )


# Same descriptive definitions used in prior RETFound analysis:
# efficiency: >=80% median Brier AND NLL recovery
# near-complete: >=95% median recovery
# reliability: >=80% observed success

benefit_efficiency_budget = first_budget(
    (
        summary[
            "median_brier_recovery"
        ]
        >= 0.80
    )
    &
    (
        summary[
            "median_nll_recovery"
        ]
        >= 0.80
    )
)

near_complete_budget = first_budget(
    (
        summary[
            "median_brier_recovery"
        ]
        >= 0.95
    )
    &
    (
        summary[
            "median_nll_recovery"
        ]
        >= 0.95
    )
)

basic_reliable_budget = first_budget(
    summary[
        "basic_success_rate"
    ]
    >= 0.80
)

strict_reliable_budget = first_budget(
    summary[
        "strict_success_rate"
    ]
    >= 0.80
)


# ============================================================
# 13. DOSE-RESPONSE REPLICATION GATES
# ============================================================

s10 = summary[
    summary["budget"] == 10
].iloc[0]

s50 = summary[
    summary["budget"] == 50
].iloc[0]

# Primary:
# 50-patient median benefit must exceed 10-patient benefit
# and recover >=80% of full100 benefit on BOTH Brier and NLL.
DOSE_RESPONSE_PASS = bool(

    s50[
        "median_brier_recovery"
    ]
    >
    s10[
        "median_brier_recovery"
    ]

    and

    s50[
        "median_nll_recovery"
    ]
    >
    s10[
        "median_nll_recovery"
    ]

    and

    s50[
        "median_brier_recovery"
    ]
    >= 0.80

    and

    s50[
        "median_nll_recovery"
    ]
    >= 0.80
)

# Exact descriptive replication of the earlier RETFound thresholds.
EXACT_RETF_THRESHOLD_REPLICATION = bool(

    benefit_efficiency_budget
    == 25

    and

    near_complete_budget
    == 50

    and

    strict_reliable_budget
    == 100
)


# ============================================================
# 14. REPORT
# ============================================================

thresholds = {

    "benefit_efficiency_budget":
        benefit_efficiency_budget,

    "near_complete_budget":
        near_complete_budget,

    "basic_reliable_budget":
        basic_reliable_budget,

    "strict_reliable_budget":
        strict_reliable_budget,

    "dose_response_pass":
        DOSE_RESPONSE_PASS,

    "exact_RETFound_threshold_replication":
        EXACT_RETF_THRESHOLD_REPLICATION,
}

report = f"""# PHASE 2A4 — VISIONFM × DEEPDRID RECALIBRATION DOSE

## Purpose

To test whether the local-label dose-response observed with RETFound
replicates using a second ophthalmic foundation-model backbone.

No VisionFM encoder or decoder weights were updated.

## Design

Calibration pool:
100 official DeepDRiD validation patients.

Frozen confirmatory test:
100 DeepDRiD evaluation patients.

Budgets:
10, 25, 50, and 100 local patients.

For 10, 25, and 50 patients:
50 nested stratified trajectories were used for each of three
independent source-decoder seeds.

For 100 patients:
the complete calibration cohort was evaluated exactly once per
source seed.

Therefore the inferential run counts are expected to be:

10: 150
25: 150
50: 150
100: 3

This explicitly prevents pseudoreplication of the full 100-patient
calibration cohort.

## Budget summary

{summary.to_string(index=False)}

## Derived thresholds

{json.dumps(thresholds, indent=2)}

## Interpretation boundary

Benefit efficiency is defined as the first budget recovering at least
80% of the full-100 median Brier and NLL benefit.

Near-complete recovery is defined as the first budget recovering at
least 95% of both median Brier and NLL benefit.

Reliability is reported separately under a basic multicomponent rule
and a stricter rule additionally requiring a positive lower bound for
the paired bootstrap Brier-gain confidence interval.

Failure to reproduce the exact RETFound 25/50/100 thresholds should
not be treated as study failure. Such a result would indicate that
annotation requirements are backbone-dependent.

The primary cross-backbone dose-response criterion is recovery of at
least 80% of full-cohort Brier and NLL benefit by 50 local patients,
with greater median recovery at 50 than at 10 patients.
"""

REPORT = (
    OUT
    / "PHASE_2A4_VISIONFM_DEEPDRID_DOSE_REPORT.md"
)

REPORT.write_text(
    report,
    encoding="utf-8",
)


# ============================================================
# 15. FINAL QC
# ============================================================

expected_runs = {
    10: 150,
    25: 150,
    50: 150,
    100: 3,
}

for budget,n in expected_runs.items():

    observed = int(
        summary[
            summary[
                "budget"
            ]
            == budget
        ][
            "n_unique_runs"
        ]
        .iloc[0]
    )

    assert observed == n, (
        f"Budget {budget}: "
        f"expected {n}, observed {observed}"
    )


print(
    "\n" + "="*80
)

print(
    "PHASE 2A4 FINAL"
)

print(
    "="*80
)

print(
    "\nCORRECTED BUDGET SUMMARY:"
)

print(
    summary.to_string(
        index=False
    )
)

print(
    "\nDERIVED THRESHOLDS:"
)

for k,v in thresholds.items():

    print(
        f"{k}: {v}"
    )

print(
    "\nRUN COUNTS:"
)

for budget,n in expected_runs.items():

    print(
        budget,
        "patients:",
        n,
        "unique inferential runs"
    )

print(
    "\nVERDICT:",
    (
        "VISIONFM DOSE-RESPONSE REPLICATION — PASS"
        if DOSE_RESPONSE_PASS
        else
        "VISIONFM DOSE-RESPONSE REPLICATION — NOT CONFIRMED"
    )
)

print(
    "\nREPORT:",
    REPORT
)

print(
    "\n✅ PHASE 2A4 COMPLETE"
)