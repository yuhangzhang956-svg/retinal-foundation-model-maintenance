
import os
import sys
import shutil
import zipfile
import warnings
from pathlib import Path
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# LOCKS
# ============================================================

ROOT = Path(
    "/content/drive_c2/MyDrive/"
    "Retinal_Retraining_Phase1A1"
)

assert ROOT.exists(), ROOT

DROOT = ROOT / "data/deepdrid"

ZIP = DROOT / "DeepDRiD-v1.1.zip"

EVAL_MANIFEST = (
    DROOT /
    "deepdrid_frozen_confirmatory_manifest.csv"
)

P1_METRICS = (
    ROOT /
    "results/03_seed_level_metrics.csv"
)

OUT = (
    ROOT /
    "results/phase1c2_deepdrid_recalibration"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

SEEDS = [
    20260906,
    20260907,
    20260908,
]

N_BOOT = 10000

REPO = "/content/RETFound"
sys.path.insert(0, REPO)

import timm
assert timm.__version__ == "0.9.2"

import models_vit
from util.datasets import build_transform

# ============================================================
# HELPERS
# ============================================================

def find_member(names, suffix):
    hits = [
        x for x in names
        if x.endswith(suffix)
    ]

    if len(hits) != 1:
        raise RuntimeError(
            f"Expected one {suffix}; "
            f"found {len(hits)}"
        )

    return hits[0]


def logit_np(p):

    p = np.clip(
        np.asarray(p),
        1e-7,
        1 - 1e-7
    )

    return np.log(
        p / (1-p)
    )


def sigmoid(x):
    return 1 / (
        1 + np.exp(-x)
    )


def calibration_intercept_slope(
    y,
    p
):

    X = (
        logit_np(p)
        .reshape(-1,1)
    )

    y = np.asarray(y).astype(int)

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000
    )

    model.fit(
        X,
        y
    )

    return (
        float(model.intercept_[0]),
        float(model.coef_[0][0])
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


def fit_platt(
    y,
    raw_p
):

    X = (
        logit_np(raw_p)
        .reshape(-1,1)
    )

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000
    )

    model.fit(
        X,
        np.asarray(y).astype(int)
    )

    return model


def apply_platt(
    model,
    raw_p
):

    X = (
        logit_np(raw_p)
        .reshape(-1,1)
    )

    return (
        model.predict_proba(X)[:,1]
    )


def threshold_for_sensitivity(
    y,
    p,
    target=0.90
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
            if tp+fn else np.nan
        )

        spec = (
            tn/(tn+fp)
            if tn+fp else np.nan
        )

        if sens >= target:

            candidate = (
                spec,
                th
            )

            if (
                best is None
                or candidate[0] > best[0]
            ):
                best = candidate

    return (
        float(best[1])
        if best is not None
        else 0.0
    )


def operating_metrics(
    y,
    p,
    th
):

    y = np.asarray(y).astype(int)

    pred = (
        np.asarray(p) >= th
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

    return {
        "sensitivity":
            sens,

        "specificity":
            spec,

        "balanced_accuracy":
            (sens+spec)/2,

        "ppv":
            tp/(tp+fp)
            if tp+fp else np.nan,

        "npv":
            tn/(tn+fn)
            if tn+fn else np.nan,
    }


def calibration_metrics(
    y,
    p
):

    y = np.asarray(y).astype(int)

    p = np.clip(
        np.asarray(p),
        1e-7,
        1-1e-7
    )

    ci, cs = (
        calibration_intercept_slope(
            y,
            p
        )
    )

    return {
        "auroc":
            roc_auc_score(y,p),

        "auprc":
            average_precision_score(y,p),

        "brier":
            brier_score_loss(y,p),

        "nll":
            log_loss(
                y,p,
                labels=[0,1]
            ),

        "ece":
            ece_score(y,p),

        "calibration_intercept":
            ci,

        "calibration_slope":
            cs,
    }

# ============================================================
# VALIDATION COHORT
# ============================================================

with zipfile.ZipFile(
    ZIP,
    "r"
) as z:

    names = z.namelist()

    val_csv_name = find_member(
        names,
        "regular_fundus_images/"
        "regular-fundus-validation/"
        "regular-fundus-validation.csv"
    )

    val = pd.read_csv(
        BytesIO(
            z.read(val_csv_name)
        )
    )

    val_members = [
        x for x in names
        if (
            "/regular-fundus-validation/"
            in x
            and
            Path(x).suffix.lower()
            == ".jpg"
        )
    ]

assert len(val) == 400
assert val["patient_id"].nunique() == 100

val["referable_dr_patient"] = (
    val["patient_DR_Level"]
    .astype(int)
    .ge(2)
    .astype(int)
)

val_map = {
    Path(x).name: x
    for x in val_members
}

val["filename"] = (
    val["image_id"].astype(str)
    + ".jpg"
)

val["zip_member"] = (
    val["filename"].map(val_map)
)

assert (
    val["zip_member"]
    .notna()
    .all()
)

# ============================================================
# EVALUATION COHORT
# ============================================================

ev = pd.read_csv(
    EVAL_MANIFEST
)

assert len(ev) == 400
assert ev["patient_id"].nunique() == 100

# ============================================================
# EXTRACT BOTH COHORTS
# ============================================================

DATA = Path(
    "/content/deepdrid_c2"
)

if DATA.exists():
    raise FileExistsError("Candidate safety guard: inspect or relocate existing extraction directory before rerun")

VAL_DIR = DATA / "validation"
EVAL_DIR = DATA / "evaluation"

VAL_DIR.mkdir(
    parents=True
)

EVAL_DIR.mkdir(
    parents=True
)

with zipfile.ZipFile(
    ZIP,
    "r"
) as z:

    for member in val[
        "zip_member"
    ]:

        dst = (
            VAL_DIR /
            Path(member).name
        )

        with z.open(member) as src, \
             open(dst,"wb") as out:

            shutil.copyfileobj(
                src,out
            )

    for member in ev[
        "zip_member"
    ]:

        dst = (
            EVAL_DIR /
            Path(member).name
        )

        with z.open(member) as src, \
             open(dst,"wb") as out:

            shutil.copyfileobj(
                src,out
            )

val["_runtime_path"] = (
    val["filename"].map(
        lambda x:
        str(
            VAL_DIR /
            Path(str(x)).name
        )
    )
)

ev["_runtime_path"] = (
    ev["filename"].map(
        lambda x:
        str(
            EVAL_DIR /
            Path(str(x)).name
        )
    )
)

# ============================================================
# DATASET
# ============================================================

args = SimpleNamespace(
    input_size=224,
    color_jitter=None,
    aa="rand-m9-mstd0.5-inc1",
    reprob=0.25,
    remode="pixel",
    recount=1,
)

EVAL_TF = build_transform(
    "val",
    args
)


class DSDataset(Dataset):

    def __init__(
        self,
        df,
        ycol
    ):

        self.df = (
            df.reset_index(
                drop=True
            ).copy()
        )

        self.ycol = ycol


    def __len__(self):
        return len(self.df)


    def __getitem__(self,i):

        r = self.df.iloc[i]

        image = Image.open(
            r["_runtime_path"]
        ).convert("RGB")

        return (
            EVAL_TF(image),
            i
        )


def loader_for(
    df,
    ycol
):

    ds = DSDataset(
        df,
        ycol
    )

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return ds,loader

# ============================================================
# RETFOUND
# ============================================================

def extract_state(ckpt):

    if isinstance(ckpt,dict):

        for key in [
            "model",
            "state_dict",
            "model_state_dict"
        ]:

            if (
                key in ckpt
                and
                isinstance(
                    ckpt[key],
                    dict
                )
            ):

                state = ckpt[key]
                break

        else:
            state = ckpt

    else:
        state = ckpt

    return {
        k[7:]
        if k.startswith("module.")
        else k:v

        for k,v in state.items()
        if torch.is_tensor(v)
    }


def load_model(seed):

    path = (
        ROOT /
        "results/checkpoints" /
        f"RETFound_BRSET_seed{seed}_best.pth"
    )

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    state = extract_state(
        ckpt
    )

    model = models_vit.RETFound_mae(
        img_size=224,
        num_classes=int(
            state[
                "head.weight"
            ].shape[0]
        ),
        drop_path_rate=0.2,
        global_pool=True
    )

    missing,unexpected = (
        model.load_state_dict(
            state,
            strict=False
        )
    )

    assert not missing
    assert not unexpected

    return model.cuda().eval()


def prob(logits):

    if isinstance(
        logits,
        (list,tuple)
    ):
        logits=logits[0]

    return torch.softmax(
        logits,
        dim=1
    )[:,1]


@torch.inference_mode()
def predict(
    model,
    df,
    patient_col,
    y_col
):

    ds,loader = loader_for(
        df,
        y_col
    )

    rows=[]

    for xb,ii in loader:

        xb=xb.cuda(
            non_blocking=True
        )

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16
        ):

            logits=model(xb)

        p=prob(
            logits
        ).float().cpu().numpy()

        for idx,pred in zip(
            ii.numpy(),
            p
        ):

            r=ds.df.iloc[
                int(idx)
            ]

            rows.append({
                "patient_id":
                    str(
                        r[
                            patient_col
                        ]
                    ),

                "y":
                    int(
                        r[y_col]
                    ),

                "p_raw":
                    float(pred)
            })

    img=pd.DataFrame(
        rows
    )

    return (
        img.groupby(
            "patient_id",
            as_index=False
        )
        .agg(
            y=("y","first"),
            p_raw=("p_raw","max")
        )
    )

# ============================================================
# SOURCE CALIBRATION
# ============================================================

p1=pd.read_csv(
    P1_METRICS
)


def sourcecal(
    seed,p
):

    row=p1[
        p1["seed"].astype(int)
        ==seed
    ].iloc[0]

    return sigmoid(
        float(
            row["platt_intercept"]
        )
        +
        float(
            row["platt_slope"]
        )
        *
        logit_np(p)
    )

# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

def bootstrap_improvement(
    y,
    source_p,
    local_p,
    n=10000,
    seed=1
):

    y=np.asarray(y)
    source_p=np.asarray(source_p)
    local_p=np.asarray(local_p)

    rng=np.random.default_rng(
        seed
    )

    brier_gain=[]
    nll_gain=[]

    N=len(y)

    for _ in range(n):

        ii=rng.integers(
            0,N,N
        )

        yy=y[ii]
        ps=source_p[ii]
        pl=local_p[ii]

        brier_gain.append(
            brier_score_loss(
                yy,ps
            )
            -
            brier_score_loss(
                yy,pl
            )
        )

        nll_gain.append(
            log_loss(
                yy,ps,
                labels=[0,1]
            )
            -
            log_loss(
                yy,pl,
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
# THREE SEEDS
# ============================================================

results=[]
predictions=[]

for seed in SEEDS:

    print(
        "\n"+"="*70
    )
    print(
        "SEED",seed
    )
    print(
        "="*70
    )

    model=load_model(
        seed
    )

    val_pt=predict(
        model,
        val,
        "patient_id",
        "referable_dr_patient"
    )

    eval_pt=predict(
        model,
        ev,
        "patient_id",
        "referable_dr_patient"
    )

    # Source BRSET calibration
    eval_pt[
        "p_sourcecal"
    ]=sourcecal(
        seed,
        eval_pt["p_raw"]
    )

    # Local 100-patient calibration
    local_platt=fit_platt(
        val_pt["y"],
        val_pt["p_raw"]
    )

    val_pt[
        "p_localcal"
    ]=apply_platt(
        local_platt,
        val_pt["p_raw"]
    )

    eval_pt[
        "p_localcal"
    ]=apply_platt(
        local_platt,
        eval_pt["p_raw"]
    )

    source_metrics=calibration_metrics(
        eval_pt["y"],
        eval_pt["p_sourcecal"]
    )

    local_metrics=calibration_metrics(
        eval_pt["y"],
        eval_pt["p_localcal"]
    )

    local_threshold=(
        threshold_for_sensitivity(
            val_pt["y"],
            val_pt["p_localcal"],
            target=0.90
        )
    )

    source_threshold=float(
        p1[
            p1["seed"].astype(int)
            ==seed
        ].iloc[0][
            "threshold_source_validation_calibrated"
        ]
    )

    source_op=operating_metrics(
        eval_pt["y"],
        eval_pt["p_sourcecal"],
        source_threshold
    )

    local_op=operating_metrics(
        eval_pt["y"],
        eval_pt["p_localcal"],
        local_threshold
    )

    boot=bootstrap_improvement(
        eval_pt["y"],
        eval_pt["p_sourcecal"],
        eval_pt["p_localcal"],
        seed=seed
    )

    result={
        "seed":seed,

        "AUROC":
            local_metrics["auroc"],

        "sourcecal_Brier":
            source_metrics["brier"],

        "localcal_Brier":
            local_metrics["brier"],

        "Brier_gain":
            source_metrics["brier"]
            -
            local_metrics["brier"],

        "sourcecal_NLL":
            source_metrics["nll"],

        "localcal_NLL":
            local_metrics["nll"],

        "NLL_gain":
            source_metrics["nll"]
            -
            local_metrics["nll"],

        "sourcecal_ECE":
            source_metrics["ece"],

        "localcal_ECE":
            local_metrics["ece"],

        "sourcecal_slope":
            source_metrics[
                "calibration_slope"
            ],

        "localcal_slope":
            local_metrics[
                "calibration_slope"
            ],

        "sourcecal_intercept":
            source_metrics[
                "calibration_intercept"
            ],

        "localcal_intercept":
            local_metrics[
                "calibration_intercept"
            ],

        "source_threshold_sensitivity":
            source_op[
                "sensitivity"
            ],

        "source_threshold_specificity":
            source_op[
                "specificity"
            ],

        "source_threshold_balanced_accuracy":
            source_op[
                "balanced_accuracy"
            ],

        "local_threshold_sensitivity":
            local_op[
                "sensitivity"
            ],

        "local_threshold_specificity":
            local_op[
                "specificity"
            ],

        "local_threshold_balanced_accuracy":
            local_op[
                "balanced_accuracy"
            ],

        **boot
    }

    results.append(
        result
    )

    eval_pt["seed"]=seed

    predictions.append(
        eval_pt
    )

    print(
        "Brier:",
        round(
            source_metrics[
                "brier"
            ],
            4
        ),
        "→",
        round(
            local_metrics[
                "brier"
            ],
            4
        )
    )

    print(
        "NLL:",
        round(
            source_metrics[
                "nll"
            ],
            4
        ),
        "→",
        round(
            local_metrics[
                "nll"
            ],
            4
        )
    )

    print(
        "Calibration slope:",
        round(
            source_metrics[
                "calibration_slope"
            ],
            3
        ),
        "→",
        round(
            local_metrics[
                "calibration_slope"
            ],
            3
        )
    )

    print(
        "Local threshold sens/spec:",
        round(
            local_op[
                "sensitivity"
            ],
            3
        ),
        "/",
        round(
            local_op[
                "specificity"
            ],
            3
        )
    )

    del model
    torch.cuda.empty_cache()

# ============================================================
# GATE
# ============================================================

res=pd.DataFrame(
    results
)

pred=pd.concat(
    predictions,
    ignore_index=True
)

res.to_csv(
    OUT /
    "01_seed_level_recalibration_metrics.csv",
    index=False
)

pred.to_csv(
    OUT /
    "02_eval_predictions.csv",
    index=False
)

res[
    "brier_success"
]=(
    (res["Brier_gain"]>=0.01)
    &
    (res["brier_gain_CI_low"]>0)
)

res[
    "nll_success"
]=(
    res["NLL_gain"]>0
)

res[
    "slope_success"
]=(
    abs(
        res["localcal_slope"]-1
    )
    <
    abs(
        res["sourcecal_slope"]-1
    )
)

res[
    "threshold_safe"
]=(
    res[
        "local_threshold_balanced_accuracy"
    ]
    >=
    res[
        "source_threshold_balanced_accuracy"
    ]
    -0.05
)

res[
    "recalibration_success"
]=(
    res["brier_success"]
    &
    res["nll_success"]
    &
    res["slope_success"]
    &
    res["threshold_safe"]
)

n_success=int(
    res[
        "recalibration_success"
    ].sum()
)

if n_success>=2:

    VERDICT=(
        "STRONG RECALIBRATION "
        "REPLICATION — PASS"
    )

    EYE_FLOOR=(
        "PASS TO COMPLETE "
        "FINAL MULTI-DOMAIN STUDY"
    )

elif (
    res["Brier_gain"].gt(0)
    .sum()
    >=2
):

    VERDICT=(
        "PARTIAL RECALIBRATION "
        "REPLICATION — BORDERLINE"
    )

    EYE_FLOOR="CONDITIONAL"

else:

    VERDICT=(
        "RECALIBRATION "
        "REPLICATION FAILED"
    )

    EYE_FLOOR=(
        "FAIL — REASSESS "
        "PROJECT"
    )

# ============================================================
# REPORT
# ============================================================

report=f"""# PHASE 1C2 — DEEPDRID 100-LABEL RECALIBRATION REPLICATION

## Design

Target calibration cohort:

100 official DeepDRiD validation patients.

Independent evaluation cohort:

100 official Online-Challenge evaluation patients.

No model weight was updated.

The recalibration-only maintenance strategy and the
100-local-patient budget were selected in the prior mBRSET
method-development phase.

## Seed-level results

{res.to_string(index=False)}

## Successful recalibration seeds

{n_success}/3

## Phase 1C2 verdict

**{VERDICT}**

## Eye and Vision floor

**{EYE_FLOOR}**

A PASS means that the low-cost recalibration maintenance
strategy selected in mBRSET reproduced in an independent,
operationally label-harmonized target domain without
retraining the foundation model.

It does not guarantee journal acceptance.

## Important interpretation

AUROC is expected to remain unchanged under monotonic
recalibration.

The primary evidence is improvement in probability
calibration and deployment operating characteristics, not
a change in discrimination.
"""

report_path=(
    OUT /
    "PHASE_1C2_RECALIBRATION_REPORT.md"
)

report_path.write_text(
    report,
    encoding="utf-8"
)

print(
    "\n"+"="*80
)

print(
    "PHASE 1C2 FINAL"
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
    "\nSUCCESSFUL SEEDS:",
    n_success,
    "/3"
)

print(
    "\nVERDICT:",
    VERDICT
)

print(
    "\nEYE AND VISION FLOOR:",
    EYE_FLOOR
)

print(
    "\nREPORT:",
    report_path
)

print(
    "\nSTOP — DO NOT START LoRA YET."
)
