"""New RETFound last-block ablation on locked patients.

Target-test data are never opened here. Both modes start from the same byte-
verified BRSET source checkpoint and use identical image transforms and roles.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from gpu_single_batch_feasibility_20261109 import BASE, build_model, configure, torch
from train_visionfm_unified_20261109 import PROJECT, ZIP_PATH, load_roles
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(BASE / "reproducibility" / "vendor" / "RETFound"))
from util.datasets import build_transform

SOURCE_FILE = BASE / "private_checkpoints" / "RETFound_BRSET_seed20260906_best.pth"
EXPECTED_SHA = "b10e6b0155bcc4ffb3ecfe8b3bb4f43363545d5dd17a86deded7ac6c396740f9"
TF_ARGS = SimpleNamespace(input_size=224, color_jitter=None, aa="rand-m9-mstd0.5-inc1",
                          reprob=0.25, remode="pixel", recount=1)
TRAIN_TF = build_transform("train", TF_ARGS)
EVAL_TF = build_transform("val", TF_ARGS)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class LegacyPosixPath(PurePosixPath):
    """Inert stand-in for Python 3.13 pathlib._local.PosixPath in old checkpoint."""


LegacyPosixPath.__module__ = "pathlib._local"
LegacyPosixPath.__name__ = "PosixPath"


def source_model():
    assert SOURCE_FILE.stat().st_size == 1213311981
    assert sha(SOURCE_FILE) == EXPECTED_SHA
    torch.serialization.add_safe_globals([LegacyPosixPath])
    checkpoint = torch.load(SOURCE_FILE, map_location="cpu", weights_only=True, mmap=True)
    assert checkpoint["seed"] == 20260906
    model = build_model("RETFound")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


class ZipImages(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.archive = zipfile.ZipFile(ZIP_PATH)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        member = "mbrset_1.0.0/" + row.image_path.replace("\\", "/")
        with Image.open(io.BytesIO(self.archive.read(member))) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(row.referable_dr_image), str(row.patient_id), int(row.referable_dr_patient), str(row.image_id)


def predict(model, dataloader):
    model.eval()
    image_rows = []
    patients = {}
    with torch.inference_mode():
        for x, _yi, pid, yp, image_id in dataloader:
            logits = model(x.cuda(non_blocking=True)).float()
            if logits.ndim == 3 and logits.shape[1] == 1:
                logits = logits[:, 0]
            assert logits.shape[1] == 2
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_half = torch.softmax(logits.half(), dim=1)[:, 1].float()
            for i, key in enumerate(pid):
                row = {"patient_id": key, "image_id": image_id[i], "y": int(yp[i]),
                       "logit0_float32": float(logits[i, 0].cpu()),
                       "logit1_float32": float(logits[i, 1].cpu()),
                       "logit_margin_float32": float((logits[i, 1] - logits[i, 0]).cpu()),
                       "p_float32": float(probs[i].cpu()),
                       "p_float16_diagnostic": float(probs_half[i].cpu())}
                image_rows.append(row)
                if key not in patients or row["p_float32"] > patients[key]["p_float32"]:
                    patients[key] = row
                else:
                    assert patients[key]["y"] == row["y"]
    rows = list(patients.values())
    y = np.asarray([r["y"] for r in rows], dtype=np.int64)
    p = np.asarray([r["p_float32"] for r in rows], dtype=np.float64)
    return image_rows, {"patients": len(rows), "positive": int(y.sum()),
                        "auroc": float(roc_auc_score(y, p)),
                        "nll": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
                        "p_zero": int(np.sum(p == 0)), "p_one": int(np.sum(p == 1))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["last1", "last3"], required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    assert args.seed == 20260906
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, valid, calibr = load_roles()
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(ZipImages(train, TRAIN_TF), batch_size=2, shuffle=True,
                              generator=g, num_workers=0, pin_memory=True)
    val_loader = DataLoader(ZipImages(valid, EVAL_TF), batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=True)
    cal_loader = DataLoader(ZipImages(calibr, EVAL_TF), batch_size=8, shuffle=False,
                            num_workers=0, pin_memory=True)
    model = source_model()
    total, trainable = configure(model, "RETFound", args.mode)
    model.cuda()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                   lr=1e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    loss_fn = torch.nn.CrossEntropyLoss()
    run_id = f"UNIFIED_R20261110_RTF_ABL_{args.mode}_seed{args.seed}"
    out = BASE / "unified_experiment"
    out.mkdir(exist_ok=True)
    config = {"run_id": run_id, "source_model_sha256": EXPECTED_SHA,
              "source_seed": 20260906, "adaptation_seed": args.seed,
              "mode": args.mode, "epochs": 10, "batch_size": 2,
              "lr": 1e-5, "weight_decay": 0.01, "optimizer": "AdamW",
              "schedule": "cosine_10_epochs", "amp": "bfloat16",
              "gradient_clipping_norm": 1.0,
              "frozen_encoder_blocks_eval": True,
              "protocol_timing": "revision-period exploratory; specified after RETFound one-seed head/LoRA test, before either ablation test",
              "train_transform": "RETFound util.datasets.build_transform train",
              "eval_transform": "RETFound util.datasets.build_transform val",
              "model_selection": "highest patient AUROC on separate 200-patient validation; tie lowest NLL then earliest",
              "train_patients": 100, "train_images": len(train),
              "validation_patients": 200, "validation_images": len(valid),
              "calibration_patients": 200, "calibration_images": len(calibr),
              "test_data_used": False, "target_test_opened": False,
              "roles_sha256": sha(BASE / "private_manifests" / "unified_experiment_patient_roles_PRIVATE.csv"),
              "total_parameters": total, "trainable_parameters": trainable,
              "torch_version": str(torch.__version__)}
    config_path = out / f"{run_id}_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    best = None
    ckpt_path = BASE / "private_checkpoints" / f"{run_id}_best.pth"
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(1, 11):
        model.train()
        unfrozen_n = 1 if args.mode == "last1" else 3
        for block in model.blocks[:-unfrozen_n]:
            block.eval()
        losses = []
        for x, yi, _pid, _yp, _img in train_loader:
            x = x.cuda(non_blocking=True)
            yi = yi.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                if logits.ndim == 3 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                loss = loss_fn(logits, yi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        _, m = predict(model, val_loader)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "validation_patients": m["patients"], "validation_positive": m["positive"],
               "validation_auroc": m["auroc"], "validation_nll": m["nll"],
               "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
               "elapsed_seconds": round(time.perf_counter() - started, 1)}
        history.append(row)
        score = (m["auroc"], -m["nll"], -epoch)
        if best is None or score > best["score"]:
            best = {"score": score, "epoch": epoch}
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config, "best_epoch": epoch}, ckpt_path)
        with (out / f"{run_id}_history.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps({"run_id": run_id, **row}), flush=True)
    torch.serialization.add_safe_globals([LegacyPosixPath])
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.cuda().eval()
    val_rows, val_m = predict(model, val_loader)
    cal_rows, cal_m = predict(model, cal_loader)
    for role, rows in (("MODEL_VALIDATION", val_rows), ("CALIBRATION_THRESHOLD", cal_rows)):
        with (BASE / "private_predictions" / f"{run_id}_{role}_image_scores_PRIVATE.csv").open(
                "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"run_id": run_id,
               "status": "TRAINING_AND_INDEPENDENT_DEV_INFERENCE_COMPLETE_TEST_NOT_OPENED",
               "best_epoch": best["epoch"], "best_validation_metrics": val_m,
               "calibration_role_counts_only_no_fit": {k: cal_m[k] for k in ("patients", "positive")},
               "peak_gpu_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
               "elapsed_seconds": round(time.perf_counter() - started, 1),
               "checkpoint_path": str(ckpt_path.relative_to(BASE)),
               "checkpoint_sha256": sha(ckpt_path), "test_data_used": False}
    (out / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
