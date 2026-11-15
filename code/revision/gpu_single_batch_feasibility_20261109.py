"""One-batch GPU architecture feasibility; synthetic inputs, no patient data.

Every invocation is an independent process. No scientific performance metric is
computed. Random initial weights are used only for GPU memory/runtime testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

BASE = Path(__file__).resolve().parent
CUDA_VENDOR = BASE / "analyses" / "torch_cuda_vendor"
DEPS_VENDOR = BASE / "analyses" / "gpu_deps_vendor"
assert CUDA_VENDOR.exists(), f"CUDA PyTorch not installed at {CUDA_VENDOR}"
sys.path.insert(0, str(CUDA_VENDOR))
sys.path.insert(0, str(DEPS_VENDOR))
# Reuse pure-Python dependencies installed for the separate CPU diagnostic;
# the CUDA torch directory above it remains the first torch package found.
sys.path.insert(2, str(BASE / "analyses" / "torch_vendor"))
if os.name == "nt":
    os.add_dll_directory(str(CUDA_VENDOR / "torch" / "lib"))

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Executed VisionFM Phase2A2 qkv LoRA form, reused for resource testing."""
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        z = nn.functional.linear(self.dropout(x), self.lora_A)
        z = nn.functional.linear(z, self.lora_B)
        return self.base(x) + self.scaling * z


class VisionFMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_bn = nn.BatchNorm1d(3072, eps=1e-6, momentum=0.99)
        self.classifier = nn.Sequential(nn.Linear(3072, 1536), nn.GELU(), nn.Dropout(0.1),
                                        nn.Linear(1536, 768), nn.GELU(), nn.Dropout(0.1),
                                        nn.Linear(768, 1))

    def forward(self, x):
        return self.classifier(self.channel_bn(x)).squeeze(-1)


class VisionFMClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = VisionFMHead()

    def forward(self, x):
        blocks = self.encoder.get_intermediate_layers(x, 4)
        feat = torch.cat([b[:, 0] for b in blocks], dim=-1)
        return self.head(feat)


def build_model(backbone):
    if backbone == "VisionFM":
        # Model import needs only this helper. Checkpoint loading is not part of
        # the memory test, so random initialization is explicitly recorded.
        stub = types.ModuleType("utils")
        stub.trunc_normal_ = torch.nn.init.trunc_normal_
        sys.modules["utils"] = stub
        sys.path.insert(0, str(BASE / "reproducibility" / "vendor" / "VisionFM"))
        import models
        encoder = models.__dict__["vit_base"](img_size=[224], patch_size=16,
                                                 num_classes=0, use_mean_pooling=False)
        return VisionFMClassifier(encoder)
    assert backbone == "RETFound"
    sys.path.insert(0, str(BASE / "reproducibility" / "vendor" / "RETFound"))
    import models_vit
    return models_vit.RETFound_mae(img_size=224, num_classes=2,
                                   drop_path_rate=0.2, global_pool=True)


def configure(model, backbone, mode):
    for p in model.parameters():
        p.requires_grad = False
    encoder = model.encoder if backbone == "VisionFM" else model
    head = model.head
    if mode == "head_only":
        pass
    elif mode == "lora":
        for block in encoder.blocks:
            block.attn.qkv = LoRALinear(block.attn.qkv)
    elif mode in ("last1", "last3"):
        n = 1 if mode == "last1" else 3
        for block in encoder.blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        # VisionFM applies this norm to the last four blocks' CLS tokens;
        # RETFound applies fc_norm to globally pooled tokens.
        for p in (encoder.norm if backbone == "VisionFM" else encoder.fc_norm).parameters():
            p.requires_grad = True
    elif mode == "full_ft":
        for p in encoder.parameters():
            p.requires_grad = True
    else:
        raise ValueError(mode)
    for p in head.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)


def memory(stage):
    free, total = torch.cuda.mem_get_info()
    return {"stage": stage, "allocated_mib": round(torch.cuda.memory_allocated()/2**20, 2),
            "reserved_mib": round(torch.cuda.memory_reserved()/2**20, 2),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated()/2**20, 2),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved()/2**20, 2),
            "device_free_mib": round(free/2**20, 2), "device_total_mib": round(total/2**20, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["VisionFM", "RETFound"], required=True)
    parser.add_argument("--mode", choices=["head_only", "lora", "last1", "last3", "full_ft"], required=True)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    assert args.batch in (1, 2, 3, 4, 8)
    outdir = BASE / "gpu_feasibility"
    outdir.mkdir(exist_ok=True)
    result = {"backbone": args.backbone, "mode": args.mode, "batch": args.batch,
              "input": "synthetic_random_224x224_rgb", "weight_state": "RANDOM_INIT_ARCHITECTURE_ONLY",
              "training_state": "NOT_STARTED", "torch_version": torch.__version__,
              "cuda_runtime": torch.version.cuda, "peak_allocated_mib": None,
              "peak_reserved_mib": None, "error": None, "stages": []}
    started = time.perf_counter()
    try:
        assert torch.cuda.is_available(), "CUDA_NOT_AVAILABLE"
        result["gpu_name"] = torch.cuda.get_device_name(0)
        torch.manual_seed(20261109)
        torch.cuda.manual_seed_all(20261109)
        torch.set_num_threads(4)
        torch.cuda.reset_peak_memory_stats()
        result["stages"].append(memory("before_model"))
        model = build_model(args.backbone)
        total, trainable = configure(model, args.backbone, args.mode)
        result["total_parameters"] = total
        result["trainable_parameters"] = trainable
        model = model.cuda()
        model.train()
        if args.batch == 1 and args.backbone == "VisionFM":
            model.head.channel_bn.eval()
            result["bn_note"] = "BN_RUNNING_STATS_FROZEN_FOR_BATCH1_SMOKE_ONLY"
        result["training_state"] = "MODEL_ON_GPU"
        result["stages"].append(memory("model_on_gpu"))
        x = torch.randn(args.batch, 3, 224, 224, device="cuda")
        if args.backbone == "VisionFM":
            y = torch.tensor([i % 2 for i in range(args.batch)], device="cuda", dtype=torch.float32)
            loss_fn = nn.BCEWithLogitsLoss()
            amp_dtype = torch.float16
            lr = 1e-5 if args.mode == "full_ft" else 3e-4
            scaler = torch.amp.GradScaler("cuda", init_scale=128.0)
        else:
            y = torch.tensor([i % 2 for i in range(args.batch)], device="cuda", dtype=torch.long)
            loss_fn = nn.CrossEntropyLoss()
            amp_dtype = torch.bfloat16
            lr = 1e-5 if args.mode == "full_ft" else 3e-4
            scaler = None
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=lr, weight_decay=0.01)
        probe = next(p for p in model.parameters() if p.requires_grad)
        probe_before = probe.detach().flatten()[:32].clone()
        optimizer.zero_grad(set_to_none=True)
        result["stages"].append(memory("input_optimizer_ready"))
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(x)
            if args.backbone == "RETFound" and logits.ndim == 3 and logits.shape[1] == 1:
                logits = logits[:, 0]
            loss = loss_fn(logits, y)
        result["training_state"] = "FORWARD_DONE"
        result["loss"] = float(loss.detach())
        result["stages"].append(memory("forward_done"))
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            result["scaler_scale_before_step"] = float(scaler.get_scale())
        result["training_state"] = "BACKWARD_DONE"
        result["stages"].append(memory("backward_done"))
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        result["optimizer_state_parameter_count"] = len(optimizer.state)
        result["probe_parameter_changed"] = bool(torch.any(probe.detach().flatten()[:32] != probe_before))
        if scaler is not None:
            result["scaler_scale_after_step"] = float(scaler.get_scale())
        result["training_state"] = ("OPTIMIZER_STEP_DONE" if optimizer.state and result["probe_parameter_changed"]
                                    else "OPTIMIZER_STEP_SKIPPED_OR_NO_PARAMETER_UPDATE")
        result["stages"].append(memory("optimizer_step_done"))
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)[-3000:]
        result["traceback_tail"] = traceback.format_exc()[-3000:]
    finally:
        if torch.cuda.is_available():
            result["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated()/2**20, 2)
            result["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved()/2**20, 2)
        result["elapsed_seconds"] = round(time.perf_counter()-started, 2)
        out = outdir / f"{args.backbone}_{args.mode}_batch{args.batch}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("backbone", "mode", "batch", "training_state", "peak_allocated_mib", "peak_reserved_mib", "error", "elapsed_seconds")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
