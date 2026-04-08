"""
EvalGuard — Experiment Script (Section VI) [v5]

[v5] Logit-Space Confidence Shift Watermarking:
  - Embedding: target-class logit boost BEFORE softmax (T-invariant)
  - Verification: Mann-Whitney U test with configurable verify_temperature
  - Parameters: delta_logit (logit shift), beta (safety factor)
"""

import sys, os, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import copy
import time
import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from evalguard import (
    obfuscate_model_vectorized, recover_weights,
    WatermarkModule, LatentExtractor, verify_ownership,
)
from evalguard.watermark import (
    verify_ownership_own_data, reconstruct_triggers_from_own_data,
)
from evalguard.crypto import keygen, kdf
from evalguard.configs import (
    CONFIGS, cifar10_data, cifar100_data,
    create_student,
)
from evalguard.attacks import (
    collect_soft_labels, soft_label_distillation,
    collect_hard_labels, hard_label_extraction,
    fine_tune_surrogate,
)

import numpy as np

RESULTS_DIR = Path("results")
CKPT_DIR = Path("checkpoints")


# ============================================================
# Dataset detection
# ============================================================

MODEL_REGISTRY = {
    "cifar10_resnet20":  ("cifar10_resnet20",  "cifar10"),
    "cifar10_vgg11":     ("cifar10_vgg11",     "cifar10"),
    "cifar100_resnet20": ("cifar100_resnet20", "cifar100"),
    "cifar100_vgg11":    ("cifar100_vgg11",    "cifar100"),
    "cifar100_resnet56": ("cifar100_resnet56", "cifar100"),
    "cifar100_wrn2810":  ("cifar100_wrn2810",  "cifar100"),
    "resnet50":          ("imagenet_resnet50", "imagenet"),
}

DATA_FN = {
    "cifar10":  cifar10_data,
    "cifar100": cifar100_data,
}


def resolve_model(name, dataset_override=None):
    if name not in MODEL_REGISTRY:
        raise ValueError("Unknown model: {}. Available: {}".format(
            name, list(MODEL_REGISTRY.keys())))
    config_key, ds = MODEL_REGISTRY[name]
    if dataset_override:
        ds = dataset_override
    short = name.replace("cifar10_", "").replace("cifar100_", "").replace("imagenet_", "")
    return config_key, ds, short


# ============================================================
# Naming helpers
# ============================================================

def _delta_str(delta):
    log2 = round(math.log2(delta))
    return "2e{}".format(log2)

def _cl_str(class_level):
    return "__CL" if class_level else ""

def stem_fidelity(teacher, eps, delta):
    return "T_{}__eps{}__delta{}".format(teacher, eps, _delta_str(delta))

def stem_finetune(teacher, eps, delta, ft_epochs):
    return "T_{}__eps{}__delta{}__ftEp{}".format(teacher, eps, _delta_str(delta), ft_epochs)

def stem_distill(teacher, student, rw, nq, T, dist_epochs, dist_lr, delta_logit=2.0, vT=5.0, tag=""):
    base = "T_{}__S_{}__rw{}__nq{}__T{}__distEp{}__distLr{}__d{}__vT{}".format(
        teacher, student, rw, nq, T, dist_epochs, dist_lr, delta_logit, vT)
    if tag:
        base = "{}__tag_{}".format(base, tag)
    return base

def stem_surrogate_ft(teacher, student, rw, nq, T, dist_epochs, dist_lr, ft_epochs, ft_lr, delta_logit=2.0, vT=5.0, trigger_mode="rec_trigger", trigger_size=0, tag=""):
    base = "T_{}__S_{}__rw{}__nq{}__T{}__distEp{}__distLr{}__ftEp{}__ftLr{}__d{}__vT{}".format(
        teacher, student, rw, nq, T, dist_epochs, dist_lr, ft_epochs, ft_lr, delta_logit, vT)
    if trigger_mode == "own_trigger":
        base = "{}__trig_own{}".format(base, trigger_size if trigger_size > 0 else "all")
    else:
        base = "{}__trig_rec{}".format(base, trigger_size if trigger_size > 0 else "all")
    if tag:
        base = "{}__tag_{}".format(base, tag)
    return base

def stem_overhead(teacher, eps):
    return "T_{}__eps{}".format(teacher, eps)

def stem_ckpt(teacher, student, rw, nq, T, dist_epochs, dist_lr, delta_logit=2.0, tag=""):
    """Checkpoint stem: no verify_temperature (checkpoints are T-independent).

    Prefix 'v6_' marks the checkpoint format version that includes the
    persisted watermark key K_w (.key file). Caches without the key file
    will be ignored on load (auto-invalidating older v5 caches).
    """
    base = "v6__T_{}__S_{}__rw{}__nq{}__T{}__distEp{}__distLr{}__d{}".format(
        teacher, student, rw, nq, T, dist_epochs, dist_lr, delta_logit)
    if tag:
        base = "{}__tag_{}".format(base, tag)
    return base


# ============================================================
# Save / Load
# ============================================================

def save_result(data, dataset, subdir, filename, teacher_name=None, student_arch=None, label_mode=None):
    """
    Save result JSON with directory structure:
      results/<subdir>/<dataset>/<teacher>_<student>/<label_mode>/<filename>.json

    Falls back to old structure if teacher_name/student_arch/label_mode not provided.
    """
    if teacher_name and student_arch and label_mode:
        model_dir = "{}_{}".format(teacher_name, student_arch)
        path = RESULTS_DIR / subdir / dataset / model_dir / label_mode / "{}.json".format(filename)
    elif teacher_name and label_mode:
        path = RESULTS_DIR / subdir / dataset / teacher_name / label_mode / "{}.json".format(filename)
    else:
        path = RESULTS_DIR / subdir / dataset / "{}.json".format(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print("  -> Saved: {}".format(path))


def save_checkpoint(model, triggers, K_w, dataset, stem, meta=None, subdir="distill",
                    teacher_name=None, student_arch=None, label_mode=None):
    """
    Save checkpoint with directory structure:
      checkpoints/<subdir>/<dataset>/<teacher>_<student>/<label_mode>/<stem>.{pt,pkl,key,meta.json}

    K_w is persisted as a separate .key file (raw bytes). Without this file,
    own_trigger verification cannot reproduce the target-class mapping.

    Falls back to old structure if teacher_name/student_arch/label_mode not provided.
    """
    if teacher_name and student_arch and label_mode:
        model_dir = "{}_{}".format(teacher_name, student_arch)
        ckpt_dir = CKPT_DIR / subdir / dataset / model_dir / label_mode
    else:
        ckpt_dir = CKPT_DIR / subdir / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "{}.pt".format(stem))
    with open(ckpt_dir / "{}.pkl".format(stem), "wb") as f:
        pickle.dump(triggers, f)
    # Persist watermark key (raw bytes)
    with open(ckpt_dir / "{}.key".format(stem), "wb") as f:
        f.write(K_w)
    if meta:
        # Embed key hex in meta for human inspection (file is the source of truth)
        meta = dict(meta)
        meta["K_w_hex"] = K_w.hex()
        meta["checkpoint_version"] = "v6"
        with open(ckpt_dir / "{}.meta.json".format(stem), "w") as f:
            json.dump(meta, f, indent=2, default=str)
    print("  -> Cached: {}/{}.pt (+ .pkl, .key)".format(ckpt_dir, stem))


def load_checkpoint(dataset, stem, num_classes, student_arch, subdir="distill",
                    teacher_name=None, label_mode=None):
    """
    Load checkpoint. Returns (model, triggers, K_w) or (None, None, None) if
    any required artifact (.pt / .pkl / .key) is missing.

    The .key file is REQUIRED — caches written before v6 are silently
    invalidated to prevent verification with the wrong watermark key.
    """
    if teacher_name and student_arch and label_mode:
        model_dir = "{}_{}".format(teacher_name, student_arch)
        ckpt_dir = CKPT_DIR / subdir / dataset / model_dir / label_mode
    else:
        ckpt_dir = CKPT_DIR / subdir / dataset
    pt_path = ckpt_dir / "{}.pt".format(stem)
    pkl_path = ckpt_dir / "{}.pkl".format(stem)
    key_path = ckpt_dir / "{}.key".format(stem)
    if not (pt_path.exists() and pkl_path.exists() and key_path.exists()):
        return None, None, None
    model = create_student(num_classes=num_classes, arch=student_arch)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    with open(pkl_path, "rb") as f:
        # Custom unpickler to remap CUDA tensors to CPU
        import pickle as _pickle
        class CPUUnpickler(_pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
                return super().find_class(module, name)
        triggers = CPUUnpickler(f).load()
    with open(key_path, "rb") as f:
        K_w = f.read()
    # Ensure all trigger queries are on CPU
    for entry in triggers:
        if hasattr(entry, 'query') and isinstance(entry.query, torch.Tensor):
            entry.query = entry.query.cpu()
    print("  -> Loaded cache: {} (K_w={}...)".format(pt_path, K_w.hex()[:12]))
    return model, triggers, K_w


# ============================================================
# Helpers
# ============================================================

def get_model(config_key, pretrained_path=None):
    config = CONFIGS[config_key]
    if pretrained_path:
        model, ll = config["model_fn"](pretrained=False)
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    else:
        model, ll = config["model_fn"](pretrained=True)
    return model, ll, config


def train_model(model, loader, epochs=50, lr=0.1, device="cpu"):
    model.to(device).train()
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        sch.step()
        if (ep + 1) % 10 == 0:
            print("    Epoch {}/{}, Loss: {:.4f}".format(ep+1, epochs, loss_sum/len(loader)))
    return model


def evaluate_accuracy(model, loader, device="cpu"):
    model.to(device).eval()
    c, t = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += model(x).max(1)[1].eq(y).sum().item()
            t += y.size(0)
    return c / t


def model_info(model, name):
    n = sum(p.numel() for p in model.parameters())
    return {"name": name, "num_parameters": n, "num_parameters_human": "{:,}".format(n)}


# ============================================================
# Verification helpers (v5: logit-space confidence shift)
# ============================================================

def verify_and_format(triggers, model, num_classes, K_w, eta, device,
                      control_queries=None, control_top1=None,
                      verify_temperature=5.0):
    """
    Run confidence shift verification and format results.
    Uses verify_temperature to amplify sub-dominant probability differences.
    """
    if len(triggers) == 0:
        return {"confidence_shift": 0.0, "n_trigger": 0, "n_control": 0,
                "p_value": 1.0, "verified": False}

    vr = verify_ownership(
        triggers, model,
        control_queries=control_queries,
        control_top1_classes=control_top1,
        K_w=K_w,
        num_classes=num_classes,
        eta=eta,
        device=device,
        verify_temperature=verify_temperature,
    )
    return {
        "confidence_shift": vr["confidence_shift"],
        "mean_trigger_conf": vr["mean_trigger_conf"],
        "mean_control_conf": vr["mean_control_conf"],
        "n_trigger": vr["n_trigger"],
        "n_control": vr["n_control"],
        "test": vr.get("test", "wilcoxon_signed_rank"),
        "statistic": vr.get("statistic", 0.0),
        "p_value": vr["p_value"],
        "log10_p_value": round(math.log10(max(vr["p_value"], 1e-300)), 2),
        "verified": vr["verified"],
        "verify_temperature": verify_temperature,
        "trigger_source": "rec_trigger",
    }


def verify_own_data_and_format(
    owner_model, own_loader, suspect_model,
    K_w, r_w, num_classes, latent_extractor, layer_name,
    delta_logit, beta, eta, device, verify_temperature=5.0,
    max_triggers=0,
):
    """
    Own-data verification: reconstruct triggers from Owner's data,
    then verify against suspect model. Zero D_eval leakage.

    Args:
        max_triggers: collect exactly this many triggers then stop.
                      0 = collect all from dataloader.
    """
    vr = verify_ownership_own_data(
        owner_model, own_loader, suspect_model,
        K_w=K_w, r_w=r_w, num_classes=num_classes,
        latent_extractor=latent_extractor, layer_name=layer_name,
        delta_logit=delta_logit, beta=beta,
        eta=eta, device=device,
        verify_temperature=verify_temperature,
        max_triggers=max_triggers,
    )
    return {
        "confidence_shift": vr["confidence_shift"],
        "mean_trigger_conf": vr["mean_trigger_conf"],
        "mean_control_conf": vr["mean_control_conf"],
        "n_trigger": vr["n_trigger"],
        "n_control": vr["n_control"],
        "test": vr.get("test", "wilcoxon_signed_rank"),
        "statistic": vr.get("statistic", 0.0),
        "p_value": vr["p_value"],
        "log10_p_value": round(math.log10(max(vr["p_value"], 1e-300)), 2),
        "verified": vr["verified"],
        "verify_temperature": verify_temperature,
        "trigger_source": "own_trigger",
    }


def watermark_config_dict(rw, delta_logit, beta, num_classes, verify_temperature=5.0, delta_min=None):
    cfg = {
        "r_w": rw, "r_w_percent": "{}%".format(rw * 100),
        "delta_logit": delta_logit,
        "beta": beta,
        "num_classes": num_classes,
        "method": "logit_space_confidence_shift",
        "verification": "wilcoxon_signed_rank_paired",
        "verify_temperature": verify_temperature,
    }
    if delta_min is not None:
        cfg["delta_min"] = delta_min
    return cfg


# ============================================================
# Table IV: Fidelity
# ============================================================

def experiment_fidelity(model, testloader, device, epsilon, delta,
                        teacher_name, dataset, config):
    print("\n" + "=" * 60)
    print("Table IV: Obfuscation & Fidelity")
    print("  Dataset: {}, Teacher: {}, eps={}, delta={}".format(
        dataset, teacher_name, epsilon, _delta_str(delta)))
    print("=" * 60)

    acc_orig = evaluate_accuracy(model, testloader, device)
    ms = copy.deepcopy(model)
    ms, sec = obfuscate_model_vectorized(ms, epsilon=epsilon, delta=delta, model_id="evalguard")
    acc_obf = evaluate_accuracy(ms, testloader, device)
    recover_weights(ms, sec, vectorized=True)
    acc_rec = evaluate_accuracy(ms, testloader, device)

    print("  Orig={:.2f}%, Obf={:.2f}%, Rec={:.2f}%, Loss={:.4f}%".format(
        acc_orig*100, acc_obf*100, acc_rec*100, (acc_orig-acc_rec)*100))

    fname = stem_fidelity(teacher_name, epsilon, delta)
    save_result({
        "experiment": "fidelity", "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model": model_info(model, teacher_name),
        "parameters": {"epsilon": epsilon, "delta": str(delta), "delta_str": _delta_str(delta)},
        "results": {
            "acc_original": round(acc_orig, 6),
            "acc_obfuscated": round(acc_obf, 6),
            "acc_recovered": round(acc_rec, 6),
            "fidelity_loss": round(acc_orig - acc_rec, 6),
            "num_classes": config["num_classes"],
            "random_guess": config["random_guess"],
        },
    }, dataset, "fidelity", fname)


# ============================================================
# Table VI: Distillation + Watermark Verification
# ============================================================

def experiment_distillation(model, trainset, testloader, latent_layer, device,
                            teacher_name, student_arch, dataset, config,
                            temperatures, n_queries, rw,
                            eta, dist_epochs, dist_lr, dist_batch,
                            delta_logit=2.0, beta=0.4, delta_min=0.5,
                            verify_temperature=5.0,
                            label_mode="soft", tag=""):
    """
    Table VI: Distillation + Confidence Shift Verification.

    Args:
        label_mode: "soft" for KL-divergence soft-label distillation (E2s),
                    "hard" for cross-entropy hard-label extraction (E2h).
    """
    nc = config["num_classes"]

    print("\n" + "=" * 60)
    print("Table VI: Distillation + Confidence Shift Verification [{}]".format(label_mode.upper()))
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("  rw={}, nq={}, dist_epochs={}, dist_lr={}, delta_logit={}, beta={}".format(
        rw, n_queries, dist_epochs, dist_lr, delta_logit, beta))
    print("  Label mode: {}".format(label_mode))
    if label_mode == "soft":
        print("  Temperatures: {}".format(temperatures))
    print("=" * 60)

    teacher = copy.deepcopy(model)
    acc_t = evaluate_accuracy(teacher, testloader, device)

    sl = DataLoader(Subset(trainset, list(range(min(2000, len(trainset))))),
                    batch_size=64, shuffle=False, num_workers=2)
    ext = LatentExtractor()
    ext.compute_median(teacher, sl, latent_layer, device)
    ql = DataLoader(Subset(trainset, list(range(min(n_queries, len(trainset))))),
                    batch_size=64, shuffle=True, num_workers=2)

    st_tmp = create_student(nc, student_arch)
    sp = sum(p.numel() for p in st_tmp.parameters())
    del st_tmp

    print("  Teacher: {} ({:,}p, acc={:.2f}%)".format(
        teacher_name, sum(p.numel() for p in model.parameters()), acc_t*100))
    print("  Student: {} ({:,}p)".format(student_arch, sp))

    # For hard-label mode, temperature doesn't affect training (CE loss),
    # but we still iterate for consistency; use T=1 internally.
    temp_list = temperatures if label_mode == "soft" else [1]

    for T in temp_list:
        if label_mode == "soft":
            print("\n--- T={} [soft-label] ---".format(T))
        else:
            print("\n--- [hard-label] ---")

        ck_stem = stem_ckpt(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, delta_logit, tag=tag)

        cached_s, cached_trig, cached_kw = load_checkpoint(
            dataset, ck_stem, nc, student_arch, subdir="distill",
            teacher_name=teacher_name, label_mode="{}_label".format(label_mode))

        if cached_s is not None:
            student, triggers, Kw = cached_s, cached_trig, cached_kw
            acc_s = evaluate_accuracy(student, testloader, device)
            print("  Cached surrogate: acc={:.1f}%, {} triggers, K_w={}...".format(
                acc_s*100, len(triggers), Kw.hex()[:12]))
        else:
            Kw = kdf(keygen(256), "watermark")
            wm = WatermarkModule(
                K_w=Kw, r_w=rw, delta_logit=delta_logit, beta=beta,
                delta_min=delta_min,
                num_classes=nc,
                latent_extractor=ext, layer_name=latent_layer,
            )

            if label_mode == "soft":
                # Soft-label: KL divergence distillation
                inp, sl_out, nwm = collect_soft_labels(teacher, ql, device, wm, T)
                nt = len(wm.trigger_set)
                print("  {} queries, {} wm ({:.2f}%), {} triggers, delta_logit={}".format(
                    len(inp), nwm, nwm/len(inp)*100, nt, delta_logit))

                student = create_student(nc, student_arch)
                student, dist_loss_history = soft_label_distillation(
                    student, inp, sl_out, T, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)
            else:
                # Hard-label: Cross-entropy extraction with watermark
                inp, hard_labels, nwm = collect_hard_labels(teacher, ql, device, wm)
                nt = len(wm.trigger_set)
                print("  {} queries, {} wm ({:.2f}%), {} triggers, delta_logit={}".format(
                    len(inp), nwm, nwm/len(inp)*100, nt, delta_logit))

                student = create_student(nc, student_arch)
                student, dist_loss_history = hard_label_extraction(
                    student, inp, hard_labels, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)

            acc_s = evaluate_accuracy(student, testloader, device)
            triggers = wm.trigger_set

            save_checkpoint(student, triggers, Kw, dataset, ck_stem, meta={
                "dataset": dataset, "teacher": teacher_name, "student": student_arch,
                "T": T, "rw": rw, "nq": n_queries, "delta_logit": delta_logit, "beta": beta,
                "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                "accuracy": round(acc_s, 6), "n_triggers": nt,
                "label_mode": label_mode, "tag": tag,
            }, subdir="distill",
                teacher_name=teacher_name, student_arch=student_arch,
                label_mode="{}_label".format(label_mode))

        vr = verify_and_format(triggers, student, nc, Kw, eta, device,
                               verify_temperature=verify_temperature)
        print("  Acc={:.1f}%, Shift={:.4f}, p={:.2e}, V={}".format(
            acc_s*100, vr["confidence_shift"], vr["p_value"], vr["verified"]))

        fname = stem_distill(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, delta_logit, vT=verify_temperature, tag=tag)
        save_result({
            "experiment": "distillation", "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "label_mode": label_mode,
            "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6),
                        "latent_layer": latent_layer},
            "student": {"architecture": student_arch, "num_parameters": sp},
            "watermark_config": watermark_config_dict(rw, delta_logit, beta, nc, verify_temperature),
            "distillation_config": {
                "temperature": T, "n_queries": n_queries,
                "epochs": dist_epochs, "lr": dist_lr, "batch_size": dist_batch,
                "label_mode": label_mode,
            },
            "result": {
                "temperature": T,
                "student_accuracy": round(acc_s, 6),
                "n_triggers": len(triggers),
                **vr,
            },
        }, dataset, "distill", fname,
            teacher_name=teacher_name, student_arch=student_arch,
            label_mode="{}_label".format(label_mode))


# ============================================================
# Table VII: Surrogate Fine-Tuning Attack
# ============================================================

def experiment_surrogate_ft(model, trainset, testset, testloader, latent_layer, device,
                            teacher_name, student_arch, dataset, config,
                            temperatures, n_queries, rw,
                            eta, dist_epochs, dist_lr, dist_batch,
                            ft_fractions, ft_epochs, ft_lr,
                            delta_logit=2.0, beta=0.4, delta_min=0.5,
                            verify_temperature=5.0,
                            label_mode="soft", trigger_mode="rec_trigger",
                            own_trigger_size=0, rec_trigger_size=0, tag=""):
    """
    Table VII: Surrogate Fine-Tuning Attack.

    Args:
        label_mode: "soft" or "hard" — controls the initial distillation method.
        trigger_mode: "rec_trigger" = use original trigger set,
                      "own_trigger" = reconstruct from testset (Owner's data),
                      "both" = run both for comparison.
    """
    nc = config["num_classes"]

    print("\n" + "=" * 60)
    print("Table VII: Surrogate Fine-Tuning Attack [{}]".format(label_mode.upper()))
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("  rw={}, dist_epochs={}, ft_epochs={}, ft_lr={}, ft_fractions={}".format(
        rw, dist_epochs, ft_epochs, ft_lr, ft_fractions))
    print("  delta_logit={}, beta={}, label_mode={}".format(delta_logit, beta, label_mode))
    print("=" * 60)

    teacher = copy.deepcopy(model)
    acc_t = evaluate_accuracy(teacher, testloader, device)

    ext = None
    temp_list = temperatures if label_mode == "soft" else [1]

    for T in temp_list:
        if label_mode == "soft":
            print("\n--- T={} [soft-label] ---".format(T))
        else:
            print("\n--- [hard-label] ---")

        ck_stem = stem_ckpt(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, delta_logit, tag=tag)

        surrogate, triggers, Kw = load_checkpoint(
            dataset, ck_stem, nc, student_arch, subdir="distill",
            teacher_name=teacher_name, label_mode="{}_label".format(label_mode))

        if surrogate is None:
            print("  No cache. Distilling ({})...".format(label_mode))
            if ext is None:
                sl = DataLoader(Subset(trainset, list(range(min(2000, len(trainset))))),
                                batch_size=64, shuffle=False, num_workers=2)
                ext = LatentExtractor()
                ext.compute_median(teacher, sl, latent_layer, device)

            ql = DataLoader(Subset(trainset, list(range(min(n_queries, len(trainset))))),
                            batch_size=64, shuffle=True, num_workers=2)
            Kw = kdf(keygen(256), "watermark")
            wm = WatermarkModule(
                K_w=Kw, r_w=rw, delta_logit=delta_logit, beta=beta,
                delta_min=delta_min,
                num_classes=nc,
                latent_extractor=ext, layer_name=latent_layer,
            )

            if label_mode == "soft":
                inp, sl_out, nwm = collect_soft_labels(teacher, ql, device, wm, T)
                surrogate = create_student(nc, student_arch)
                surrogate, dist_loss_history = soft_label_distillation(
                    surrogate, inp, sl_out, T, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)
            else:
                inp, hard_labels, nwm = collect_hard_labels(teacher, ql, device, wm)
                surrogate = create_student(nc, student_arch)
                surrogate, dist_loss_history = hard_label_extraction(
                    surrogate, inp, hard_labels, dist_epochs,
                    batch_size=dist_batch, lr=dist_lr, device=device)

            triggers = wm.trigger_set

            acc_d = evaluate_accuracy(surrogate, testloader, device)
            save_checkpoint(surrogate, triggers, Kw, dataset, ck_stem, meta={
                "dataset": dataset, "teacher": teacher_name, "student": student_arch,
                "T": T, "rw": rw, "nq": n_queries, "delta_logit": delta_logit,
                "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                "accuracy": round(acc_d, 6), "n_triggers": len(triggers),
                "label_mode": label_mode, "tag": tag,
            }, subdir="distill",
                teacher_name=teacher_name, student_arch=student_arch,
                label_mode="{}_label".format(label_mode))

            vr_d = verify_and_format(triggers, surrogate, nc, Kw, eta, device,
                                     verify_temperature=verify_temperature)
            dfname = stem_distill(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, delta_logit, vT=verify_temperature, tag=tag)
            save_result({
                "experiment": "distillation",
                "timestamp": datetime.now().isoformat(),
                "note": "Auto-generated during surrogate_ft",
                "dataset": dataset,
                "label_mode": label_mode,
                "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6)},
                "student": {"architecture": student_arch},
                "watermark_config": watermark_config_dict(rw, delta_logit, beta, nc, verify_temperature),
                "distillation_config": {
                    "temperature": T, "n_queries": n_queries,
                    "epochs": dist_epochs, "lr": dist_lr, "batch_size": dist_batch,
                    "label_mode": label_mode,
                },
                "result": {
                    "temperature": T,
                    "student_accuracy": round(acc_d, 6),
                    "n_triggers": len(triggers),
                    **vr_d,
                },
            }, dataset, "distill", dfname,
                teacher_name=teacher_name, student_arch=student_arch,
                label_mode="{}_label".format(label_mode))
        # else: Kw was loaded from the .key file by load_checkpoint above

        acc_base = evaluate_accuracy(surrogate, testloader, device)
        nt = len(triggers)

        # Determine which trigger modes to run
        if trigger_mode == "both":
            t_modes = ["rec_trigger", "own_trigger"]
        else:
            t_modes = [trigger_mode]

        for t_mode in t_modes:
            print("\n  --- Verification: trigger_mode={} ---".format(t_mode))

            # Prepare own_trigger mode
            # No Phi(x) filtering — every sample is a trigger candidate
            # No latent extractor needed
            if t_mode == "own_trigger":
                own_loader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)
                vr_base = verify_own_data_and_format(
                    model, own_loader, surrogate,
                    Kw, rw, nc, ext, latent_layer,
                    delta_logit, beta, eta, device, verify_temperature,
                    max_triggers=own_trigger_size)
                trigger_size_used = vr_base["n_trigger"]
                trigger_pool_size = len(trainset)
            else:
                # Cap recorded triggers if rec_trigger_size > 0
                use_triggers = triggers
                if rec_trigger_size > 0 and len(triggers) > rec_trigger_size:
                    use_triggers = triggers[:rec_trigger_size]
                vr_base = verify_and_format(use_triggers, surrogate, nc, Kw, eta, device,
                                            verify_temperature=verify_temperature)
                trigger_size_used = len(use_triggers)
                trigger_pool_size = len(triggers)

            print("  Base: acc={:.1f}%, triggers={}/{}, shift={:.4f}, p={:.2e}, V={} [{}]".format(
                acc_base*100, trigger_size_used, trigger_pool_size,
                vr_base["confidence_shift"],
                vr_base["p_value"], vr_base["verified"], t_mode))

            ft_results = []
            for frac in ft_fractions:
                if frac == 0.0:
                    acc_ft = acc_base
                    vr_ft = vr_base
                    label = "baseline"
                else:
                    nft = int(len(trainset) * frac)
                    fl = DataLoader(Subset(trainset, list(range(nft))),
                                    batch_size=128, shuffle=True)
                    surr_ft, ft_loss_history = fine_tune_surrogate(
                        copy.deepcopy(surrogate), fl, ft_epochs, ft_lr, device)
                    acc_ft = evaluate_accuracy(surr_ft, testloader, device)

                    if t_mode == "own_trigger":
                        vr_ft = verify_own_data_and_format(
                            model, own_loader, surr_ft,
                            Kw, rw, nc, ext, latent_layer,
                            delta_logit, beta, eta, device, verify_temperature,
                            max_triggers=own_trigger_size)
                    else:
                        vr_ft = verify_and_format(use_triggers, surr_ft, nc, Kw, eta, device,
                                                  verify_temperature=verify_temperature)
                    label = "{}% ({})".format(int(frac*100), nft)

                print("    FT {}: acc={:.1f}%, shift={:.4f}, p={:.2e}, V={} [{}]".format(
                    label, acc_ft*100, vr_ft["confidence_shift"],
                    vr_ft["p_value"], vr_ft["verified"], t_mode))

                ft_results.append({
                    "ft_fraction": frac,
                    "ft_samples": int(len(trainset) * frac),
                    "accuracy": round(acc_ft, 6),
                    **vr_ft,
                })

            # Save with trigger_mode in directory path
            save_label = "{}_label__{}".format(label_mode, t_mode)

            fname = stem_surrogate_ft(
                teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr,
                ft_epochs, ft_lr, delta_logit, vT=verify_temperature,
                trigger_mode=t_mode, trigger_size=trigger_size_used, tag=tag)
            save_result({
                "experiment": "surrogate_finetune",
                "timestamp": datetime.now().isoformat(),
                "dataset": dataset,
                "label_mode": label_mode,
                "trigger_mode": t_mode,
                "trigger_config": {
                    "trigger_source": t_mode,
                    "trigger_size_used": trigger_size_used,
                    "trigger_pool_size": trigger_pool_size,
                    "own_trigger_size": own_trigger_size if t_mode == "own_trigger" else None,
                    "rec_trigger_size": rec_trigger_size if t_mode == "rec_trigger" else None,
                },
                "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6)},
                "student": {"architecture": student_arch},
                "watermark_config": watermark_config_dict(rw, delta_logit, beta, nc, verify_temperature),
                "distillation_baseline": {
                    "temperature": T, "n_queries": n_queries,
                    "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                    "surrogate_accuracy": round(acc_base, 6),
                    "n_triggers_recorded": nt,
                    "n_triggers_used": trigger_size_used,
                    **vr_base,
                    "label_mode": label_mode,
                    "trigger_mode": t_mode,
                },
                "ft_config": {
                    "ft_fractions": list(ft_fractions),
                    "ft_epochs": ft_epochs,
                    "ft_lr": ft_lr,
                },
                "ft_results": ft_results,
            }, dataset, "surrogate_ft", fname,
                teacher_name=teacher_name, student_arch=student_arch,
                label_mode=save_label)


# ============================================================
# Table VIII: Overhead
# ============================================================

def experiment_overhead(model, testloader, device, epsilon, delta,
                        teacher_name, dataset, config):
    print("\n" + "=" * 60)
    print("Table VIII: Overhead")
    print("  Dataset: {}, Teacher: {}, eps={}".format(dataset, teacher_name, epsilon))
    print("=" * 60)

    np_ = sum(p.numel() for p in model.parameters())
    m = copy.deepcopy(model)

    t0 = time.time()
    m, sec = obfuscate_model_vectorized(m, epsilon=epsilon, delta=delta, model_id="o")
    ot = time.time() - t0

    t0 = time.time()
    recover_weights(m, sec, vectorized=True)
    rt = time.time() - t0

    m.to(device).eval()
    x, _ = next(iter(testloader))
    x = x[:1].to(device)
    with torch.no_grad():
        for _ in range(10):
            m(x)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            m(x)
    it = (time.time() - t0) / 100 * 1000

    ss = len(sec.K_obf) + 8 + len(sec.K_w) + len(sec.model_id.encode())
    print("  Obf={:.3f}s, Rec={:.3f}s, Inf={:.2f}ms, |S|={}B".format(ot, rt, it, ss))

    fname = stem_overhead(teacher_name, epsilon)
    save_result({
        "experiment": "overhead", "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model": model_info(model, teacher_name),
        "parameters": {"epsilon": epsilon, "delta": str(delta)},
        "results": {
            "obfuscation_time_s": round(ot, 4),
            "recovery_time_s": round(rt, 4),
            "inference_ms": round(it, 4),
            "secret_size_bytes": ss,
            "num_parameters": np_,
        },
    }, dataset, "overhead", fname)


# ============================================================
# Main
# ============================================================

def parse_list_float(s):
    return [float(x.strip()) for x in s.split(",")]

def parse_list_int(s):
    return [int(x.strip()) for x in s.split(",")]

def main():
    pa = argparse.ArgumentParser(
        description="EvalGuard Experiments (v4 - Confidence Shift)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    pa.add_argument("--experiment", default="distill",
                    choices=["fidelity", "finetune", "distill", "surrogate_ft",
                             "overhead", "all"])
    pa.add_argument("--model", default="cifar10_vgg11")
    pa.add_argument("--dataset", default=None)
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--pretrained", default=None)

    pa.add_argument("--epsilon", type=float, default=50.0)
    pa.add_argument("--delta", type=float, default=2**(-32))

    # Watermark parameters (v5: logit-space confidence shift)
    pa.add_argument("--rw", type=float, default=0.50)
    pa.add_argument("--delta_logit", type=float, default=2.0,
                    help="Logit-space shift amount (applied before softmax)")
    pa.add_argument("--beta", type=float, default=0.3,
                    help="Safety factor: delta = min(delta_logit, beta * logit_margin)")
    pa.add_argument("--delta_min", type=float, default=0.5,
                    help="Minimum effective delta. Queries whose safe delta would "
                         "fall below this are NOT recorded as triggers (avoids "
                         "polluting the trigger set with near-zero signals).")
    pa.add_argument("--verify_temperature", type=float, default=5.0,
                    help="Temperature used during verification softmax")

    pa.add_argument("--student_arch", default="vgg11",
                    choices=["resnet20", "resnet56", "vgg11", "mobilenetv2", "resnet18"])
    pa.add_argument("--nq", type=int, default=50000)
    pa.add_argument("--temperatures", type=str, default="5,10,20")
    pa.add_argument("--dist_epochs", type=int, default=80)
    pa.add_argument("--dist_lr", type=float, default=0.002)
    pa.add_argument("--dist_batch", type=int, default=128)

    pa.add_argument("--ft_epochs", type=int, default=20)
    pa.add_argument("--ft_lr", type=float, default=0.0005)
    pa.add_argument("--ft_fractions", type=str, default="0.0,0.01,0.05,0.10")

    pa.add_argument("--label_mode", default="soft",
                    choices=["soft", "hard", "both"],
                    help="Label mode: 'soft' for KL-divergence, 'hard' for CE, 'both' to run both")

    pa.add_argument("--trigger_mode", default="rec_trigger",
                    choices=["rec_trigger", "own_trigger", "both"],
                    help="Trigger source for verification: "
                         "'rec_trigger' = recorded trigger set from embedding phase, "
                         "'own_trigger' = reconstruct from Owner's data (zero D_eval leakage), "
                         "'both' = run both for comparison")

    pa.add_argument("--own_trigger_size", type=int, default=0,
                    help="Number of own triggers to collect for own_trigger verification. "
                         "0 = collect all from testset. "
                         "Scans testset until this many triggers are found.")

    pa.add_argument("--rec_trigger_size", type=int, default=0,
                    help="Number of recorded triggers to use for rec_trigger verification. "
                         "0 = use all recorded triggers.")

    pa.add_argument("--epsilons", type=str, default="1,10,50,100,200")

    pa.add_argument("--tag", type=str, default="",
                    help="Free-form experiment tag appended to checkpoint and result "
                         "filenames. Use this to distinguish runs with different "
                         "watermark parameters (e.g. 'paper', 'amplified', 'smoke').")

    a = pa.parse_args()

    config_key, dataset, teacher_short = resolve_model(a.model, a.dataset)
    config = CONFIGS[config_key]

    temperatures = parse_list_int(a.temperatures)
    ft_fractions = parse_list_float(a.ft_fractions)

    print("=" * 60)
    print("EvalGuard Experiment Runner (v5 - Logit-Space Shift)")
    print("  Dataset: {}".format(dataset))
    print("  Teacher: {} (config: {})".format(teacher_short, config_key))
    print("  Student: {}".format(a.student_arch))
    print("  Device: {}".format(a.device))
    print("  Experiment: {}".format(a.experiment))
    print("  Label mode: {}".format(a.label_mode))
    print("  Trigger mode: {}".format(a.trigger_mode))
    print("  rw={}, delta_logit={}, beta={}, delta_min={}, verify_T={}".format(
        a.rw, a.delta_logit, a.beta, a.delta_min, a.verify_temperature))
    print("  tag={}".format(a.tag if a.tag else "(none)"))
    print("  dist: epochs={}, lr={}, batch={}".format(a.dist_epochs, a.dist_lr, a.dist_batch))
    print("  ft: epochs={}, lr={}, fractions={}".format(a.ft_epochs, a.ft_lr, ft_fractions))
    print("  temperatures={}".format(temperatures))
    print("=" * 60)

    if dataset not in DATA_FN:
        print("[WARN] No data loader for dataset '{}'. Some experiments may fail.".format(dataset))
        trainset, testset, trainloader, testloader = None, None, None, None
    else:
        trainset, testset, trainloader, testloader = DATA_FN[dataset]()

    model, latent_layer, _ = get_model(config_key, a.pretrained)
    model.to(a.device)
    if testloader:
        print("Baseline accuracy: {:.2f}%\n".format(
            evaluate_accuracy(model, testloader, a.device) * 100))

    # Determine label modes to run
    if a.label_mode == "both":
        label_modes = ["soft", "hard"]
    else:
        label_modes = [a.label_mode]

    if a.experiment in ("fidelity", "all"):
        experiment_fidelity(model, testloader, a.device, a.epsilon, a.delta,
                            teacher_short, dataset, config)

    if a.experiment in ("distill", "all"):
        for lm in label_modes:
            experiment_distillation(model, trainset, testloader, latent_layer, a.device,
                                    teacher_short, a.student_arch, dataset, config,
                                    temperatures, a.nq, a.rw,
                                    2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                    delta_logit=a.delta_logit, beta=a.beta,
                                    delta_min=a.delta_min,
                                    verify_temperature=a.verify_temperature,
                                    label_mode=lm, tag=a.tag)

    if a.experiment in ("surrogate_ft", "all"):
        for lm in label_modes:
            experiment_surrogate_ft(model, trainset, testset, testloader, latent_layer, a.device,
                                    teacher_short, a.student_arch, dataset, config,
                                    temperatures, a.nq, a.rw,
                                    2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                    ft_fractions, a.ft_epochs, a.ft_lr,
                                    delta_logit=a.delta_logit, beta=a.beta,
                                    delta_min=a.delta_min,
                                    verify_temperature=a.verify_temperature,
                                    label_mode=lm, trigger_mode=a.trigger_mode,
                                    own_trigger_size=a.own_trigger_size,
                                    rec_trigger_size=a.rec_trigger_size, tag=a.tag)

    if a.experiment in ("overhead", "all"):
        experiment_overhead(model, testloader, a.device, a.epsilon, a.delta,
                            teacher_short, dataset, config)

    print("\nResults: ./results/{}/{}/   Checkpoints: ./checkpoints/distill/{}/".format(
        a.experiment, dataset, dataset))


if __name__ == "__main__":
    main()