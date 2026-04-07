"""
EvalGuard — Experiment Script (Section VI) [v2]
"""

import sys, os
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
    WatermarkModule, LatentExtractor, verify_ownership, compute_null_probability,
)
from evalguard.crypto import keygen, kdf
from evalguard.configs import (
    CONFIGS, cifar10_data, cifar100_data,
    create_student,
)
from evalguard.attacks import (
    collect_soft_labels, soft_label_distillation, fine_tune_surrogate,
)

import numpy as np

RESULTS_DIR = Path("results")
CKPT_DIR = Path("checkpoints")


# ============================================================
# Dataset detection
# ============================================================

# 更新后的模型注册表，名字完全对应
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

def _amp_str(alpha):
    """Return '__amp{alpha}' if alpha > 1.0, else empty string."""
    if alpha is not None and alpha > 1.0:
        return "__amp{}".format(alpha)
    return ""

def stem_fidelity(teacher, eps, delta):
    return "T_{}__eps{}__delta{}".format(teacher, eps, _delta_str(delta))

def stem_finetune(teacher, eps, delta, ft_epochs):
    return "T_{}__eps{}__delta{}__ftEp{}".format(teacher, eps, _delta_str(delta), ft_epochs)

def stem_distill(teacher, student, rw, nq, T, dist_epochs, dist_lr, alpha=1.0):
    return "T_{}__S_{}__rw{}__nq{}__T{}__distEp{}__distLr{}{}".format(
        teacher, student, rw, nq, T, dist_epochs, dist_lr, _amp_str(alpha))

def stem_surrogate_ft(teacher, student, rw, nq, T, dist_epochs, dist_lr, ft_epochs, ft_lr, alpha=1.0):
    return "T_{}__S_{}__rw{}__nq{}__T{}__distEp{}__distLr{}__ftEp{}__ftLr{}{}".format(
        teacher, student, rw, nq, T, dist_epochs, dist_lr, ft_epochs, ft_lr, _amp_str(alpha))

def stem_overhead(teacher, eps):
    return "T_{}__eps{}".format(teacher, eps)

def stem_null(teacher, k, theta, n_models):
    return "T_{}__k{}__theta{}__nModels{}".format(teacher, k, theta, n_models)

def stem_ckpt(teacher, student, rw, nq, T, dist_epochs, dist_lr, alpha=1.0):
    return stem_distill(teacher, student, rw, nq, T, dist_epochs, dist_lr, alpha)


# ============================================================
# Save / Load
# ============================================================

def save_result(data, dataset, subdir, filename):
    # 修改目录层级：先 experiment_type (subdir)，再 dataset
    path = RESULTS_DIR / subdir / dataset / "{}.json".format(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print("  -> Saved: {}".format(path))


def save_checkpoint(model, triggers, dataset, stem, meta=None, subdir="distill"):
    # 修改目录层级：先 experiment_type (subdir)，再 dataset
    ckpt_dir = CKPT_DIR / subdir / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "{}.pt".format(stem))
    with open(ckpt_dir / "{}.pkl".format(stem), "wb") as f:
        pickle.dump(triggers, f)
    if meta:
        with open(ckpt_dir / "{}.meta.json".format(stem), "w") as f:
            json.dump(meta, f, indent=2, default=str)
    print("  -> Cached: {}/{}.pt".format(ckpt_dir, stem))


def load_checkpoint(dataset, stem, num_classes, student_arch, subdir="distill"):
    # 修改目录层级：先 experiment_type (subdir)，再 dataset
    ckpt_dir = CKPT_DIR / subdir / dataset
    pt_path = ckpt_dir / "{}.pt".format(stem)
    pkl_path = ckpt_dir / "{}.pkl".format(stem)
    if not pt_path.exists() or not pkl_path.exists():
        return None, None
    model = create_student(num_classes=num_classes, arch=student_arch)
    model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
    with open(pkl_path, "rb") as f:
        triggers = pickle.load(f)
    print("  -> Loaded cache: {}".format(pt_path))
    return model, triggers


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


def verify_and_format(triggers, model, k, theta, eta, p0, device):
    if len(triggers) == 0:
        return {"watermark_retention": 0.0, "n_match": 0, "n_total": 0,
                "null_expected": 0, "p_value": 1.0, "verified": False}
    vr = verify_ownership(triggers, model, k=k, theta=theta, eta=eta, device=device)
    return {
        "watermark_retention": round(vr["match_rate"], 6),
        "n_match": vr["n_match"],
        "n_total": vr["n_total"],
        "null_expected": round(vr["n_total"] * p0, 1),
        "p_value": vr["p_value"],
        "log10_p_value": round(math.log10(max(vr["p_value"], 1e-300)), 2),
        "verified": vr["verified"],
    }


def watermark_config_dict(rw, k, theta, eta, p0):
    return {
        "r_w": rw, "r_w_percent": "{}%".format(rw * 100),
        "k": k, "bits_per_sample": round(math.log2(math.factorial(k)), 2),
        "theta": theta,
        "eta": "2^-64", "eta_numeric": eta,
        "p0": round(p0, 6), "p0_human": "1/{}! = 1/{}".format(k, math.factorial(k)),
    }


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
# Table V: Fine-Tuning Attack on M*
# ============================================================

def experiment_finetune(model, trainset, testloader, device,
                        teacher_name, dataset, config,
                        epsilon, delta, epsilons, ft_fractions, ft_epochs):
    print("\n" + "=" * 60)
    print("Table V: Fine-Tuning Attack on M*")
    print("  Dataset: {}, Teacher: {}, epsilons={}, ft_epochs={}".format(
        dataset, teacher_name, epsilons, ft_epochs))
    print("=" * 60)

    acc_orig = evaluate_accuracy(model, testloader, device)
    attacks = []

    for eps in epsilons:
        ms = copy.deepcopy(model)
        ms, _ = obfuscate_model_vectorized(ms, epsilon=eps, delta=delta, model_id="evalguard")
        acc_obf = evaluate_accuracy(ms, testloader, device)

        for frac in ft_fractions:
            n = int(len(trainset) * frac)
            fl = DataLoader(Subset(trainset, list(range(n))), batch_size=64, shuffle=True)

            af = evaluate_accuracy(
                train_model(copy.deepcopy(ms), fl, ft_epochs, 0.001, device),
                testloader, device)

            m2 = copy.deepcopy(ms)
            for nm, p in m2.named_parameters():
                if "fc" not in nm and "classifier" not in nm:
                    p.requires_grad = False
            asf = evaluate_accuracy(
                train_model(m2, fl, ft_epochs, 0.001, device),
                testloader, device)

            print("  eps={}, {}%: Obf={:.2f}%, FTAL={:.2f}%, SFT={:.2f}%".format(
                eps, int(frac*100), acc_obf*100, af*100, asf*100))

            attacks.append({
                "epsilon": eps, "ft_fraction": frac, "ft_samples": n,
                "acc_obfuscated": round(acc_obf, 6),
                "acc_FTAL": round(af, 6),
                "acc_SFT": round(asf, 6),
            })

    fname = stem_finetune(teacher_name, "multi", delta, ft_epochs)
    save_result({
        "experiment": "finetune_attack_on_M*", "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "model": model_info(model, teacher_name),
        "teacher_accuracy": round(acc_orig, 6),
        "parameters": {
            "delta": str(delta), "delta_str": _delta_str(delta),
            "epsilons": list(epsilons),
            "ft_fractions": list(ft_fractions),
            "ft_epochs": ft_epochs,
        },
        "attack_results": attacks,
    }, dataset, "finetune", fname)


# ============================================================
# Table VI: Distillation + Watermark Retention
# ============================================================

def experiment_distillation(model, trainset, testloader, latent_layer, device,
                            teacher_name, student_arch, dataset, config,
                            temperatures, n_queries, k, rw,
                            theta, eta, dist_epochs, dist_lr, dist_batch,
                            alpha=1.0):
    nc = config["num_classes"]
    p0 = compute_null_probability(k, theta)

    print("\n" + "=" * 60)
    print("Table VI: Distillation + Watermark Retention")
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("  rw={}, k={}, nq={}, dist_epochs={}, dist_lr={}, dist_batch={}, alpha={}".format(
        rw, k, n_queries, dist_epochs, dist_lr, dist_batch, alpha))
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

    for T in temperatures:
        print("\n--- T={} ---".format(T))
        ck_stem = stem_ckpt(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, alpha)

        cached_s, cached_trig = load_checkpoint(dataset, ck_stem, nc, student_arch, subdir="distill")

        if cached_s is not None:
            student, triggers = cached_s, cached_trig
            acc_s = evaluate_accuracy(student, testloader, device)
            print("  Cached surrogate: acc={:.1f}%, {} triggers".format(acc_s*100, len(triggers)))
        else:
            Kw = kdf(keygen(256), "watermark")
            wm = WatermarkModule(K_w=Kw, r_w=rw, k=k, alpha=alpha,
                                 latent_extractor=ext, layer_name=latent_layer)
            inp, sl_out, nwm = collect_soft_labels(teacher, ql, device, wm, T)
            nt = len(wm.trigger_set)
            print("  {} queries, {} wm ({:.2f}%), {} triggers, alpha={}".format(
                len(inp), nwm, nwm/len(inp)*100, nt, alpha))

            student = create_student(nc, student_arch)
            student = soft_label_distillation(
                student, inp, sl_out, T, dist_epochs,
                batch_size=dist_batch, lr=dist_lr, device=device)
            acc_s = evaluate_accuracy(student, testloader, device)
            triggers = wm.trigger_set

            save_checkpoint(student, triggers, dataset, ck_stem, meta={
                "dataset": dataset, "teacher": teacher_name, "student": student_arch,
                "T": T, "rw": rw, "nq": n_queries, "k": k, "alpha": alpha,
                "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                "accuracy": round(acc_s, 6), "n_triggers": nt,
            }, subdir="distill")

        vr = verify_and_format(triggers, student, k, theta, eta, p0, device)
        print("  Acc={:.1f}%, Ret={:.1f}% ({}/{}), p={:.2e}, V={}".format(
            acc_s*100, vr["watermark_retention"]*100, vr["n_match"], vr["n_total"],
            vr["p_value"], vr["verified"]))

        fname = stem_distill(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, alpha)
        save_result({
            "experiment": "distillation", "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6),
                        "latent_layer": latent_layer},
            "student": {"architecture": student_arch, "num_parameters": sp},
            "watermark_config": {**watermark_config_dict(rw, k, theta, eta, p0), "alpha": alpha},
            "distillation_config": {
                "temperature": T, "n_queries": n_queries,
                "epochs": dist_epochs, "lr": dist_lr, "batch_size": dist_batch,
            },
            "result": {
                "temperature": T,
                "student_accuracy": round(acc_s, 6),
                "n_triggers": len(triggers),
                **vr,
            },
        }, dataset, "distill", fname)


# ============================================================
# Table VII: Surrogate Fine-Tuning Attack
# ============================================================

def experiment_surrogate_ft(model, trainset, testloader, latent_layer, device,
                            teacher_name, student_arch, dataset, config,
                            temperatures, n_queries, k, rw,
                            theta, eta, dist_epochs, dist_lr, dist_batch,
                            ft_fractions, ft_epochs, ft_lr,
                            alpha=1.0):
    nc = config["num_classes"]
    p0 = compute_null_probability(k, theta)

    print("\n" + "=" * 60)
    print("Table VII: Surrogate Fine-Tuning Attack")
    print("  Dataset: {}, T: {} -> S: {}".format(dataset, teacher_name, student_arch))
    print("  rw={}, dist_epochs={}, ft_epochs={}, ft_lr={}, ft_fractions={}".format(
        rw, dist_epochs, ft_epochs, ft_lr, ft_fractions))
    print("=" * 60)

    teacher = copy.deepcopy(model)
    acc_t = evaluate_accuracy(teacher, testloader, device)

    ext = None

    for T in temperatures:
        print("\n--- T={} ---".format(T))
        ck_stem = stem_ckpt(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, alpha)

        surrogate, triggers = load_checkpoint(dataset, ck_stem, nc, student_arch, subdir="distill")

        if surrogate is None:
            print("  No cache. Distilling...")
            if ext is None:
                sl = DataLoader(Subset(trainset, list(range(min(2000, len(trainset))))),
                                batch_size=64, shuffle=False, num_workers=2)
                ext = LatentExtractor()
                ext.compute_median(teacher, sl, latent_layer, device)

            ql = DataLoader(Subset(trainset, list(range(min(n_queries, len(trainset))))),
                            batch_size=64, shuffle=True, num_workers=2)
            Kw = kdf(keygen(256), "watermark")
            wm = WatermarkModule(K_w=Kw, r_w=rw, k=k, alpha=alpha,
                                 latent_extractor=ext, layer_name=latent_layer)
            inp, sl_out, nwm = collect_soft_labels(teacher, ql, device, wm, T)

            surrogate = create_student(nc, student_arch)
            surrogate, dist_loss_history = soft_label_distillation(
                surrogate, inp, sl_out, T, dist_epochs,
                batch_size=dist_batch, lr=dist_lr, device=device)
            triggers = wm.trigger_set

            acc_d = evaluate_accuracy(surrogate, testloader, device)
            save_checkpoint(surrogate, triggers, dataset, ck_stem, meta={
                "dataset": dataset, "teacher": teacher_name, "student": student_arch,
                "T": T, "rw": rw, "nq": n_queries, "k": k, "alpha": alpha,
                "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                "accuracy": round(acc_d, 6), "n_triggers": len(triggers),
            }, subdir="distill")

            vr_d = verify_and_format(triggers, surrogate, k, theta, eta, p0, device)
            dfname = stem_distill(teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, alpha)
            save_result({
                "experiment": "distillation",
                "timestamp": datetime.now().isoformat(),
                "note": "Auto-generated during surrogate_ft",
                "dataset": dataset,
                "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6)},
                "student": {"architecture": student_arch},
                "watermark_config": {**watermark_config_dict(rw, k, theta, eta, p0), "alpha": alpha},
                "distillation_config": {
                    "temperature": T, "n_queries": n_queries,
                    "epochs": dist_epochs, "lr": dist_lr, "batch_size": dist_batch,
                },
                "result": {
                    "temperature": T,
                    "student_accuracy": round(acc_d, 6),
                    "n_triggers": len(triggers),
                    **vr_d,
                },
            }, dataset, "distill", dfname)

        acc_base = evaluate_accuracy(surrogate, testloader, device)
        nt = len(triggers)
        vr_base = verify_and_format(triggers, surrogate, k, theta, eta, p0, device)
        print("  Base: acc={:.1f}%, ret={:.1f}% ({}/{}), V={}".format(
            acc_base*100, vr_base["watermark_retention"]*100,
            vr_base["n_match"], vr_base["n_total"], vr_base["verified"]))

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
                vr_ft = verify_and_format(triggers, surr_ft, k, theta, eta, p0, device)
                label = "{}% ({})".format(int(frac*100), nft)

            print("    FT {}: acc={:.1f}%, ret={:.1f}% ({}/{}), p={:.2e}, V={}".format(
                label, acc_ft*100, vr_ft["watermark_retention"]*100,
                vr_ft["n_match"], vr_ft["n_total"], vr_ft["p_value"], vr_ft["verified"]))

            ft_results.append({
                "ft_fraction": frac,
                "ft_samples": int(len(trainset) * frac),
                "accuracy": round(acc_ft, 6),
                **vr_ft,
            })

        fname = stem_surrogate_ft(
            teacher_name, student_arch, rw, n_queries, T, dist_epochs, dist_lr, ft_epochs, ft_lr, alpha)
        save_result({
            "experiment": "surrogate_finetune",
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset,
            "teacher": {**model_info(model, teacher_name), "accuracy": round(acc_t, 6)},
            "student": {"architecture": student_arch},
            "watermark_config": {**watermark_config_dict(rw, k, theta, eta, p0), "alpha": alpha},
            "distillation_baseline": {
                "temperature": T, "n_queries": n_queries,
                "dist_epochs": dist_epochs, "dist_lr": dist_lr, "dist_batch": dist_batch,
                "surrogate_accuracy": round(acc_base, 6),
                "n_triggers": nt, **vr_base,
            },
            "ft_config": {
                "ft_fractions": list(ft_fractions),
                "ft_epochs": ft_epochs,
                "ft_lr": ft_lr,
            },
            "ft_results": ft_results,
        }, dataset, "surrogate_ft", fname)


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
# Empirical Null p₀
# ============================================================

def experiment_null(trainset, testset, testloader, device,
                    teacher_name, dataset, config,
                    k, theta, n_models, null_epochs):
    print("\n" + "=" * 60)
    print("Empirical Null p₀")
    print("  Dataset: {}, Arch: {}, k={}, theta={}, n_models={}, epochs={}".format(
        dataset, teacher_name, k, theta, n_models, null_epochs))
    print("=" * 60)

    config_key, _, _ = resolve_model(teacher_name)
    model_fn = CONFIGS[config_key]["model_fn"]

    models, accs = [], []
    for i in range(n_models):
        m, _ = model_fn(pretrained=False)
        m = train_model(
            m, DataLoader(trainset, 128, shuffle=True, num_workers=2),
            null_epochs, 0.1, device)
        a = evaluate_accuracy(m, testloader, device)
        models.append(m)
        accs.append(a)
        print("  Model {}: {:.2f}%".format(i + 1, a * 100))

    p0_theory = compute_null_probability(k, theta)
    pairs = []
    ld = DataLoader(Subset(testset, list(range(min(5000, len(testset))))),
                    batch_size=64, num_workers=2)

    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            mt, tt = 0, 0
            models[i].to(device).eval()
            models[j].to(device).eval()
            with torch.no_grad():
                for x, _ in ld:
                    x = x.to(device)
                    ri = np.argsort(
                        -torch.softmax(models[i](x), -1).cpu().numpy(),
                        axis=1)[:, 1:k+1]
                    rj = np.argsort(
                        -torch.softmax(models[j](x), -1).cpu().numpy(),
                        axis=1)[:, 1:k+1]
                    mt += (ri == rj).all(axis=1).sum()
                    tt += x.size(0)
            emp_p0 = mt / tt
            pairs.append({
                "model_i": i + 1, "model_j": j + 1,
                "acc_i": round(accs[i], 4), "acc_j": round(accs[j], 4),
                "empirical_p0": round(float(emp_p0), 6),
            })
            print("  Pair ({},{}): empirical_p0={:.4f}% (theory={:.4f}%)".format(
                i+1, j+1, emp_p0*100, p0_theory*100))

    fname = stem_null(teacher_name, k, theta, n_models)
    save_result({
        "experiment": "empirical_null_p0",
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset,
        "description": (
            "Verify theoretical p0=1/k! by measuring sub-dominant rank agreement "
            "between independently trained models. If empirical p0 >> theoretical p0, "
            "the minimum trigger set size |T|_min must be increased."
        ),
        "parameters": {
            "k": k, "theta": theta,
            "theoretical_p0": round(p0_theory, 6),
            "n_models": n_models,
            "null_epochs": null_epochs,
            "architecture": teacher_name,
        },
        "model_accuracies": [round(a, 4) for a in accs],
        "pair_results": pairs,
    }, dataset, "null", fname)


# ============================================================
# Main
# ============================================================

def parse_list_float(s):
    return [float(x.strip()) for x in s.split(",")]

def parse_list_int(s):
    return [int(x.strip()) for x in s.split(",")]

def main():
    pa = argparse.ArgumentParser(
        description="EvalGuard Experiments (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    pa.add_argument("--experiment", default="fidelity",
                    choices=["fidelity", "finetune", "distill", "surrogate_ft",
                             "overhead", "null", "all"])
    # 默认教师模型改为 cifar10_vgg11，避免旧版的简称混淆
    pa.add_argument("--model", default="cifar10_vgg11")
    pa.add_argument("--dataset", default=None)
    pa.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--pretrained", default=None)

    pa.add_argument("--epsilon", type=float, default=50.0)
    pa.add_argument("--delta", type=float, default=2**(-32))

    pa.add_argument("--rw", type=float, default=0.01)
    pa.add_argument("--k", type=int, default=4)
    pa.add_argument("--theta", type=float, default=1.0)
    pa.add_argument("--alpha", type=float, default=1.0,
                    help="Amplification factor (1.0=off, >1.0=amplify sub-dominant gaps for large num_classes)")

    pa.add_argument("--student_arch", default="resnet20",
                    choices=["resnet20", "resnet56", "vgg11", "mobilenetv2", "resnet18"])
    pa.add_argument("--nq", type=int, default=50000)
    pa.add_argument("--temperatures", type=str, default="5,10,20")
    pa.add_argument("--dist_epochs", type=int, default=80)
    pa.add_argument("--dist_lr", type=float, default=0.002)
    pa.add_argument("--dist_batch", type=int, default=128)

    pa.add_argument("--ft_epochs", type=int, default=20)
    pa.add_argument("--ft_lr", type=float, default=0.0005)
    pa.add_argument("--ft_fractions", type=str, default="0.0,0.01,0.05,0.10")

    pa.add_argument("--epsilons", type=str, default="1,10,50,100,200")

    pa.add_argument("--null_models", type=int, default=3)
    pa.add_argument("--null_epochs", type=int, default=30)

    a = pa.parse_args()

    config_key, dataset, teacher_short = resolve_model(a.model, a.dataset)
    config = CONFIGS[config_key]

    temperatures = parse_list_int(a.temperatures)
    ft_fractions = parse_list_float(a.ft_fractions)
    epsilons = parse_list_int(a.epsilons)

    print("=" * 60)
    print("EvalGuard Experiment Runner (v3)")
    print("  Dataset: {}".format(dataset))
    print("  Teacher: {} (config: {})".format(teacher_short, config_key))
    print("  Student: {}".format(a.student_arch))
    print("  Device: {}".format(a.device))
    print("  Experiment: {}".format(a.experiment))
    print("  rw={}, k={}, alpha={}".format(a.rw, a.k, a.alpha))
    print("  dist: epochs={}, lr={}, batch={}".format(a.dist_epochs, a.dist_lr, a.dist_batch))
    print("  ft: epochs={}, lr={}, fractions={}".format(a.ft_epochs, a.ft_lr, ft_fractions))
    print("  temperatures={}".format(temperatures))
    if a.alpha > 1.0:
        print("  *** Amplification ENABLED (alpha={}) ***".format(a.alpha))
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

    if a.experiment in ("fidelity", "all"):
        experiment_fidelity(model, testloader, a.device, a.epsilon, a.delta,
                            teacher_short, dataset, config)

    if a.experiment in ("finetune", "all"):
        experiment_finetune(model, trainset, testloader, a.device,
                            teacher_short, dataset, config,
                            a.epsilon, a.delta, epsilons, ft_fractions, a.ft_epochs)

    if a.experiment in ("distill", "all"):
        experiment_distillation(model, trainset, testloader, latent_layer, a.device,
                                teacher_short, a.student_arch, dataset, config,
                                temperatures, a.nq, a.k, a.rw,
                                a.theta, 2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                alpha=a.alpha)

    if a.experiment in ("surrogate_ft", "all"):
        experiment_surrogate_ft(model, trainset, testloader, latent_layer, a.device,
                                teacher_short, a.student_arch, dataset, config,
                                temperatures, a.nq, a.k, a.rw,
                                a.theta, 2**(-64), a.dist_epochs, a.dist_lr, a.dist_batch,
                                ft_fractions, a.ft_epochs, a.ft_lr,
                                alpha=a.alpha)

    if a.experiment in ("overhead", "all"):
        experiment_overhead(model, testloader, a.device, a.epsilon, a.delta,
                            teacher_short, dataset, config)

    if a.experiment in ("null", "all"):
        experiment_null(trainset, testset, testloader, a.device,
                        a.model, dataset, config,
                        a.k, a.theta, a.null_models, a.null_epochs)

    print("\nResults: ./results/{}/{}/   Checkpoints: ./checkpoints/distill/{}/".format(a.experiment, dataset, dataset))


if __name__ == "__main__":
    main()