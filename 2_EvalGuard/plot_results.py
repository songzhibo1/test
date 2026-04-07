"""
EvalGuard — Result Visualization

Generates tables and plots from experiment results (results.json).
Uses matplotlib for plots, falls back to ASCII tables if unavailable.

Usage:
    python plot_results.py                     # ASCII tables from results.json
    python plot_results.py --plot              # Generate matplotlib figures
    python plot_results.py --file my_results.json
"""

import argparse
import json
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import numpy as np


def load_results(path="results.json"):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Table IV: Obfuscation & Fidelity
# ============================================================

def print_table_iv(results):
    """Format Table IV from fidelity experiment results."""
    if "fidelity" not in results:
        print("No fidelity data found.")
        return

    r = results["fidelity"]
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║           Table IV: Obfuscation and Fidelity            ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Original accuracy:     {r['acc_original']*100:>7.2f}%                       ║")
    print(f"║  Obfuscated M* (no TEE):{r['acc_obfuscated']*100:>7.2f}%                       ║")
    print(f"║  Recovered (TEE):       {r['acc_recovered']*100:>7.2f}%                       ║")
    print(f"║  Fidelity loss:         {r['fidelity_loss']*100:>7.4f}%                       ║")
    print("║                                                          ║")
    print("║  EvalGuard fidelity loss = 0% by construction            ║")
    print("║  DAWN fidelity loss ≈ r_w (top-1 modification)           ║")
    print("╚══════════════════════════════════════════════════════════╝")


# ============================================================
# Table V: Fine-Tuning Attack
# ============================================================

def print_table_v(results):
    """Format Table V from fine-tuning attack results."""
    if "finetune" not in results:
        print("No finetune data found.")
        return

    rows = results["finetune"]
    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│     Table V: Fine-Tuning Attack on M* (CIFAR-10)       │")
    print("├──────┬──────────┬──────────┬──────────┬────────────────┤")
    print("│  ε   │ Obf.Acc  │ FTAL 1%  │ FTAL 10% │   SFT 10%    │")
    print("├──────┼──────────┼──────────┼──────────┼────────────────┤")

    # Group by epsilon
    by_eps = {}
    for r in rows:
        eps = r["eps"]
        if eps not in by_eps:
            by_eps[eps] = {"obf": r["obf"]}
        frac = r["frac"]
        by_eps[eps][f"ftal_{frac}"] = r["ftal"]
        by_eps[eps][f"sft_{frac}"] = r.get("sft", 0)

    for eps, d in sorted(by_eps.items()):
        ftal_1 = d.get("ftal_0.01", 0) * 100
        ftal_10 = d.get("ftal_0.1", 0) * 100
        sft_10 = d.get("sft_0.1", 0) * 100
        print(f"│ {eps:>4.0f} │  {d['obf']*100:>5.1f}%  │  {ftal_1:>5.1f}%  │  {ftal_10:>5.1f}%  │    {sft_10:>5.1f}%     │")

    print("└──────┴──────────┴──────────┴──────────┴────────────────┘")
    print("  For ε ≤ 50, fine-tuning with 10% data recovers < 50%.")
    print("  SFT is ineffective: all weights perturbed, no clean subset.")


# ============================================================
# Table VI: Distillation + Watermark Retention
# ============================================================

def print_table_vi(results):
    """Format Table VI from distillation results."""
    if "distillation" not in results:
        print("No distillation data found.")
        return

    rows = results["distillation"]
    print("\n┌────────────────────────────────────────────────────────┐")
    print("│  Table VI: Watermark Retention Under Distillation      │")
    print("├──────┬──────────┬───────────┬───────────┬─────────────┤")
    print("│  T   │ Surr.Acc │ WM Ret.   │ p-value   │ Verified    │")
    print("├──────┼──────────┼───────────┼───────────┼─────────────┤")

    for r in rows:
        T = r["T"]
        acc = r["acc"] * 100
        ret = r.get("retention", 0) * 100
        pv = r.get("p_value", 1.0)
        ver = r.get("verified", False)
        pv_str = f"{pv:.1e}" if pv < 0.01 else f"{pv:.4f}"
        ver_str = "  ✓  " if ver else "  ✗  "
        print(f"│  {T:>3} │  {acc:>5.1f}%  │  {ret:>5.1f}%   │ {pv_str:>9s} │  {ver_str}      │")

    print("└──────┴──────────┴───────────┴───────────┴─────────────┘")
    print("  T>1: watermark retention high (sub-dominant ranks preserved).")
    print("  T=1: softmax is peaked, sub-dominant differences suppressed.")


# ============================================================
# Table VIII: Overhead
# ============================================================

def print_table_viii(results):
    if "overhead" not in results:
        print("No overhead data found.")
        return

    r = results["overhead"]
    print("\n┌──────────────────────────────────────────────────┐")
    print("│           Table VIII: Overhead                    │")
    print("├──────────────────────┬────────────────────────────┤")
    print(f"│ Obfuscation (1-time) │ {r['obf_s']:>8.3f} s                  │")
    print(f"│ Recovery (1-time)    │ {r['rec_s']:>8.3f} s                  │")
    print(f"│ Inference (per-query)│ {r['inf_ms']:>8.2f} ms                 │")
    print(f"│ Secret package |S|   │ {r['secret_bytes']:>8d} bytes              │")
    print(f"│ Model parameters     │ {r['params']:>8,}                  │")
    print("└──────────────────────┴────────────────────────────┘")


# ============================================================
# Matplotlib Plots
# ============================================================

def plot_epsilon_vs_accuracy(results, save_path="fig_epsilon_attack.png"):
    """Plot ε vs accuracy under fine-tuning attack (Table V)."""
    if not HAS_MPL or "finetune" not in results:
        return

    rows = results["finetune"]
    by_eps = {}
    for r in rows:
        eps = r["eps"]
        if eps not in by_eps:
            by_eps[eps] = {"obf": r["obf"]}
        by_eps[eps][f"ftal_{r['frac']}"] = r["ftal"]

    epsilons = sorted(by_eps.keys())
    obf_acc = [by_eps[e]["obf"] * 100 for e in epsilons]
    ftal_1 = [by_eps[e].get("ftal_0.01", 0) * 100 for e in epsilons]
    ftal_10 = [by_eps[e].get("ftal_0.1", 0) * 100 for e in epsilons]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(epsilons, obf_acc, "o-", label="M* (no attack)", color="#E24B4A")
    ax.plot(epsilons, ftal_1, "s--", label="FTAL 1% data", color="#378ADD")
    ax.plot(epsilons, ftal_10, "^--", label="FTAL 10% data", color="#1D9E75")
    ax.axhline(y=10, color="gray", linestyle=":", alpha=0.5, label="Random guess")
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fine-tuning attack resilience (CIFAR-10)")
    ax.legend()
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_temperature_vs_retention(results, save_path="fig_distillation.png"):
    """Plot distillation temperature vs watermark retention (Table VI)."""
    if not HAS_MPL or "distillation" not in results:
        return

    rows = results["distillation"]
    temps = [r["T"] for r in rows]
    accs = [r["acc"] * 100 for r in rows]
    rets = [r.get("retention", 0) * 100 for r in rows]

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))
    color_acc = "#378ADD"
    color_ret = "#D85A30"

    ax1.set_xlabel("Distillation temperature T")
    ax1.set_ylabel("Surrogate accuracy (%)", color=color_acc)
    ax1.plot(temps, accs, "o-", color=color_acc, label="Surrogate accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Watermark retention (%)", color=color_ret)
    ax2.plot(temps, rets, "s--", color=color_ret, label="WM retention")
    ax2.tick_params(axis="y", labelcolor=color_ret)

    ax1.set_title("Accuracy vs watermark retention under distillation")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EvalGuard Result Visualization")
    parser.add_argument("--file", default="results.json", help="Path to results JSON")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib figures")
    args = parser.parse_args()

    try:
        results = load_results(args.file)
    except FileNotFoundError:
        print(f"File not found: {args.file}")
        print("Run experiments first: python experiments.py --experiment all")
        sys.exit(1)

    # Print all available tables
    print_table_iv(results)
    print_table_v(results)
    print_table_vi(results)
    print_table_viii(results)

    # Generate plots
    if args.plot:
        if not HAS_MPL:
            print("\nmatplotlib not installed. pip install matplotlib")
        else:
            plot_epsilon_vs_accuracy(results)
            plot_temperature_vs_retention(results)
            print("\nAll figures saved.")


if __name__ == "__main__":
    main()
