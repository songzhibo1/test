"""
EvalGuard — Baselines (Section VI-A)

1. DAWN [Szyller et al., ACM MM 2021]:
   Flips top-1 prediction for fraction r_w of queries.
   Fidelity loss ≈ r_w (Table IV).
   Watermark survives soft-label distillation but degrades accuracy.

2. No Protection:
   Original model in TEE, no obfuscation, no watermark.
   Provides overhead lower bound.
"""
from __future__ import annotations

import hmac
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn

from .watermark_math import binomial_p_value


# ============================================================
# DAWN Baseline
# ============================================================

@dataclass
class DAWNTriggerEntry:
    """DAWN trigger: query + original top-1 + flipped class."""
    query: torch.Tensor
    original_class: int
    flipped_class: int


@dataclass
class DAWNWatermark:
    """
    DAWN: flip top-1 for r_w fraction of queries.

    Unlike EvalGuard which only permutes sub-dominant ranks,
    DAWN modifies the argmax → fidelity loss ≈ r_w.

    Parameters:
        K_w: watermark key for deterministic decision
        r_w: flip ratio (default 0.5%)
    """
    K_w: bytes
    r_w: float = 0.005
    trigger_set: List[DAWNTriggerEntry] = field(default_factory=list)

    def embed(self, x: torch.Tensor, p: np.ndarray) -> np.ndarray:
        """
        DAWN embedding: if Φ(x)=1, swap top-1 and top-2 probabilities.

        Returns modified probability vector.
        """
        x_hash = hashlib.sha256(x.cpu().numpy().tobytes()).digest()
        h = hmac.new(self.K_w, x_hash, hashlib.sha256).digest()
        decision_val = int.from_bytes(h[:16], "big")
        threshold = int(self.r_w * (2 ** 128))

        if decision_val >= threshold:
            return p  # No watermark

        q = p.copy()
        rho = np.argsort(-p)
        # Swap top-1 and top-2 probability values
        q[rho[0]], q[rho[1]] = p[rho[1]], p[rho[0]]

        self.trigger_set.append(DAWNTriggerEntry(
            query=x.clone(),
            original_class=int(rho[0]),
            flipped_class=int(rho[1]),
        ))
        return q

    def verify(
        self,
        suspect_model: nn.Module,
        device: str = "cpu",
        eta: float = 2 ** (-64),
        num_classes: int = 10,
    ) -> Dict:
        """
        DAWN verification: check if suspect model returns the flipped class.

        Args:
            num_classes: total number of classes (used for the H0 probability
                         p0 = 1/C). Defaults to 10 for CIFAR-10.
        """
        suspect_model.to(device).eval()
        n_match = 0

        for entry in self.trigger_set:
            x = entry.query.unsqueeze(0).to(device) if entry.query.dim() == 3 else entry.query.to(device)
            with torch.no_grad():
                pred = suspect_model(x).argmax(dim=-1).item()
            # DAWN checks if the suspect returns the flipped class
            if pred == entry.flipped_class:
                n_match += 1

        n_total = len(self.trigger_set)
        # Under H_0, probability of returning any specific class ≈ 1/C
        p0 = 1.0 / num_classes
        p_value = binomial_p_value(n_match, n_total, p0) if n_total > 0 else 1.0

        return {
            "verified": p_value < eta,
            "n_match": n_match,
            "n_total": n_total,
            "match_rate": n_match / n_total if n_total > 0 else 0.0,
            "p_value": p_value,
        }


def measure_dawn_fidelity(
    model: nn.Module,
    testloader,
    K_w: bytes,
    r_w: float = 0.005,
    device: str = "cpu",
) -> Dict:
    """
    Measure DAWN's fidelity loss: fraction of queries where
    the watermarked output disagrees with the original top-1.

    Expected: fidelity_loss ≈ r_w (Table IV).
    """
    model.to(device).eval()
    dawn = DAWNWatermark(K_w=K_w, r_w=r_w)

    total = 0
    flipped = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for i in range(inputs.size(0)):
                original_pred = np.argmax(probs[i])
                q = dawn.embed(inputs[i], probs[i])
                watermarked_pred = np.argmax(q)

                if watermarked_pred != original_pred:
                    flipped += 1
                total += 1

    return {
        "total_queries": total,
        "flipped": flipped,
        "fidelity_loss": flipped / total if total > 0 else 0.0,
        "expected_loss": r_w,
        "n_triggers": len(dawn.trigger_set),
    }


# ============================================================
# Comparison Table (Table IX)
# ============================================================

COMPARISON_TABLE = {
    "NNSplitter":  {"type": "Obfuscation",  "TF": True,  "fid": True,  "E01": "~", "E2": False, "verif": None},
    "O2Splitter":  {"type": "Obfuscation",  "TF": True,  "fid": True,  "E01": True, "E2": False, "verif": None},
    "Uchida":      {"type": "Param WM",     "TF": False, "fid": True,  "E01": False,"E2": False, "verif": "WB"},
    "Adi":         {"type": "Param WM",     "TF": False, "fid": "~",   "E01": False,"E2": False, "verif": "BB"},
    "EWE":         {"type": "Param WM",     "TF": False, "fid": "~",   "E01": False,"E2": "~",   "verif": "BB"},
    "DAWN":        {"type": "Output WM",    "TF": True,  "fid": False, "E01": False,"E2": True,  "verif": "BB"},
    "GINSEW":      {"type": "Output WM",    "TF": True,  "fid": True,  "E01": False,"E2": True,  "verif": "WB"},
    "EaaW":        {"type": "Output WM",    "TF": True,  "fid": True,  "E01": False,"E2": True,  "verif": "WB"},
    "ModelGuard":  {"type": "Perturbation", "TF": True,  "fid": False, "E01": False,"E2": "~",   "verif": None},
    "EvalGuard":   {"type": "Obf+WM",      "TF": True,  "fid": True,  "E01": True, "E2": True,  "verif": "BB"},
}


def print_comparison_table():
    """Print Table IX from the paper."""
    header = f"{'Method':<12} {'Type':<14} {'TF':>3} {'Fid':>4} {'E0/1':>5} {'E2':>4} {'Verif':>6}"
    print(header)
    print("-" * len(header))
    for name, attrs in COMPARISON_TABLE.items():
        def fmt(v):
            if v is True: return "✓"
            if v is False: return "✗"
            if v is None: return "✗"
            return str(v)
        print(f"{name:<12} {attrs['type']:<14} {fmt(attrs['TF']):>3} "
              f"{fmt(attrs['fid']):>4} {fmt(attrs['E01']):>5} "
              f"{fmt(attrs['E2']):>4} {fmt(attrs['verif']):>6}")