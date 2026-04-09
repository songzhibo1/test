"""
EvalGuard Phase 2 & 3: Output Watermark Embedding (Algorithm 2)
                        + Ownership Verification (Algorithm 3)

[v5] Logit-Space Confidence Shift Watermarking:
  Replaces probability-space shift with logit-space shift.

  Key changes from v4:
  - Embedding: add delta_logit to target class LOGIT before softmax
    instead of adding delta to probability after softmax
  - This makes the watermark signal T-invariant: same logit shift
    regardless of distillation temperature
  - No manual clip/renormalize needed (softmax handles it)
  - Verification: unchanged (Mann-Whitney U test)

  Design rationale:
  - v4 probability-space shift works at T=1 but vanishes at T>=5
    because softmax(logits/T) compresses sub-dominant probabilities
  - Logit-space shift is preserved regardless of T: student learns
    the logit offset directly via KL divergence
"""
from __future__ import annotations

import hmac
import hashlib
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from scipy import stats

import numpy as np
import torch
import torch.nn as nn

from .crypto import prf


# ============================================================
# Perturbation-Resilient Mapping R(x) — Eq. (11)
# ============================================================

@dataclass
class LatentExtractor:
    """
    Eq. (11): R(x) = Binarize(L(f, x), μ)
    Unchanged from v3/v4.
    """
    median: np.ndarray = None

    def compute_median(self, model: nn.Module, dataloader, layer_name: str, device: str = "cpu"):
        latents = []
        def hook_fn(module, input, output):
            out = output.detach().cpu()
            if out.dim() > 2:
                out = out.flatten(start_dim=2).mean(dim=2)
            latents.append(out.numpy())
        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        model.to(device).eval()
        with torch.no_grad():
            for batch_x, _ in dataloader:
                model(batch_x.to(device))
        handle.remove()
        all_latents = np.concatenate(latents, axis=0)
        self.median = np.median(all_latents, axis=0)

    def extract_and_binarize(self, model: nn.Module, x: torch.Tensor, layer_name: str, device: str = "cpu") -> bytes:
        latent = None
        def hook_fn(module, input, output):
            nonlocal latent
            out = output.detach().cpu()
            if out.dim() > 2:
                out = out.flatten(start_dim=2).mean(dim=2)
            latent = out.numpy()[0]
        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        model.eval()
        with torch.no_grad():
            inp = x.unsqueeze(0).to(device) if x.dim() == 3 else x.to(device)
            model(inp)
        handle.remove()
        binary = (latent > self.median).astype(np.uint8)
        return np.packbits(binary).tobytes()

    def extract_and_binarize_batch(self, model: nn.Module, batch_x: torch.Tensor, layer_name: str, device: str = "cpu"):
        batch_latent = None
        def hook_fn(module, input, output):
            nonlocal batch_latent
            out = output.detach().cpu()
            if out.dim() > 2:
                out = out.flatten(start_dim=2).mean(dim=2)
            batch_latent = out.numpy()
        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        model.eval()
        with torch.no_grad():
            model(batch_x.to(device))
        handle.remove()
        results = []
        for i in range(batch_latent.shape[0]):
            binary = (batch_latent[i] > self.median).astype(np.uint8)
            results.append(np.packbits(binary).tobytes())
        return results


# ============================================================
# Target-Class Mapping — Definition 8
# ============================================================

def derive_target_class(K_w: bytes, top1_class: int, num_classes: int) -> int:
    """
    Definition 8: Target-Class Mapping.

    t(c) = HMAC(K_w, "target:c") mod (C-1), adjusted to skip c itself.
    Deterministic: same key + same top-1 class → same target class.
    """
    h = hmac.new(K_w, ("target:" + str(top1_class)).encode(), hashlib.sha256).digest()
    t = int.from_bytes(h[:4], "big") % (num_classes - 1)
    if t >= top1_class:
        t += 1
    return t


def derive_control_class(K_w: bytes, top1_class: int, num_classes: int) -> int:
    """
    Deterministic control class for paired verification.

    Picks a class that is neither the top-1 class c nor the target class t(c),
    using a domain-separated HMAC so it is reproducible across runs and
    cryptographically tied to K_w.

    Used by verify_ownership to construct the paired control sample for
    Wilcoxon signed-rank testing.
    """
    target = derive_target_class(K_w, top1_class, num_classes)
    h = hmac.new(K_w, ("control:" + str(top1_class)).encode(), hashlib.sha256).digest()
    # Pick from C-2 candidates (excluding c and t(c))
    raw = int.from_bytes(h[:4], "big") % (num_classes - 2)
    # Map raw index in [0, C-2) onto the C classes minus {c, t(c)}
    excluded = sorted({top1_class, target})
    cls = raw
    for ex in excluded:
        if cls >= ex:
            cls += 1
    return cls


# ============================================================
# Algorithm 2: Logit-Space Confidence Shift Watermark Embedding
# ============================================================

@dataclass
class TriggerEntry:
    """One entry in trigger set T."""
    query: torch.Tensor
    target_class: int
    top1_class: int


@dataclass
class WatermarkModule:
    """
    Implements Algorithm 2: Logit-Space Confidence Shift Embedding.

    For selected queries (decided by HMAC), boost the target class LOGIT
    by delta_logit before applying softmax. This produces a T-invariant
    watermark signal that survives distillation at any temperature.

    Parameters:
        K_w: watermark key
        r_w: watermark ratio (default 0.005)
        delta_logit: logit-space shift amount (default 2.0)
        beta: safety factor to cap shift below logit margin (default 0.3)
        delta_min: minimum effective delta — queries whose safe delta would
                   fall below this threshold are NOT recorded as triggers
                   (avoids polluting the trigger set with near-zero signals
                   on low-margin samples). Default 0.5.
        num_classes: total number of classes
        latent_extractor: for R(x) computation
        layer_name: which hidden layer for latent extraction
    """
    K_w: bytes
    r_w: float = 0.005
    delta_logit: float = 2.0
    beta: float = 0.3
    delta_min: float = 0.5
    num_classes: int = 10
    latent_extractor: LatentExtractor = None
    layer_name: str = ""
    trigger_set: List[TriggerEntry] = field(default_factory=list)
    _target_class_cache: dict = field(default_factory=dict, repr=False)

    def _get_target_class(self, top1_class: int) -> int:
        if top1_class not in self._target_class_cache:
            self._target_class_cache[top1_class] = derive_target_class(
                self.K_w, top1_class, self.num_classes)
        return self._target_class_cache[top1_class]

    def watermark_decision(self, r_x: bytes) -> bool:
        """Eq. (8): Φ(x) = 1 if HMAC(K_w, R(x))[0:128] < r_w × 2^128"""
        h = hmac.new(self.K_w, r_x, hashlib.sha256).digest()
        decision_value = int.from_bytes(h[:16], "big")
        threshold = int(self.r_w * (2 ** 128))
        return decision_value < threshold

    def _compute_delta_logit(self, logits: np.ndarray) -> float:
        """
        Compute the safe logit shift, ensuring top-1 is not flipped.

        delta = min(delta_logit, beta × logit_margin)
        where logit_margin = logit[top1] - logit[second]
        """
        sorted_logits = np.sort(logits)[::-1]
        margin = sorted_logits[0] - sorted_logits[1]
        return min(self.delta_logit, self.beta * margin)

    def embed_logits(self, model: nn.Module, x: torch.Tensor,
                     logits: np.ndarray, temperature: float = 1.0,
                     device: str = "cpu") -> np.ndarray:
        """
        Algorithm 2: Logit-Space Embedding (single sample).

        1. Decide whether to watermark this query
        2. Identify top-1 class c and target class t(c)
        3. Compute safe delta; if below delta_min, skip (do not record)
        4. Add delta to logits[t(c)]
        5. Apply softmax(logits / T) to get probabilities
        6. Record trigger entry

        Args:
            logits: raw logits for this sample (1D numpy array)
            temperature: distillation temperature T

        Returns:
            probability vector after softmax (with or without watermark)
        """
        r_x = self.latent_extractor.extract_and_binarize(model, x, self.layer_name, device)
        should_wm = self.watermark_decision(r_x)

        if not should_wm:
            # No watermark: just return softmax(logits / T)
            logits_t = torch.tensor(logits).unsqueeze(0) / temperature
            return torch.softmax(logits_t, dim=-1).squeeze(0).numpy()

        # Compute safe delta in logit space; reject low-signal samples
        delta = self._compute_delta_logit(logits)
        if delta < self.delta_min:
            logits_t = torch.tensor(logits).unsqueeze(0) / temperature
            return torch.softmax(logits_t, dim=-1).squeeze(0).numpy()

        top1_class = int(np.argmax(logits))
        target_class = self._get_target_class(top1_class)

        # Apply logit-space shift
        logits_wm = logits.copy()
        logits_wm[target_class] += delta

        # Softmax (automatically normalizes, no clip needed)
        logits_t = torch.tensor(logits_wm).unsqueeze(0) / temperature
        q = torch.softmax(logits_t, dim=-1).squeeze(0).numpy()

        # Record trigger
        self.trigger_set.append(TriggerEntry(
            query=x.clone() if isinstance(x, torch.Tensor) else x,
            target_class=target_class,
            top1_class=top1_class,
        ))
        return q

    def embed_batch_logits(self, model: nn.Module, batch_x: torch.Tensor,
                           batch_logits: np.ndarray, temperature: float = 1.0,
                           device: str = "cpu"):
        """
        Batch version of embed_logits().

        Args:
            batch_logits: raw logits (batch_size, num_classes) numpy array
            temperature: distillation temperature T

        Returns: (probabilities, n_watermarked)
        """
        r_x_list = self.latent_extractor.extract_and_binarize_batch(
            model, batch_x, self.layer_name, device)

        batch_logits_wm = batch_logits.copy()
        n_watermarked = 0

        for i in range(len(r_x_list)):
            should_wm = self.watermark_decision(r_x_list[i])
            if not should_wm:
                continue

            logits_i = batch_logits[i]

            # Compute safe delta in logit space; skip if too weak
            delta = self._compute_delta_logit(logits_i)
            if delta < self.delta_min:
                continue

            n_watermarked += 1
            top1_class = int(np.argmax(logits_i))
            target_class = self._get_target_class(top1_class)

            # Apply logit-space shift
            batch_logits_wm[i][target_class] += delta

            # Record trigger
            self.trigger_set.append(TriggerEntry(
                query=batch_x[i].clone(),
                target_class=target_class,
                top1_class=top1_class,
            ))

        # Apply softmax(logits / T) to entire batch at once
        logits_t = torch.tensor(batch_logits_wm) / temperature
        batch_probs = torch.softmax(logits_t, dim=-1).numpy()

        return batch_probs, n_watermarked

# ============================================================
# Algorithm 3: Ownership Verification (Confidence Shift)
# ============================================================

def verify_ownership(
    trigger_set: List[TriggerEntry],
    suspect_model: nn.Module,
    control_queries: torch.Tensor = None,
    control_top1_classes: List[int] = None,
    K_w: bytes = None,
    num_classes: int = 10,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    verify_temperature: float = 5.0,
    batch_size: int = 64,
) -> Dict:
    """
    Algorithm 3: Ownership Verification via Confidence Shift Detection.

    Statistical design (paired vs independent):

      1. PAIRED MODE (default — control_queries is None):
         For each trigger query, in a single forward pass we read
         BOTH p[target_class] (trigger sample) and p[control_class]
         (paired control sample), where control_class is derived
         deterministically from K_w via derive_control_class().

         Test: scipy.stats.wilcoxon(trigger - control, alternative='greater')
         (Wilcoxon signed-rank test, the correct paired analogue of
         Mann-Whitney U).

      2. INDEPENDENT MODE (control_queries provided):
         Compute p[derive_target_class(K_w, top1)] separately on a
         held-out pool of independent inputs.

         Test: scipy.stats.mannwhitneyu(trigger, control, alternative='greater').

    The verify_temperature is used to spread out sub-dominant probabilities;
    the logit-space watermark signal is T-invariant, so any T > 1 works.

    Args:
        trigger_set: list of TriggerEntry from embedding phase
        suspect_model: the model to verify
        control_queries: optional independent inputs for the control group
        control_top1_classes: optional top-1 hint for independent controls
        K_w: watermark key (REQUIRED for control_class derivation)
        num_classes: total number of classes
        eta: significance threshold
        device: computation device
        verify_temperature: T used to compute softmax during verification
        batch_size: batch size for forward passes
    """
    if K_w is None:
        raise ValueError("K_w is required for verify_ownership (control class derivation).")

    suspect_model.to(device).eval()
    vT = verify_temperature

    # ----------------------------------------------------------
    # PAIRED MODE: single forward pass per trigger, collect both
    # target_class and control_class confidences from the same pmf.
    # ----------------------------------------------------------
    if control_queries is None:
        trigger_confidences: List[float] = []
        control_confidences: List[float] = []
        if len(trigger_set) == 0:
            return _empty_verification_result(eta, vT, mode="paired_wilcoxon")

        # Batch the trigger queries to amortize forward cost
        n = len(trigger_set)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_entries = trigger_set[start:end]
                # Stack queries into a batch
                batch_x = torch.stack([
                    e.query if isinstance(e.query, torch.Tensor)
                    else torch.tensor(e.query)
                    for e in batch_entries
                ]).to(device)
                logits = suspect_model(batch_x)
                probs = torch.softmax(logits / vT, dim=-1).cpu().numpy()
                for i, entry in enumerate(batch_entries):
                    trigger_confidences.append(float(probs[i, entry.target_class]))
                    control_class = derive_control_class(
                        K_w, entry.top1_class, num_classes)
                    control_confidences.append(float(probs[i, control_class]))

        trigger_arr = np.array(trigger_confidences)
        control_arr = np.array(control_confidences)
        diffs = trigger_arr - control_arr

        # Wilcoxon signed-rank test (paired, one-sided)
        # zero_method='wilcox' drops zero differences (standard practice)
        try:
            res = stats.wilcoxon(diffs, alternative='greater', zero_method='wilcox')
            stat_val = float(res.statistic)
            p_value = float(res.pvalue)
        except ValueError:
            # All differences are zero -> no signal
            stat_val, p_value = 0.0, 1.0

        return {
            "verified": p_value < eta,
            "p_value": p_value,
            "test": "wilcoxon_signed_rank",
            "statistic": stat_val,
            "mean_trigger_conf": round(float(trigger_arr.mean()), 6),
            "mean_control_conf": round(float(control_arr.mean()), 6),
            "confidence_shift": round(float(diffs.mean()), 6),
            "n_trigger": int(len(trigger_arr)),
            "n_control": int(len(control_arr)),
            "n_pairs": int(len(diffs)),
            "eta": eta,
            "verify_temperature": vT,
        }

    # ----------------------------------------------------------
    # INDEPENDENT MODE: trigger group on trigger inputs,
    # control group on independent held-out inputs.
    # ----------------------------------------------------------
    trigger_confidences = []
    if len(trigger_set) > 0:
        n = len(trigger_set)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_entries = trigger_set[start:end]
                batch_x = torch.stack([
                    e.query if isinstance(e.query, torch.Tensor)
                    else torch.tensor(e.query)
                    for e in batch_entries
                ]).to(device)
                logits = suspect_model(batch_x)
                probs = torch.softmax(logits / vT, dim=-1).cpu().numpy()
                for i, entry in enumerate(batch_entries):
                    trigger_confidences.append(float(probs[i, entry.target_class]))

    control_confidences = []
    if isinstance(control_queries, torch.Tensor):
        n_ctrl = control_queries.size(0)
        with torch.no_grad():
            for start in range(0, n_ctrl, batch_size):
                end = min(start + batch_size, n_ctrl)
                batch_x = control_queries[start:end].to(device)
                logits = suspect_model(batch_x)
                probs = torch.softmax(logits / vT, dim=-1).cpu().numpy()
                for i in range(end - start):
                    if control_top1_classes is not None:
                        top1_c = int(control_top1_classes[start + i])
                    else:
                        top1_c = int(np.argmax(probs[i]))
                    t_c = derive_target_class(K_w, top1_c, num_classes)
                    control_confidences.append(float(probs[i, t_c]))

    trigger_arr = np.array(trigger_confidences)
    control_arr = np.array(control_confidences)

    if len(trigger_arr) > 0 and len(control_arr) > 0:
        u_stat, p_value = stats.mannwhitneyu(
            trigger_arr, control_arr, alternative='greater')
        u_stat, p_value = float(u_stat), float(p_value)
    else:
        u_stat, p_value = 0.0, 1.0

    return {
        "verified": p_value < eta,
        "p_value": p_value,
        "test": "mann_whitney_u",
        "statistic": u_stat,
        "mean_trigger_conf": round(float(trigger_arr.mean()) if len(trigger_arr) else 0.0, 6),
        "mean_control_conf": round(float(control_arr.mean()) if len(control_arr) else 0.0, 6),
        "confidence_shift": round(
            (float(trigger_arr.mean()) - float(control_arr.mean()))
            if len(trigger_arr) and len(control_arr) else 0.0,
            6),
        "n_trigger": int(len(trigger_arr)),
        "n_control": int(len(control_arr)),
        "eta": eta,
        "verify_temperature": vT,
    }


def _empty_verification_result(eta, vT, mode="paired_wilcoxon"):
    return {
        "verified": False, "p_value": 1.0, "test": mode, "statistic": 0.0,
        "mean_trigger_conf": 0.0, "mean_control_conf": 0.0,
        "confidence_shift": 0.0, "n_trigger": 0, "n_control": 0, "n_pairs": 0,
        "eta": eta, "verify_temperature": vT,
    }


# ============================================================
# Multi-design verification (diagnostic)
# ============================================================
#
# The original paired Wilcoxon verification uses a SINGLE control class per
# top-1 class (derive_control_class). With only C distinct (target, control)
# pairs (one per top-1), any structural class-similarity bias between the
# learned representation and the HMAC-derived control can either hide or
# invert the watermark signal — especially when the watermark signal is weak
# (e.g. high distillation temperature, small delta_logit).
#
# verify_ownership_all_designs runs three orthogonal control designs on the
# SAME forward pass, so a single verification call reports:
#   A. single_ctrl   — the original design (for backward compatibility)
#   B. mean_rest     — p[target] vs mean(p over all classes != top1, target)
#   C. suspect_top1  — re-derive BOTH target and control using the suspect
#                      model's own argmax (fair when teacher and suspect
#                      disagree, which happens in own_trigger mode).
#
# Comparing A/B/C lets you tell "no signal learned" from "single-control
# class-bias artefact".

def _wilcoxon_paired(trig: np.ndarray, ctrl: np.ndarray, label: str,
                     eta: float, vT: float) -> Dict:
    diffs = trig - ctrl
    try:
        r = stats.wilcoxon(diffs, alternative='greater', zero_method='wilcox')
        stat_val = float(r.statistic)
        p_value = float(r.pvalue)
    except ValueError:
        stat_val, p_value = 0.0, 1.0
    return {
        "verified": p_value < eta,
        "p_value": p_value,
        "log10_p_value": round(math.log10(max(p_value, 1e-300)), 2),
        "test": "wilcoxon_signed_rank",
        "statistic": stat_val,
        "control_design": label,
        "mean_trigger_conf": round(float(trig.mean()), 6),
        "mean_control_conf": round(float(ctrl.mean()), 6),
        "confidence_shift": round(float(diffs.mean()), 6),
        "median_shift": round(float(np.median(diffs)), 6),
        "n_trigger": int(len(trig)),
        "n_control": int(len(ctrl)),
        "n_pairs": int(len(diffs)),
        "eta": eta,
        "verify_temperature": vT,
    }


def verify_ownership_all_designs(
    trigger_set: List[TriggerEntry],
    suspect_model: nn.Module,
    K_w: bytes,
    num_classes: int = 10,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    verify_temperature: float = 5.0,
    batch_size: int = 64,
) -> Dict:
    """
    Run Wilcoxon paired verification under THREE different control designs
    on the SAME forward pass, to disentangle single-control class-bias from
    a true absence of watermark signal.

    Designs (all one-sided Wilcoxon signed-rank, alternative='greater'):
      A 'single_ctrl':  original — derive_control_class(K_w, entry.top1_class).
                        One deterministic control per top-1 class.
      B 'mean_rest':    p[target] - mean(p[c]) over c != top1 != target.
                        Robust to per-class natural bias of a single control.
      C 'suspect_top1': re-derive target AND control using the suspect model's
                        own argmax on the sample. Fair when teacher and suspect
                        disagree (important for own_trigger mode).

    Returns {'single_ctrl', 'mean_rest', 'suspect_top1'} where each value is
    a dict with the same fields as verify_ownership() paired mode, plus
    'control_design' and 'median_shift'.
    """
    suspect_model.to(device).eval()
    vT = verify_temperature
    n = len(trigger_set)
    if n == 0:
        empty = _empty_verification_result(eta, vT, mode="paired_wilcoxon")
        return {
            "single_ctrl":  dict(empty, control_design="single_ctrl"),
            "mean_rest":    dict(empty, control_design="mean_rest"),
            "suspect_top1": dict(empty, control_design="suspect_top1"),
        }

    # Single forward pass over all triggers; keep full probability vectors.
    all_probs = np.zeros((n, num_classes), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_entries = trigger_set[start:end]
            batch_x = torch.stack([
                e.query if isinstance(e.query, torch.Tensor)
                else torch.tensor(e.query)
                for e in batch_entries
            ]).to(device)
            logits = suspect_model(batch_x)
            probs = torch.softmax(logits / vT, dim=-1).cpu().numpy()
            all_probs[start:end] = probs

    suspect_top1 = all_probs.argmax(axis=1)

    A_trig = np.empty(n, dtype=np.float64)
    A_ctrl = np.empty(n, dtype=np.float64)
    B_trig = np.empty(n, dtype=np.float64)
    B_ctrl = np.empty(n, dtype=np.float64)
    C_trig = np.empty(n, dtype=np.float64)
    C_ctrl = np.empty(n, dtype=np.float64)

    # Cache per-top1 derivations to avoid repeated HMACs
    target_cache: Dict[int, int] = {}
    control_cache: Dict[int, int] = {}

    def _target(c: int) -> int:
        if c not in target_cache:
            target_cache[c] = derive_target_class(K_w, c, num_classes)
        return target_cache[c]

    def _control(c: int) -> int:
        if c not in control_cache:
            control_cache[c] = derive_control_class(K_w, c, num_classes)
        return control_cache[c]

    for i, entry in enumerate(trigger_set):
        # Design A: original single control
        cc_A = _control(entry.top1_class)
        A_trig[i] = all_probs[i, entry.target_class]
        A_ctrl[i] = all_probs[i, cc_A]

        # Design B: mean over all classes except top1 and target
        row = all_probs[i]
        denom = num_classes - 2
        rest_sum = row.sum() - row[entry.top1_class] - row[entry.target_class]
        B_trig[i] = row[entry.target_class]
        B_ctrl[i] = rest_sum / denom

        # Design C: re-derive target & control from suspect top1
        st1 = int(suspect_top1[i])
        tgt_C = _target(st1)
        cc_C = _control(st1)
        C_trig[i] = row[tgt_C]
        C_ctrl[i] = row[cc_C]

    return {
        "single_ctrl":  _wilcoxon_paired(A_trig, A_ctrl, "single_ctrl",  eta, vT),
        "mean_rest":    _wilcoxon_paired(B_trig, B_ctrl, "mean_rest",    eta, vT),
        "suspect_top1": _wilcoxon_paired(C_trig, C_ctrl, "suspect_top1", eta, vT),
    }


def verify_ownership_own_data_all_designs(
    owner_model: nn.Module,
    own_dataloader,
    suspect_model: nn.Module,
    K_w: bytes,
    num_classes: int,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    verify_temperature: float = 5.0,
    max_triggers: int = 0,
) -> Dict:
    """
    Own-data all-designs variant:
      1. Reconstruct triggers from owner_model's data (no Phi(x) filter —
         every sample becomes a candidate, see reconstruct_triggers_from_own_data).
      2. Verify against suspect_model under the three control designs.

    Returns {'single_ctrl', 'mean_rest', 'suspect_top1'} each with
    'trigger_source'='own_trigger' and 'n_own_data_scanned' populated.
    """
    triggers = reconstruct_triggers_from_own_data(
        owner_model, own_dataloader, K_w,
        r_w=0.0, num_classes=num_classes,
        latent_extractor=None, layer_name="",
        delta_logit=0.0, beta=0.0,
        device=device, max_triggers=max_triggers,
    )

    if len(triggers) == 0:
        empty = _empty_verification_result(eta, verify_temperature, mode="paired_wilcoxon")
        stub = dict(empty, trigger_source="own_trigger", n_own_data_scanned=0)
        return {
            "single_ctrl":  dict(stub, control_design="single_ctrl"),
            "mean_rest":    dict(stub, control_design="mean_rest"),
            "suspect_top1": dict(stub, control_design="suspect_top1"),
        }

    out = verify_ownership_all_designs(
        triggers, suspect_model,
        K_w=K_w, num_classes=num_classes,
        eta=eta, device=device,
        verify_temperature=verify_temperature,
    )
    for k in out:
        out[k]["trigger_source"] = "own_trigger"
        out[k]["n_own_data_scanned"] = len(triggers)
    return out


# ============================================================
# Own-Data Verification (no D_eval leakage)
# ============================================================

def reconstruct_triggers_from_own_data(
    owner_model: nn.Module,
    own_dataloader,
    K_w: bytes,
    r_w: float,
    num_classes: int,
    latent_extractor: LatentExtractor,
    layer_name: str,
    delta_logit: float = 2.0,
    beta: float = 0.3,
    device: str = "cpu",
    max_triggers: int = 0,
) -> List[TriggerEntry]:
    """
    Reconstruct trigger set from Owner's own data WITHOUT accessing D_eval.

    Key insight: the watermark signal is model-level, not per-sample.
    During evaluation, only Phi(x)=1 queries got watermarked, but the
    surrogate learns a global pattern: "when top-1 is class c, target
    class t(c) has elevated logit." This pattern generalizes to ALL
    inputs predicted as class c, regardless of whether they were
    watermarked during evaluation.

    Therefore, we do NOT filter by Phi(x). Every Owner sample with a
    valid top-1 prediction becomes a trigger for verification.

    Args:
        max_triggers: stop after collecting this many triggers.
                      0 = collect all from the dataloader.
        r_w, latent_extractor, layer_name, delta_logit, beta:
                      retained for API compatibility but NOT used
                      for filtering (no Phi(x) decision).
    """
    owner_model.to(device).eval()
    triggers = []

    with torch.no_grad():
        for batch_x, _ in own_dataloader:
            batch_x = batch_x.to(device)
            logits = owner_model(batch_x)

            for i in range(batch_x.size(0)):
                top1 = int(logits[i].argmax().item())
                target = derive_target_class(K_w, top1, num_classes)
                triggers.append(TriggerEntry(
                    query=batch_x[i].cpu().clone(),
                    target_class=target,
                    top1_class=top1,
                ))
                if max_triggers > 0 and len(triggers) >= max_triggers:
                    return triggers

    return triggers


def verify_ownership_own_data(
    owner_model: nn.Module,
    own_dataloader,
    suspect_model: nn.Module,
    K_w: bytes,
    r_w: float,
    num_classes: int,
    latent_extractor: LatentExtractor,
    layer_name: str,
    delta_logit: float = 2.0,
    beta: float = 0.3,
    eta: float = 2 ** (-64),
    device: str = "cpu",
    verify_temperature: float = 5.0,
    max_triggers: int = 0,
) -> Dict:
    """
    Full own-data verification pipeline:
      1. Reconstruct triggers from Owner's data (stop at max_triggers)
      2. Query suspect model with these triggers
      3. Mann-Whitney U test

    No D_eval samples are needed. Zero privacy leakage.

    Args:
        max_triggers: collect exactly this many triggers then stop.
                      0 = collect all from dataloader.
    """
    triggers = reconstruct_triggers_from_own_data(
        owner_model, own_dataloader, K_w, r_w, num_classes,
        latent_extractor, layer_name, delta_logit, beta, device,
        max_triggers=max_triggers,
    )

    if len(triggers) == 0:
        empty = _empty_verification_result(eta, verify_temperature)
        empty["trigger_source"] = "own_trigger"
        empty["n_own_data_scanned"] = 0
        return empty

    # Step 2 & 3: verify using reconstructed triggers (reuse existing function)
    result = verify_ownership(
        triggers, suspect_model,
        K_w=K_w, num_classes=num_classes,
        eta=eta, device=device,
        verify_temperature=verify_temperature,
    )
    result["trigger_source"] = "own_trigger"
    # Every scanned input becomes a trigger (no Phi(x) filter), so
    # the scan count equals len(triggers).
    result["n_own_data_scanned"] = len(triggers)
    return result