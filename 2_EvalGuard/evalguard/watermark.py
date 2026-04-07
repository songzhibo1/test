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

    t(c) = HMAC(K_w, c) mod (C-1), adjusted to skip c itself.
    Deterministic: same key + same top-1 class → same target class.
    """
    h = hmac.new(K_w, str(top1_class).encode(), hashlib.sha256).digest()
    t = int.from_bytes(h[:4], "big") % (num_classes - 1)
    if t >= top1_class:
        t += 1
    return t


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
        num_classes: total number of classes
        latent_extractor: for R(x) computation
        layer_name: which hidden layer for latent extraction
    """
    K_w: bytes
    r_w: float = 0.005
    delta_logit: float = 2.0
    beta: float = 0.3
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
        3. Add delta to logits[t(c)]
        4. Apply softmax(logits / T) to get probabilities
        5. Record trigger entry

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

        top1_class = int(np.argmax(logits))
        target_class = self._get_target_class(top1_class)

        # Compute safe delta in logit space
        delta = self._compute_delta_logit(logits)

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

            n_watermarked += 1
            logits_i = batch_logits[i]
            top1_class = int(np.argmax(logits_i))
            target_class = self._get_target_class(top1_class)

            # Compute safe delta in logit space
            delta = self._compute_delta_logit(logits_i)

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

    # Keep old interface for backward compatibility
    def embed(self, model, x, p, device="cpu"):
        """Backward-compatible wrapper. Use embed_logits for new code."""
        return self.embed_logits(model, x, p, temperature=1.0, device=device)

    def embed_batch(self, model, batch_x, batch_p, device="cpu"):
        """Backward-compatible wrapper. Use embed_batch_logits for new code."""
        return self.embed_batch_logits(model, batch_x, batch_p, temperature=1.0, device=device)


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
) -> Dict:
    """
    Algorithm 3: Ownership Verification via Confidence Shift Detection.

    Uses a verification temperature to amplify sub-dominant probability
    differences. The logit-space watermark signal is T-invariant, so
    any T > 1 that spreads out sub-dominant probabilities works.

    Args:
        trigger_set: list of TriggerEntry from embedding phase
        suspect_model: the model to verify
        control_queries: non-trigger queries for the control group
        control_top1_classes: top-1 classes for control queries
        K_w: watermark key
        num_classes: total number of classes
        eta: significance threshold
        device: computation device
        verify_temperature: T used to compute softmax during verification
            (higher T amplifies sub-dominant differences)
    """
    suspect_model.to(device).eval()
    vT = verify_temperature

    # Collect trigger group
    trigger_confidences = []

    for entry in trigger_set:
        x = entry.query
        if isinstance(x, torch.Tensor):
            x_input = x.unsqueeze(0).to(device) if x.dim() == 3 else x.to(device)
        else:
            x_input = torch.tensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = suspect_model(x_input)
            p = torch.softmax(logits / vT, dim=-1).cpu().numpy().flatten()

        trigger_confidences.append(float(p[entry.target_class]))

    # Collect control group
    control_confidences = []

    if control_queries is not None:
        if isinstance(control_queries, torch.Tensor):
            n_control = control_queries.size(0)
            for i in range(n_control):
                x = control_queries[i]
                x_input = x.unsqueeze(0).to(device) if x.dim() == 3 else x.to(device)

                with torch.no_grad():
                    logits = suspect_model(x_input)
                    p = torch.softmax(logits / vT, dim=-1).cpu().numpy().flatten()

                if control_top1_classes is not None:
                    top1_c = control_top1_classes[i]
                else:
                    top1_c = int(np.argmax(p))

                t_c = derive_target_class(K_w, top1_c, num_classes)
                control_confidences.append(float(p[t_c]))
    else:
        # Fallback: use trigger queries, read a different class as control
        for entry in trigger_set:
            x = entry.query
            if isinstance(x, torch.Tensor):
                x_input = x.unsqueeze(0).to(device) if x.dim() == 3 else x.to(device)
            else:
                x_input = torch.tensor(x).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = suspect_model(x_input)
                p = torch.softmax(logits / vT, dim=-1).cpu().numpy().flatten()

            # Pick a deterministic non-target, non-top1 class as control
            candidates = [c for c in range(num_classes)
                          if c != entry.top1_class and c != entry.target_class]
            if isinstance(x, torch.Tensor):
                hash_val = hash(x.cpu().numpy().tobytes())
            else:
                hash_val = hash(x)
            control_class = candidates[hash_val % len(candidates)]
            control_confidences.append(float(p[control_class]))

    trigger_confidences = np.array(trigger_confidences)
    control_confidences = np.array(control_confidences)

    # Mann-Whitney U test (one-sided: trigger > control)
    if len(trigger_confidences) > 0 and len(control_confidences) > 0:
        u_stat, p_value = stats.mannwhitneyu(
            trigger_confidences, control_confidences,
            alternative='greater'
        )
    else:
        u_stat, p_value = 0, 1.0

    mean_trigger = float(np.mean(trigger_confidences)) if len(trigger_confidences) > 0 else 0.0
    mean_control = float(np.mean(control_confidences)) if len(control_confidences) > 0 else 0.0

    return {
        "verified": p_value < eta,
        "p_value": float(p_value),
        "u_statistic": float(u_stat),
        "mean_trigger_conf": round(mean_trigger, 6),
        "mean_control_conf": round(mean_control, 6),
        "confidence_shift": round(mean_trigger - mean_control, 6),
        "n_trigger": len(trigger_confidences),
        "n_control": len(control_confidences),
        "eta": eta,
        "verify_temperature": vT,
    }


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
        return {
            "verified": False, "p_value": 1.0, "u_statistic": 0.0,
            "mean_trigger_conf": 0.0, "mean_control_conf": 0.0,
            "confidence_shift": 0.0, "n_trigger": 0, "n_control": 0,
            "eta": eta, "verify_temperature": verify_temperature,
            "trigger_source": "own_trigger",
        }

    # Step 2 & 3: verify using reconstructed triggers (reuse existing function)
    result = verify_ownership(
        triggers, suspect_model,
        K_w=K_w, num_classes=num_classes,
        eta=eta, device=device,
        verify_temperature=verify_temperature,
    )
    result["trigger_source"] = "own_trigger"
    result["n_own_data_scanned"] = sum(
        len(dl) for dl in [own_dataloader]) * own_dataloader.batch_size
    return result