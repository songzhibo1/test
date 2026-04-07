"""
EvalGuard Phase 2 & 3: Output Watermark Embedding (Algorithm 2)
                        + Ownership Verification (Algorithm 3)

[v3] Added amplification (alpha parameter):
  alpha=1.0: original behavior (permute only)
  alpha>1.0: after permutation, amplify gaps between sub-dominant probabilities
             to strengthen watermark signal for large num_classes

Key insight: Watermark is embedded in sub-dominant ranks (rank 2 to k+1),
NOT in top-1 prediction → zero fidelity loss (Proposition 1).
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

from .crypto import prf, keyed_fisher_yates


# ============================================================
# Perturbation-Resilient Mapping R(x) — Eq. (11)
# ============================================================

@dataclass
class LatentExtractor:
    """
    Eq. (11): R(x) = Binarize(L(f, x), μ)
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
# Amplification helper
# ============================================================

def amplify_sub_dominant(p: np.ndarray, rho: np.ndarray, permuted_indices: list,
                         pi: list, k: int, alpha: float) -> np.ndarray:
    """
    After permutation, amplify the gaps between sub-dominant probabilities.

    Args:
        p: original full probability vector
        rho: argsort indices (descending)
        permuted_indices: rho[1:k+1] — the sub-dominant class indices
        pi: permutation applied
        k: permutation depth
        alpha: amplification factor (1.0 = no amplification)

    Returns:
        q: modified probability vector with amplified sub-dominant gaps

    Method:
        1. Get permuted sub-dominant probabilities
        2. Amplify: new_p[i] = mean + alpha * (old_p[i] - mean)
        3. Clip to [eps, top1_prob * 0.95] (keep top-1 unchanged)
        4. Redistribute the probability mass difference to maintain sum=1
    """
    q = p.copy()
    sub_ranks = permuted_indices
    original_sub_probs = p[sub_ranks].copy()

    # Step 1: apply permutation
    permuted_probs = original_sub_probs[pi]

    if alpha <= 1.0:
        # No amplification, just permute
        for i in range(k):
            q[sub_ranks[i]] = permuted_probs[i]
        return q

    # Step 2: amplify around mean
    mean_p = permuted_probs.mean()
    amplified = mean_p + alpha * (permuted_probs - mean_p)

    # Step 3: clip
    eps = 1e-7
    top1_prob = p[rho[0]]
    amplified = np.clip(amplified, eps, top1_prob * 0.95)

    # Step 4: redistribute — keep top-1 and non-sub-dominant classes unchanged
    # Original sub-dominant sum
    old_sub_sum = original_sub_probs.sum()
    new_sub_sum = amplified.sum()
    # Difference needs to be absorbed by the remaining classes (rank k+2 onwards)
    diff = new_sub_sum - old_sub_sum
    remaining_indices = rho[k + 1:]  # classes outside top-1 and sub-dominant
    if len(remaining_indices) > 0 and abs(diff) > eps:
        remaining_probs = q[remaining_indices].copy()
        remaining_sum = remaining_probs.sum()
        if remaining_sum > eps:
            # Scale remaining classes to absorb the difference
            target_remaining = remaining_sum - diff
            if target_remaining > eps:
                scale = target_remaining / remaining_sum
                q[remaining_indices] = remaining_probs * scale
            else:
                # Can't absorb — reduce amplification to maintain valid distribution
                amplified = amplified * (old_sub_sum / new_sub_sum)

    # Apply amplified values
    for i in range(k):
        q[sub_ranks[i]] = amplified[i]

    return q


# ============================================================
# Algorithm 2: Output Watermark Embedding
# ============================================================

@dataclass
class TriggerEntry:
    """One entry in trigger set T."""
    query: torch.Tensor
    expected_ordering: List[int]
    permutation: List[int]


@dataclass
class WatermarkModule:
    """
    Implements Algorithm 2: Output Watermark Embedding.

    Parameters:
        K_w: watermark key
        r_w: watermark ratio (default 0.5% = 0.005)
        k: permutation depth (default 4)
        alpha: amplification factor (default 1.0 = no amplification)
               alpha > 1.0 amplifies sub-dominant probability gaps
               Recommended: alpha=3~10 for num_classes >= 100
        latent_extractor: for R(x) computation
        layer_name: which hidden layer for latent extraction
    """
    K_w: bytes
    r_w: float = 0.005
    k: int = 4
    alpha: float = 1.0
    latent_extractor: LatentExtractor = None
    layer_name: str = ""
    trigger_set: List[TriggerEntry] = field(default_factory=list)

    def watermark_decision(self, r_x: bytes) -> Tuple[bool, bytes]:
        """Eq. (8): Φ(x) = 1 if HMAC(K_w, R(x))[0:128] < r_w × 2^128"""
        h = hmac.new(self.K_w, r_x, hashlib.sha256).digest()
        decision_value = int.from_bytes(h[:16], "big")
        threshold = int(self.r_w * (2 ** 128))
        return decision_value < threshold, h

    def embed(self, model: nn.Module, x: torch.Tensor, p: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        Algorithm 2 main entry (single sample).
        If alpha > 1.0, applies amplification after permutation.
        """
        r_x = self.latent_extractor.extract_and_binarize(model, x, self.layer_name, device)
        should_wm, h = self.watermark_decision(r_x)

        if not should_wm:
            return p

        rho = np.argsort(-p)
        K_pi = h[16:32]
        pi = keyed_fisher_yates(self.k, K_pi)

        sub_ranks = rho[1: self.k + 1]

        if self.alpha > 1.0:
            q = amplify_sub_dominant(p, rho, sub_ranks, pi, self.k, self.alpha)
        else:
            q = p.copy()
            original_probs = p[sub_ranks].copy()
            for i in range(self.k):
                q[sub_ranks[i]] = original_probs[pi[i]]

        # Record the PERMUTED ordering: re-sort sub_ranks by watermarked probabilities
        # This is what the student will learn via distillation
        permuted_probs = q[sub_ranks]
        permuted_order = np.argsort(-permuted_probs)
        expected = sub_ranks[permuted_order]

        self.trigger_set.append(TriggerEntry(
            query=x.clone() if isinstance(x, torch.Tensor) else x,
            expected_ordering=expected.tolist(),
            permutation=pi,
        ))
        return q

    def embed_batch(self, model: nn.Module, batch_x: torch.Tensor, batch_p: np.ndarray, device: str = "cpu"):
        """
        Batch version of embed().
        If alpha > 1.0, applies amplification after permutation.
        """
        r_x_list = self.latent_extractor.extract_and_binarize_batch(
            model, batch_x, self.layer_name, device)

        batch_q = batch_p.copy()
        n_watermarked = 0

        for i in range(len(r_x_list)):
            should_wm, h = self.watermark_decision(r_x_list[i])
            if not should_wm:
                continue

            n_watermarked += 1
            p = batch_p[i]
            rho = np.argsort(-p)
            K_pi = h[16:32]
            pi = keyed_fisher_yates(self.k, K_pi)

            sub_ranks = rho[1: self.k + 1]

            if self.alpha > 1.0:
                batch_q[i] = amplify_sub_dominant(p, rho, sub_ranks, pi, self.k, self.alpha)
            else:
                original_probs = p[sub_ranks].copy()
                for j in range(self.k):
                    batch_q[i, sub_ranks[j]] = original_probs[pi[j]]

            # Record the PERMUTED ordering: re-sort sub_ranks by watermarked probabilities
            permuted_probs = batch_q[i, sub_ranks]
            permuted_order = np.argsort(-permuted_probs)
            expected = sub_ranks[permuted_order]

            self.trigger_set.append(TriggerEntry(
                query=batch_x[i].clone(),
                expected_ordering=expected.tolist(),
                permutation=pi,
            ))

        return batch_q, n_watermarked


# ============================================================
# Algorithm 3: Ownership Verification
# ============================================================

def kendall_tau(alpha: List[int], gamma: List[int]) -> float:
    """Definition 7: Kendall's Rank Correlation τ."""
    k = len(alpha)
    assert len(gamma) == k
    rank_gamma = {v: i for i, v in enumerate(gamma)}
    concordant, discordant = 0, 0
    for i in range(k):
        for j in range(i + 1, k):
            g_rank_i = rank_gamma.get(alpha[i])
            g_rank_j = rank_gamma.get(alpha[j])
            if g_rank_i is not None and g_rank_j is not None:
                if g_rank_i < g_rank_j:
                    concordant += 1
                else:
                    discordant += 1
    total_pairs = k * (k - 1) // 2
    return (concordant - discordant) / total_pairs if total_pairs else 0.0


def compute_null_probability(k: int, theta: float) -> float:
    """Compute p_0 = Pr[τ(α, γ_rand) ≥ θ] under H_0. (Table II)"""
    from itertools import permutations
    ref = list(range(k))
    total, match = 0, 0
    for perm in permutations(range(k)):
        total += 1
        if kendall_tau(ref, list(perm)) >= theta:
            match += 1
    return match / total


def binomial_p_value(n_match: int, n_total: int, p0: float) -> float:
    """Eq. (14): One-sided binomial test p-value."""
    return stats.binom.sf(n_match - 1, n_total, p0)


def verify_ownership(
    trigger_set: List[TriggerEntry],
    suspect_model: nn.Module,
    k: int = 4,
    theta: float = 1.0,
    eta: float = 2 ** (-64),
    device: str = "cpu",
) -> Dict:
    """Algorithm 3: Ownership Verification."""
    p0 = compute_null_probability(k, theta)
    n_match = 0
    suspect_model.to(device).eval()

    for entry in trigger_set:
        x = entry.query
        if isinstance(x, torch.Tensor):
            x_input = x.unsqueeze(0).to(device) if x.dim() == 3 else x.to(device)
        else:
            x_input = torch.tensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = suspect_model(x_input)
            p_prime = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

        rho_prime = np.argsort(-p_prime)
        gamma_x = rho_prime[1: k + 1].tolist()
        alpha_x = entry.expected_ordering
        tau = kendall_tau(alpha_x, gamma_x)

        if tau >= theta:
            n_match += 1

    p_value = binomial_p_value(n_match, len(trigger_set), p0)

    return {
        "verified": p_value < eta,
        "n_match": n_match,
        "n_total": len(trigger_set),
        "p_value": p_value,
        "p0": p0,
        "eta": eta,
        "match_rate": n_match / len(trigger_set) if trigger_set else 0.0,
    }