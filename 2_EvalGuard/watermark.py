"""
EvalGuard Phase 2 & 3: Output Watermark Embedding (Algorithm 2)
                        + Ownership Verification (Algorithm 3)

Key insight: Watermark is embedded in sub-dominant ranks (rank 2 to k+1),
NOT in top-1 prediction → zero fidelity loss (Proposition 1).

Watermark decision: Φ(x) via HMAC on binarized latent embedding R(x)
Rank permutation: Keyed Fisher-Yates on sub-dominant probability ranks
Verification: Kendall's τ + one-sided binomial test
"""
from __future__ import annotations

import hmac
import hashlib
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from scipy import stats

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from evalguard.crypto import keyed_fisher_yates


# ============================================================
# Perturbation-Resilient Mapping R(x)
# ============================================================

@dataclass
class LatentExtractor:
    """
    Eq. (11): R(x) = Binarize(L(f, x), μ)

    L(f, x) extracts latent representation from a hidden layer.
    μ is component-wise median over D_train.
    Binarize: component → 1 if > μ, else 0.

    This ensures R(x) = R(x + δ) for small perturbations δ,
    preventing the adversary from detecting watermarked queries
    via perturbation comparison.
    """
    median: np.ndarray = None  # μ: component-wise median over D_train
    hook_handle: object = None
    _latent: np.ndarray = None

    def compute_median(self, model: nn.Module, dataloader, layer_name: str, device: str = "cpu"):
        """
        Compute μ = median of L(f, x) over D_train.
        Must be called once during setup (Phase 1).
        """
        latents = []

        def hook_fn(module, input, output):
            # output: (batch, features, ...) → flatten spatial dims
            out = output.detach().cpu()
            if out.dim() > 2:
                out = out.flatten(start_dim=2).mean(dim=2)  # global avg pool
            latents.append(out.numpy())

        # Register forward hook on target layer
        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)

        model.eval()
        with torch.no_grad():
            for batch_x, _ in dataloader:
                model(batch_x.to(device))

        handle.remove()

        all_latents = np.concatenate(latents, axis=0)
        self.median = np.median(all_latents, axis=0)

    def extract_and_binarize(self, model: nn.Module, x: torch.Tensor,
                              layer_name: str, device: str = "cpu") -> bytes:
        """
        R(x) = Binarize(L(f, x), μ)
        Returns binarized embedding as bytes for HMAC input.
        """
        latent = None

        def hook_fn(module, input, output):
            nonlocal latent
            out = output.detach().cpu()
            if out.dim() > 2:
                out = out.flatten(start_dim=2).mean(dim=2)
            latent = out.numpy()[0]  # single sample

        target_layer = dict(model.named_modules())[layer_name]
        handle = target_layer.register_forward_hook(hook_fn)

        model.eval()
        with torch.no_grad():
            model(x.unsqueeze(0).to(device) if x.dim() == len(x.shape) else x.to(device))

        handle.remove()

        # Binarize: 1 if > median, else 0
        binary = (latent > self.median).astype(np.uint8)
        return np.packbits(binary).tobytes()


# ============================================================
# Algorithm 2: Output Watermark Embedding
# ============================================================

@dataclass
class TriggerEntry:
    """One entry in trigger set T."""
    query: torch.Tensor          # x
    expected_ordering: List[int] # α_x: expected sub-dominant rank ordering
    permutation: List[int]       # π applied


@dataclass
class WatermarkModule:
    """
    Implements Algorithm 2: Output Watermark Embedding.

    Parameters:
    - K_w: watermark key
    - r_w: watermark ratio (default 0.5% = 0.005)
    - k: permutation depth (default 4, gives log2(4!) ≈ 4.6 bits/sample)
    - latent_extractor: for R(x) computation
    - layer_name: which layer to extract latents from
    """
    K_w: bytes
    r_w: float = 0.005
    k: int = 4
    latent_extractor: LatentExtractor = None
    layer_name: str = ""
    trigger_set: List[TriggerEntry] = field(default_factory=list)

    def watermark_decision(self, r_x: bytes) -> Tuple[bool, bytes]:
        """
        Eq. (8): Φ(x) = 1 if HMAC(K_w, R(x))[0:128] < r_w × 2^128

        Returns: (should_watermark, full_hmac_digest)
        """
        h = hmac.new(self.K_w, r_x, hashlib.sha256).digest()
        # First 128 bits = first 16 bytes
        decision_value = int.from_bytes(h[:16], "big")
        threshold = int(self.r_w * (2 ** 128))
        return decision_value < threshold, h

    def embed(self, model: nn.Module, x: torch.Tensor, p: np.ndarray,
              device: str = "cpu") -> np.ndarray:
        """
        Algorithm 2: Main watermark embedding.

        Input:
            p: output probability vector f(x; W), shape (num_classes,)
            x: input query
        Output:
            q: watermarked output (same shape as p)

        Steps:
        1. r ← Binarize(L(f, x), μ)           [Eq. 11]
        2. h ← HMAC(K_w, r)
        3. If h[0:127] < r_w × 2^128:          [Eq. 8]
           a. ρ ← argsort(p, desc)
           b. K_π ← h[128:255]                  [Eq. 9]
           c. π ← FY(k, K_π)                    [Def. 6]
           d. Permute ranks 1..k (0-indexed)     [Eq. 10]
           e. Record (x, ρ[1:k+1], π) in T
        4. Else return p unchanged
        """
        # Step 1: Compute perturbation-resilient mapping
        r_x = self.latent_extractor.extract_and_binarize(
            model, x, self.layer_name, device
        )

        # Step 2-3: Watermark decision
        should_wm, h = self.watermark_decision(r_x)

        if not should_wm:
            return p  # No watermark, return original

        # Step 3a: Get descending rank ordering
        rho = np.argsort(-p)  # ρ: indices sorted by descending probability

        # Step 3b: Derive permutation key from second half of HMAC
        K_pi = h[16:32]  # h[128:255] in bits = h[16:32] in bytes

        # Step 3c: Keyed Fisher-Yates shuffle on k sub-dominant ranks
        pi = keyed_fisher_yates(self.k, K_pi)

        # Step 3d: Apply permutation (Eq. 10)
        q = p.copy()
        # rho[0] = top-1 class → preserved (first case of Eq. 10)
        # rho[1] to rho[k] = sub-dominant ranks → permute probability VALUES
        sub_ranks = rho[1 : self.k + 1]  # 0-indexed: positions 1..k
        original_probs = p[sub_ranks].copy()

        for i in range(self.k):
            q[sub_ranks[i]] = original_probs[pi[i]]

        # Step 3e: Record in trigger set T
        self.trigger_set.append(TriggerEntry(
            query=x.clone() if isinstance(x, torch.Tensor) else x,
            expected_ordering=sub_ranks.tolist(),
            permutation=pi
        ))

        return q


# ============================================================
# Algorithm 3: Ownership Verification
# ============================================================

def kendall_tau(alpha: List[int], gamma: List[int]) -> float:
    """
    Definition 7: Kendall's Rank Correlation τ.

    τ(α, γ) = (C - D) / C(k, 2)

    C: concordant pairs, D: discordant pairs.
    τ = 1 → identical orderings
    τ = -1 → reversed orderings
    """
    k = len(alpha)
    assert len(gamma) == k

    # Build rank maps
    rank_alpha = {v: i for i, v in enumerate(alpha)}
    rank_gamma = {v: i for i, v in enumerate(gamma)}

    concordant = 0
    discordant = 0
    for i in range(k):
        for j in range(i + 1, k):
            a_i, a_j = alpha[i], alpha[j]
            # In alpha, a_i has rank i, a_j has rank j (i < j)
            # In gamma, check relative order
            g_rank_i = rank_gamma.get(a_i)
            g_rank_j = rank_gamma.get(a_j)
            if g_rank_i is not None and g_rank_j is not None:
                if g_rank_i < g_rank_j:
                    concordant += 1
                else:
                    discordant += 1

    total_pairs = k * (k - 1) // 2
    if total_pairs == 0:
        return 0.0
    return (concordant - discordant) / total_pairs


def compute_null_probability(k: int, theta: float) -> float:
    """
    Compute p_0 = Pr[τ(α, γ_rand) ≥ θ] under H_0.

    Under null hypothesis: γ is a random permutation of k elements.
    Enumerate all k! permutations and count those with τ ≥ θ.

    Table II values:
    k=4, θ=1.0 → p_0 = 1/24 ≈ 4.17%
    k=4, θ≥2/3 → p_0 = 5/24 ≈ 20.8%
    """
    from itertools import permutations

    ref = list(range(k))
    total = 0
    match = 0

    for perm in permutations(range(k)):
        total += 1
        tau = kendall_tau(ref, list(perm))
        if tau >= theta:
            match += 1

    return match / total


def binomial_p_value(n_match: int, n_total: int, p0: float) -> float:
    """
    Eq. (14): One-sided binomial test p-value.

    Pr(N_match ≥ n | H_0) = Σ_{i=n}^{|T|} C(|T|, i) * p_0^i * (1-p_0)^(|T|-i)

    We reject H_0 if this is < η (e.g., 2^{-64}).
    """
    # Use survival function: P(X ≥ n) = 1 - P(X ≤ n-1)
    return stats.binom.sf(n_match - 1, n_total, p0)


def verify_ownership(
    trigger_set: List[TriggerEntry],
    suspect_model: nn.Module,
    k: int = 4,
    theta: float = 1.0,
    eta: float = 2**(-64),
    device: str = "cpu",
) -> Dict:
    """
    Algorithm 3: Ownership Verification.

    Input:
        trigger_set T with expected orderings {α_x}
        suspect model API
        threshold θ, false-positive bound η
    Output:
        VERIFIED or NOT VERIFIED

    Steps:
    1. For each x ∈ T:
       a. Query suspect model → p'
       b. ρ' ← argsort(p', desc)
       c. γ_x ← ρ'[1:k+1]
       d. If τ(α_x, γ_x) ≥ θ: match++
    2. Compute p-value via Eq. (14)
    3. If p-value < η: VERIFIED
    """
    p0 = compute_null_probability(k, theta)
    n_match = 0

    suspect_model.eval()
    for entry in trigger_set:
        x = entry.query
        if isinstance(x, torch.Tensor):
            x_input = x.unsqueeze(0).to(device) if x.dim() == 3 else x.to(device)
        else:
            x_input = torch.tensor(x).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = suspect_model(x_input)
            p_prime = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

        # Get sub-dominant ordering from suspect
        rho_prime = np.argsort(-p_prime)
        gamma_x = rho_prime[1 : k + 1].tolist()

        # Compare with expected ordering
        alpha_x = entry.expected_ordering
        tau = kendall_tau(alpha_x, gamma_x)

        if tau >= theta:
            n_match += 1

    # Compute p-value
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