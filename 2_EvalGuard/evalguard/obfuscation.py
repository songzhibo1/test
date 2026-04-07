"""
EvalGuard Phase 1: Weight Obfuscation (Algorithm 1)

Core idea: Add PRF-compressed Gaussian noise to EVERY weight.
- O(1) secret storage: only (K_obf, σ, K_w) regardless of model size
- Satisfies (ε, δ + 2δ_PRF)-DP (Theorem 1)
- Recovery is exact: same PRF key → same noise → subtract to recover

Eq. (4): s_i = PRF(K_obf, ID_i),  m_i = σ · N(0, I; s_i)
Eq. (5): w*_i = w_i + m_i
Eq. (6): w_i = w*_i - σ · N(0, I; PRF(K_obf, ID_i))
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .crypto import keygen, kdf, prf_to_seed, generate_gaussian_noise


@dataclass
class SecretPackage:
    """
    S = {K_obf, σ, K_w} — provisioned into TEE via remote attestation.
    O(1) storage regardless of model size.
    """
    K_obf: bytes       # Obfuscation PRF key
    sigma: float       # Noise scale
    K_w: bytes          # Watermark key, derived: K_w = KDF(K_obf, "watermark")
    model_id: str       # Unique model identifier for ID_i construction


def compute_l2_sensitivity(model: nn.Module) -> float:
    """
    Definition 2: ℓ2 local sensitivity.
    Δ_{g,2} = max_{W,W'} ||g(W) - g(W')||_2

    For identity release function g(w_i) = w_i, the sensitivity is
    the range of each weight coordinate over the trained model.
    """
    max_abs = 0.0
    for param in model.parameters():
        max_abs = max(max_abs, param.data.abs().max().item())
    return 2.0 * max_abs


def compute_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """
    Gaussian mechanism noise scale (Definition 3):
    σ = (Δ_{g,2} / ε) * sqrt(2 * ln(1.25/δ))
    """
    return (sensitivity / epsilon) * math.sqrt(2.0 * math.log(1.25 / delta))


def make_weight_id(model_id: str, layer_idx: int, weight_idx: int) -> bytes:
    """
    ID_i = (model_id || layer_idx || weight_idx)
    Algorithm 1, line 5.
    """
    return f"{model_id}:{layer_idx}:{weight_idx}".encode("utf-8")


def obfuscate_model(
    model: nn.Module,
    epsilon: float = 50.0,
    delta: float = 2 ** (-32),
    security_param: int = 256,
    model_id: str = "model_0",
) -> tuple:
    """
    Algorithm 1: Weight Obfuscation (per-weight PRF, faithful to paper).

    Input:  Model M with weights W, parameters (ε, δ, λ)
    Output: Obfuscated model M*, secret package S
    """
    K_obf = keygen(security_param)
    sensitivity = compute_l2_sensitivity(model)
    sigma = compute_sigma(sensitivity, epsilon, delta)
    K_w = kdf(K_obf, "watermark")

    secret = SecretPackage(K_obf=K_obf, sigma=sigma, K_w=K_w, model_id=model_id)

    with torch.no_grad():
        for layer_idx, (name, param) in enumerate(model.named_parameters()):
            shape = param.data.shape
            flat = param.data.cpu().numpy().flatten()
            for weight_idx in range(len(flat)):
                id_bytes = make_weight_id(model_id, layer_idx, weight_idx)
                seed = prf_to_seed(K_obf, id_bytes)
                noise = generate_gaussian_noise(seed, (1,), sigma)
                flat[weight_idx] += noise[0]
            param.data = torch.from_numpy(flat.reshape(shape)).to(param.device)

    return model, secret


def obfuscate_model_vectorized(
    model: nn.Module,
    epsilon: float = 50.0,
    delta: float = 2 ** (-32),
    security_param: int = 256,
    model_id: str = "model_0",
) -> tuple:
    """
    Vectorized (fast) version of Algorithm 1.

    Per-layer PRF call instead of per-weight. Practical optimization:
    per-weight HMAC is O(n) calls, too slow for millions of weights.
    """
    K_obf = keygen(security_param)
    sensitivity = compute_l2_sensitivity(model)
    sigma = compute_sigma(sensitivity, epsilon, delta)
    K_w = kdf(K_obf, "watermark")

    secret = SecretPackage(K_obf=K_obf, sigma=sigma, K_w=K_w, model_id=model_id)

    with torch.no_grad():
        for layer_idx, (name, param) in enumerate(model.named_parameters()):
            layer_id = make_weight_id(model_id, layer_idx, 0)
            seed = prf_to_seed(K_obf, layer_id)
            noise = generate_gaussian_noise(seed, param.shape, sigma)
            param.data += torch.from_numpy(noise).float().to(param.device)

    return model, secret


def recover_weights(
    model: nn.Module,
    secret: SecretPackage,
    vectorized: bool = True,
):
    """
    Eq. (6): Weight recovery inside TEE.
    w_i = w*_i - σ · N(0, I; PRF(K_obf, ID_i))

    Deterministic: same key + same ID → same noise → exact subtraction.
    """
    with torch.no_grad():
        for layer_idx, (name, param) in enumerate(model.named_parameters()):
            if vectorized:
                layer_id = make_weight_id(secret.model_id, layer_idx, 0)
                seed = prf_to_seed(secret.K_obf, layer_id)
                noise = generate_gaussian_noise(seed, param.shape, secret.sigma)
                param.data -= torch.from_numpy(noise).float().to(param.device)
            else:
                shape = param.data.shape
                flat = param.data.cpu().numpy().flatten()
                for weight_idx in range(len(flat)):
                    id_bytes = make_weight_id(secret.model_id, layer_idx, weight_idx)
                    seed = prf_to_seed(secret.K_obf, id_bytes)
                    noise = generate_gaussian_noise(seed, (1,), secret.sigma)
                    flat[weight_idx] -= noise[0]
                param.data = torch.from_numpy(flat.reshape(shape)).to(param.device)
