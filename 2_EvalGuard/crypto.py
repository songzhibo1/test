"""
EvalGuard - Cryptographic Primitives (Definitions 4-6 in the paper)

PRF: HMAC-SHA-256
KDF: HKDF-SHA-256
Keyed Fisher-Yates Shuffle: HMAC-driven deterministic permutation
"""

import hmac
import hashlib
import struct
from typing import List

import numpy as np


def prf(key: bytes, input_data: bytes) -> bytes:
    """
    PRF(K, x) = HMAC-SHA-256(K, x)
    Definition 5: computationally indistinguishable from random for any PPT adversary.
    """
    return hmac.new(key, input_data, hashlib.sha256).digest()


def keygen(security_param: int = 256) -> bytes:
    """
    Gen(1^λ) → K: generate a λ-bit secret key.
    Definition 4.
    """
    return np.random.bytes(security_param // 8)


def kdf(key: bytes, context: str) -> bytes:
    """
    Key Derivation Function: K_w = KDF(K_obf, "watermark")
    Eq. (7). Uses HKDF-Extract then HKDF-Expand (RFC 5869) simplified.
    """
    # HKDF-Extract
    prk = hmac.new(b"\x00" * 32, key, hashlib.sha256).digest()
    # HKDF-Expand
    info = context.encode("utf-8")
    okm = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()
    return okm


def prf_to_seed(key: bytes, id_bytes: bytes) -> int:
    """
    PRF(K_obf, ID_i) → deterministic seed for numpy RNG.
    Eq. (4): s_i = PRF(K_obf, ID_i)

    Maps 32-byte HMAC output to a 128-bit integer seed for numpy SeedSequence.
    """
    h = prf(key, id_bytes)
    # Use first 16 bytes as seed (128-bit)
    return int.from_bytes(h[:16], "big")


def keyed_fisher_yates(k: int, key_pi: bytes) -> List[int]:
    """
    Definition 6: Keyed Fisher-Yates Shuffle.

    Given k elements and key K_π, produce permutation π of {0, ..., k-1}.
    For i = k-1 down to 1:
        j = HMAC(K_π, i) mod (i+1)
        swap positions i and j

    When K_π is secret, each of k! permutations is equally likely
    from the adversary's perspective.

    NOTE: Paper uses 1-indexed {1,...,k}, we use 0-indexed {0,...,k-1}.
    """
    perm = list(range(k))
    for i in range(k - 1, 0, -1):
        # j = HMAC(K_π, i) mod (i+1)
        h = hmac.new(key_pi, struct.pack(">I", i), hashlib.sha256).digest()
        j = int.from_bytes(h[:8], "big") % (i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def generate_gaussian_noise(seed: int, shape: tuple, sigma: float) -> np.ndarray:
    """
    Generate deterministic Gaussian noise from PRF seed.
    Eq. (4): m_i = σ · N(0, I; s_i)

    Same seed always produces same noise → exact weight recovery.
    """
    ss = np.random.SeedSequence(seed)
    rng = np.random.Generator(np.random.PCG64(ss))
    return sigma * rng.standard_normal(shape)