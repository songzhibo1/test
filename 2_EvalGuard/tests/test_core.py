"""
EvalGuard — Unit Tests

Tests the core invariants without requiring GPU:
1. PRF / KDF / Fisher-Yates determinism
2. Gaussian noise statistics
3. Weight obfuscation → recovery = identity
4. Kendall's τ + null probability (Table II)
5. Binomial p-value (Proposition 2)
6. Watermark decision rate matches r_w
7. Rank permutation preserves top-1 (Proposition 1)

Run:
    cd EvalGuard/
    python -m pytest tests/test_core.py -v
    # or simply:
    python tests/test_core.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from evalguard.crypto import (
    prf, keygen, kdf, prf_to_seed,
    keyed_fisher_yates, generate_gaussian_noise,
)
from evalguard import kendall_tau, compute_null_probability, binomial_p_value


def test_prf_determinism():
    key = keygen(256)
    x = b"test_input_42"
    assert prf(key, x) == prf(key, x)
    assert prf(key, b"other_input") != prf(key, x)
    print("PASS  PRF determinism")


def test_keyed_fisher_yates():
    key = b"\x01" * 32
    pi = keyed_fisher_yates(4, key)
    assert sorted(pi) == [0, 1, 2, 3]
    assert keyed_fisher_yates(4, key) == pi
    for size in [3, 5, 8, 10]:
        assert sorted(keyed_fisher_yates(size, key)) == list(range(size))
    print(f"PASS  Keyed Fisher-Yates: k=4, π={pi}")


def test_fisher_yates_uniformity():
    counts = {}
    for _ in range(6000):
        pi = tuple(keyed_fisher_yates(3, keygen(256)))
        counts[pi] = counts.get(pi, 0) + 1
    assert len(counts) == 6
    for perm, count in counts.items():
        assert 500 < count < 1500
    print(f"PASS  Fisher-Yates uniformity: {len(counts)} permutations")


def test_gaussian_noise_determinism():
    a = generate_gaussian_noise(12345, (100,), 1.0)
    b = generate_gaussian_noise(12345, (100,), 1.0)
    assert np.allclose(a, b)
    c = generate_gaussian_noise(12346, (100,), 1.0)
    assert not np.allclose(a, c)
    print("PASS  Gaussian noise determinism")


def test_gaussian_noise_stats():
    noise = generate_gaussian_noise(42, (100000,), sigma=2.5)
    assert abs(noise.mean()) < 0.05
    assert abs(noise.std() - 2.5) < 0.05
    print(f"PASS  Gaussian stats: mean={noise.mean():.4f}, std={noise.std():.4f}")


def test_kdf():
    K = keygen(256)
    assert kdf(K, "watermark") != kdf(K, "other")
    assert len(kdf(K, "watermark")) == 32
    assert kdf(K, "watermark") == kdf(K, "watermark")
    print("PASS  KDF derivation")


def test_weight_obfuscation_recovery():
    K_obf = keygen(256)
    sigma = 5.0
    weights = np.random.randn(100).astype(np.float64)
    original = weights.copy()

    seeds = []
    for i in range(len(weights)):
        seed = prf_to_seed(K_obf, f"model:0:{i}".encode())
        seeds.append(seed)
        weights[i] += generate_gaussian_noise(seed, (1,), sigma)[0]

    assert not np.allclose(weights, original)

    for i in range(len(weights)):
        weights[i] -= generate_gaussian_noise(seeds[i], (1,), sigma)[0]

    assert np.allclose(weights, original, atol=1e-12)
    print("PASS  Obfuscation → recovery is exact")


def test_kendall_tau():
    assert kendall_tau([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0
    assert kendall_tau([0, 1, 2, 3], [3, 2, 1, 0]) == -1.0
    assert abs(kendall_tau([0, 1, 2, 3], [0, 2, 1, 3]) - 2 / 3) < 1e-10
    print("PASS  Kendall's τ")


def test_null_probability():
    """Table II verification."""
    assert abs(compute_null_probability(3, 1.0) - 1 / 6) < 1e-10
    assert abs(compute_null_probability(4, 1.0) - 1 / 24) < 1e-10
    assert abs(compute_null_probability(5, 1.0) - 1 / 120) < 1e-10
    print("PASS  Null probabilities match Table II")


def test_binomial_p_value():
    p8 = binomial_p_value(8, 60, 1 / 24)
    assert p8 < 0.01
    p26 = binomial_p_value(26, 60, 1 / 24)
    assert p26 < 2 ** (-64)
    assert binomial_p_value(0, 60, 1 / 24) == 1.0
    print(f"PASS  Binomial p-value: p(n=8)={p8:.2e}, p(n=26)={p26:.2e}")


def test_watermark_decision_rate():
    import hmac as _hmac, hashlib as _hashlib
    K_w, r_w = keygen(256), 0.005
    threshold = int(r_w * (2 ** 128))
    n_wm = sum(
        int.from_bytes(_hmac.new(K_w, np.random.bytes(32), _hashlib.sha256).digest()[:16], "big") < threshold
        for _ in range(100000)
    )
    rate = n_wm / 100000
    assert abs(rate - r_w) < 0.003
    print(f"PASS  Watermark rate: {rate:.4f} (expected ~{r_w})")


def test_rank_permutation_preserves_top1():
    p = np.array([0.50, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.01])
    rho = np.argsort(-p)
    pi = keyed_fisher_yates(4, keygen(256))

    q = p.copy()
    sub = rho[1:5]
    orig = p[sub].copy()
    for i in range(4):
        q[sub[i]] = orig[pi[i]]

    assert np.argmax(q) == np.argmax(p)
    assert q[np.argmax(p)] == p[np.argmax(p)]
    assert set(q[sub].tolist()) == set(orig.tolist())
    print(f"PASS  Top-1 preserved (Proposition 1), π={pi}")


def test_trigger_set_minimum():
    """Table II: |T|_min for η < 2^{-64}."""
    eta = 2 ** (-64)
    for k, p0, t_min in [(3, 1 / 6, 105), (4, 1 / 24, 60), (5, 1 / 120, 42)]:
        assert binomial_p_value(t_min, t_min, p0) < eta
    print("PASS  Minimum trigger-set sizes (Table II)")


if __name__ == "__main__":
    print("=" * 55)
    print("EvalGuard Unit Tests")
    print("=" * 55)

    test_prf_determinism()
    test_keyed_fisher_yates()
    test_fisher_yates_uniformity()
    test_gaussian_noise_determinism()
    test_gaussian_noise_stats()
    test_kdf()
    test_weight_obfuscation_recovery()
    test_kendall_tau()
    test_null_probability()
    test_binomial_p_value()
    test_watermark_decision_rate()
    test_rank_permutation_preserves_top1()
    test_trigger_set_minimum()

    print("\n" + "=" * 55)
    print("ALL 13 TESTS PASSED")
    print("=" * 55)
