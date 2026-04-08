"""
EvalGuard — Unit Tests (v6)

Covers the v5/v6 logit-space confidence-shift watermark and the
paired Wilcoxon verification path.

What is tested (CPU-only, no real datasets):
  1. Crypto primitives — PRF, KDF, keyed Fisher-Yates, Gaussian noise
  2. Weight obfuscation -> recovery is exact
  3. derive_target_class:    determinism, never maps c -> c, range
  4. derive_control_class:   determinism, disjoint from {c, t(c)}
  5. _compute_delta_logit:   safe under beta cap, respects delta_logit cap,
                             never flips top-1 after embedding
  6. WatermarkModule.embed_batch_logits:
        - top-1 of returned probabilities equals top-1 of input logits
        - n_watermarked rate ~ r_w (large-N statistical check)
        - delta_min filter rejects low-margin samples
  7. verify_ownership (paired Wilcoxon):
        - clean model -> p_value not significant (no false positive)
        - watermarked teacher -> p_value < eta (true positive)

Run:
    cd 2_EvalGuard/
    python tests/test_core.py
    # or:
    python -m pytest tests/test_core.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import torch
import torch.nn as nn

from evalguard.crypto import (
    prf, keygen, kdf, prf_to_seed,
    keyed_fisher_yates, generate_gaussian_noise,
)
from evalguard.watermark_math import binomial_p_value
from evalguard.watermark import (
    LatentExtractor, WatermarkModule, TriggerEntry,
    derive_target_class, derive_control_class,
    verify_ownership,
)


# ============================================================
# 1. Crypto primitives
# ============================================================

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
    print("PASS  Keyed Fisher-Yates: k=4, pi={}".format(pi))


def test_fisher_yates_uniformity():
    counts = {}
    for _ in range(6000):
        pi = tuple(keyed_fisher_yates(3, keygen(256)))
        counts[pi] = counts.get(pi, 0) + 1
    assert len(counts) == 6
    for perm, count in counts.items():
        assert 500 < count < 1500
    print("PASS  Fisher-Yates uniformity: {} permutations".format(len(counts)))


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
    print("PASS  Gaussian stats: mean={:.4f}, std={:.4f}".format(noise.mean(), noise.std()))


def test_kdf():
    K = keygen(256)
    assert kdf(K, "watermark") != kdf(K, "other")
    assert len(kdf(K, "watermark")) == 32
    assert kdf(K, "watermark") == kdf(K, "watermark")
    print("PASS  KDF derivation")


def test_keygen_uses_csprng():
    # Should produce distinct outputs every call (no PRNG seeding accident)
    keys = {keygen(256) for _ in range(1000)}
    assert len(keys) == 1000
    print("PASS  keygen() returns 1000 distinct keys")


# ============================================================
# 2. Weight obfuscation -> recovery
# ============================================================

def test_weight_obfuscation_recovery():
    K_obf = keygen(256)
    sigma = 5.0
    weights = np.random.randn(100).astype(np.float64)
    original = weights.copy()

    seeds = []
    for i in range(len(weights)):
        seed = prf_to_seed(K_obf, "model:0:{}".format(i).encode())
        seeds.append(seed)
        weights[i] += generate_gaussian_noise(seed, (1,), sigma)[0]

    assert not np.allclose(weights, original)

    for i in range(len(weights)):
        weights[i] -= generate_gaussian_noise(seeds[i], (1,), sigma)[0]

    assert np.allclose(weights, original, atol=1e-12)
    print("PASS  Obfuscation -> recovery is exact")


# ============================================================
# 3. derive_target_class / derive_control_class
# ============================================================

def test_derive_target_class_determinism_and_no_self_map():
    K = keygen(256)
    for C in (10, 100, 1000):
        for c in range(C):
            t1 = derive_target_class(K, c, C)
            t2 = derive_target_class(K, c, C)
            assert t1 == t2, "non-deterministic at c={}".format(c)
            assert 0 <= t1 < C
            assert t1 != c, "target class collides with top1 at c={}".format(c)
    print("PASS  derive_target_class: deterministic, no self-map (C=10/100/1000)")


def test_derive_target_class_distribution():
    # Over many keys, the marginal distribution should be roughly uniform
    C = 10
    counts = [0] * C
    n_keys = 5000
    for _ in range(n_keys):
        K = keygen(256)
        t = derive_target_class(K, 0, C)
        counts[t] += 1
    counts[0] = 0  # cannot be 0 by construction
    expected = n_keys / (C - 1)
    for i, cnt in enumerate(counts):
        if i == 0:
            assert cnt == 0
            continue
        # Within +-30% of expected (loose, n=5000)
        assert 0.7 * expected < cnt < 1.3 * expected, \
            "target class {} count {} far from expected {}".format(i, cnt, expected)
    print("PASS  derive_target_class distribution within +-30% of uniform")


def test_derive_control_class_disjoint():
    K = keygen(256)
    for C in (10, 100):
        for c in range(C):
            t = derive_target_class(K, c, C)
            ctrl = derive_control_class(K, c, C)
            assert 0 <= ctrl < C
            assert ctrl != c, "control collides with top1 at c={}".format(c)
            assert ctrl != t, "control collides with target at c={}".format(c)
            # Determinism
            assert ctrl == derive_control_class(K, c, C)
    print("PASS  derive_control_class: deterministic, disjoint from {c, t(c)}")


# ============================================================
# 4. _compute_delta_logit safety
# ============================================================

def test_compute_delta_logit_caps_at_beta_margin():
    wm = WatermarkModule(K_w=keygen(256), delta_logit=5.0, beta=0.3, delta_min=0.0)
    # margin = 10 - 4 = 6, beta*margin = 1.8 < 5.0 -> delta = 1.8
    logits = np.array([10.0, 4.0, 1.0, 0.0])
    assert abs(wm._compute_delta_logit(logits) - 1.8) < 1e-9
    print("PASS  _compute_delta_logit capped by beta * margin")


def test_compute_delta_logit_caps_at_delta_logit():
    wm = WatermarkModule(K_w=keygen(256), delta_logit=2.0, beta=0.9, delta_min=0.0)
    # margin = 10 - 0 = 10, beta*margin = 9.0 > 2.0 -> delta = 2.0
    logits = np.array([10.0, 0.0, -1.0])
    assert abs(wm._compute_delta_logit(logits) - 2.0) < 1e-9
    print("PASS  _compute_delta_logit capped by delta_logit")


# ============================================================
# 5. WatermarkModule.embed_batch_logits
# ============================================================

class _FakeLatentExtractor(LatentExtractor):
    """Hands out unique random bytes per sample so watermark_decision
    behaves uniformly without needing a real model hook."""

    def extract_and_binarize_batch(self, model, batch_x, layer_name, device="cpu"):
        out = []
        for i in range(batch_x.size(0)):
            out.append(os.urandom(16))
        return out


class _IdentityModel(nn.Module):
    """Lets WatermarkModule call .eval() and produces no actual hook events."""
    def forward(self, x):
        return x


def test_embed_batch_does_not_flip_top1():
    torch.manual_seed(0)
    np.random.seed(0)
    n, C = 200, 10
    # Generate logits with a clear top-1 and a non-trivial margin
    base = np.random.randn(n, C) * 0.5
    base[np.arange(n), np.random.randint(0, C, size=n)] += 5.0  # boost a random class
    original_top1 = base.argmax(axis=1)

    wm = WatermarkModule(
        K_w=keygen(256), r_w=1.0,           # always-on for this test
        delta_logit=2.0, beta=0.3,
        delta_min=0.0,
        num_classes=C,
        latent_extractor=_FakeLatentExtractor(),
        layer_name="",
    )

    batch_x = torch.zeros(n, 3, 8, 8)  # dummy
    probs, n_wm = wm.embed_batch_logits(_IdentityModel(), batch_x, base, temperature=1.0)
    assert probs.shape == (n, C)
    new_top1 = probs.argmax(axis=1)
    # top-1 must NEVER flip after embedding (Proposition 1 analogue)
    assert (new_top1 == original_top1).all(), \
        "top-1 flipped on {} samples".format(int((new_top1 != original_top1).sum()))
    print("PASS  embed_batch preserves top-1 on {} samples (n_wm={})".format(n, n_wm))


def test_embed_batch_watermark_rate():
    torch.manual_seed(1)
    np.random.seed(1)
    n, C = 5000, 10
    base = np.random.randn(n, C).astype(np.float64) * 0.5
    base[np.arange(n), np.random.randint(0, C, size=n)] += 5.0

    target_rate = 0.10
    wm = WatermarkModule(
        K_w=keygen(256), r_w=target_rate,
        delta_logit=2.0, beta=0.3, delta_min=0.0,
        num_classes=C,
        latent_extractor=_FakeLatentExtractor(),
        layer_name="",
    )

    _, n_wm = wm.embed_batch_logits(
        _IdentityModel(), torch.zeros(n, 3, 8, 8), base, temperature=1.0)
    actual = n_wm / n
    # Within 0.02 (binomial 95% CI for n=5000, p=0.1 is ~+-0.008)
    assert abs(actual - target_rate) < 0.02, \
        "rate {:.4f} far from {}".format(actual, target_rate)
    print("PASS  watermark rate {:.4f} ~ r_w={}".format(actual, target_rate))


def test_delta_min_filter():
    # Tight margins -> beta*margin < delta_min -> nothing recorded
    np.random.seed(2)
    n, C = 500, 10
    base = np.random.randn(n, C) * 0.05  # very small margins
    wm = WatermarkModule(
        K_w=keygen(256), r_w=1.0,
        delta_logit=2.0, beta=0.3,
        delta_min=1.0,                    # require delta >= 1.0
        num_classes=C,
        latent_extractor=_FakeLatentExtractor(),
        layer_name="",
    )
    _, n_wm = wm.embed_batch_logits(
        _IdentityModel(), torch.zeros(n, 3, 8, 8), base, temperature=1.0)
    assert n_wm == 0, "delta_min filter should have rejected all {} samples, got {}".format(n, n_wm)
    print("PASS  delta_min filter rejects low-margin samples")


# ============================================================
# 6. verify_ownership end-to-end (paired Wilcoxon)
# ============================================================

class _LookupModel(nn.Module):
    """Maps each input row (treated as an index) back to a fixed logits matrix."""
    def __init__(self, logits_matrix: np.ndarray):
        super().__init__()
        self.logits_matrix = torch.tensor(logits_matrix, dtype=torch.float32)

    def forward(self, x):
        # x is (B, ...) — first element of each sample is the row index
        if x.dim() > 1:
            idx = x.flatten(start_dim=1)[:, 0].long()
        else:
            idx = x.long()
        return self.logits_matrix[idx]


def test_verify_clean_model_no_false_positive():
    """Clean (unwatermarked) model: paired test should NOT verify."""
    torch.manual_seed(3)
    np.random.seed(3)
    C = 10
    n_trig = 200
    K = keygen(256)
    # "Clean" logits: random, no preferential boosting of any HMAC-derived class
    clean_logits = np.random.randn(n_trig, C).astype(np.float32)
    # Make sure each row has a stable top-1
    top1 = clean_logits.argmax(axis=1)

    # Build a fake trigger set whose target_class is HMAC-derived from K
    triggers = []
    for i in range(n_trig):
        t = derive_target_class(K, int(top1[i]), C)
        # Index encoding: query[0,0,0] = i (so the LookupModel knows which row to return)
        q = torch.zeros(3, 4, 4)
        q[0, 0, 0] = float(i)
        triggers.append(TriggerEntry(query=q, target_class=t, top1_class=int(top1[i])))

    model = _LookupModel(clean_logits)
    res = verify_ownership(triggers, model, K_w=K, num_classes=C,
                           verify_temperature=5.0, batch_size=64)
    # Confidence shift should be near zero
    assert res["test"] == "wilcoxon_signed_rank"
    assert res["verified"] is False, \
        "FALSE POSITIVE on clean model: p={}, shift={}".format(
            res["p_value"], res["confidence_shift"])
    print("PASS  clean model not verified (p={:.2e}, shift={:.4f})".format(
        res["p_value"], res["confidence_shift"]))


def test_verify_watermarked_model_true_positive():
    """Watermarked model: paired test SHOULD verify."""
    torch.manual_seed(4)
    np.random.seed(4)
    C = 10
    n_trig = 200
    K = keygen(256)
    # Random base logits with a clear top-1
    logits = np.random.randn(n_trig, C).astype(np.float32) * 0.5
    rand_top = np.random.randint(0, C, size=n_trig)
    logits[np.arange(n_trig), rand_top] += 5.0
    # Now boost the target class for each row by a fixed delta (logit space)
    boosted = logits.copy()
    triggers = []
    for i in range(n_trig):
        top1 = int(boosted[i].argmax())
        t = derive_target_class(K, top1, C)
        boosted[i, t] += 1.5  # noticeable but not flipping
        # confirm top-1 not flipped
        assert int(boosted[i].argmax()) == top1
        q = torch.zeros(3, 4, 4)
        q[0, 0, 0] = float(i)
        triggers.append(TriggerEntry(query=q, target_class=t, top1_class=top1))

    model = _LookupModel(boosted)
    res = verify_ownership(triggers, model, K_w=K, num_classes=C,
                           verify_temperature=5.0, batch_size=64)
    assert res["test"] == "wilcoxon_signed_rank"
    assert res["verified"] is True, \
        "FALSE NEGATIVE on watermarked model: p={}, shift={}".format(
            res["p_value"], res["confidence_shift"])
    assert res["confidence_shift"] > 0
    print("PASS  watermarked model verified (p={:.2e}, shift={:.4f})".format(
        res["p_value"], res["confidence_shift"]))


# ============================================================
# 7. Auxiliary
# ============================================================

def test_binomial_p_value():
    p8 = binomial_p_value(8, 60, 1 / 24)
    assert p8 < 0.01
    p26 = binomial_p_value(26, 60, 1 / 24)
    assert p26 < 2 ** (-64)
    assert binomial_p_value(0, 60, 1 / 24) == 1.0
    print("PASS  Binomial p-value: p(n=8)={:.2e}, p(n=26)={:.2e}".format(p8, p26))


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EvalGuard Unit Tests (v6)")
    print("=" * 60)

    test_prf_determinism()
    test_keyed_fisher_yates()
    test_fisher_yates_uniformity()
    test_gaussian_noise_determinism()
    test_gaussian_noise_stats()
    test_kdf()
    test_keygen_uses_csprng()
    test_weight_obfuscation_recovery()

    test_derive_target_class_determinism_and_no_self_map()
    test_derive_target_class_distribution()
    test_derive_control_class_disjoint()

    test_compute_delta_logit_caps_at_beta_margin()
    test_compute_delta_logit_caps_at_delta_logit()

    test_embed_batch_does_not_flip_top1()
    test_embed_batch_watermark_rate()
    test_delta_min_filter()

    test_verify_clean_model_no_false_positive()
    test_verify_watermarked_model_true_positive()

    test_binomial_p_value()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)