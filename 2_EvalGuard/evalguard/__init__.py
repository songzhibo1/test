"""
EvalGuard: Guard Model IP During Third-Party Evaluation

Three-phase framework:
  Phase 1: Weight Obfuscation (Algorithm 1)
  Phase 2: Confidence Shift Watermark Embedding (Algorithm 2, v4)
  Phase 3: Ownership Verification via Mann-Whitney U Test (Algorithm 3, v4)

[v4] Replaced rank permutation with confidence shift.
"""

from .crypto import keygen, kdf, prf

# Pure math functions — always available, no torch needed
from .watermark_math import (
    binomial_p_value,
    mannwhitney_p_value,
)

# Torch-dependent modules: check before importing.
import importlib as _il
if _il.util.find_spec("torch") is not None:
    from .obfuscation import (
        obfuscate_model,
        obfuscate_model_vectorized,
        recover_weights,
        SecretPackage,
        compute_sigma,
        compute_l2_sensitivity,
    )
    from .watermark import (
        WatermarkModule,
        LatentExtractor,
        TriggerEntry,
        verify_ownership,
        verify_ownership_all_designs,
        verify_ownership_own_data_all_designs,
        derive_target_class,
        derive_control_class,
    )

__all__ = [
    "keygen", "kdf", "prf",
    "obfuscate_model", "obfuscate_model_vectorized",
    "recover_weights", "SecretPackage",
    "compute_sigma", "compute_l2_sensitivity",
    "WatermarkModule", "LatentExtractor", "TriggerEntry",
    "verify_ownership",
    "verify_ownership_all_designs", "verify_ownership_own_data_all_designs",
    "derive_target_class", "derive_control_class",
    "binomial_p_value", "mannwhitney_p_value",
]