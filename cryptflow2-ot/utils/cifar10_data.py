"""CIFAR10 asset discovery and preprocessing metadata for Phase 07 gates."""

from __future__ import annotations

import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from format_convert import inspect_dense_relu_mlp_asset


REQUIRED_CIFAR_MODELS = ("5x100", "7x100", "10x200")
CIFAR_INPUT_DIM = 32 * 32 * 3
CIFAR_TEST_BATCH = "cifar-10-batches-py/test_batch"


@dataclass(frozen=True)
class CifarPreprocessSpec:
    name: str
    channel_order: str
    flatten_order: str
    scaling: str
    normalization: str
    source_path: str
    provenance: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class CifarAssetStatus:
    status: str
    source_root: str
    required_models: tuple[str, ...]
    found_models: dict[str, str]
    missing_models: tuple[str, ...]
    secure_cifar_allowed: bool
    candidate_preprocessing: tuple[CifarPreprocessSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["required_models"] = list(self.required_models)
        payload["missing_models"] = list(self.missing_models)
        payload["candidate_preprocessing"] = [
            spec.to_dict() for spec in self.candidate_preprocessing
        ]
        return payload


def candidate_preprocess_specs(source_root: Path) -> tuple[CifarPreprocessSpec, ...]:
    root = Path(source_root)
    return (
        CifarPreprocessSpec(
            name="torchvision_to_nhwc_0_1",
            channel_order="NHWC",
            flatten_order="row_major_hwc",
            scaling="float32_0_1",
            normalization="none",
            source_path=str(root / "ml/datasets/cifar10/dump.py"),
            provenance=(
                "torchvision.transforms.ToTensor() then images.permute(0, 2, 3, 1) "
                "and NumPy flatten()"
            ),
        ),
        CifarPreprocessSpec(
            name="pickle_chw_minus_half",
            channel_order="CHW",
            flatten_order="row_major_chw",
            scaling="float32_0_1_minus_0_5",
            normalization="minus_0_5",
            source_path=str(root / "shark_crown_ml/Convert-for-crown-mpc.py"),
            provenance=(
                "CIFAR pickle test_batch data array flattened as stored by CIFAR10, "
                "scaled with (x / 255) - 0.5"
            ),
        ),
    )


def _is_exact_model_candidate(path: Path, model: str) -> bool:
    name = path.name.lower()
    token = model.lower()
    compact = token.replace("x", "")
    return (
        token in name
        or f"_{token}_" in name
        or f"-{token}-" in name
        or f"_{compact}_" in name
    )


def _candidate_model_paths(source_root: Path, model: str) -> list[Path]:
    suffixes = {".onnx", ".crownmlp", ".h5", ".hdf5", ".keras", ".dat"}
    if not source_root.exists():
        return []
    matches: list[Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if _is_exact_model_candidate(path, model):
            matches.append(path)
    return sorted(matches)


def discover_shark_cifar_assets(source_root: Path) -> CifarAssetStatus:
    root = Path(source_root).expanduser()
    found: dict[str, str] = {}
    missing: list[str] = []
    for model in REQUIRED_CIFAR_MODELS:
        candidates = _candidate_model_paths(root, model)
        if candidates:
            found[model] = str(candidates[0])
        else:
            missing.append(model)

    status = "found" if not missing else "missing_exact_assets"
    return CifarAssetStatus(
        status=status,
        source_root=str(root),
        required_models=REQUIRED_CIFAR_MODELS,
        found_models=found,
        missing_models=tuple(missing),
        secure_cifar_allowed=(status == "found"),
        candidate_preprocessing=candidate_preprocess_specs(root),
    )


def validate_dense_relu_mlp_asset(
    path: Path, *, expected_input_dim: int = CIFAR_INPUT_DIM
) -> dict[str, object]:
    info = inspect_dense_relu_mlp_asset(path)
    input_dim = int(info["input_dim"])
    if input_dim != expected_input_dim:
        raise ValueError(
            f"Dense ReLU MLP input_dim {input_dim} != expected CIFAR input_dim {expected_input_dim}"
        )
    return info


def load_cifar10_test_set(
    data_root: Path,
    *,
    spec: CifarPreprocessSpec,
    num_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    test_file = Path(data_root) / CIFAR_TEST_BATCH
    if not test_file.is_file():
        raise FileNotFoundError(f"CIFAR10 test_batch not found: {test_file}")

    with test_file.open("rb") as f:
        data = pickle.load(f, encoding="bytes")

    images = np.asarray(data[b"data"], dtype=np.float32)
    labels = np.asarray(data[b"labels"], dtype=np.int64)
    if images.ndim != 2 or images.shape[1] != CIFAR_INPUT_DIM:
        raise ValueError(f"CIFAR10 data shape must be (N, 3072), got {images.shape}")

    if spec.name == "pickle_chw_minus_half":
        transformed = (images / 255.0) - 0.5
    elif spec.name == "torchvision_to_nhwc_0_1":
        chw = images.reshape((-1, 3, 32, 32))
        transformed = np.transpose(chw, (0, 2, 3, 1)).reshape((-1, CIFAR_INPUT_DIM))
        transformed = transformed / 255.0
    else:
        raise ValueError(f"Unsupported CIFAR preprocessing spec: {spec.name}")

    if num_images is not None:
        if num_images <= 0:
            raise ValueError("num_images must be positive when provided")
        transformed = transformed[:num_images]
        labels = labels[:num_images]
    return transformed.astype(np.float32).copy(), labels.copy()
