"""NumPy evaluation helpers for CROWN-compatible MLP models."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from format_convert import (  # noqa: E402
    canonical_dense_weight_bias,
    load_pure_relu_linear_onnx,
)


Layer = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class CandidateSample:
    index: int
    label: int
    pred: int
    x: np.ndarray


class ModelEvaluator:
    def __init__(self, onnx_path: Path):
        bundle = load_pure_relu_linear_onnx(onnx_path)
        self.layers: list[Layer] = [
            canonical_dense_weight_bias(layer) for layer in bundle.layers
        ]

    @property
    def input_dim(self) -> int:
        return int(self.layers[0][0].shape[1])

    def logits_batch(self, x: np.ndarray) -> np.ndarray:
        return logits_batch_from_layers(self.layers, x)

    def predict_batch(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits_batch(x), axis=1).astype(np.int64)


def logits_batch_from_layers(layers: Sequence[Layer], x: np.ndarray) -> np.ndarray:
    if not layers:
        raise ValueError("logits_batch_from_layers: empty layers")
    h = np.asarray(x, dtype=np.float64)
    if h.ndim == 1:
        h = h.reshape(1, -1)
    for w, b in layers[:-1]:
        h = h @ np.asarray(w, dtype=np.float64).T + np.asarray(b, dtype=np.float64)
        h = np.maximum(h, 0.0)
    w_last, b_last = layers[-1]
    logits = h @ np.asarray(w_last, dtype=np.float64).T + np.asarray(
        b_last, dtype=np.float64
    )
    return logits


def predict_batch_from_layers(layers: Sequence[Layer], x: np.ndarray) -> np.ndarray:
    logits = logits_batch_from_layers(layers, x)
    return np.argmax(logits, axis=1).astype(np.int64)


def runner_up_targets_from_logits(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("runner_up_targets_from_logits expects a 2D logits array")
    if scores.shape[1] < 2:
        raise ValueError("runner-up target selection requires at least 2 classes")
    return np.argsort(scores, axis=1)[:, -2].astype(np.int64)


def min_logit_targets_from_logits(
    logits: np.ndarray, preds: np.ndarray | None = None
) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("min_logit_targets_from_logits expects a 2D logits array")
    if scores.shape[1] < 2:
        raise ValueError("minimum-logit target selection requires at least 2 classes")
    if preds is None:
        return np.argmin(scores, axis=1).astype(np.int64)

    pred_arr = np.asarray(preds, dtype=np.int64)
    if pred_arr.shape != (scores.shape[0],):
        raise ValueError("preds must have one entry per logits row")
    masked = scores.copy()
    rows = np.arange(scores.shape[0])
    masked[rows, pred_arr] = np.inf
    return np.argmin(masked, axis=1).astype(np.int64)


def select_candidate_samples(
    images: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    *,
    start: int,
    num: int,
    filter_correct: bool = True,
) -> list[CandidateSample]:
    if start < 0:
        raise ValueError("start must be >= 0")
    if num <= 0:
        raise ValueError("num must be > 0")
    end = start + num
    if end > images.shape[0] or end > labels.shape[0] or end > preds.shape[0]:
        raise ValueError(
            f"candidate interval [{start}, {end}) exceeds dataset size "
            f"{min(images.shape[0], labels.shape[0], preds.shape[0])}"
        )

    selected: list[CandidateSample] = []
    for i in range(start, end):
        label = int(labels[i])
        pred = int(preds[i])
        if filter_correct and pred != label:
            continue
        selected.append(CandidateSample(index=i, label=label, pred=pred, x=images[i]))
    return selected
