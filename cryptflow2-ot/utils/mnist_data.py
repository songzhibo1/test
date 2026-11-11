"""MNIST IDX gzip loading helpers."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np


TEST_IMAGES_FILE = "t10k-images-idx3-ubyte.gz"
TEST_LABELS_FILE = "t10k-labels-idx1-ubyte.gz"


def load_idx_images(path: Path, *, subtract: float = 0.0) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image file magic for {path}: {magic}")
        raw = f.read(rows * cols * num)
        if len(raw) != rows * cols * num:
            raise ValueError(f"Short image file read for {path}")
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(num, rows * cols)
    images = (arr.astype(np.float32) / 255.0).copy()
    if subtract != 0.0:
        images -= np.float32(subtract)
    return images


def load_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label file magic for {path}: {magic}")
        raw = f.read(num)
        if len(raw) != num:
            raise ValueError(f"Short label file read for {path}")
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int64).copy()


def load_mnist_test_set(
    data_root: Path, *, subtract: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    mnist_dir = data_root / "mnist"
    images = load_idx_images(mnist_dir / TEST_IMAGES_FILE, subtract=subtract)
    labels = load_idx_labels(mnist_dir / TEST_LABELS_FILE)
    if images.shape[0] != labels.shape[0]:
        raise ValueError(
            f"MNIST image/label count mismatch: {images.shape[0]} != {labels.shape[0]}"
        )
    return images, labels
