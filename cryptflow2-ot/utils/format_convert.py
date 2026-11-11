"""ONNX helpers: load models that are pure affine + ReLU stacks and extract weights.

CROWN MLP binary (``.crownmlp`` / arbitrary path), version 1 — little-endian::

    Bytes 0..7:   magic ASCII ``CROWNMLP`` (exactly 8 bytes).
    Bytes 8..11:  uint32 ``version`` (must be 1).
    Bytes 12..15: uint32 ``num_layers``.
    Bytes 16..19: uint32 ``input_dim`` (must equal first layer ``in_dim``; flatten后的向量维).
    Then ``num_layers`` times:
        uint32 ``in_dim``, uint32 ``out_dim``,
        ``in_dim * out_dim`` float64 row-major ``W[i,j]`` = row ``i`` output neuron, col ``j`` input
        (same layout as C++ ``DenseLayerPublic::W[out][in]``),
        ``out_dim`` float64 biases.

All integers and floats are little-endian. Floats are IEEE754 binary64.
"""

from __future__ import annotations

import struct
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import numpy_helper

__all__ = [
    "CROWN_MLP_MAGIC",
    "CROWN_MLP_VERSION",
    "LinearLayerWeights",
    "PureReluLinearOnnx",
    "export_crown_mlp_bin",
    "inspect_dense_relu_mlp_asset",
    "load_pure_relu_linear_onnx",
    "predict_label_numpy_mlp",
    "read_crown_mlp_bin",
]

# CROWN packed MLP on disk (see module docstring).
CROWN_MLP_MAGIC = b"CROWNMLP"
CROWN_MLP_VERSION = 1

_ALLOWED_OPS = frozenset({"Gemm", "MatMul", "Relu", "Add", "Flatten"})


@dataclass(frozen=True)
class LinearLayerWeights:
    """One affine map from an ONNX ``Gemm`` or ``MatMul`` (+ optional ``Add`` bias)."""

    weight: np.ndarray
    bias: np.ndarray | None
    op_type: str
    gemm_trans_a: int | None
    gemm_trans_b: int | None
    gemm_alpha: float | None = None
    matmul_weight_input_index: int | None = None


@dataclass(frozen=True)
class PureReluLinearOnnx:
    """Loaded ONNX plus ordered linear layer weights (ReLU layers carry no parameters)."""

    model: onnx.ModelProto
    layers: tuple[LinearLayerWeights, ...]
    ordered_nodes: tuple[onnx.NodeProto, ...]


def _get_attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(onnx.helper.get_attribute_value(attr))
    return default


def _get_attr_float(node: onnx.NodeProto, name: str, default: float) -> float:
    for attr in node.attribute:
        if attr.name == name:
            return float(onnx.helper.get_attribute_value(attr))
    return default


def _assert_single_use_activations(graph: onnx.GraphProto) -> None:
    """Each activation tensor (non-initializer) may feed at most one node — a pure chain."""
    init_names = {i.name for i in graph.initializer}
    in_names = {i.name for i in graph.input}
    fanout: dict[str, int] = defaultdict(int)
    for node in graph.node:
        for inp in node.input:
            if not inp or inp in init_names:
                continue
            fanout[inp] += 1
    for name, count in fanout.items():
        if name in in_names:
            if count > 1:
                raise ValueError(
                    f"Graph input {name!r} feeds {count} nodes; expected a single-consumer chain."
                )
            continue
        if count > 1:
            raise ValueError(
                f"Activation tensor {name!r} feeds {count} nodes; expected a pure sequential MLP."
            )


def _reject_subgraphs(nodes: Iterable[onnx.NodeProto]) -> None:
    for node in nodes:
        for attr in node.attribute:
            if attr.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError(
                    f"Node {node.name!r} ({node.op_type}) contains a subgraph; "
                    "only a flat affine+ReLU graph is supported."
                )


def _initializer_map(graph: onnx.GraphProto) -> dict[str, np.ndarray]:
    return {init.name: numpy_helper.to_array(init).copy() for init in graph.initializer}


def _topological_order(graph: onnx.GraphProto) -> list[onnx.NodeProto]:
    nodes = list(graph.node)
    init_names = {i.name for i in graph.initializer}
    input_names = {i.name for i in graph.input}
    sources = init_names | input_names

    producers: dict[str, int] = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                producers[out] = idx

    pending: list[int] = []
    for node in nodes:
        deps = 0
        for inp in node.input:
            if not inp or inp in sources:
                continue
            if inp not in producers:
                raise ValueError(
                    f"Node {node.name!r} ({node.op_type}) references unknown tensor {inp!r}."
                )
            deps += 1
        pending.append(deps)

    consumers: dict[str, list[int]] = defaultdict(list)
    for idx, node in enumerate(nodes):
        for inp in node.input:
            if inp and inp not in sources and inp in producers:
                consumers[inp].append(idx)

    queue = deque(i for i, d in enumerate(pending) if d == 0)
    order_idx: list[int] = []
    while queue:
        i = queue.popleft()
        order_idx.append(i)
        for out in nodes[i].output:
            if not out:
                continue
            for c in consumers[out]:
                pending[c] -= 1
                if pending[c] == 0:
                    queue.append(c)

    if len(order_idx) != len(nodes):
        raise ValueError("Graph contains a cycle or unresolved dependencies.")
    return [nodes[i] for i in order_idx]


def _validate_add_bias_only(node: onnx.NodeProto, init_names: set[str]) -> None:
    init_inputs = [i for i in node.input if i and i in init_names]
    if len(init_inputs) != 1:
        raise ValueError(
            f"Add node {node.name!r} must have exactly one initializer input (bias). "
            f"Got initializer inputs: {init_inputs!r}."
        )


def _linear_layers_from_nodes(
    graph: onnx.GraphProto, ordered_nodes: list[onnx.NodeProto], arrays: dict[str, np.ndarray]
) -> tuple[LinearLayerWeights, ...]:
    init_names = set(arrays)
    layers: list[LinearLayerWeights] = []

    i = 0
    while i < len(ordered_nodes):
        node = ordered_nodes[i]

        if node.op_type in ("Relu", "Flatten"):
            i += 1
            continue

        if node.op_type == "Add":
            raise ValueError(
                f"Standalone Add {node.name!r} is not allowed unless it immediately "
                "follows a MatMul as a bias term."
            )

        if node.op_type == "Gemm":
            if len(node.input) < 2:
                raise ValueError(f"Gemm {node.name!r} needs at least two inputs.")
            w_name, b_name = node.input[1], node.input[2] if len(node.input) > 2 else None
            if w_name not in arrays:
                raise ValueError(f"Gemm {node.name!r} weight {w_name!r} is not an initializer.")
            if b_name is not None and b_name not in arrays:
                raise ValueError(f"Gemm {node.name!r} bias {b_name!r} is not an initializer.")
            weight = arrays[w_name]
            bias = arrays[b_name] if b_name else None
            layers.append(
                LinearLayerWeights(
                    weight=weight,
                    bias=bias,
                    op_type="Gemm",
                    gemm_trans_a=_get_attr_int(node, "transA", 0),
                    gemm_trans_b=_get_attr_int(node, "transB", 0),
                    gemm_alpha=_get_attr_float(node, "alpha", 1.0),
                    matmul_weight_input_index=None,
                )
            )
            i += 1
            continue

        if node.op_type == "MatMul":
            init_inputs = [x for x in node.input if x and x in init_names]
            if len(init_inputs) != 1:
                raise ValueError(
                    f"MatMul {node.name!r} must have exactly one initializer input (weight)."
                )
            w_name = init_inputs[0]
            weight = arrays[w_name]
            w_slot = 0 if (node.input[0] and node.input[0] in init_names) else 1
            bias: np.ndarray | None = None
            j = i + 1
            if j < len(ordered_nodes) and ordered_nodes[j].op_type == "Add":
                add_node = ordered_nodes[j]
                _validate_add_bias_only(add_node, init_names)
                bias_name = next(x for x in add_node.input if x and x in init_names)
                bias = arrays[bias_name]
                j += 1
            layers.append(
                LinearLayerWeights(
                    weight=weight,
                    bias=bias,
                    op_type="MatMul",
                    gemm_trans_a=None,
                    gemm_trans_b=None,
                    gemm_alpha=None,
                    matmul_weight_input_index=w_slot,
                )
            )
            i = j
            continue

        raise ValueError(f"Unexpected node {node.name!r} of type {node.op_type!r}.")

    return tuple(layers)


def load_pure_relu_linear_onnx(path: str | Path) -> PureReluLinearOnnx:
    """Load an ONNX file, verify it is only Gemm/MatMul + ReLU (+ MatMul bias Add), and return weights.

    The graph is restricted to these operators:

    - ``Gemm`` / ``MatMul`` (optional ``Add`` immediately after ``MatMul`` for bias)
    - ``Relu``
    - ``Flatten`` (no parameters; reshapes activations, e.g. before the first ``Gemm``)

    Any other operator, subgraphs, control flow, or ``Add`` that is not a single-initializer
    bias fused after ``MatMul`` causes a :class:`ValueError`.

    Weight/bias arrays are NumPy copies of ONNX initializers. For ``Gemm``, see ONNX ``Gemm``
    attributes ``transA`` / ``transB`` on each :class:`LinearLayerWeights` when applying the map.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"ONNX path does not exist or is not a file: {p}")

    model = onnx.load(p)
    graph = model.graph
    _reject_subgraphs(graph.node)
    for node in graph.node:
        if node.op_type not in _ALLOWED_OPS:
            raise ValueError(
                f"Unsupported operator {node.op_type!r} in node {node.name!r}. "
                f"Only {sorted(_ALLOWED_OPS)} are allowed for a pure ReLU + affine network."
            )

    _assert_single_use_activations(graph)
    arrays = _initializer_map(graph)
    ordered = _topological_order(graph)

    for node in ordered:
        if node.op_type == "Add":
            _validate_add_bias_only(node, set(arrays))

    layers = _linear_layers_from_nodes(graph, ordered, arrays)
    if not layers:
        raise ValueError("No Gemm or MatMul layers found in the graph.")

    return PureReluLinearOnnx(model=model, layers=layers, ordered_nodes=tuple(ordered))


def _validate_crown_mlp_topology(ordered_nodes: list[onnx.NodeProto], n_affine: int) -> None:
    """Match C++ ``PredictLabel`` / ``CrownStandardBounds``: ReLU after each hidden affine, none after logits."""
    relus = sum(1 for n in ordered_nodes if n.op_type == "Relu")
    if relus != n_affine - 1:
        raise ValueError(
            f"CROWN MLP expects exactly {n_affine - 1} Relu for {n_affine} affine layers; found {relus}."
        )
    last = ordered_nodes[-1]
    if last.op_type == "Relu":
        raise ValueError(
            "Last operator may not be Relu: C++ applies no ReLU after the final affine (logits) layer."
        )


def canonical_dense_weight_bias(layer: LinearLayerWeights) -> tuple[np.ndarray, np.ndarray]:
    """Return ``W`` (out_dim, in_dim) and ``b`` (out_dim,) matching C++ ``DenseLayerPublic`` / ``matvec``."""
    if layer.op_type == "Gemm":
        ta = layer.gemm_trans_a or 0
        if ta != 0:
            raise ValueError(
                f"CROWN export does not support Gemm with transA={ta}; re-export ONNX with transA=0."
            )
        alpha = float(layer.gemm_alpha if layer.gemm_alpha is not None else 1.0)
        tb = layer.gemm_trans_b or 0
        w_raw = np.asarray(layer.weight, dtype=np.float64)
        b_onnx = w_raw.T if tb else w_raw
        w_cpp = (alpha * b_onnx).T
        out_d = int(w_cpp.shape[0])
        if layer.bias is not None:
            bvec = np.asarray(layer.bias, dtype=np.float64).reshape(-1)
            if bvec.shape[0] != out_d:
                raise ValueError(f"Gemm bias length {bvec.shape[0]} != out_dim {out_d}.")
        else:
            bvec = np.zeros(out_d, dtype=np.float64)
        return w_cpp, bvec

    if layer.op_type == "MatMul":
        idx = layer.matmul_weight_input_index
        if idx is None:
            raise ValueError("MatMul layer missing matmul_weight_input_index (internal error).")
        w_raw = np.asarray(layer.weight, dtype=np.float64)
        w_cpp = w_raw.T.copy() if idx == 1 else w_raw.copy()
        out_d = int(w_cpp.shape[0])
        if layer.bias is not None:
            bvec = np.asarray(layer.bias, dtype=np.float64).reshape(-1)
            if bvec.shape[0] != out_d:
                raise ValueError(f"MatMul bias length {bvec.shape[0]} != out_dim {out_d}.")
        else:
            bvec = np.zeros(out_d, dtype=np.float64)
        return w_cpp, bvec

    raise ValueError(f"Unknown op_type {layer.op_type!r}")


def export_crown_mlp_bin(onnx_path: str | Path, out_path: str | Path) -> None:
    """Parse ONNX via :func:`load_pure_relu_linear_onnx`, validate CROWN topology, write packed ``.crownmlp`` binary."""
    bundle = load_pure_relu_linear_onnx(onnx_path)
    layers = bundle.layers
    if len(layers) < 2:
        raise ValueError(
            f"CROWN requires at least 2 affine layers (hidden+ReLU + logits); got {len(layers)}."
        )
    _validate_crown_mlp_topology(list(bundle.ordered_nodes), len(layers))
    canonical = [canonical_dense_weight_bias(L) for L in layers]
    for i in range(len(canonical) - 1):
        out_i = canonical[i][0].shape[0]
        in_next = canonical[i + 1][0].shape[1]
        if out_i != in_next:
            raise ValueError(
                f"Layer dimension mismatch between affine {i} and {i + 1}: "
                f"out_dim {out_i} != next in_dim {in_next}."
            )
    input_dim = int(canonical[0][0].shape[1])
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    _write_crown_mlp_file(out_p, canonical, input_dim)


def _write_crown_mlp_file(
    path: Path, canonical: list[tuple[np.ndarray, np.ndarray]], input_dim: int
) -> None:
    n = len(canonical)
    blob = bytearray()
    blob.extend(CROWN_MLP_MAGIC)
    blob.extend(struct.pack("<III", CROWN_MLP_VERSION, n, input_dim))
    for w, b in canonical:
        out_d, in_d = int(w.shape[0]), int(w.shape[1])
        if int(b.shape[0]) != out_d:
            raise ValueError("Internal error: bias length != out_dim.")
        blob.extend(struct.pack("<II", in_d, out_d))
        blob.extend(np.ascontiguousarray(w, dtype=np.float64).tobytes())
        blob.extend(np.ascontiguousarray(b, dtype=np.float64).tobytes())
    path.write_bytes(blob)


def read_crown_mlp_bin(path: str | Path) -> tuple[int, list[tuple[np.ndarray, np.ndarray]]]:
    """Read a CROWN MLP binary file; returns ``(input_dim, [(W, b), ...])`` with float64 NumPy arrays."""
    data = Path(path).read_bytes()
    if len(data) < 8 + 4 + 4 + 4:
        raise ValueError("File too small for CROWN MLP header.")
    if data[:8] != CROWN_MLP_MAGIC:
        raise ValueError(f"Bad magic: expected {CROWN_MLP_MAGIC!r}, got {data[:8]!r}.")
    version, num_layers, input_dim = struct.unpack_from("<III", data, 8)
    if version != CROWN_MLP_VERSION:
        raise ValueError(f"Unsupported CROWN MLP version {version} (expected {CROWN_MLP_VERSION}).")
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1.")
    off = 20
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    expected_in: int | None = None
    for _ in range(num_layers):
        if off + 8 > len(data):
            raise ValueError("Truncated layer header.")
        in_d, out_d = struct.unpack_from("<II", data, off)
        off += 8
        if expected_in is None:
            if int(input_dim) != int(in_d):
                raise ValueError(
                    f"input_dim field {input_dim} != first layer in_dim {in_d}."
                )
        else:
            if int(in_d) != int(expected_in):
                raise ValueError(
                    f"Layer chain mismatch: expected in_dim {expected_in}, got {in_d}."
                )
        w_count = in_d * out_d
        b_bytes = 8 * out_d
        need = 8 * w_count + b_bytes
        if off + need > len(data):
            raise ValueError("Truncated weights or bias.")
        w_flat = np.frombuffer(data, dtype="<f8", count=w_count, offset=off)
        off += 8 * w_count
        bvec = np.frombuffer(data, dtype="<f8", count=out_d, offset=off)
        off += b_bytes
        w = w_flat.reshape((out_d, in_d), order="C").copy()
        layers.append((w, bvec.copy()))
        expected_in = int(out_d)
    if off != len(data):
        raise ValueError(f"Junk at end of file ({len(data) - off} bytes).")
    return int(input_dim), layers


def inspect_dense_relu_mlp_asset(path: str | Path) -> dict[str, object]:
    """Validate an ONNX or ``.crownmlp`` asset using the existing dense ReLU MLP contracts."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".crownmlp":
        input_dim, layers = read_crown_mlp_bin(p)
        dims = [int(input_dim)]
        dims.extend(int(w.shape[0]) for w, _ in layers)
        return {
            "format": "crownmlp",
            "input_dim": int(input_dim),
            "num_layers": len(layers),
            "layer_dims": dims,
            "path": str(p),
        }

    if suffix == ".onnx":
        bundle = load_pure_relu_linear_onnx(p)
        canonical = [canonical_dense_weight_bias(layer) for layer in bundle.layers]
        dims = [int(canonical[0][0].shape[1])]
        dims.extend(int(w.shape[0]) for w, _ in canonical)
        _validate_crown_mlp_topology(list(bundle.ordered_nodes), len(canonical))
        return {
            "format": "onnx",
            "input_dim": dims[0],
            "num_layers": len(canonical),
            "layer_dims": dims,
            "path": str(p),
        }

    raise ValueError(
        f"Unsupported dense ReLU MLP asset suffix for {p}: expected .onnx or .crownmlp"
    )


def predict_label_numpy_mlp(
    layers: list[tuple[np.ndarray, np.ndarray]], x0: np.ndarray
) -> int:
    """Same semantics as C++ ``PredictLabel`` (ReLU between layers, no ReLU on logits)."""
    x = np.asarray(x0, dtype=np.float64).reshape(-1)
    for i in range(len(layers) - 1):
        w, b = layers[i]
        pre = w @ x + b
        x = np.maximum(0.0, pre)
    w, b = layers[-1]
    logits = w @ x + b
    return int(np.argmax(logits))


if __name__ == "__main__":
    import sys

    onnx_path = Path(sys.argv[1] if len(sys.argv) > 1 else "models/mnistfc/mnist-net_256x2.onnx")
    out_bin = Path(sys.argv[2] if len(sys.argv) > 2 else str(onnx_path.with_suffix(".crownmlp")))
    bundle = load_pure_relu_linear_onnx(onnx_path)
    print(f"loaded {len(bundle.layers)} affine layers from {onnx_path}")
    export_crown_mlp_bin(onnx_path, out_bin)
    print(f"wrote {out_bin}")
