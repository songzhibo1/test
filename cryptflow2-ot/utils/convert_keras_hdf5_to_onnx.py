#!/usr/bin/env python3
"""Convert legacy Keras HDF5 dense-ReLU MLPs into this repo's ONNX format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _dense_layer_names(h5_file: h5py.File) -> list[str]:
    raw_config = h5_file.attrs.get("model_config")
    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")
    if not isinstance(raw_config, str):
        return _dense_layer_names_from_weight_attrs(h5_file)
    config = json.loads(raw_config)
    layers = config.get("config")
    if not isinstance(layers, list):
        raise ValueError("Unexpected Keras model_config layout")
    names: list[str] = []
    for layer in layers:
        if not isinstance(layer, dict):
            continue
        if layer.get("class_name") != "Dense":
            continue
        layer_cfg = layer.get("config", {})
        name = layer_cfg.get("name")
        if not isinstance(name, str):
            raise ValueError("Dense layer is missing a name in model_config")
        names.append(name)
    if not names:
        raise ValueError("No Dense layers found in Keras model_config")
    return names


def _decode_h5_name(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise ValueError(f"Unsupported HDF5 layer name value: {value!r}")


def _group_has_dense_arrays(group: h5py.Group) -> bool:
    found_kernel = False
    found_bias = False

    def _visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        nonlocal found_kernel, found_bias
        if not isinstance(obj, h5py.Dataset):
            return
        base = name.rsplit("/", 1)[-1]
        found_kernel = found_kernel or base == "kernel:0"
        found_bias = found_bias or base == "bias:0"

    group.visititems(_visit)
    return found_kernel and found_bias


def _dense_layer_names_from_weight_attrs(h5_file: h5py.File) -> list[str]:
    raw_names = h5_file.attrs.get("layer_names")
    if raw_names is None:
        raise ValueError("HDF5 model is missing both model_config and layer_names metadata")
    names: list[str] = []
    for raw_name in raw_names:
        name = _decode_h5_name(raw_name)
        group = h5_file.get(name)
        if isinstance(group, h5py.Group) and _group_has_dense_arrays(group):
            names.append(name)
    if not names:
        raise ValueError("No Dense weight groups found from HDF5 layer_names metadata")
    return names


def _extract_dense_group_arrays(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    kernel: np.ndarray | None = None
    bias: np.ndarray | None = None

    def _visit(_name: str, obj: h5py.Dataset | h5py.Group) -> None:
        nonlocal kernel, bias
        if not isinstance(obj, h5py.Dataset):
            return
        base = _name.rsplit("/", 1)[-1]
        if base == "kernel:0":
            kernel = np.asarray(obj[...], dtype=np.float32)
        elif base == "bias:0":
            bias = np.asarray(obj[...], dtype=np.float32)

    group.visititems(_visit)
    if kernel is None or bias is None:
        raise ValueError(f"Could not find kernel/bias datasets under group {group.name}")
    if kernel.ndim != 2 or bias.ndim != 1:
        raise ValueError(f"Unexpected Dense tensor shapes under group {group.name}")
    if kernel.shape[1] != bias.shape[0]:
        raise ValueError(
            f"Dense kernel/bias mismatch under group {group.name}: "
            f"{kernel.shape} vs {bias.shape}"
        )
    return kernel, bias


def load_keras_dense_layers(h5_path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    with h5py.File(h5_path, "r") as h5_file:
        dense_names = _dense_layer_names(h5_file)
        model_weights = h5_file.get("model_weights")
        if isinstance(model_weights, h5py.Group):
            weights_root: h5py.Group | h5py.File = model_weights
        elif h5_file.attrs.get("model_config") is None:
            weights_root = h5_file
        else:
            raise ValueError("HDF5 model is missing model_weights group")
        layers: list[tuple[np.ndarray, np.ndarray]] = []
        for dense_name in dense_names:
            if dense_name not in weights_root:
                raise ValueError(f"Dense layer {dense_name} missing from model_weights")
            kernel_io, bias = _extract_dense_group_arrays(weights_root[dense_name])
            weight_oi = kernel_io.T.copy()
            layers.append((weight_oi, bias.copy()))
        return layers


def export_layers_to_onnx(
    layers: list[tuple[np.ndarray, np.ndarray]], out_path: Path, *, producer_name: str
) -> None:
    if not layers:
        raise ValueError("Cannot export empty layer list")
    input_dim = int(layers[0][0].shape[1])
    output_dim = int(layers[-1][1].shape[0])
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, input_dim])]
    outputs = [helper.make_tensor_value_info(f"Z{len(layers) - 1}", TensorProto.FLOAT, [None, output_dim])]
    initializers = []
    nodes = []
    prev = "input"

    for idx, (weight_oi, bias) in enumerate(layers):
        w_name = f"W{idx}"
        b_name = f"B{idx}"
        z_name = f"Z{idx}"
        initializers.append(numpy_helper.from_array(weight_oi.astype(np.float32), name=w_name))
        initializers.append(numpy_helper.from_array(bias.astype(np.float32), name=b_name))
        nodes.append(
            helper.make_node(
                "Gemm",
                inputs=[prev, w_name, b_name],
                outputs=[z_name],
                transB=1,
                alpha=1.0,
                beta=1.0,
                name=f"Gemm_{idx}",
            )
        )
        if idx < len(layers) - 1:
            relu_name = f"A{idx}"
            nodes.append(
                helper.make_node(
                    "Relu", inputs=[z_name], outputs=[relu_name], name=f"Relu_{idx}"
                )
            )
            prev = relu_name
        else:
            prev = z_name

    graph = helper.make_graph(
        nodes=nodes,
        name="LegacyKerasDenseReluMlp",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info(prev, TensorProto.FLOAT, [None, output_dim])],
        initializer=initializers,
    )
    model_proto = helper.make_model(graph, producer_name=producer_name)
    model_proto.opset_import[0].version = 13
    onnx.checker.check_model(model_proto)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_proto, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Legacy Keras HDF5 model path")
    parser.add_argument("--out", required=True, type=Path, help="Output ONNX path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layers = load_keras_dense_layers(args.input)
    export_layers_to_onnx(
        layers,
        args.out,
        producer_name="secure-crown-artifact.keras-hdf5-converter",
    )
    print(f"[ok] exported {len(layers)} affine layers to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
