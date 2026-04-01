#!/usr/bin/env python3
"""
Prepare CROWN data for MP-SPDZ.

Reads the binary MPC weight/input files from shark_crown_ml and writes
them as SPDZ player input files (Player-Data/Input-P{0,1}-0).

Player 0 provides: weights and biases (model owner)
Player 1 provides: input image (data owner)

Usage:
    python Programs/Source/crown_prepare_data.py \
        --weights-file ../shark/shark_crown_ml/crown_mpc_data/vnncomp_mnist_3layer_relu_256_best/weights/weights.dat \
        --input-file ../shark/shark_crown_ml/crown_mpc_data/vnncomp_mnist_3layer_relu_256_best/images/0.bin \
        --layer-dims 784 256 256 10 \
        --eps 0.03 \
        --true-label 7 \
        --target-label 6
"""

import numpy as np
import struct
import argparse
import os


def load_weights_mpc_format(weights_file, layer_dims):
    """Load weights from MPC binary format."""
    weights = []
    biases = []
    with open(weights_file, 'rb') as f:
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            w_data = struct.unpack(f'{out_dim * in_dim}f', f.read(out_dim * in_dim * 4))
            W = np.array(w_data, dtype=np.float64).reshape(out_dim, in_dim)
            weights.append(W)
            b_data = struct.unpack(f'{out_dim}f', f.read(out_dim * 4))
            b = np.array(b_data, dtype=np.float64)
            biases.append(b)
            print(f"  Layer {i}: W shape={W.shape}, b shape={b.shape}")
    return weights, biases


def load_input_mpc_format(input_file, input_dim=784):
    """Load input from MPC binary format."""
    with open(input_file, 'rb') as f:
        x_data = struct.unpack(f'{input_dim}f', f.read(input_dim * 4))
        x0 = np.array(x_data, dtype=np.float64)
    return x0


def write_spdz_input_file(filepath, values):
    """Write values to SPDZ text input file (space-separated)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for v in values:
            f.write(f' {v:.10f}')
        f.write('\n')
    print(f"  Written {len(values)} values to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Prepare CROWN data for MP-SPDZ')

    parser.add_argument('--weights-file', type=str, required=True,
                        help='Path to binary weights file')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Path to binary input image file')
    parser.add_argument('--layer-dims', type=int, nargs='+', required=True,
                        help='Network layer dimensions (e.g. 784 256 256 10)')
    parser.add_argument('--eps', type=float, default=0.03,
                        help='Perturbation epsilon')
    parser.add_argument('--true-label', type=int, default=7,
                        help='True class label')
    parser.add_argument('--target-label', type=int, default=6,
                        help='Target attack label')
    parser.add_argument('--output-dir', type=str, default='Player-Data',
                        help='Output directory for SPDZ input files')

    args = parser.parse_args()

    print("=" * 60)
    print("CROWN Data Preparation for MP-SPDZ")
    print("=" * 60)
    print(f"Layer dims: {args.layer_dims}")
    print(f"Eps: {args.eps}")
    print(f"True label: {args.true_label}, Target label: {args.target_label}")

    # Load data
    print("\nLoading weights...")
    weights, biases = load_weights_mpc_format(args.weights_file, args.layer_dims)

    print("\nLoading input...")
    x0 = load_input_mpc_format(args.input_file, args.layer_dims[0])
    print(f"  Input shape: {x0.shape}, range: [{x0.min():.4f}, {x0.max():.4f}]")

    # Player 0: model parameters
    # Format: [num_layers, layer_dims..., eps, true_label, target_label, W1_flat, b1, W2_flat, b2, ...]
    print("\nPreparing Player 0 input (model owner)...")
    p0_values = []
    # Metadata as fixed-point (will be read as sfix and converted)
    num_layers = len(args.layer_dims) - 1
    p0_values.append(float(num_layers))
    for d in args.layer_dims:
        p0_values.append(float(d))
    p0_values.append(args.eps)
    p0_values.append(float(args.true_label))
    p0_values.append(float(args.target_label))

    # Weights and biases
    for i in range(num_layers):
        W_flat = weights[i].flatten().tolist()
        b_flat = biases[i].flatten().tolist()
        p0_values.extend(W_flat)
        p0_values.extend(b_flat)

    write_spdz_input_file(os.path.join(args.output_dir, 'Input-P0-0'), p0_values)

    # Player 1: input data
    print("\nPreparing Player 1 input (data owner)...")
    p1_values = x0.tolist()
    write_spdz_input_file(os.path.join(args.output_dir, 'Input-P1-0'), p1_values)

    # Write a config file for the .mpc program to know dimensions
    config_path = os.path.join(args.output_dir, 'crown-config.txt')
    with open(config_path, 'w') as f:
        f.write(f"{num_layers}\n")
        f.write(f"{' '.join(map(str, args.layer_dims))}\n")
        f.write(f"{args.eps}\n")
        f.write(f"{args.true_label}\n")
        f.write(f"{args.target_label}\n")
    print(f"\nConfig written to {config_path}")

    print("\nTotal values: P0={}, P1={}".format(len(p0_values), len(p1_values)))
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
