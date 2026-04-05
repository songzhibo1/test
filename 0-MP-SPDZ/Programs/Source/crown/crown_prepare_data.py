#!/usr/bin/env python3
"""
Prepare CROWN data for MP-SPDZ.

Player 0: W1(flat row-major), b1, W2(flat), b2, ...  (model owner)
Player 1: x0(flat), eps, true_label, target_label     (data owner)

Each value on its own line (standard SPDZ input format).
"""

import numpy as np
import struct
import argparse
import os


def load_weights(weights_file, layer_dims):
    weights, biases = [], []
    with open(weights_file, 'rb') as f:
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            w = struct.unpack(f'{out_dim * in_dim}f', f.read(out_dim * in_dim * 4))
            W = np.array(w, dtype=np.float64).reshape(out_dim, in_dim)
            b = np.array(struct.unpack(f'{out_dim}f', f.read(out_dim * 4)), dtype=np.float64)
            weights.append(W)
            biases.append(b)
            print(f"  Layer {i}: W({out_dim}x{in_dim}), b({out_dim})")
    return weights, biases


def load_input(input_file, input_dim):
    with open(input_file, 'rb') as f:
        return np.array(struct.unpack(f'{input_dim}f', f.read(input_dim * 4)), dtype=np.float64)


def write_spdz(filepath, values):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for v in values:
            f.write(f'{v:.15f}\n')
    print(f"  Written {len(values)} values to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Prepare CROWN data for MP-SPDZ')
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--layer-dims', type=int, nargs='+', required=True)
    parser.add_argument('--eps', type=float, default=0.03)
    parser.add_argument('--true-label', type=int, default=7)
    parser.add_argument('--target-label', type=int, default=6)
    parser.add_argument('--output-dir', type=str, default='Player-Data')
    args = parser.parse_args()

    num_layers = len(args.layer_dims) - 1
    print(f"CROWN Data Prep: {num_layers} layers, dims={args.layer_dims}, eps={args.eps}")

    print("\nLoading weights...")
    weights, biases = load_weights(args.weights_file, args.layer_dims)
    print("\nLoading input...")
    x0 = load_input(args.input_file, args.layer_dims[0])
    print(f"  x0: shape={x0.shape}, range=[{x0.min():.4f}, {x0.max():.4f}]")

    # Player 0: metadata + weights and biases
    print("\nPreparing Player 0 input (model owner)...")
    p0 = []
    # Metadata prefix: num_layers, layer_dims, eps, true_label, target_label
    p0.append(float(num_layers))
    for d in args.layer_dims:
        p0.append(float(d))
    p0.append(args.eps)
    p0.append(float(args.true_label))
    p0.append(float(args.target_label))
    # Weights and biases
    for i in range(num_layers):
        p0.extend(weights[i].flatten().tolist())
        p0.extend(biases[i].flatten().tolist())
    write_spdz(os.path.join(args.output_dir, 'Input-P0-0'), p0)

    # Player 1: input image only
    print("Preparing Player 1 input (data owner)...")
    p1 = x0.tolist()
    write_spdz(os.path.join(args.output_dir, 'Input-P1-0'), p1)

    # Write config file for reference
    config_path = os.path.join(args.output_dir, 'crown-config.txt')
    with open(config_path, 'w') as f:
        f.write(f"num_layers={num_layers}\n")
        f.write(f"layer_dims={args.layer_dims}\n")
        f.write(f"eps={args.eps}\n")
        f.write(f"true_label={args.true_label}\n")
        f.write(f"target_label={args.target_label}\n")
    print(f"\nConfig written to {config_path}")

    print(f"\nTotal values: P0={len(p0)}, P1={len(p1)}")
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
