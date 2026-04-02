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

    # Player 0: all weights and biases in order
    print("\nPlayer 0 (model)...")
    p0 = []
    for i in range(num_layers):
        p0.extend(weights[i].flatten().tolist())
        p0.extend(biases[i].flatten().tolist())
    write_spdz(os.path.join(args.output_dir, 'Input-P0-0'), p0)

    # Player 1: input + config
    print("Player 1 (data)...")
    p1 = x0.tolist()
    write_spdz(os.path.join(args.output_dir, 'Input-P1-0'), p1)

    print(f"\nDone! P0={len(p0)}, P1={len(p1)} values")


if __name__ == "__main__":
    main()
