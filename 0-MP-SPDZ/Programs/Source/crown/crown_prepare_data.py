#!/usr/bin/env python3
"""
Prepare CROWN data for MP-SPDZ.

Player 0 (SERVER / model owner): W1(flat), b1, W2(flat), b2, ...
Player 1 (CLIENT / data owner):  x0(flat), eps, diff_vec(output_dim)

eps, true_label, target_label are CLIENT secrets -- not compile-time args.
diff_vec encodes the labels: diff_vec[true]=1, diff_vec[target]=-1, rest=0.

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
    output_dim = args.layer_dims[-1]
    print(f"CROWN Data Prep: {num_layers} layers, dims={args.layer_dims}, eps={args.eps}")

    print("\nLoading weights...")
    weights, biases = load_weights(args.weights_file, args.layer_dims)
    print("\nLoading input...")
    x0 = load_input(args.input_file, args.layer_dims[0])
    print(f"  x0: shape={x0.shape}, range=[{x0.min():.4f}, {x0.max():.4f}]")

    # Player 0 (SERVER): weights and biases only
    print("\nPreparing Player 0 input (SERVER / model owner)...")
    p0 = []
    for i in range(num_layers):
        p0.extend(weights[i].flatten().tolist())
        p0.extend(biases[i].flatten().tolist())
    write_spdz(os.path.join(args.output_dir, 'Input-P0-0'), p0)

    # Player 1 (CLIENT): x0, eps, diff_vec (all secrets)
    print("Preparing Player 1 input (CLIENT / data owner)...")
    p1 = x0.tolist()
    p1.append(args.eps)  # eps is CLIENT's secret
    # diff_vec: encodes true_label and target_label
    diff_vec = [0.0] * output_dim
    diff_vec[args.true_label] = 1.0
    diff_vec[args.target_label] = -1.0
    p1.extend(diff_vec)
    write_spdz(os.path.join(args.output_dir, 'Input-P1-0'), p1)

    print(f"\nTotal values: P0={len(p0)}, P1={len(p1)}")
    print(f"  P1 breakdown: x0={len(x0)}, eps=1, diff_vec={output_dim}")
    print("Data preparation complete!")


if __name__ == "__main__":
    main()
