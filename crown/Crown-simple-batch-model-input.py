"""
修改版CROWN简化实现 - 支持MPC数据格式和5层网络
基于第一个简化实现代码，添加MPC二进制文件读取功能
适配: weights_file, input_file, layer_dims = [784, 256, 256, 256, 256, 10]
"""

import numpy as np
import argparse
import sys
import struct
import os

# ==================== MPC 数据加载函数 ====================

def load_weights_mpc_format(weights_file, layer_dims):
    """
    加载 MPC 格式的权重文件
    文件格式: 对于每层，先存储 W (out_dim x in_dim)，再存储 b (out_dim)
    返回: weights 列表 (每个元素形状为 (out_dim, in_dim))，biases 列表
    """
    weights = []
    biases = []

    print(f"Loading weights from: {weights_file}")

    with open(weights_file, 'rb') as f:
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]

            print(f"  Layer {i+1}: ({in_dim} -> {out_dim})")

            # 读取 W (out_dim x in_dim 格式，行优先)
            w_data = struct.unpack(f'{out_dim * in_dim}f', f.read(out_dim * in_dim * 4))
            W = np.array(w_data, dtype=np.float32).reshape(out_dim, in_dim)
            weights.append(W)

            # 读取 b (out_dim)
            b_data = struct.unpack(f'{out_dim}f', f.read(out_dim * 4))
            b = np.array(b_data, dtype=np.float32)
            biases.append(b)

            print(f"    W{i+1}: {W.shape}, b{i+1}: {b.shape}")
            print(f"    W first 5 (flattened): {W.flatten()[:5]}")
            print(f"    b first 5: {b[:5]}")

    return weights, biases


def load_input_mpc_format(input_file, input_dim=784):
    """加载MPC格式的输入文件"""
    print(f"Loading input from: {input_file}")

    with open(input_file, 'rb') as f:
        x_data = struct.unpack(f'{input_dim}f', f.read(input_dim * 4))
        x0 = np.array(x_data, dtype=np.float32)

    print(f"Loaded input: shape={x0.shape}, first 10: {x0[:10]}")
    return x0


# ==================== 辅助函数 ====================

def ReLU(x):
    """ReLU 激活函数"""
    return np.maximum(x, 0)


def forward(x, weights, biases):
    """前向传播"""
    a = x
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = np.dot(W, a) + b
        if i < len(weights) - 1:  # 最后一层没有 ReLU
            a = ReLU(z)
        else:
            a = z
    return a


# ==================== MPC Debug 辅助函数 ====================

def print_mpc_style(name, data):
    """
    辅助函数：以 C++ MPC 代码的格式打印数据
    格式: Name: [val1, val2, ...] (保留6位小数)
    """
    if isinstance(data, (list, np.ndarray)):
        # 展平并取前 10 个元素，与 MPC 代码保持一致的预览风格
        flat_data = np.array(data).flatten()
        preview_items = [f"{x:.6f}" for x in flat_data[:10]]
        preview = ", ".join(preview_items)
        if len(flat_data) > 10:
            preview += ", ..."
        print(f"{name}: [{preview}]")
    else:
        print(f"{name}: [{data:.6f}]")


# ==================== CROWN 核心函数（简化版本）====================

def get_layer_bound_first(W, b, x0, eps, debug=False, layer_idx=0):
    """计算第一层的 pre-ReLU 边界"""
    Ax0 = np.dot(W, x0)
    dualnorms = np.sum(np.abs(W), axis=1)

    if debug:
        print_mpc_style(f"Layer{layer_idx}_Ax0", Ax0)
        print_mpc_style(f"Layer{layer_idx}_dualnorm", dualnorms)

    radius = eps * dualnorms

    if debug:
        # MPC 代码里 radius 经过了 ReLU (虽然理论上是非负的)
        print_mpc_style(f"Layer{layer_idx}_radius", ReLU(radius))

    UB = Ax0 + radius + b
    LB = Ax0 - radius + b

    if debug:
        print_mpc_style(f"Layer{layer_idx}_UB", UB)
        print_mpc_style(f"Layer{layer_idx}_LB", LB)

    return UB, LB


def get_neuron_states(UB, LB):
    """根据 pre-ReLU 边界确定神经元状态"""
    states = np.zeros_like(UB, dtype=np.int8)
    UB_relu = ReLU(UB)
    LB_relu = ReLU(LB)
    states[UB_relu == 0] = -1
    states[LB_relu > 0] = +1
    return states


def get_layer_bound_crown(Ws, bs, UBs, LBs, neuron_states, x0, eps, debug=False, layer_idx=-1):
    """CROWN 方法计算某一层的 pre-ReLU 边界"""
    nlayer = len(Ws)
    assert nlayer >= 2

    # 计算松弛斜率 α
    diags = []
    for i in range(nlayer):
        if i == 0:
            diags.append(np.ones(Ws[0].shape[1]))
        else:
            state = neuron_states[i-1]
            alpha = state.astype(np.float64)
            alpha = np.maximum(alpha, 0)
            idx_unsure = np.where(state == 0)[0]
            if len(idx_unsure) > 0:
                alpha[idx_unsure] = UBs[i][idx_unsure] / (UBs[i][idx_unsure] - LBs[i][idx_unsure])

            # 打印当前层计算用的上一层的 alpha (对应 MPC 的 LayerX_alpha)
            if debug and i == nlayer - 1:
                 print_mpc_style(f"Layer{layer_idx-1}_alpha", alpha)

            diags.append(alpha)

    # 初始化
    constants = np.copy(bs[-1])
    UB_final = np.zeros_like(constants, dtype=np.float64)
    LB_final = np.zeros_like(constants, dtype=np.float64)

    # 记录修正项用于调试
    lb_corrections = np.zeros_like(constants, dtype=np.float64)
    ub_corrections = np.zeros_like(constants, dtype=np.float64)

    # 反向传播构建 A 矩阵
    # 初始 A 就是当前层的权重 W * alpha
    A = Ws[-1] * diags[-1]
    if debug:
        print_mpc_style(f"Layer{layer_idx}_A_matrix", A)

    for i in range(nlayer - 1, 0, -1):
        # 累加 Bias: constants += A * b_{i-1}
        constants = constants + np.dot(A, bs[i-1])

        idx_unsure = np.where(neuron_states[i-1] == 0)[0]

        if len(idx_unsure) > 0:
            for j in range(A.shape[0]):
                l_ub = np.zeros_like(LBs[i], dtype=np.float64)
                l_lb = np.zeros_like(LBs[i], dtype=np.float64)

                A_unsure = A[j][idx_unsure]
                pos = np.where(A_unsure > 0)[0]
                neg = np.where(A_unsure < 0)[0]

                idx_unsure_pos = idx_unsure[pos]
                idx_unsure_neg = idx_unsure[neg]

                l_ub[idx_unsure_pos] = LBs[i][idx_unsure_pos]
                l_lb[idx_unsure_neg] = LBs[i][idx_unsure_neg]

                # 计算并累加修正项
                ub_corr_val = -np.dot(A[j], l_ub)
                lb_corr_val = -np.dot(A[j], l_lb)

                UB_final[j] += ub_corr_val
                LB_final[j] += lb_corr_val

                # 如果是反向传播的第一步，记录修正项以便打印
                if i == nlayer - 1:
                    ub_corrections[j] += ub_corr_val
                    lb_corrections[j] += lb_corr_val

        # 更新 A 矩阵
        if i > 1:
            A = np.dot(A, Ws[i-1] * diags[i-1])
        else:
            A = np.dot(A, Ws[i-1])

    # 打印中间结果
    if debug:
        prefix = f"Layer{layer_idx}" if layer_idx >= 0 else "Final"
        print_mpc_style(f"{prefix}_constants", constants)

        if layer_idx >= 0:
            print_mpc_style(f"{prefix}_lb_corr", lb_corrections)
            print_mpc_style(f"{prefix}_ub_corr", ub_corrections)
        else:
            # 最后一层的命名约定
            print_mpc_style("Final_lb_corr_layer1", lb_corrections)
            print_mpc_style("Final_ub_corr_layer1", ub_corrections)

    UB_final = UB_final + constants
    LB_final = LB_final + constants

    Ax0 = np.dot(A, x0)
    dualnorms = np.sum(np.abs(A), axis=1)
    radius = eps * dualnorms

    if debug:
        prefix = f"Layer{layer_idx}" if layer_idx >= 0 else "Final"
        print_mpc_style(f"{prefix}_Ax0", Ax0)
        print_mpc_style(f"{prefix}_dualnorm", dualnorms)
        print_mpc_style(f"{prefix}_radius", ReLU(radius))

    UB_final = UB_final + Ax0 + radius
    LB_final = LB_final + Ax0 - radius

    if debug and layer_idx >= 0:
        print_mpc_style(f"Layer{layer_idx}_UB", UB_final)
        print_mpc_style(f"Layer{layer_idx}_LB", LB_final)

    return UB_final, LB_final


def compute_worst_bound_simplified(weights, biases, x0, eps, true_label, target_label, verbose=False, debug_mpc=False):
    """
    简化版本：计算 f_c(x) - f_j(x) 的下界
    支持任意层数的网络
    """
    numlayer = len(weights)

    if debug_mpc:
        print("\n============================================")
        print(f"Debug Output (Python Simplified) - {numlayer} Layers")
        print("============================================")
        print(f"Network structure: {x0.shape[0]} -> {' -> '.join([str(w.shape[0]) for w in weights])}")

    # 第一层
    UB, LB = get_layer_bound_first(weights[0], biases[0], x0, eps, debug=debug_mpc, layer_idx=0)
    preReLU_UBs = [UB]
    preReLU_LBs = [LB]
    states = get_neuron_states(UB, LB)
    neuron_states = [states]

    if verbose:
        print(f"  Layer 0: {np.sum(states == -1)} never activated, {np.sum(states == 1)} always activated")

    # 中间层
    UBs = [x0 + eps] + preReLU_UBs
    LBs = [x0 - eps] + preReLU_LBs

    for layer in range(1, numlayer - 1):
        UB, LB = get_layer_bound_crown(
            weights[:layer+1], biases[:layer+1],
            UBs, LBs, neuron_states, x0, eps,
            debug=debug_mpc, layer_idx=layer
        )
        preReLU_UBs.append(UB)
        preReLU_LBs.append(LB)
        UBs = [x0 + eps] + preReLU_UBs
        LBs = [x0 - eps] + preReLU_LBs
        states = get_neuron_states(UB, LB)
        neuron_states.append(states)

        if verbose:
            print(f"  Layer {layer}: {np.sum(states == -1)} never activated, {np.sum(states == 1)} always activated")

    # 最后一层 (Margin层)
    W_last = weights[-1]
    b_last = biases[-1]
    W_diff = (W_last[true_label] - W_last[target_label]).reshape(1, -1)
    b_diff = np.array([b_last[true_label] - b_last[target_label]])

    if debug_mpc:
        # 手动计算并打印最后一层的中间值，以匹配 MPC 逻辑
        # 1. Alpha
        state = neuron_states[-1]
        alpha = state.astype(np.float64)
        alpha = np.maximum(alpha, 0)
        idx_unsure = np.where(state == 0)[0]
        if len(idx_unsure) > 0:
            alpha[idx_unsure] = preReLU_UBs[-1][idx_unsure] / (preReLU_UBs[-1][idx_unsure] - preReLU_LBs[-1][idx_unsure])
        print_mpc_style(f"Layer{numlayer-2}_alpha", alpha)

        # 2. W_diff, b_diff
        print_mpc_style("Final_W_diff", W_diff)
        print_mpc_style("Final_b_diff", b_diff)

        # 3. A_final (W_diff * alpha)
        A_final = W_diff * alpha
        print_mpc_style("Final_A_final", A_final)

        # 4. Initial Constants
        initial_constants = np.dot(A_final, biases[-2]) + b_diff
        print_mpc_style("Final_constants_initial", initial_constants)

    UB, LB = get_layer_bound_crown(
        weights[:-1] + [W_diff], biases[:-1] + [b_diff],
        UBs, LBs, neuron_states, x0, eps,
        debug=debug_mpc, layer_idx=-1
    )

    return LB[0], UB[0]


# ==================== 主要测试函数 ====================

def run_mpc_test(weights_file, input_file, layer_dims, eps_list, true_label, target_label, verbose=False, debug_mpc=False):
    """
    运行MPC格式数据的测试
    """
    print("\n" + "=" * 80)
    print("CROWN MPC数据格式测试")
    print("=" * 80)

    # 检查文件是否存在
    if not os.path.exists(weights_file):
        print(f"错误: 权重文件不存在: {weights_file}")
        return None

    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return None

    print(f"网络结构: {' -> '.join(map(str, layer_dims))}")
    print(f"Eps 设置: {eps_list}")
    print(f"True label: {true_label}, Target label: {target_label}")

    # 加载权重和输入
    weights, biases = load_weights_mpc_format(weights_file, layer_dims)
    x0 = load_input_mpc_format(input_file, layer_dims[0])

    # 计算网络输出验证
    output = forward(x0, weights, biases)
    pred_label = np.argmax(output)

    print(f"\n网络前向传播验证:")
    print(f"网络输出: {output}")
    print(f"预测类别: {pred_label}")
    print(f"真实标签匹配: {'✓' if pred_label == true_label else '✗'}")

    # 运行CROWN测试
    print(f"\n{'='*60}")
    print("CROWN边界计算结果:")
    print(f"{'='*60}")
    print(f"{'Eps':>10} | {'Lower Bound':>15} | {'Upper Bound':>15} | {'鲁棒性':>8}")
    print("-" * 60)

    results = []
    for eps in eps_list:
        lb, ub = compute_worst_bound_simplified(
            weights, biases, x0, eps, true_label, target_label,
            verbose=verbose, debug_mpc=debug_mpc
        )

        robust = "✓" if lb > 0 else "✗"
        print(f"{eps:>10.3f} | {lb:>15.6f} | {ub:>15.6f} | {robust:>8}")

        results.append({
            'eps': eps,
            'lb': lb,
            'ub': ub,
            'robust': lb > 0
        })

    print("-" * 60)

    # 统计
    robust_count = sum(1 for r in results if r['robust'])
    print(f"\n统计结果:")
    print(f"鲁棒测试: {robust_count}/{len(results)} 通过")

    if robust_count > 0:
        max_robust_eps = max(r['eps'] for r in results if r['robust'])
        print(f"最大鲁棒ε: {max_robust_eps:.3f}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CROWN MPC数据格式测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python script.py  # 使用默认参数
  python script.py --eps 0.01 0.05 0.1 --verbose
  python script.py --debug-mpc  # 开启调试输出
        """
    )

    # # 文件路径参数
    # parser.add_argument('--weights-file', type=str,
    #                     default="shark/shark_crown_ml/crown_mpc_data/vnncomp_mnist_5layer_relu_256_best/weights/weights.dat",
    #                     help='权重文件路径')
    # parser.add_argument('--input-file', type=str,
    #                     default="shark/shark_crown_ml/crown_mpc_data/vnncomp_mnist_5layer_relu_256_best/images/0.bin",
    #                     help='输入文件路径')
    # parser.add_argument('--layer-dims', type=int, nargs='+',
    #                     default=[784, 256, 256, 256, 256, 10],
    #                     help='网络层维度')

    # 文件路径参数
    parser.add_argument('--weights-file', type=str,
                        default="shark/shark_crown_ml/crown_mpc_data/eran_cifar_5layer_relu_100_best/weights/weights.dat",
                        help='权重文件路径')
    parser.add_argument('--input-file', type=str,
                        default="shark/shark_crown_ml/crown_mpc_data/eran_cifar_5layer_relu_100_best/images/0.bin",
                        help='输入文件路径')
    parser.add_argument('--layer-dims', type=int, nargs='+',
                        default=[3072, 100, 100, 100, 100, 10],
                        help='网络层维度')

    # Eps 配置参数
    parser.add_argument('--eps', type=float, nargs='+',
                        default=[0.005, 0.01, 0.025, 0.05, 0.08, 0.1],
                        help='扰动半径列表')

    # 标签参数
    parser.add_argument('--true-label', type=int, default=7,
                        help='真实标签')
    parser.add_argument('--target-label', type=int, default=3,
                        help='目标攻击标签')

    # 调试参数
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出模式')
    parser.add_argument('--debug-mpc', action='store_true',
                        help='开启MPC风格的调试输出')

    args = parser.parse_args()

    # 运行测试
    try:
        results = run_mpc_test(
            args.weights_file, args.input_file, args.layer_dims,
            args.eps, args.true_label, args.target_label,
            verbose=args.verbose, debug_mpc=args.debug_mpc
        )

        if results is None:
            print("测试失败!")
            return 1

        print(f"\n测试完成! 共计算了 {len(results)} 个epsilon值。")
        return 0

    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())