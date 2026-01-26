#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/relu.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/reciprocal.hpp>
#include <shark/protocols/ars.hpp>
#include <shark/utils/timer.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <stdexcept>

using u64 = shark::u64;
using namespace shark::protocols;

const int f = 28;
const u64 SCALAR_ONE = 1ULL << f;

// ==================== 文件加载器类 ====================

class Loader {
    std::ifstream file;
    std::string filename;
public:
    Loader(const std::string &fname) : filename(fname) {
        file.open(fname, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << fname << std::endl;
            std::exit(1);
        }
        std::cout << "  Opened file: " << fname << std::endl;
    }

    // 加载数据并转换为定点数
    void load(shark::span<u64> &X, int precision) {
        int size = X.size();
        for (int i = 0; i < size; i++) {
            float fval;
            file.read((char *)&fval, sizeof(float));
            if (file.fail()) {
                std::cerr << "Error reading from file: " << filename << " at position " << i << std::endl;
                std::exit(1);
            }
            X[i] = (u64)(int64_t)(fval * (1ULL << precision));
        }
    }

    // 加载单个浮点值
    float load_float() {
        float fval;
        file.read((char *)&fval, sizeof(float));
        return fval;
    }

    // 检查是否到达文件末尾
    bool eof() {
        return file.eof();
    }

    ~Loader() {
        if (file.is_open()) {
            file.close();
        }
    }
};

// ==================== 配置结构体 ====================

struct NetworkConfig {
    int num_layers;                    // 网络层数
    std::vector<int> layer_dims;       // 每层维度 [input_dim, hidden1, hidden2, ..., output_dim]
    std::string weights_file;          // 权重文件路径
    std::string input_file;            // 输入文件路径
    float eps;                         // 扰动半径
    int true_label;                    // 真实标签
    int target_label;                  // 目标标签
};

// ==================== 基础工具函数 ====================

u64 float_to_fixed(double val) {
    int64_t sval = (int64_t)(val * SCALAR_ONE);
    return (u64)sval;
}

double fixed_to_float(u64 val) {
    int64_t sval = (int64_t)val;
    return (double)sval / (double)SCALAR_ONE;
}

shark::span<u64> secure_sub(shark::span<u64>& A, shark::span<u64>& B) {
    shark::span<u64> B_neg(B.size());
    for(size_t i = 0; i < B.size(); ++i) B_neg[i] = -B[i];
    return add::call(A, B_neg);
}

shark::span<u64> secure_abs(shark::span<u64>& W) {
    shark::span<u64> W_neg(W.size());
    for(size_t i = 0; i < W.size(); ++i) W_neg[i] = -W[i];
    auto pos = relu::call(W);
    auto neg = relu::call(W_neg);
    return add::call(pos, neg);
}

shark::span<u64> broadcast_scalar(shark::span<u64>& scalar_share, int size) {
    shark::span<u64> vec(size);
    u64 val = scalar_share[0];
    for(int i = 0; i < size; ++i) vec[i] = val;
    return vec;
}

shark::span<u64> compute_alpha_secure(shark::span<u64>& U, shark::span<u64>& L, shark::span<u64>& epsilon_share) {
    size_t size = U.size();
    auto num = relu::call(U);

    shark::span<u64> L_neg(size);
    for(size_t i = 0; i < size; ++i) L_neg[i] = -L[i];
    auto term2 = relu::call(L_neg);

    auto den = add::call(num, term2);
    auto eps_vec = broadcast_scalar(epsilon_share, size);
    den = add::call(den, eps_vec);

    auto den_inv = reciprocal::call(den, f);
    auto alpha = mul::call(num, den_inv);
    return ars::call(alpha, f);
}

shark::span<u64> scale_matrix_by_alpha(shark::span<u64>& W, shark::span<u64>& alpha, int rows, int cols) {
    shark::span<u64> result(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            shark::span<u64> w_elem(1), a_elem(1);
            w_elem[0] = W[r * cols + c];
            a_elem[0] = alpha[c];
            auto prod = mul::call(w_elem, a_elem);
            prod = ars::call(prod, f);
            result[r * cols + c] = prod[0];
        }
    }
    return result;
}

shark::span<u64> dot_product(shark::span<u64>& A, shark::span<u64>& B, int size) {
    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        shark::span<u64> a_elem(1), b_elem(1);
        a_elem[0] = A[i];
        b_elem[0] = B[i];
        auto prod = mul::call(a_elem, b_elem);
        prod = ars::call(prod, f);
        result = add::call(result, prod);
    }
    return result;
}

shark::span<u64> sum_abs(shark::span<u64>& A, int size) {
    auto A_abs = secure_abs(A);
    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        shark::span<u64> elem(1);
        elem[0] = A_abs[i];
        result = add::call(result, elem);
    }
    return result;
}

// ==================== LB/UB Correction 函数 ====================

shark::span<u64> compute_lb_correction_vec(shark::span<u64>& A, shark::span<u64>& LB, int size) {
    shark::span<u64> A_neg(size);
    for(int i = 0; i < size; ++i) A_neg[i] = -A[i];
    auto relu_neg_A = relu::call(A_neg);

    shark::span<u64> LB_neg(size);
    for(int i = 0; i < size; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    auto correction_vec = mul::call(relu_neg_A, relu_neg_LB);
    correction_vec = ars::call(correction_vec, f);

    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        shark::span<u64> elem(1);
        elem[0] = correction_vec[i];
        result = add::call(result, elem);
    }
    return result;
}

shark::span<u64> compute_lb_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB, int rows, int cols) {
    shark::span<u64> corrections(rows);

    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    for(int r = 0; r < rows; ++r) {
        shark::span<u64> A_row(cols);
        for(int c = 0; c < cols; ++c) {
            A_row[c] = A[r * cols + c];
        }

        shark::span<u64> A_row_neg(cols);
        for(int c = 0; c < cols; ++c) A_row_neg[c] = -A_row[c];
        auto relu_neg_A_row = relu::call(A_row_neg);

        auto correction_vec = mul::call(relu_neg_A_row, relu_neg_LB);
        correction_vec = ars::call(correction_vec, f);

        shark::span<u64> sum(1);
        sum[0] = 0;
        for(int c = 0; c < cols; ++c) {
            shark::span<u64> elem(1);
            elem[0] = correction_vec[c];
            sum = add::call(sum, elem);
        }
        corrections[r] = sum[0];
    }
    return corrections;
}

shark::span<u64> compute_ub_correction_vec(shark::span<u64>& A, shark::span<u64>& LB, int size) {
    auto relu_A = relu::call(A);

    shark::span<u64> LB_neg(size);
    for(int i = 0; i < size; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    auto correction_vec = mul::call(relu_A, relu_neg_LB);
    correction_vec = ars::call(correction_vec, f);

    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        shark::span<u64> elem(1);
        elem[0] = correction_vec[i];
        result = add::call(result, elem);
    }
    return result;
}

shark::span<u64> compute_ub_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB, int rows, int cols) {
    shark::span<u64> corrections(rows);

    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    for(int r = 0; r < rows; ++r) {
        shark::span<u64> A_row(cols);
        for(int c = 0; c < cols; ++c) {
            A_row[c] = A[r * cols + c];
        }

        auto relu_A_row = relu::call(A_row);

        auto correction_vec = mul::call(relu_A_row, relu_neg_LB);
        correction_vec = ars::call(correction_vec, f);

        shark::span<u64> sum(1);
        sum[0] = 0;
        for(int c = 0; c < cols; ++c) {
            shark::span<u64> elem(1);
            elem[0] = correction_vec[c];
            sum = add::call(sum, elem);
        }
        corrections[r] = sum[0];
    }
    return corrections;
}

// ==================== 层信息结构体 ====================

struct LayerInfo {
    shark::span<u64> W;      // 权重矩阵
    shark::span<u64> b;      // 偏置向量
    int input_dim;           // 输入维度
    int output_dim;          // 输出维度
};

struct LayerBounds {
    shark::span<u64> UB;     // 上界
    shark::span<u64> LB;     // 下界
    shark::span<u64> alpha;  // 松弛斜率
};

// ==================== CROWN 核心计算类 ====================

class CROWNComputer {
public:
    std::vector<LayerInfo> layers;
    std::vector<LayerBounds> layer_bounds;
    shark::span<u64> x0;
    shark::span<u64> eps_share;
    shark::span<u64> epsilon_share;
    shark::span<u64> ones_input;
    int input_dim;
    int num_layers;

    CROWNComputer(int input_dim_) : input_dim(input_dim_), num_layers(0) {
        ones_input = shark::span<u64>(input_dim);
    }

    void set_input(shark::span<u64>& x0_, shark::span<u64>& eps_, shark::span<u64>& epsilon_, shark::span<u64>& ones_) {
        x0 = x0_;
        eps_share = eps_;
        epsilon_share = epsilon_;
        for(int i = 0; i < input_dim; ++i) {
            ones_input[i] = ones_[i];
        }
    }

    void add_layer(shark::span<u64>& W, shark::span<u64>& b, int in_dim, int out_dim) {
        LayerInfo layer;
        layer.W = W;
        layer.b = b;
        layer.input_dim = in_dim;
        layer.output_dim = out_dim;
        layers.push_back(layer);
        num_layers++;
    }

    // 计算第一层边界 (IBP)
    LayerBounds compute_first_layer_bounds() {
        LayerBounds bounds;
        LayerInfo& layer = layers[0];
        int out_dim = layer.output_dim;
        int in_dim = layer.input_dim;

        // Ax0 = W @ x0
        auto Ax0 = matmul::call(out_dim, in_dim, 1, layer.W, x0);
        Ax0 = ars::call(Ax0, f);

        // dualnorm = sum(|W|, axis=1)
        auto W_abs = secure_abs(layer.W);
        auto dualnorm = matmul::call(out_dim, in_dim, 1, W_abs, ones_input);
        dualnorm = ars::call(dualnorm, f);

        // radius = eps * dualnorm
        auto eps_vec = broadcast_scalar(eps_share, out_dim);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);

        // UB = Ax0 + radius + b
        auto temp1 = add::call(Ax0, radius);
        bounds.UB = add::call(temp1, layer.b);

        // LB = Ax0 - radius + b
        auto temp2 = secure_sub(Ax0, radius);
        bounds.LB = add::call(temp2, layer.b);

        // alpha
        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    // 计算中间层边界 (CROWN) - 与第一个文件一致的版本
    LayerBounds compute_middle_layer_bounds(int layer_idx) {
        LayerBounds bounds;
        LayerInfo& curr_layer = layers[layer_idx];
        LayerBounds& prev_bounds = layer_bounds[layer_idx - 1];

        int out_dim = curr_layer.output_dim;
        int in_dim = curr_layer.input_dim;

        // A = W * diag(alpha_prev)
        auto A = scale_matrix_by_alpha(curr_layer.W, prev_bounds.alpha, out_dim, in_dim);

        // constants = b + A @ b_prev
        shark::span<u64> constants(out_dim);
        for(int i = 0; i < out_dim; ++i) constants[i] = curr_layer.b[i];

        auto Ab_prev = matmul::call(out_dim, in_dim, 1, A, layers[layer_idx - 1].b);
        Ab_prev = ars::call(Ab_prev, f);
        constants = add::call(constants, Ab_prev);

        // Compute corrections for previous layer
        auto lb_corr = compute_lb_correction_matrix(A, prev_bounds.LB, out_dim, in_dim);
        auto ub_corr = compute_ub_correction_matrix(A, prev_bounds.LB, out_dim, in_dim);

        // Backpropagate A through all previous layers to input
        auto A_prop = A;
        for(int i = layer_idx - 1; i >= 0; --i) {
            if(i > 0) {
                // Scale by alpha of layer i-1
                auto A_scaled = scale_matrix_by_alpha(layers[i].W, layer_bounds[i - 1].alpha,
                                                      layers[i].output_dim, layers[i].input_dim);
                A_prop = matmul::call(out_dim, layers[i].output_dim, layers[i].input_dim, A_prop, A_scaled);
            } else {
                A_prop = matmul::call(out_dim, layers[i].output_dim, layers[i].input_dim, A_prop, layers[i].W);
            }
            A_prop = ars::call(A_prop, f);
        }

        // Ax0 = A_prop @ x0
        auto Ax0 = matmul::call(out_dim, input_dim, 1, A_prop, x0);
        Ax0 = ars::call(Ax0, f);

        // dualnorm = sum(|A_prop|, axis=1)
        auto A_abs = secure_abs(A_prop);
        auto dualnorm = matmul::call(out_dim, input_dim, 1, A_abs, ones_input);
        dualnorm = ars::call(dualnorm, f);

        // radius = eps * dualnorm
        auto eps_vec = broadcast_scalar(eps_share, out_dim);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);

        // base = Ax0 + constants
        auto base = add::call(Ax0, constants);

        // UB = base + ub_corr + radius
        bounds.UB = add::call(base, ub_corr);
        bounds.UB = add::call(bounds.UB, radius);

        // LB = base - lb_corr - radius
        bounds.LB = secure_sub(base, lb_corr);
        bounds.LB = secure_sub(bounds.LB, radius);

        // alpha
        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    // 计算最终层的差分验证边界 - 与第一个文件一致的版本
    std::pair<shark::span<u64>, shark::span<u64>> compute_final_diff_bounds(
        shark::span<u64>& diff_vec, int true_label, int target_label) {

        int last_idx = num_layers - 1;
        LayerInfo& last_layer = layers[last_idx];
        int out_dim = last_layer.output_dim;
        int in_dim = last_layer.input_dim;

        // W_diff = W_last^T @ diff_vec
        shark::span<u64> W_diff(in_dim);
        for(int i = 0; i < in_dim; ++i) {
            shark::span<u64> sum(1);
            sum[0] = 0;
            for(int j = 0; j < out_dim; ++j) {
                shark::span<u64> w_elem(1), d_elem(1);
                w_elem[0] = last_layer.W[j * in_dim + i];
                d_elem[0] = diff_vec[j];
                auto prod = mul::call(w_elem, d_elem);
                prod = ars::call(prod, f);
                sum = add::call(sum, prod);
            }
            W_diff[i] = sum[0];
        }

        // b_diff = b_last^T @ diff_vec
        shark::span<u64> b_diff(1);
        b_diff[0] = 0;
        for(int j = 0; j < out_dim; ++j) {
            shark::span<u64> b_elem(1), d_elem(1);
            b_elem[0] = last_layer.b[j];
            d_elem[0] = diff_vec[j];
            auto prod = mul::call(b_elem, d_elem);
            prod = ars::call(prod, f);
            b_diff = add::call(b_diff, prod);
        }

        // A_final = W_diff * alpha_{last-1}
        LayerBounds& prev_bounds = layer_bounds[last_idx - 1];
        shark::span<u64> A_final(in_dim);
        for(int i = 0; i < in_dim; ++i) {
            shark::span<u64> w_elem(1), a_elem(1);
            w_elem[0] = W_diff[i];
            a_elem[0] = prev_bounds.alpha[i];
            auto prod = mul::call(w_elem, a_elem);
            prod = ars::call(prod, f);
            A_final[i] = prod[0];
        }

        // constants = A_final @ b_{last-1} + b_diff
        auto constants = dot_product(A_final, layers[last_idx - 1].b, in_dim);
        constants = add::call(constants, b_diff);

        // Compute corrections for layer last-1
        auto lb_corr = compute_lb_correction_vec(A_final, prev_bounds.LB, in_dim);
        auto ub_corr = compute_ub_correction_vec(A_final, prev_bounds.LB, in_dim);

        // Initialize UB_final and LB_final
        shark::span<u64> UB_final(1);
        shark::span<u64> LB_final(1);
        UB_final[0] = ub_corr[0];
        LB_final[0] = 0;
        LB_final = secure_sub(LB_final, lb_corr);

        // Backpropagate through remaining layers
        // A_current = A_final (as vector, 1 x in_dim)
        shark::span<u64> A_current(in_dim);
        for(int i = 0; i < in_dim; ++i) A_current[i] = A_final[i];

        for(int layer_idx = last_idx - 1; layer_idx >= 1; --layer_idx) {
            LayerInfo& curr_layer = layers[layer_idx];
            LayerBounds& curr_prev_bounds = layer_bounds[layer_idx - 1];

            // A_scaled = W * diag(alpha_{layer_idx-1})
            auto A_scaled = scale_matrix_by_alpha(curr_layer.W, curr_prev_bounds.alpha,
                                                   curr_layer.output_dim, curr_layer.input_dim);

            // A_current = A_current @ A_scaled
            auto A_new = matmul::call(1, curr_layer.output_dim, curr_layer.input_dim, A_current, A_scaled);
            A_new = ars::call(A_new, f);

            // Add A_new @ b_{layer_idx-1} to constants
            shark::span<u64> A_new_vec(curr_layer.input_dim);
            for(int i = 0; i < curr_layer.input_dim; ++i) A_new_vec[i] = A_new[i];

            auto const_add = dot_product(A_new_vec, layers[layer_idx - 1].b, curr_layer.input_dim);
            constants = add::call(constants, const_add);

            // Compute corrections for layer layer_idx-1
            auto lb_corr_layer = compute_lb_correction_vec(A_new_vec, curr_prev_bounds.LB, curr_layer.input_dim);
            auto ub_corr_layer = compute_ub_correction_vec(A_new_vec, curr_prev_bounds.LB, curr_layer.input_dim);

            UB_final = add::call(UB_final, ub_corr_layer);
            LB_final = secure_sub(LB_final, lb_corr_layer);

            // Update A_current
            A_current = shark::span<u64>(curr_layer.input_dim);
            for(int i = 0; i < curr_layer.input_dim; ++i) A_current[i] = A_new[i];
        }

        // Final backprop to input: A_current = A_current @ W0
        auto A_input = matmul::call(1, layers[0].output_dim, layers[0].input_dim, A_current, layers[0].W);
        A_input = ars::call(A_input, f);

        // Ax0 = A_input @ x0
        shark::span<u64> A_input_vec(input_dim);
        for(int i = 0; i < input_dim; ++i) A_input_vec[i] = A_input[i];

        auto Ax0 = dot_product(A_input_vec, x0, input_dim);
        auto dualnorm = sum_abs(A_input_vec, input_dim);
        auto radius = mul::call(dualnorm, eps_share);
        radius = ars::call(radius, f);

        // Final LB = LB_final + constants + Ax0 - radius
        auto base = add::call(Ax0, constants);
        base = add::call(base, LB_final);
        auto final_LB = secure_sub(base, radius);

        // Final UB = UB_final + constants + Ax0 + radius
        auto base_ub = add::call(Ax0, constants);
        base_ub = add::call(base_ub, UB_final);
        auto final_UB = add::call(base_ub, radius);

        return std::make_pair(final_LB, final_UB);
    }

    // 主计算函数 - 类似 Python 的 compute_worst_bound_simplified
    std::pair<shark::span<u64>, shark::span<u64>> compute_worst_bound(
        shark::span<u64>& diff_vec, int true_label, int target_label) {

        // Step 1: 计算第一层边界 (IBP)
        compute_first_layer_bounds();

        // Step 2: 计算中间层边界 (CROWN)
        for(int layer_idx = 1; layer_idx < num_layers - 1; ++layer_idx) {
            compute_middle_layer_bounds(layer_idx);
        }

        // Step 3: 计算最终差分边界
        return compute_final_diff_bounds(diff_vec, true_label, target_label);
    }
};

// ==================== 主函数 ====================

int main(int argc, char **argv) {
    init::from_args(argc, argv);

    // ==================== 网络配置 ====================
    NetworkConfig config;
    config.num_layers = 3;
    config.layer_dims = {784, 20, 20, 10};  // [input, hidden1, hidden2, output]

    // 从 build/ 目录到数据目录的相对路径
    config.weights_file = "shark_crown_ml/crown_mpc_data/mnist_3layer_relu_20_best/weights/weights.dat";
    config.input_file = "shark_crown_ml/crown_mpc_data/mnist_3layer_relu_20_best/images/0.bin";
    config.eps = 0.05;           // 扰动半径
    config.true_label = 7;       // 真实标签 (需要根据实际输入修改)
    config.target_label = 6;     // 目标标签

    // 从命令行参数解析配置 (可选)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--weights=") == 0) {
            config.weights_file = arg.substr(10);
        } else if (arg.find("--input=") == 0) {
            config.input_file = arg.substr(8);
        } else if (arg.find("--eps=") == 0) {
            config.eps = std::stof(arg.substr(6));
        } else if (arg.find("--true_label=") == 0) {
            config.true_label = std::stoi(arg.substr(13));
        } else if (arg.find("--target_label=") == 0) {
            config.target_label = std::stoi(arg.substr(15));
        }
    }

    // 计算网络维度
    int input_dim = config.layer_dims[0];
    int output_dim = config.layer_dims[config.layer_dims.size() - 1];
    int max_dim = 0;
    for (int dim : config.layer_dims) {
        if (dim > max_dim) max_dim = dim;
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "CROWN MPC - File Loading Mode" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Network configuration:" << std::endl;
    std::cout << "  Layers: " << config.num_layers << std::endl;
    std::cout << "  Dimensions: ";
    for (size_t i = 0; i < config.layer_dims.size(); ++i) {
        std::cout << config.layer_dims[i];
        if (i < config.layer_dims.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    std::cout << "  Weights file: " << config.weights_file << std::endl;
    std::cout << "  Input file: " << config.input_file << std::endl;
    std::cout << "  Epsilon: " << config.eps << std::endl;
    std::cout << "  True label: " << config.true_label << std::endl;
    std::cout << "  Target label: " << config.target_label << std::endl;
    std::cout << "============================================\n" << std::endl;

    // ==================== 分配内存 ====================

    // 输入数据
    shark::span<u64> x0(input_dim);
    shark::span<u64> eps_share(1);
    shark::span<u64> epsilon_share(1);
    shark::span<u64> diff_vec(output_dim);
    shark::span<u64> ones_global(max_dim);

    // 权重和偏置 (动态分配)
    std::vector<shark::span<u64>> weights(config.num_layers);
    std::vector<shark::span<u64>> biases(config.num_layers);

    for (int i = 0; i < config.num_layers; ++i) {
        int in_dim = config.layer_dims[i];
        int out_dim = config.layer_dims[i + 1];
        weights[i] = shark::span<u64>(out_dim * in_dim);
        biases[i] = shark::span<u64>(out_dim);
    }

    // ==================== CLIENT 初始化 (加载输入) ====================

    if (party == CLIENT) {
        std::cout << "[CLIENT] Loading input data..." << std::endl;

        // 加载输入图像
        Loader input_loader(config.input_file);
        input_loader.load(x0, f);
        std::cout << "  Loaded input: " << input_dim << " values" << std::endl;

        // 设置 epsilon (扰动半径)
        eps_share[0] = float_to_fixed(config.eps);
        epsilon_share[0] = float_to_fixed(0.000001);  // 数值稳定性

        // 设置 ones 向量
        for (int i = 0; i < max_dim; ++i) {
            ones_global[i] = SCALAR_ONE;
        }

        // 设置 diff_vec (用于鲁棒性验证)
        for (int i = 0; i < output_dim; ++i) {
            diff_vec[i] = 0;
        }
        diff_vec[config.true_label] = float_to_fixed(1.0);
        diff_vec[config.target_label] = float_to_fixed(-1.0);

        std::cout << "[CLIENT] Input loading complete." << std::endl;
    }

    // ==================== SERVER 初始化 (加载权重) ====================

    if (party == SERVER) {
        std::cout << "[SERVER] Loading model weights..." << std::endl;

        Loader weights_loader(config.weights_file);

        for (int i = 0; i < config.num_layers; ++i) {
            int in_dim = config.layer_dims[i];
            int out_dim = config.layer_dims[i + 1];

            // 加载权重矩阵
            weights_loader.load(weights[i], f);
            std::cout << "  W" << (i+1) << ": " << out_dim << " x " << in_dim
                      << " = " << (out_dim * in_dim) << " values" << std::endl;

            // 加载偏置向量
            weights_loader.load(biases[i], f);
            std::cout << "  b" << (i+1) << ": " << out_dim << " values" << std::endl;
        }

        // SERVER 端的 ones_global 初始化为 0
        for (int i = 0; i < max_dim; ++i) {
            ones_global[i] = 0;
        }

        std::cout << "[SERVER] Weight loading complete." << std::endl;
    }

    // ==================== 秘密共享输入阶段 ====================

    std::cout << "\n[MPC] Starting secret sharing..." << std::endl;
    shark::utils::start_timer("input");

    // CLIENT 数据秘密共享
    input::call(x0, CLIENT);
    input::call(eps_share, CLIENT);
    input::call(epsilon_share, CLIENT);
    input::call(diff_vec, CLIENT);
    input::call(ones_global, CLIENT);

    // SERVER 权重秘密共享
    for (int i = 0; i < config.num_layers; ++i) {
        input::call(weights[i], SERVER);
        input::call(biases[i], SERVER);
    }

    shark::utils::stop_timer("input");
    std::cout << "[MPC] Secret sharing complete." << std::endl;

    if (party != DEALER) peer->sync();

    // ==================== CROWN 计算阶段 ====================

    std::cout << "\n[MPC] Starting CROWN computation..." << std::endl;
    shark::utils::start_timer("crown_mpc");

    // 创建 CROWN 计算器
    CROWNComputer crown(input_dim);

    // 设置输入
    crown.set_input(x0, eps_share, epsilon_share, ones_global);

    // 添加网络层
    for (int i = 0; i < config.num_layers; ++i) {
        int in_dim = config.layer_dims[i];
        int out_dim = config.layer_dims[i + 1];
        crown.add_layer(weights[i], biases[i], in_dim, out_dim);
    }

    // 计算边界
    auto [final_LB, final_UB] = crown.compute_worst_bound(diff_vec, config.true_label, config.target_label);

    shark::utils::stop_timer("crown_mpc");
    std::cout << "[MPC] CROWN computation complete." << std::endl;

    // ==================== 输出结果阶段 ====================

    output::call(final_LB);
    output::call(final_UB);

    if (party != DEALER) {
        double lb_val = fixed_to_float(final_LB[0]);
        double ub_val = fixed_to_float(final_UB[0]);

        std::cout << "\n============================================" << std::endl;
        std::cout << "CROWN MPC Results" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "True label: " << config.true_label << std::endl;
        std::cout << "Target label: " << config.target_label << std::endl;
        std::cout << "Epsilon: " << config.eps << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "Lower Bound (LB): " << std::fixed << std::setprecision(6) << lb_val << std::endl;
        std::cout << "Upper Bound (UB): " << std::fixed << std::setprecision(6) << ub_val << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "Robust against target " << config.target_label << ": "
                  << (lb_val > 0 ? "YES ✓" : "NO ✗") << std::endl;
        std::cout << "============================================" << std::endl;
    }

    finalize::call();
    shark::utils::print_all_timers();

    return 0;
}