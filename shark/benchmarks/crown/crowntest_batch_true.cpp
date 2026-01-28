/**
 * crowntest_batch_true.cpp
 *
 * 真正的批量化CROWN实现 - 完整保留原始算法逻辑
 *
 * 与 crowntest_batch.cpp 的区别：
 * - crowntest_batch: 串行处理每张图片，每张图片完整运行CROWN → N次协议调用
 * - crowntest_batch_true: 将N张图片打包成矩阵，一次运行CROWN → 1次协议调用
 *
 * 关键：所有计算逻辑与原始 crowntest.cpp / crowntest_optimized.cpp 完全相同
 * 只是数据维度从 (dim,) 变成 (dim * batch_size)
 */

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
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>

using u64 = shark::u64;
using namespace shark::protocols;

const int f = 26;
const u64 SCALAR_ONE = 1ULL << f;

// ==================== Batch Config ====================
struct ImageConfig {
    int image_id;
    int true_label;
    int target_label;
};

std::vector<ImageConfig> load_batch_config(const std::string& config_file) {
    std::vector<ImageConfig> configs;
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open batch config file: " << config_file << std::endl;
        return configs;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        ImageConfig cfg;
        char comma;
        if (iss >> cfg.image_id >> comma >> cfg.true_label >> comma >> cfg.target_label) {
            configs.push_back(cfg);
        }
    }
    return configs;
}

// ==================== Loader ====================
class Loader {
    std::ifstream file;
public:
    Loader(const std::string &fname) {
        file.open(fname, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open file: " << fname << std::endl;
            std::exit(1);
        }
    }
    void load(shark::span<u64> &X, int precision) {
        for (size_t i = 0; i < X.size(); i++) {
            float fval;
            file.read((char *)&fval, sizeof(float));
            X[i] = (u64)(int64_t)(fval * (1ULL << precision));
        }
    }
    ~Loader() { if (file.is_open()) file.close(); }
};

// ==================== Utils ====================
double fixed_to_float(u64 val) {
    return (double)(int64_t)val / (double)SCALAR_ONE;
}

u64 float_to_fixed(double val) {
    return (u64)(int64_t)(val * SCALAR_ONE);
}

// ==================== 批量化工具函数 ====================
// 所有函数与原始版本逻辑相同，只是处理 (dim * B) 的数据

// 批量减法: A - B
shark::span<u64> secure_sub_batch(shark::span<u64>& A, shark::span<u64>& B) {
    shark::span<u64> B_neg(B.size());
    for(size_t i = 0; i < B.size(); ++i) B_neg[i] = -B[i];
    return add::call(A, B_neg);
}

// 批量绝对值: |W|
shark::span<u64> secure_abs_batch(shark::span<u64>& W) {
    shark::span<u64> W_neg(W.size());
    for(size_t i = 0; i < W.size(); ++i) W_neg[i] = -W[i];
    auto pos = relu::call(W);
    auto neg = relu::call(W_neg);
    return add::call(pos, neg);
}

// 广播标量到批量: scalar (1,) -> vec (size * B,)
shark::span<u64> broadcast_scalar_batch(shark::span<u64>& scalar, int size, int B) {
    shark::span<u64> result(size * B);
    u64 val = scalar[0];
    for(int i = 0; i < size * B; ++i) result[i] = val;
    return result;
}

// 广播bias到批量: b (out_dim,) -> b_batch (out_dim * B,)
// 存储格式: [b_img0, b_img1, ..., b_imgB-1] 每个b_img是 (out_dim,)
shark::span<u64> broadcast_bias_batch(shark::span<u64>& b, int out_dim, int B) {
    shark::span<u64> result(out_dim * B);
    for(int img = 0; img < B; ++img) {
        for(int i = 0; i < out_dim; ++i) {
            result[img * out_dim + i] = b[i];
        }
    }
    return result;
}

// 批量行和: 计算每行的和
// W_abs: (rows * cols), 结果: (rows,) 然后广播到 (rows * B)
shark::span<u64> compute_row_sum_broadcast(shark::span<u64>& W_abs, int rows, int cols, int B) {
    shark::span<u64> result(rows * B);
    for(int row = 0; row < rows; ++row) {
        u64 sum = 0;
        for(int col = 0; col < cols; ++col) {
            sum += W_abs[row * cols + col];
        }
        // 广播到所有图片
        for(int img = 0; img < B; ++img) {
            result[img * rows + row] = sum;
        }
    }
    return result;
}

// ==================== 改进的 Reciprocal ====================
shark::span<u64> newton_refine_batch(shark::span<u64>& a, shark::span<u64>& x_n,
                                      shark::span<u64>& two_share, int size, int B) {
    auto ax = mul::call(a, x_n);
    ax = ars::call(ax, f);
    auto two_vec = broadcast_scalar_batch(two_share, size, B);
    auto diff = secure_sub_batch(two_vec, ax);
    auto x_next = mul::call(x_n, diff);
    return ars::call(x_next, f);
}

shark::span<u64> improved_reciprocal_batch(shark::span<u64>& a, shark::span<u64>& two_share,
                                            int size, int B, int iterations = 1) {
    auto x = reciprocal::call(a, f);
    for(int i = 0; i < iterations; ++i) {
        x = newton_refine_batch(a, x, two_share, size, B);
    }
    return x;
}

// ==================== Alpha 计算 (与原版完全相同的逻辑) ====================
shark::span<u64> secure_clamp_upper_1_batch(shark::span<u64>& x, shark::span<u64>& one_share,
                                             int size, int B) {
    auto ones = broadcast_scalar_batch(one_share, size, B);
    auto one_minus_x = secure_sub_batch(ones, x);
    auto relu_part = relu::call(one_minus_x);
    return secure_sub_batch(ones, relu_part);
}

shark::span<u64> compute_alpha_batch(shark::span<u64>& UB, shark::span<u64>& LB,
                                      shark::span<u64>& epsilon_share,
                                      shark::span<u64>& one_share,
                                      shark::span<u64>& two_share,
                                      int size, int B) {
    // alpha = relu(UB) / (relu(UB) + relu(-LB) + epsilon)
    auto num = relu::call(UB);

    shark::span<u64> LB_neg(size * B);
    for(int i = 0; i < size * B; ++i) LB_neg[i] = -LB[i];
    auto term2 = relu::call(LB_neg);

    auto den = add::call(num, term2);
    auto eps_vec = broadcast_scalar_batch(epsilon_share, size, B);
    den = add::call(den, eps_vec);

    auto den_inv = improved_reciprocal_batch(den, two_share, size, B, 3);
    auto alpha = mul::call(num, den_inv);
    alpha = ars::call(alpha, f);

    return secure_clamp_upper_1_batch(alpha, one_share, size, B);
}

// ==================== 批量 scale_matrix_by_alpha ====================
// 与原版相同: W_scaled[r][c] = W[r][c] * alpha[c]
// 批量版本: 对每张图片分别缩放
shark::span<u64> scale_matrix_by_alpha_batch(shark::span<u64>& W, shark::span<u64>& alpha,
                                              int rows, int cols, int B) {
    // W: (rows * cols) - 共享权重
    // alpha: (cols * B) - 每张图片的alpha
    // 结果: (rows * cols * B)
    shark::span<u64> alpha_expanded(rows * cols * B);
    for(int img = 0; img < B; ++img) {
        for(int r = 0; r < rows; ++r) {
            for(int c = 0; c < cols; ++c) {
                alpha_expanded[img * rows * cols + r * cols + c] = alpha[img * cols + c];
            }
        }
    }

    // 广播W到每张图片
    shark::span<u64> W_expanded(rows * cols * B);
    for(int img = 0; img < B; ++img) {
        for(int i = 0; i < rows * cols; ++i) {
            W_expanded[img * rows * cols + i] = W[i];
        }
    }

    auto result = mul::call(W_expanded, alpha_expanded);
    return ars::call(result, f);
}

// ==================== 批量点积 ====================
shark::span<u64> dot_product_batch(shark::span<u64>& A, shark::span<u64>& B_vec, int size, int batch) {
    auto prod = mul::call(A, B_vec);
    prod = ars::call(prod, f);
    shark::span<u64> result(batch);
    for(int img = 0; img < batch; ++img) {
        u64 sum = 0;
        for(int i = 0; i < size; ++i) {
            sum += prod[img * size + i];
        }
        result[img] = sum;
    }
    return result;
}

// ==================== 批量 correction 计算 ====================
// 与原版完全相同的逻辑

// LB correction for vector
shark::span<u64> compute_lb_correction_vec_batch(shark::span<u64>& A, shark::span<u64>& LB,
                                                   int size, int B) {
    // A: (size * B), LB: (size * B)
    shark::span<u64> A_neg(size * B);
    for(int i = 0; i < size * B; ++i) A_neg[i] = -A[i];
    auto relu_neg_A = relu::call(A_neg);

    shark::span<u64> LB_neg(size * B);
    for(int i = 0; i < size * B; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    auto correction_vec = mul::call(relu_neg_A, relu_neg_LB);
    correction_vec = ars::call(correction_vec, f);

    // 对每张图片求和
    shark::span<u64> result(B);
    for(int img = 0; img < B; ++img) {
        u64 sum = 0;
        for(int i = 0; i < size; ++i) {
            sum += correction_vec[img * size + i];
        }
        result[img] = sum;
    }
    return result;
}

shark::span<u64> compute_ub_correction_vec_batch(shark::span<u64>& A, shark::span<u64>& LB,
                                                   int size, int B) {
    auto relu_A = relu::call(A);

    shark::span<u64> LB_neg(size * B);
    for(int i = 0; i < size * B; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    auto correction_vec = mul::call(relu_A, relu_neg_LB);
    correction_vec = ars::call(correction_vec, f);

    shark::span<u64> result(B);
    for(int img = 0; img < B; ++img) {
        u64 sum = 0;
        for(int i = 0; i < size; ++i) {
            sum += correction_vec[img * size + i];
        }
        result[img] = sum;
    }
    return result;
}

// LB/UB correction for matrix (用于中间层)
shark::span<u64> compute_lb_correction_matrix_batch(shark::span<u64>& A, shark::span<u64>& LB,
                                                      int rows, int cols, int B) {
    // A: (rows * cols * B), LB: (cols * B)
    // LB_neg
    shark::span<u64> LB_neg(cols * B);
    for(int i = 0; i < cols * B; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    // A_neg
    shark::span<u64> A_neg(rows * cols * B);
    for(int i = 0; i < rows * cols * B; ++i) A_neg[i] = -A[i];
    auto relu_neg_A = relu::call(A_neg);

    // 扩展 relu_neg_LB 到 (rows * cols * B)
    shark::span<u64> relu_neg_LB_expanded(rows * cols * B);
    for(int img = 0; img < B; ++img) {
        for(int r = 0; r < rows; ++r) {
            for(int c = 0; c < cols; ++c) {
                relu_neg_LB_expanded[img * rows * cols + r * cols + c] = relu_neg_LB[img * cols + c];
            }
        }
    }

    auto correction_matrix = mul::call(relu_neg_A, relu_neg_LB_expanded);
    correction_matrix = ars::call(correction_matrix, f);

    // 对每张图片的每行求和
    shark::span<u64> corrections(rows * B);
    for(int img = 0; img < B; ++img) {
        for(int r = 0; r < rows; ++r) {
            u64 sum = 0;
            for(int c = 0; c < cols; ++c) {
                sum += correction_matrix[img * rows * cols + r * cols + c];
            }
            corrections[img * rows + r] = sum;
        }
    }
    return corrections;
}

shark::span<u64> compute_ub_correction_matrix_batch(shark::span<u64>& A, shark::span<u64>& LB,
                                                      int rows, int cols, int B) {
    shark::span<u64> LB_neg(cols * B);
    for(int i = 0; i < cols * B; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    auto relu_A = relu::call(A);

    shark::span<u64> relu_neg_LB_expanded(rows * cols * B);
    for(int img = 0; img < B; ++img) {
        for(int r = 0; r < rows; ++r) {
            for(int c = 0; c < cols; ++c) {
                relu_neg_LB_expanded[img * rows * cols + r * cols + c] = relu_neg_LB[img * cols + c];
            }
        }
    }

    auto correction_matrix = mul::call(relu_A, relu_neg_LB_expanded);
    correction_matrix = ars::call(correction_matrix, f);

    shark::span<u64> corrections(rows * B);
    for(int img = 0; img < B; ++img) {
        for(int r = 0; r < rows; ++r) {
            u64 sum = 0;
            for(int c = 0; c < cols; ++c) {
                sum += correction_matrix[img * rows * cols + r * cols + c];
            }
            corrections[img * rows + r] = sum;
        }
    }
    return corrections;
}

// ==================== 批量 CROWN 计算器 ====================
// 完整保留原始CROWN算法的所有逻辑

struct LayerInfo {
    shark::span<u64> W;  // (out_dim * in_dim) - 共享
    shark::span<u64> b;  // (out_dim) - 共享
    int input_dim;
    int output_dim;
};

struct LayerBoundsBatch {
    shark::span<u64> UB;     // (out_dim * B)
    shark::span<u64> LB;     // (out_dim * B)
    shark::span<u64> alpha;  // (out_dim * B)
};

class BatchCROWNComputer {
public:
    std::vector<LayerInfo> layers;
    std::vector<LayerBoundsBatch> layer_bounds;

    shark::span<u64> X0;           // (input_dim * B) - 所有图片
    shark::span<u64> diff_vecs;    // (output_dim * B) - 所有diff向量
    shark::span<u64> eps_share;
    shark::span<u64> epsilon_share;
    shark::span<u64> one_share;
    shark::span<u64> two_share;

    int input_dim;
    int num_layers;
    int B;  // batch size

    BatchCROWNComputer(int input_dim_, int num_layers_, int B_)
        : input_dim(input_dim_), num_layers(num_layers_), B(B_) {}

    void set_input(shark::span<u64>& x0_batch, shark::span<u64>& diff_batch,
                   shark::span<u64>& eps_, shark::span<u64>& epsilon_,
                   shark::span<u64>& one_, shark::span<u64>& two_) {
        X0 = x0_batch;
        diff_vecs = diff_batch;
        eps_share = eps_;
        epsilon_share = epsilon_;
        one_share = one_;
        two_share = two_;
    }

    void add_layer(shark::span<u64>& W, shark::span<u64>& b, int in_dim, int out_dim) {
        LayerInfo layer;
        layer.W = W;
        layer.b = b;
        layer.input_dim = in_dim;
        layer.output_dim = out_dim;
        layers.push_back(layer);
    }

    void reset_bounds() {
        layer_bounds.clear();
    }

    // ========== 第一层边界 (与原版逻辑完全相同) ==========
    LayerBoundsBatch compute_first_layer_bounds_batch() {
        LayerBoundsBatch bounds;
        LayerInfo& layer = layers[0];
        int out_dim = layer.output_dim;
        int in_dim = layer.input_dim;

        // Ax0 = W @ X0, X0: (in_dim, B)
        // matmul(M, K, N, A, B): A(M×K) @ B(K×N) -> (M×N)
        auto Ax0 = matmul::call(out_dim, in_dim, B, layer.W, X0);
        Ax0 = ars::call(Ax0, f);

        // 计算 ||W||_∞ (行范数的绝对值)
        auto W_abs = secure_abs_batch(layer.W);
        auto dualnorm = compute_row_sum_broadcast(W_abs, out_dim, in_dim, B);

        // radius = dualnorm * eps
        auto eps_vec = broadcast_scalar_batch(eps_share, out_dim, B);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);
        radius = relu::call(radius);

        // 广播 bias
        auto b_batch = broadcast_bias_batch(layer.b, out_dim, B);

        // UB = Ax0 + b + radius
        // LB = Ax0 + b - radius
        auto temp1 = add::call(Ax0, radius);
        bounds.UB = add::call(temp1, b_batch);

        auto temp2 = secure_sub_batch(Ax0, radius);
        bounds.LB = add::call(temp2, b_batch);

        // 计算 alpha
        bounds.alpha = compute_alpha_batch(bounds.UB, bounds.LB, epsilon_share,
                                            one_share, two_share, out_dim, B);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    // ========== 中间层边界 (与原版逻辑完全相同) ==========
    // 注意：所有协议调用必须在批量级别，不能在per-image循环内
    LayerBoundsBatch compute_middle_layer_bounds_batch(int layer_idx) {
        LayerBoundsBatch bounds;
        LayerInfo& curr_layer = layers[layer_idx];
        int out_dim = curr_layer.output_dim;

        // A_prop 初始化为当前层权重，需要扩展到 B 张图片
        // A_prop: (out_dim * prev_dim * B)
        int prev_dim = curr_layer.input_dim;
        shark::span<u64> A_prop(out_dim * prev_dim * B);
        for(int img = 0; img < B; ++img) {
            for(int i = 0; i < out_dim * prev_dim; ++i) {
                A_prop[img * out_dim * prev_dim + i] = curr_layer.W[i];
            }
        }

        // constants: (out_dim * B)
        shark::span<u64> constants(out_dim * B);
        for(int img = 0; img < B; ++img) {
            for(int i = 0; i < out_dim; ++i) {
                constants[img * out_dim + i] = curr_layer.b[i];
            }
        }

        // corrections: (out_dim * B)
        shark::span<u64> lb_corr_total(out_dim * B);
        shark::span<u64> ub_corr_total(out_dim * B);
        for(int i = 0; i < out_dim * B; ++i) {
            lb_corr_total[i] = 0;
            ub_corr_total[i] = 0;
        }

        // 反向传播
        for(int i = layer_idx - 1; i >= 0; --i) {
            int this_out = layers[i].output_dim;
            int this_in = layers[i].input_dim;

            // A_prop = A_prop * diag(alpha)
            A_prop = scale_matrix_by_alpha_batch(A_prop, layer_bounds[i].alpha, out_dim, this_out, B);

            // 计算 corrections
            auto lb_c = compute_lb_correction_matrix_batch(A_prop, layer_bounds[i].LB, out_dim, this_out, B);
            auto ub_c = compute_ub_correction_matrix_batch(A_prop, layer_bounds[i].LB, out_dim, this_out, B);

            lb_corr_total = add::call(lb_corr_total, lb_c);
            ub_corr_total = add::call(ub_corr_total, ub_c);

            // constants += A_prop @ b[i]
            // 批量计算: 对每张图片的A_prop乘以b，然后本地累加
            // A_prop: (out_dim * this_out * B), b: (this_out)
            // 我们需要计算每张图片的 sum_j(A_prop[row,j] * b[j])
            // 这是一个本地操作，不需要协议调用
            for(int img = 0; img < B; ++img) {
                for(int row = 0; row < out_dim; ++row) {
                    u64 dot = 0;
                    for(int j = 0; j < this_out; ++j) {
                        // A_prop[img, row, j] * b[j] (本地乘法，因为b是明文共享的一部分)
                        dot += (A_prop[img * out_dim * this_out + row * this_out + j] *
                                layers[i].b[j]) >> f;
                    }
                    constants[img * out_dim + row] += dot;
                }
            }

            // A_prop = A_prop @ W[i]
            // 批量matmul: 每张图片的A_prop (out_dim x this_out) @ W (this_out x this_in)
            // = 结果 (out_dim x this_in x B)
            // 使用批量matmul: matmul(out_dim * B, this_out, this_in, A_prop_reshaped, W)
            // 但这需要重新组织数据...

            // 简化方案: 本地矩阵乘法 (因为W是共享的，A_prop也是共享的)
            // A_next[img, row, col] = sum_k A_prop[img, row, k] * W[k, col]
            shark::span<u64> A_next(out_dim * this_in * B);
            for(int img = 0; img < B; ++img) {
                for(int row = 0; row < out_dim; ++row) {
                    for(int col = 0; col < this_in; ++col) {
                        u64 sum = 0;
                        for(int k = 0; k < this_out; ++k) {
                            sum += (A_prop[img * out_dim * this_out + row * this_out + k] *
                                    layers[i].W[k * this_in + col]) >> f;
                        }
                        A_next[img * out_dim * this_in + row * this_in + col] = sum;
                    }
                }
            }
            A_prop = A_next;
        }

        // Ax0 = A_prop @ X0
        // A_prop: (out_dim * input_dim * B), X0: (input_dim * B)
        // 每张图片: A_prop[img] (out_dim x input_dim) @ x0[img] (input_dim) = (out_dim)
        // 这需要协议调用因为X0是秘密共享的
        // 但我们不能在循环内调用协议...

        // 解决方案: 使用单次批量matmul
        // 重组数据使得可以用一次matmul处理所有图片
        // A_prop_flat: (out_dim * B, input_dim), X0: (input_dim, B)
        // matmul(out_dim * B, input_dim, 1, ...) 不对...

        // 正确方案: 逐元素乘法 + 本地求和
        // Ax0[img, row] = sum_col A_prop[img, row, col] * X0[img, col]
        // = sum over col of (A_prop * X0_expanded)[img, row, col]

        // 扩展 X0 到 (out_dim * input_dim * B)
        shark::span<u64> X0_expanded(out_dim * input_dim * B);
        for(int img = 0; img < B; ++img) {
            for(int row = 0; row < out_dim; ++row) {
                for(int col = 0; col < input_dim; ++col) {
                    X0_expanded[img * out_dim * input_dim + row * input_dim + col] = X0[img * input_dim + col];
                }
            }
        }

        // 批量乘法: A_prop * X0_expanded
        auto AX_prod = mul::call(A_prop, X0_expanded);
        AX_prod = ars::call(AX_prod, f);

        // 本地求和得到 Ax0
        shark::span<u64> Ax0(out_dim * B);
        for(int img = 0; img < B; ++img) {
            for(int row = 0; row < out_dim; ++row) {
                u64 sum = 0;
                for(int col = 0; col < input_dim; ++col) {
                    sum += AX_prod[img * out_dim * input_dim + row * input_dim + col];
                }
                Ax0[img * out_dim + row] = sum;
            }
        }

        // 计算 dualnorm (对每张图片的 A_prop) - 本地操作
        auto A_abs = secure_abs_batch(A_prop);
        shark::span<u64> dualnorm(out_dim * B);
        for(int img = 0; img < B; ++img) {
            for(int row = 0; row < out_dim; ++row) {
                u64 sum = 0;
                for(int col = 0; col < input_dim; ++col) {
                    sum += A_abs[img * out_dim * input_dim + row * input_dim + col];
                }
                dualnorm[img * out_dim + row] = sum;
            }
        }

        // radius
        auto eps_vec = broadcast_scalar_batch(eps_share, out_dim, B);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);
        radius = relu::call(radius);

        // 计算 bounds
        auto base = add::call(Ax0, constants);

        bounds.UB = add::call(base, ub_corr_total);
        bounds.UB = add::call(bounds.UB, radius);

        bounds.LB = secure_sub_batch(base, lb_corr_total);
        bounds.LB = secure_sub_batch(bounds.LB, radius);

        bounds.alpha = compute_alpha_batch(bounds.UB, bounds.LB, epsilon_share,
                                            one_share, two_share, out_dim, B);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    // ========== 最终层边界 (与原版逻辑完全相同) ==========
    // 注意：所有协议调用必须在批量级别，不能在per-image循环内
    std::pair<shark::span<u64>, shark::span<u64>> compute_final_diff_bounds_batch() {
        int last_idx = num_layers - 1;
        LayerInfo& last_layer = layers[last_idx];
        int in_dim = last_layer.input_dim;
        int out_dim = last_layer.output_dim;

        // W_diff = diff_vec^T @ W (批量计算)
        // diff_vecs: (out_dim * B), W: (out_dim * in_dim)
        // W_diff[img, j] = sum_k diff_vecs[img, k] * W[k, j]
        // 使用批量乘法: 扩展W到(out_dim * in_dim * B), 扩展diff到(out_dim * in_dim * B)
        shark::span<u64> W_expanded(out_dim * in_dim * B);
        shark::span<u64> diff_expanded(out_dim * in_dim * B);
        for(int img = 0; img < B; ++img) {
            for(int k = 0; k < out_dim; ++k) {
                for(int j = 0; j < in_dim; ++j) {
                    W_expanded[img * out_dim * in_dim + k * in_dim + j] = last_layer.W[k * in_dim + j];
                    diff_expanded[img * out_dim * in_dim + k * in_dim + j] = diff_vecs[img * out_dim + k];
                }
            }
        }

        auto W_diff_prod = mul::call(W_expanded, diff_expanded);
        W_diff_prod = ars::call(W_diff_prod, f);

        // 求和得到 W_diff: (in_dim * B)
        shark::span<u64> W_diff(in_dim * B);
        for(int img = 0; img < B; ++img) {
            for(int j = 0; j < in_dim; ++j) {
                u64 sum = 0;
                for(int k = 0; k < out_dim; ++k) {
                    sum += W_diff_prod[img * out_dim * in_dim + k * in_dim + j];
                }
                W_diff[img * in_dim + j] = sum;
            }
        }

        // b_diff = dot(b, diff_vec) - 本地计算 (b是共享的一部分)
        shark::span<u64> constants(B);
        for(int img = 0; img < B; ++img) {
            u64 sum = 0;
            for(int j = 0; j < out_dim; ++j) {
                sum += (last_layer.b[j] * diff_vecs[img * out_dim + j]) >> f;
            }
            constants[img] = sum;
        }

        // corrections
        shark::span<u64> lb_corr_total(B);
        shark::span<u64> ub_corr_total(B);
        for(int i = 0; i < B; ++i) {
            lb_corr_total[i] = 0;
            ub_corr_total[i] = 0;
        }

        // A_prop: (in_dim * B) - 但在反向传播中维度会变化
        auto A_prop = W_diff;
        int curr_dim = in_dim;

        // 反向传播
        for(int i = last_idx - 1; i >= 0; --i) {
            int this_out = layers[i].output_dim;
            int this_in = layers[i].input_dim;

            // 此时 A_prop 应该是 (this_out * B)
            // A_prop = A_prop * alpha (逐元素) - 批量协议调用
            auto A_scaled = mul::call(A_prop, layer_bounds[i].alpha);
            A_prop = ars::call(A_scaled, f);

            // corrections - 批量协议调用
            auto lb_c = compute_lb_correction_vec_batch(A_prop, layer_bounds[i].LB, this_out, B);
            auto ub_c = compute_ub_correction_vec_batch(A_prop, layer_bounds[i].LB, this_out, B);

            for(int img = 0; img < B; ++img) {
                lb_corr_total[img] += lb_c[img];
                ub_corr_total[img] += ub_c[img];
            }

            // constants += A_prop @ b - 本地计算
            for(int img = 0; img < B; ++img) {
                u64 dot = 0;
                for(int j = 0; j < this_out; ++j) {
                    dot += (A_prop[img * this_out + j] * layers[i].b[j]) >> f;
                }
                constants[img] += dot;
            }

            // A_prop = A_prop @ W - 本地矩阵乘法
            // A_prop: (this_out * B), W: (this_out * this_in)
            // A_next[img, j] = sum_k A_prop[img, k] * W[k, j]
            shark::span<u64> A_next(this_in * B);
            for(int img = 0; img < B; ++img) {
                for(int j = 0; j < this_in; ++j) {
                    u64 sum = 0;
                    for(int k = 0; k < this_out; ++k) {
                        sum += (A_prop[img * this_out + k] * layers[i].W[k * this_in + j]) >> f;
                    }
                    A_next[img * this_in + j] = sum;
                }
            }
            A_prop = A_next;
            curr_dim = this_in;
        }

        // Ax0 = A_prop @ x0 - 批量乘法 + 本地求和
        // A_prop: (input_dim * B), X0: (input_dim * B)
        auto ax_prod = mul::call(A_prop, X0);
        ax_prod = ars::call(ax_prod, f);

        shark::span<u64> Ax0(B);
        for(int img = 0; img < B; ++img) {
            u64 sum = 0;
            for(int j = 0; j < input_dim; ++j) {
                sum += ax_prod[img * input_dim + j];
            }
            Ax0[img] = sum;
        }

        // dualnorm = sum(|A_prop|) - 批量绝对值
        auto A_abs = secure_abs_batch(A_prop);
        shark::span<u64> dualnorm(B);
        for(int img = 0; img < B; ++img) {
            u64 sum = 0;
            for(int j = 0; j < input_dim; ++j) {
                sum += A_abs[img * input_dim + j];
            }
            dualnorm[img] = sum;
        }

        // radius = dualnorm * eps
        auto radius = mul::call(dualnorm, eps_share);
        radius = ars::call(radius, f);
        radius = relu::call(radius);

        // final bounds
        shark::span<u64> base(B);
        for(int img = 0; img < B; ++img) {
            base[img] = Ax0[img] + constants[img];
        }

        shark::span<u64> final_LB(B);
        shark::span<u64> final_UB(B);
        for(int img = 0; img < B; ++img) {
            final_LB[img] = base[img] - lb_corr_total[img] - radius[img];
            final_UB[img] = base[img] + ub_corr_total[img] + radius[img];
        }

        return std::make_pair(final_LB, final_UB);
    }

    std::pair<shark::span<u64>, shark::span<u64>> compute_worst_bound_batch() {
        reset_bounds();
        compute_first_layer_bounds_batch();

        for(int layer_idx = 1; layer_idx < num_layers - 1; ++layer_idx) {
            compute_middle_layer_bounds_batch(layer_idx);
        }

        return compute_final_diff_bounds_batch();
    }
};

// ==================== Main ====================
int main(int argc, char **argv) {
    init::from_args(argc, argv);

    std::string model_name = "eran_cifar_5layer_relu_100_best";
    int input_dim = 3072;
    int output_dim = 10;
    int num_layers = 5;
    int hidden_dim = 100;
    float eps = 0.002;
    std::string batch_config_file = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--model=") == 0) model_name = arg.substr(8);
        else if (arg.find("--num_layers=") == 0) num_layers = std::stoi(arg.substr(13));
        else if (arg.find("--hidden_dim=") == 0) hidden_dim = std::stoi(arg.substr(13));
        else if (arg.find("--input_dim=") == 0) input_dim = std::stoi(arg.substr(12));
        else if (arg.find("--output_dim=") == 0) output_dim = std::stoi(arg.substr(13));
        else if (arg.find("--eps=") == 0) eps = std::stof(arg.substr(6));
        else if (arg.find("--batch_config=") == 0) batch_config_file = arg.substr(15);
    }

    std::vector<ImageConfig> batch_configs = load_batch_config(batch_config_file);
    if (batch_configs.empty()) {
        std::cerr << "No valid batch configs. Use --batch_config=<file>" << std::endl;
        finalize::call();
        return 1;
    }

    int B = batch_configs.size();

    std::vector<int> layer_dims;
    layer_dims.push_back(input_dim);
    for (int i = 0; i < num_layers - 1; ++i) layer_dims.push_back(hidden_dim);
    layer_dims.push_back(output_dim);

    std::string base_path = "shark_crown_ml/crown_mpc_data/" + model_name;
    std::string weights_file = base_path + "/weights/weights.dat";

    if (party != DEALER) {
        std::cout << "\n********************************************" << std::endl;
        std::cout << "  TRUE BATCH MODE: " << B << " images" << std::endl;
        std::cout << "  MODEL: " << model_name << std::endl;
        std::cout << "  Layers: " << num_layers << ", Hidden: " << hidden_dim << std::endl;
        std::cout << "  EPS: " << eps << std::endl;
        std::cout << "  Algorithm: FULL CROWN (same as crowntest.cpp)" << std::endl;
        std::cout << "********************************************" << std::endl;
    }

    // ==================== 加载权重 ====================
    shark::utils::start_timer("total_time");
    shark::utils::start_timer("input");

    std::vector<shark::span<u64>> weights(num_layers);
    std::vector<shark::span<u64>> biases(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        weights[i] = shark::span<u64>(layer_dims[i + 1] * layer_dims[i]);
        biases[i] = shark::span<u64>(layer_dims[i + 1]);
    }

    if (party == SERVER) {
        Loader weights_loader(weights_file);
        for (int i = 0; i < num_layers; ++i) {
            weights_loader.load(weights[i], f);
            weights_loader.load(biases[i], f);
        }
    }
    for (int i = 0; i < num_layers; ++i) {
        input::call(weights[i], SERVER);
        input::call(biases[i], SERVER);
    }

    // ==================== 批量加载所有图片 ====================
    shark::span<u64> X0_batch(input_dim * B);
    shark::span<u64> diff_batch(output_dim * B);

    if (party == CLIENT) {
        for (int img = 0; img < B; ++img) {
            std::string input_file = base_path + "/images/" +
                                     std::to_string(batch_configs[img].image_id) + ".bin";
            Loader input_loader(input_file);

            shark::span<u64> x0_temp(input_dim);
            input_loader.load(x0_temp, f);
            for (int i = 0; i < input_dim; ++i) {
                X0_batch[img * input_dim + i] = x0_temp[i];
            }

            for (int i = 0; i < output_dim; ++i) {
                diff_batch[img * output_dim + i] = 0;
            }
            diff_batch[img * output_dim + batch_configs[img].true_label] = float_to_fixed(1.0);
            diff_batch[img * output_dim + batch_configs[img].target_label] = float_to_fixed(-1.0);
        }
    }

    // 一次性共享所有数据
    input::call(X0_batch, CLIENT);
    input::call(diff_batch, CLIENT);

    shark::span<u64> epsilon_share(1), one_share(1), two_share(1), eps_share(1);
    if (party == CLIENT) {
        epsilon_share[0] = float_to_fixed(0.000001);
        one_share[0] = SCALAR_ONE;
        two_share[0] = float_to_fixed(2.0);
        eps_share[0] = float_to_fixed(eps);
    }
    input::call(epsilon_share, CLIENT);
    input::call(one_share, CLIENT);
    input::call(two_share, CLIENT);
    input::call(eps_share, CLIENT);

    shark::utils::stop_timer("input");

    if (party != DEALER) peer->sync();

    // ==================== 批量CROWN计算 ====================
    shark::utils::start_timer("crown_calculation");

    BatchCROWNComputer crown(input_dim, num_layers, B);
    crown.set_input(X0_batch, diff_batch, eps_share, epsilon_share, one_share, two_share);
    for (int i = 0; i < num_layers; ++i) {
        crown.add_layer(weights[i], biases[i], layer_dims[i], layer_dims[i+1]);
    }

    auto [final_LB, final_UB] = crown.compute_worst_bound_batch();

    output::call(final_LB);
    output::call(final_UB);

    shark::utils::stop_timer("crown_calculation");
    shark::utils::stop_timer("total_time");

    // ==================== 输出结果 ====================
    if (party != DEALER) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "BATCH RESULTS (TRUE CROWN ALGORITHM)" << std::endl;
        std::cout << "============================================" << std::endl;

        int verified_count = 0;
        for (int i = 0; i < B; ++i) {
            double lb = fixed_to_float(final_LB[i]);
            double ub = fixed_to_float(final_UB[i]);
            bool verified = lb > 0;
            if (verified) verified_count++;

            std::cout << "[" << i + 1 << "/" << B << "] "
                      << "id=" << batch_configs[i].image_id
                      << ", true=" << batch_configs[i].true_label
                      << ", target=" << batch_configs[i].target_label
                      << " => LB=" << std::fixed << std::setprecision(4) << lb
                      << ", UB=" << ub
                      << (verified ? " [VERIFIED]" : " [NOT VERIFIED]")
                      << std::endl;
        }

        std::cout << "\nTotal: " << B << ", Verified: " << verified_count
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * verified_count / B) << "%)" << std::endl;
        std::cout << "============================================" << std::endl;

        shark::utils::print_all_timers();
    }

    finalize::call();
    return 0;
}
