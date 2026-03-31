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
#include <map>
#include <algorithm>

using u64 = shark::u64;
using namespace shark::protocols;

// 精度设置
const int f = 26;
const u64 SCALAR_ONE = 1ULL << f;

// ==================== Debug System ====================

struct DebugRecord {
    std::string name;
    shark::span<u64> data;
    int layer_idx;

    DebugRecord(const std::string& n, shark::span<u64>& d, int layer = -1)
        : name(n), layer_idx(layer) {
        data = shark::span<u64>(d.size());
        for (size_t i = 0; i < d.size(); ++i) {
            data[i] = d[i];
        }
    }
};

std::vector<DebugRecord> debug_records;
bool enable_debug = false;

void record_value(const std::string& name, shark::span<u64>& data, int layer_idx = -1) {
    if (enable_debug) {
        debug_records.emplace_back(name, data, layer_idx);
    }
}

// ==================== Loader ====================

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
    }

    void load(shark::span<u64> &X, int precision) {
        int size = X.size();
        for (int i = 0; i < size; i++) {
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

// 优化：使用 add::call 进行向量减法
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

// ==================== 改进的 Reciprocal (牛顿迭代) ====================

shark::span<u64> newton_refine_reciprocal(shark::span<u64>& a, shark::span<u64>& x_n, shark::span<u64>& two_share) {
    size_t size = a.size();

    auto ax = mul::call(a, x_n);
    ax = ars::call(ax, f);

    shark::span<u64> two_vec(size);
    for (size_t i = 0; i < size; ++i) two_vec[i] = two_share[0];
    auto diff = secure_sub(two_vec, ax);

    auto x_next = mul::call(x_n, diff);
    return ars::call(x_next, f);
}

shark::span<u64> improved_reciprocal(shark::span<u64>& a, shark::span<u64>& two_share, int iterations = 1) {
    auto x = reciprocal::call(a, f);
    for (int i = 0; i < iterations; ++i) {
        x = newton_refine_reciprocal(a, x, two_share);
    }
    return x;
}

// ==================== Alpha 计算 ====================

shark::span<u64> secure_clamp_upper_1(shark::span<u64>& x, shark::span<u64>& one_share) {
    size_t size = x.size();
    auto ones = broadcast_scalar(one_share, size);
    auto one_minus_x = secure_sub(ones, x);
    auto relu_part = relu::call(one_minus_x);
    return secure_sub(ones, relu_part);
}

shark::span<u64> compute_alpha_secure(shark::span<u64>& U, shark::span<u64>& L,
                                       shark::span<u64>& epsilon_share,
                                       shark::span<u64>& one_share,
                                       shark::span<u64>& two_share) {
    size_t size = U.size();

    auto num = relu::call(U);

    shark::span<u64> L_neg(size);
    for(size_t i = 0; i < size; ++i) L_neg[i] = -L[i];
    auto term2 = relu::call(L_neg);

    auto den = add::call(num, term2);
    auto eps_vec = broadcast_scalar(epsilon_share, size);
    den = add::call(den, eps_vec);

    auto den_inv = improved_reciprocal(den, two_share, 3);

    auto alpha = mul::call(num, den_inv);
    alpha = ars::call(alpha, f);

    return secure_clamp_upper_1(alpha, one_share);
}

// ==================== 优化的工具函数 ====================

// 优化: 向量化的矩阵-alpha 缩放
// W: rows x cols 矩阵, alpha: cols 向量
// 结果: result[r][c] = W[r][c] * alpha[c]
shark::span<u64> scale_matrix_by_alpha(shark::span<u64>& W, shark::span<u64>& alpha, int rows, int cols) {
    // 扩展 alpha 到与 W 相同的大小: 每行都是 alpha
    shark::span<u64> alpha_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            alpha_expanded[r * cols + c] = alpha[c];
        }
    }

    // 单次向量化乘法
    auto result = mul::call(W, alpha_expanded);
    return ars::call(result, f);
}

// 优化: 向量化点积
// 单次 mul::call，然后本地求和
shark::span<u64> dot_product(shark::span<u64>& A, shark::span<u64>& B, int size) {
    // 向量化乘法
    auto prod = mul::call(A, B);
    prod = ars::call(prod, f);

    // 本地求和 (秘密份额的加法可以本地完成)
    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        result[0] += prod[i];
    }
    return result;
}

// 优化: 向量化绝对值求和
shark::span<u64> sum_abs(shark::span<u64>& A, int size) {
    auto A_abs = secure_abs(A);

    // 本地求和
    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        result[0] += A_abs[i];
    }
    return result;
}

// 优化: 本地行求和 (不需要协议调用)
shark::span<u64> compute_row_sum_manual(shark::span<u64>& A_abs, int rows, int cols) {
    shark::span<u64> result(rows);
    for (int row = 0; row < rows; ++row) {
        u64 sum = 0;
        for (int col = 0; col < cols; ++col) {
            sum += A_abs[row * cols + col];
        }
        result[row] = sum;
    }
    return result;
}

// ==================== 优化的 Corrections ====================

// 优化: 向量化 lb correction (向量版本)
shark::span<u64> compute_lb_correction_vec(shark::span<u64>& A, shark::span<u64>& LB, int size) {
    shark::span<u64> A_neg(size);
    for(int i = 0; i < size; ++i) A_neg[i] = -A[i];
    auto relu_neg_A = relu::call(A_neg);

    shark::span<u64> LB_neg(size);
    for(int i = 0; i < size; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    // 向量化乘法
    auto correction_vec = mul::call(relu_neg_A, relu_neg_LB);
    correction_vec = ars::call(correction_vec, f);

    // 本地求和
    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        result[0] += correction_vec[i];
    }
    return result;
}

// 优化: 向量化 ub correction (向量版本)
shark::span<u64> compute_ub_correction_vec(shark::span<u64>& A, shark::span<u64>& LB, int size) {
    auto relu_A = relu::call(A);

    shark::span<u64> LB_neg(size);
    for(int i = 0; i < size; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    // 向量化乘法
    auto correction_vec = mul::call(relu_A, relu_neg_LB);
    correction_vec = ars::call(correction_vec, f);

    // 本地求和
    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        result[0] += correction_vec[i];
    }
    return result;
}

// 优化: 向量化 lb correction (矩阵版本)
// A: rows x cols, LB: cols
// 对每行分别计算 correction
shark::span<u64> compute_lb_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB, int rows, int cols) {
    // 预计算 relu(-LB)
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    // 计算 -A 并做 relu (整个矩阵一次)
    shark::span<u64> A_neg(rows * cols);
    for(int i = 0; i < rows * cols; ++i) A_neg[i] = -A[i];
    auto relu_neg_A = relu::call(A_neg);

    // 扩展 relu_neg_LB 到矩阵大小
    shark::span<u64> relu_neg_LB_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            relu_neg_LB_expanded[r * cols + c] = relu_neg_LB[c];
        }
    }

    // 单次向量化乘法
    auto correction_matrix = mul::call(relu_neg_A, relu_neg_LB_expanded);
    correction_matrix = ars::call(correction_matrix, f);

    // 本地按行求和
    shark::span<u64> corrections(rows);
    for(int r = 0; r < rows; ++r) {
        u64 sum = 0;
        for(int c = 0; c < cols; ++c) {
            sum += correction_matrix[r * cols + c];
        }
        corrections[r] = sum;
    }
    return corrections;
}

// 优化: 向量化 ub correction (矩阵版本)
shark::span<u64> compute_ub_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB, int rows, int cols) {
    // 预计算 relu(-LB)
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    // 计算 relu(A) (整个矩阵一次)
    auto relu_A = relu::call(A);

    // 扩展 relu_neg_LB 到矩阵大小
    shark::span<u64> relu_neg_LB_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            relu_neg_LB_expanded[r * cols + c] = relu_neg_LB[c];
        }
    }

    // 单次向量化乘法
    auto correction_matrix = mul::call(relu_A, relu_neg_LB_expanded);
    correction_matrix = ars::call(correction_matrix, f);

    // 本地按行求和
    shark::span<u64> corrections(rows);
    for(int r = 0; r < rows; ++r) {
        u64 sum = 0;
        for(int c = 0; c < cols; ++c) {
            sum += correction_matrix[r * cols + c];
        }
        corrections[r] = sum;
    }
    return corrections;
}

// ==================== CROWN Core ====================

struct LayerInfo {
    shark::span<u64> W;
    shark::span<u64> b;
    int input_dim;
    int output_dim;
};

struct LayerBounds {
    shark::span<u64> UB;
    shark::span<u64> LB;
    shark::span<u64> alpha;
};

class CROWNComputer {
public:
    std::vector<LayerInfo> layers;
    std::vector<LayerBounds> layer_bounds;
    shark::span<u64> x0;
    shark::span<u64> eps_share;
    shark::span<u64> epsilon_share;
    shark::span<u64> one_share;
    shark::span<u64> two_share;
    shark::span<u64> ones_input;
    int input_dim;
    int num_layers;

    CROWNComputer(int input_dim_) : input_dim(input_dim_), num_layers(0) { }

    void set_input(shark::span<u64>& x0_, shark::span<u64>& eps_, shark::span<u64>& epsilon_,
                   shark::span<u64>& one_, shark::span<u64>& two_, shark::span<u64>& ones_input_) {
        x0 = x0_;
        eps_share = eps_;
        epsilon_share = epsilon_;
        one_share = one_;
        two_share = two_;
        ones_input = ones_input_;
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

    LayerBounds compute_first_layer_bounds() {
        LayerBounds bounds;
        LayerInfo& layer = layers[0];
        int out_dim = layer.output_dim;
        int in_dim = layer.input_dim;

        auto Ax0 = matmul::call(out_dim, in_dim, 1, layer.W, x0);
        Ax0 = ars::call(Ax0, f);
        record_value("Layer0_Ax0", Ax0, 0);

        auto W_abs = secure_abs(layer.W);
        auto dualnorm = compute_row_sum_manual(W_abs, out_dim, in_dim);
        record_value("Layer0_dualnorm", dualnorm, 0);

        auto eps_vec = broadcast_scalar(eps_share, out_dim);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);
        radius = relu::call(radius);
        record_value("Layer0_radius", radius, 0);

        auto temp1 = add::call(Ax0, radius);
        bounds.UB = add::call(temp1, layer.b);
        record_value("Layer0_UB", bounds.UB, 0);

        auto temp2 = secure_sub(Ax0, radius);
        bounds.LB = add::call(temp2, layer.b);
        record_value("Layer0_LB", bounds.LB, 0);

        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share, one_share, two_share);
        record_value("Layer0_alpha", bounds.alpha, 0);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    LayerBounds compute_middle_layer_bounds(int layer_idx) {
        LayerBounds bounds;
        LayerInfo& curr_layer = layers[layer_idx];
        int out_dim = curr_layer.output_dim;
        int in_dim = curr_layer.input_dim;

        auto A_prop = curr_layer.W;

        shark::span<u64> constants(out_dim);
        for(int i = 0; i < out_dim; ++i) constants[i] = curr_layer.b[i];

        shark::span<u64> lb_corr_total(out_dim);
        shark::span<u64> ub_corr_total(out_dim);
        for(int i = 0; i < out_dim; ++i) {
            lb_corr_total[i] = 0;
            ub_corr_total[i] = 0;
        }

        for(int i = layer_idx - 1; i >= 0; --i) {
            // 优化: 使用向量化的 scale_matrix_by_alpha
            A_prop = scale_matrix_by_alpha(A_prop, layer_bounds[i].alpha, out_dim, layers[i].output_dim);

            if (i == layer_idx - 1) {
                 record_value("Layer" + std::to_string(layer_idx) + "_A_matrix", A_prop, layer_idx);
            }

            // 优化: 使用向量化的 correction 计算
            auto lb_c = compute_lb_correction_matrix(A_prop, layer_bounds[i].LB, out_dim, layers[i].output_dim);
            auto ub_c = compute_ub_correction_matrix(A_prop, layer_bounds[i].LB, out_dim, layers[i].output_dim);

            lb_corr_total = add::call(lb_corr_total, lb_c);
            ub_corr_total = add::call(ub_corr_total, ub_c);

            auto Ab = matmul::call(out_dim, layers[i].output_dim, 1, A_prop, layers[i].b);
            Ab = ars::call(Ab, f);
            constants = add::call(constants, Ab);

            A_prop = matmul::call(out_dim, layers[i].output_dim, layers[i].input_dim, A_prop, layers[i].W);
            A_prop = ars::call(A_prop, f);
        }

        record_value("Layer" + std::to_string(layer_idx) + "_constants", constants, layer_idx);
        record_value("Layer" + std::to_string(layer_idx) + "_lb_corr", lb_corr_total, layer_idx);
        record_value("Layer" + std::to_string(layer_idx) + "_ub_corr", ub_corr_total, layer_idx);

        auto Ax0 = matmul::call(out_dim, input_dim, 1, A_prop, x0);
        Ax0 = ars::call(Ax0, f);
        record_value("Layer" + std::to_string(layer_idx) + "_Ax0", Ax0, layer_idx);

        auto A_abs = secure_abs(A_prop);
        auto dualnorm = compute_row_sum_manual(A_abs, out_dim, input_dim);
        record_value("Layer" + std::to_string(layer_idx) + "_dualnorm", dualnorm, layer_idx);

        auto eps_vec = broadcast_scalar(eps_share, out_dim);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);
        radius = relu::call(radius);
        record_value("Layer" + std::to_string(layer_idx) + "_radius", radius, layer_idx);

        auto base = add::call(Ax0, constants);

        bounds.UB = add::call(base, ub_corr_total);
        bounds.UB = add::call(bounds.UB, radius);
        record_value("Layer" + std::to_string(layer_idx) + "_UB", bounds.UB, layer_idx);

        bounds.LB = secure_sub(base, lb_corr_total);
        bounds.LB = secure_sub(bounds.LB, radius);
        record_value("Layer" + std::to_string(layer_idx) + "_LB", bounds.LB, layer_idx);

        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share, one_share, two_share);
        record_value("Layer" + std::to_string(layer_idx) + "_alpha", bounds.alpha, layer_idx);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    std::pair<shark::span<u64>, shark::span<u64>> compute_worst_bound(
        shark::span<u64>& diff_vec, int true_label, int target_label) {

        compute_first_layer_bounds();

        for(int layer_idx = 1; layer_idx < num_layers - 1; ++layer_idx) {
            compute_middle_layer_bounds(layer_idx);
        }

        return compute_final_diff_bounds(diff_vec, true_label, target_label);
    }

    std::pair<shark::span<u64>, shark::span<u64>> compute_final_diff_bounds(
        shark::span<u64>& diff_vec, int true_label, int target_label) {

        int last_idx = num_layers - 1;
        LayerInfo& last_layer = layers[last_idx];
        int out_dim = 1;
        int in_dim = last_layer.input_dim;

        // 优化: 使用 matmul 计算 W_diff = W^T * diff_vec
        // W: output_dim x in_dim, diff_vec: output_dim
        // W_diff[i] = sum_j(W[j][i] * diff_vec[j]) = (W^T * diff_vec)[i]
        // 即 W_diff = matmul(in_dim, output_dim, 1, W^T, diff_vec)
        // 但 W 是行主序存储，W[j][i] = W[j * in_dim + i]
        // 我们需要计算 W_diff[i] = sum_j W[j * in_dim + i] * diff_vec[j]
        // 这等价于 matmul(1, output_dim, in_dim, diff_vec, W)，结果是 1 x in_dim
        auto W_diff = matmul::call(1, last_layer.output_dim, in_dim, diff_vec, last_layer.W);
        W_diff = ars::call(W_diff, f);
        record_value("Final_W_diff", W_diff, -1);

        // 优化: 向量化计算 b_diff
        auto b_diff = dot_product(last_layer.b, diff_vec, last_layer.output_dim);
        record_value("Final_b_diff", b_diff, -1);

        auto constants = b_diff;
        shark::span<u64> lb_corr_total(1); lb_corr_total[0] = 0;
        shark::span<u64> ub_corr_total(1); ub_corr_total[0] = 0;

        auto A_prop = W_diff;

        for(int i = last_idx - 1; i >= 0; --i) {
            // 优化: 向量化的 alpha 缩放
            // A_prop: in_dim 向量, alpha: output_dim 向量
            // 需要 A_prop[k] *= alpha[k]
            auto A_scaled = mul::call(A_prop, layer_bounds[i].alpha);
            A_prop = ars::call(A_scaled, f);

            if (i == last_idx - 1) {
                record_value("Final_A_final", A_prop, -1);

                auto init_c = dot_product(A_prop, layers[i].b, layers[i].output_dim);
                init_c = add::call(init_c, b_diff);
                record_value("Final_constants_initial", init_c, -1);
            }

            auto lb_c = compute_lb_correction_vec(A_prop, layer_bounds[i].LB, layers[i].output_dim);
            auto ub_c = compute_ub_correction_vec(A_prop, layer_bounds[i].LB, layers[i].output_dim);

            if(i == last_idx - 1) {
                record_value("Final_lb_corr_layer1", lb_c, -1);
                record_value("Final_ub_corr_layer1", ub_c, -1);
            }

            // 本地加法
            lb_corr_total[0] += lb_c[0];
            ub_corr_total[0] += ub_c[0];

            auto Ab = dot_product(A_prop, layers[i].b, layers[i].output_dim);
            constants[0] += Ab[0];

            auto A_next = matmul::call(1, layers[i].output_dim, layers[i].input_dim, A_prop, layers[i].W);
            A_next = ars::call(A_next, f);

            shark::span<u64> A_next_vec(layers[i].input_dim);
            for(int k=0; k<layers[i].input_dim; ++k) A_next_vec[k] = A_next[k];
            A_prop = A_next_vec;
        }

        record_value("Final_constants", constants, -1);

        auto Ax0 = dot_product(A_prop, x0, input_dim);
        auto dualnorm = sum_abs(A_prop, input_dim);
        auto radius = mul::call(dualnorm, eps_share);
        radius = ars::call(radius, f);
        radius = relu::call(radius);

        record_value("Final_Ax0", Ax0, -1);
        record_value("Final_dualnorm", dualnorm, -1);
        record_value("Final_radius", radius, -1);

        auto base = add::call(Ax0, constants);

        auto final_LB = secure_sub(base, lb_corr_total);
        final_LB = secure_sub(final_LB, radius);

        auto final_UB = add::call(base, ub_corr_total);
        final_UB = add::call(final_UB, radius);

        return std::make_pair(final_LB, final_UB);
    }
};

// ==================== Main ====================

int main(int argc, char **argv) {
    init::from_args(argc, argv);

//    std::string model_name = "eran_cifar_5layer_relu_100_best";
//    int input_dim = 3072;
//    int output_dim = 10;
//    int num_layers = 5;
//    int hidden_dim = 100;
//    int true_label = 1;
//    int target_label = 4;
//    float eps = 0.0002;
//    int image_id = 6;
//    std::string custom_input_file = "";

    std::string model_name = "vnncomp_mnist_7layer_relu_256_best";
    int input_dim = 784;
    int output_dim = 10;
    int num_layers = 7;
    int hidden_dim = 256;
    int true_label = 1;
    int target_label = 5;
    float eps = 0.015;
    int image_id = 2;
    std::string custom_input_file = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--model=") == 0) {
            model_name = arg.substr(8);
        }
        else if (arg.find("--num_layers=") == 0) {
            num_layers = std::stoi(arg.substr(13));
        }
        else if (arg.find("--hidden_dim=") == 0) {
            hidden_dim = std::stoi(arg.substr(13));
        }
        else if (arg.find("--input_dim=") == 0) {
            input_dim = std::stoi(arg.substr(12));
        }
        else if (arg.find("--output_dim=") == 0) {
            output_dim = std::stoi(arg.substr(13));
        }
        else if (arg.find("--eps=") == 0) {
            eps = std::stof(arg.substr(6));
        }
        else if (arg.find("--true_label=") == 0) {
            true_label = std::stoi(arg.substr(13));
        }
        else if (arg.find("--target_label=") == 0) {
            target_label = std::stoi(arg.substr(15));
        }
        else if (arg.find("--image_id=") == 0) {
            image_id = std::stoi(arg.substr(11));
        }
        else if (arg.find("--input_file=") == 0) {
            custom_input_file = arg.substr(13);
        }
        else if (arg == "--debug") {
            enable_debug = true;
        }
    }

    std::vector<int> layer_dims;
    layer_dims.push_back(input_dim);
    for (int i = 0; i < num_layers - 1; ++i) {
        layer_dims.push_back(hidden_dim);
    }
    layer_dims.push_back(output_dim);

    std::string base_path = "shark_crown_ml/crown_mpc_data/" + model_name;
    std::string weights_file = base_path + "/weights/weights.dat";

    std::string input_file;
    if (!custom_input_file.empty()) {
        input_file = custom_input_file;
    } else {
        input_file = base_path + "/images/" + std::to_string(image_id) + ".bin";
    }
    std::cout << "DEBUG: Loading input file: " << input_file << std::endl;

    if (party != DEALER) {
        std::cout << "\n********************************************" << std::endl;
        std::cout << "  MODEL: " << model_name << std::endl;
        std::cout << "  Layers: " << num_layers << ", Hidden: " << hidden_dim << std::endl;
        std::cout << "  Layer dims: [";
        for (size_t i = 0; i < layer_dims.size(); ++i) {
            std::cout << layer_dims[i];
            if (i < layer_dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Input file: " << input_file << std::endl;
        std::cout << "  EPS: " << eps << std::endl;
        std::cout << "  True label: " << true_label << ", Target: " << target_label << std::endl;
        std::cout << "********************************************" << std::endl;
    }

    // ==================== Input Phase 开始 ====================
    shark::utils::start_timer("End_to_end_time");  // 端到端时间包含 input
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

    shark::span<u64> epsilon_share(1), one_share(1), two_share(1);
    shark::span<u64> diff_vec(output_dim), x0(input_dim), ones_input(input_dim);
    shark::span<u64> eps_share(1);

    if (party == CLIENT) {
        Loader input_loader(input_file);
        input_loader.load(x0, f);
        epsilon_share[0] = float_to_fixed(0.000001);
        one_share[0] = SCALAR_ONE;
        two_share[0] = float_to_fixed(2.0);
        eps_share[0] = float_to_fixed(eps);
        for (int i = 0; i < input_dim; ++i) ones_input[i] = SCALAR_ONE;
        for (int i = 0; i < output_dim; ++i) diff_vec[i] = 0;
        diff_vec[true_label] = float_to_fixed(1.0);
        diff_vec[target_label] = float_to_fixed(-1.0);
    }

    input::call(x0, CLIENT);
    input::call(epsilon_share, CLIENT);
    input::call(one_share, CLIENT);
    input::call(two_share, CLIENT);
    input::call(eps_share, CLIENT);
    input::call(diff_vec, CLIENT);
    input::call(ones_input, CLIENT);

    shark::utils::stop_timer("input");
    // ==================== Input Phase 结束 ====================

    if (party != DEALER) peer->sync();
    shark::utils::start_timer("crown_calculation");

    CROWNComputer crown(input_dim);
    crown.set_input(x0, eps_share, epsilon_share, one_share, two_share, ones_input);
    for (int i = 0; i < num_layers; ++i) {
        crown.add_layer(weights[i], biases[i], layer_dims[i], layer_dims[i+1]);
    }

    auto [final_LB, final_UB] = crown.compute_worst_bound(diff_vec, true_label, target_label);

    shark::utils::stop_timer("crown_calculation");
    shark::utils::stop_timer("End_to_end_time");

    if (enable_debug) {
        for (auto& record : debug_records) {
            output::call(record.data);
        }
    }

    output::call(final_LB);
    output::call(final_UB);

    if (party != DEALER) {
        if (enable_debug) {
            std::cout << "\n============================================" << std::endl;
            std::cout << "Debug Output" << std::endl;
            std::cout << "============================================" << std::endl;
            for (auto& record : debug_records) {
                std::cout << record.name << ": [";
                int print_count = std::min((int)record.data.size(), 10);
                for (int i = 0; i < print_count; ++i) {
                    std::cout << std::fixed << std::setprecision(6) << fixed_to_float(record.data[i]);
                    if (i < print_count - 1) std::cout << ", ";
                }
                if ((int)record.data.size() > 10) std::cout << ", ...";
                std::cout << "]" << std::endl;
            }
        }
    }

    if (party != DEALER) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "MODEL: " << model_name << std::endl;
        std::cout << "IMAGE: " << (custom_input_file.empty() ? std::to_string(image_id) + ".bin" : custom_input_file) << std::endl;
        std::cout << "EPS: " << eps << " | True: " << true_label << " | Target: " << target_label << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "MPC LB: " << std::fixed << std::setprecision(6) << fixed_to_float(final_LB[0]) << std::endl;
        std::cout << "MPC UB: " << std::fixed << std::setprecision(6) << fixed_to_float(final_UB[0]) << std::endl;
        std::cout << "============================================" << std::endl;

        shark::utils::print_all_timers();
    }

    finalize::call();
    return 0;
}
