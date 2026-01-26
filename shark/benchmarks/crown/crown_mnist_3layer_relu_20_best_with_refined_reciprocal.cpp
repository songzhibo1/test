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
bool enable_debug = false;  // 设置为 true 开启调试输出

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

// 牛顿迭代: x_{n+1} = x_n * (2 - a * x_n)
shark::span<u64> newton_refine_reciprocal(shark::span<u64>& a, shark::span<u64>& x_n, shark::span<u64>& two_share) {
    size_t size = a.size();

    // ax = a * x_n
    auto ax = mul::call(a, x_n);
    ax = ars::call(ax, f);

    // diff = 2 - ax
    shark::span<u64> two_vec(size);
    for (size_t i = 0; i < size; ++i) two_vec[i] = two_share[0];
    auto diff = secure_sub(two_vec, ax);

    // x_{n+1} = x_n * diff
    auto x_next = mul::call(x_n, diff);
    return ars::call(x_next, f);
}

// 改进的倒数: 原始 reciprocal + 牛顿迭代
shark::span<u64> improved_reciprocal(shark::span<u64>& a, shark::span<u64>& two_share, int iterations = 1) {
    auto x = reciprocal::call(a, f);
    for (int i = 0; i < iterations; ++i) {
        x = newton_refine_reciprocal(a, x, two_share);
    }
    return x;
}

// ==================== Alpha 计算 (使用改进的 reciprocal) ====================

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

    // num = ReLU(U)
    auto num = relu::call(U);

    // term2 = ReLU(-L)
    shark::span<u64> L_neg(size);
    for(size_t i = 0; i < size; ++i) L_neg[i] = -L[i];
    auto term2 = relu::call(L_neg);

    // den = num + term2 + epsilon
    auto den = add::call(num, term2);
    auto eps_vec = broadcast_scalar(epsilon_share, size);
    den = add::call(den, eps_vec);

    // 使用改进的 reciprocal (1次牛顿迭代)
    auto den_inv = improved_reciprocal(den, two_share, 1);

    // alpha = num * den_inv
    auto alpha = mul::call(num, den_inv);
    alpha = ars::call(alpha, f);

    // Clamp to [0, 1]
    return secure_clamp_upper_1(alpha, one_share);
}

// ==================== 其他工具函数 ====================

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

shark::span<u64> compute_row_sum_manual(shark::span<u64>& A_abs, int rows, int cols) {
    shark::span<u64> result(rows);
    for (int row = 0; row < rows; ++row) {
        shark::span<u64> sum(1);
        sum[0] = 0;
        for (int col = 0; col < cols; ++col) {
            shark::span<u64> abs_elem(1);
            abs_elem[0] = A_abs[row * cols + col];
            sum = add::call(sum, abs_elem);
        }
        result[row] = sum[0];
    }
    return result;
}

// ==================== Corrections ====================

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

shark::span<u64> compute_lb_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB, int rows, int cols) {
    shark::span<u64> corrections(rows);
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);
    for(int r = 0; r < rows; ++r) {
        shark::span<u64> A_row(cols);
        for(int c = 0; c < cols; ++c) A_row[c] = A[r * cols + c];
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

shark::span<u64> compute_ub_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB, int rows, int cols) {
    shark::span<u64> corrections(rows);
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);
    for(int r = 0; r < rows; ++r) {
        shark::span<u64> A_row(cols);
        for(int c = 0; c < cols; ++c) A_row[c] = A[r * cols + c];
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
    shark::span<u64> two_share;  // 新增：用于牛顿迭代
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

        // 使用改进的 alpha 计算
        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share, one_share, two_share);
        record_value("Layer0_alpha", bounds.alpha, 0);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    LayerBounds compute_middle_layer_bounds(int layer_idx) {
        LayerBounds bounds;
        LayerInfo& curr_layer = layers[layer_idx];
        LayerBounds& prev_bounds = layer_bounds[layer_idx - 1];

        int out_dim = curr_layer.output_dim;
        int in_dim = curr_layer.input_dim;

        auto A = scale_matrix_by_alpha(curr_layer.W, prev_bounds.alpha, out_dim, in_dim);

        shark::span<u64> constants(out_dim);
        for(int i = 0; i < out_dim; ++i) constants[i] = curr_layer.b[i];

        auto Ab_prev = matmul::call(out_dim, in_dim, 1, A, layers[layer_idx - 1].b);
        Ab_prev = ars::call(Ab_prev, f);
        constants = add::call(constants, Ab_prev);
        record_value("Layer" + std::to_string(layer_idx) + "_constants", constants, layer_idx);

        auto lb_corr = compute_lb_correction_matrix(A, prev_bounds.LB, out_dim, in_dim);
        auto ub_corr = compute_ub_correction_matrix(A, prev_bounds.LB, out_dim, in_dim);
        record_value("Layer" + std::to_string(layer_idx) + "_lb_corr", lb_corr, layer_idx);
        record_value("Layer" + std::to_string(layer_idx) + "_ub_corr", ub_corr, layer_idx);

        auto A_prop = A;
        for(int i = layer_idx - 1; i >= 0; --i) {
            if(i > 0) {
                auto A_scaled = scale_matrix_by_alpha(layers[i].W, layer_bounds[i - 1].alpha,
                                                      layers[i].output_dim, layers[i].input_dim);
                A_prop = matmul::call(out_dim, layers[i].output_dim, layers[i].input_dim, A_prop, A_scaled);
            } else {
                A_prop = matmul::call(out_dim, layers[i].output_dim, layers[i].input_dim, A_prop, layers[i].W);
            }
            A_prop = ars::call(A_prop, f);
        }

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

        bounds.UB = add::call(base, ub_corr);
        bounds.UB = add::call(bounds.UB, radius);
        record_value("Layer" + std::to_string(layer_idx) + "_UB", bounds.UB, layer_idx);

        bounds.LB = secure_sub(base, lb_corr);
        bounds.LB = secure_sub(bounds.LB, radius);
        record_value("Layer" + std::to_string(layer_idx) + "_LB", bounds.LB, layer_idx);

        // 使用改进的 alpha 计算
        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share, one_share, two_share);
        record_value("Layer" + std::to_string(layer_idx) + "_alpha", bounds.alpha, layer_idx);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    std::pair<shark::span<u64>, shark::span<u64>> compute_final_diff_bounds(
        shark::span<u64>& diff_vec, int true_label, int target_label) {

        int last_idx = num_layers - 1;
        LayerInfo& last_layer = layers[last_idx];
        int out_dim = last_layer.output_dim;
        int in_dim = last_layer.input_dim;

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
        record_value("Final_W_diff", W_diff, -1);

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
        record_value("Final_b_diff", b_diff, -1);

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
        record_value("Final_A_final", A_final, -1);

        auto constants = dot_product(A_final, layers[last_idx - 1].b, in_dim);
        constants = add::call(constants, b_diff);
        record_value("Final_constants_initial", constants, -1);

        auto lb_corr = compute_lb_correction_vec(A_final, prev_bounds.LB, in_dim);
        auto ub_corr = compute_ub_correction_vec(A_final, prev_bounds.LB, in_dim);
        record_value("Final_lb_corr_layer1", lb_corr, -1);
        record_value("Final_ub_corr_layer1", ub_corr, -1);

        shark::span<u64> UB_final(1);
        shark::span<u64> LB_final(1);
        UB_final[0] = ub_corr[0];
        LB_final[0] = 0;
        LB_final = secure_sub(LB_final, lb_corr);

        shark::span<u64> A_current(in_dim);
        for(int i = 0; i < in_dim; ++i) A_current[i] = A_final[i];

        for(int layer_idx = last_idx - 1; layer_idx >= 1; --layer_idx) {
            LayerInfo& curr_layer = layers[layer_idx];
            LayerBounds& curr_prev_bounds = layer_bounds[layer_idx - 1];

            auto A_scaled = scale_matrix_by_alpha(curr_layer.W, curr_prev_bounds.alpha,
                                                   curr_layer.output_dim, curr_layer.input_dim);

            auto A_new = matmul::call(1, curr_layer.output_dim, curr_layer.input_dim, A_current, A_scaled);
            A_new = ars::call(A_new, f);

            shark::span<u64> A_new_vec(curr_layer.input_dim);
            for(int i = 0; i < curr_layer.input_dim; ++i) A_new_vec[i] = A_new[i];

            auto const_add = dot_product(A_new_vec, layers[layer_idx - 1].b, curr_layer.input_dim);
            constants = add::call(constants, const_add);

            auto lb_corr_layer = compute_lb_correction_vec(A_new_vec, curr_prev_bounds.LB, curr_layer.input_dim);
            auto ub_corr_layer = compute_ub_correction_vec(A_new_vec, curr_prev_bounds.LB, curr_layer.input_dim);

            UB_final = add::call(UB_final, ub_corr_layer);
            LB_final = secure_sub(LB_final, lb_corr_layer);

            A_current = shark::span<u64>(curr_layer.input_dim);
            for(int i = 0; i < curr_layer.input_dim; ++i) A_current[i] = A_new[i];
        }

        auto A_input = matmul::call(1, layers[0].output_dim, layers[0].input_dim, A_current, layers[0].W);
        A_input = ars::call(A_input, f);

        shark::span<u64> A_input_vec(input_dim);
        for(int i = 0; i < input_dim; ++i) A_input_vec[i] = A_input[i];

        auto Ax0 = dot_product(A_input_vec, x0, input_dim);
        auto dualnorm = sum_abs(A_input_vec, input_dim);
        auto radius = mul::call(dualnorm, eps_share);
        radius = ars::call(radius, f);
        radius = relu::call(radius);

        record_value("Final_Ax0", Ax0, -1);
        record_value("Final_dualnorm", dualnorm, -1);
        record_value("Final_radius", radius, -1);
        record_value("Final_constants_final", constants, -1);
        record_value("Final_UB_corrections", UB_final, -1);
        record_value("Final_LB_corrections", LB_final, -1);

        auto base = add::call(Ax0, constants);
        base = add::call(base, LB_final);
        auto final_LB = secure_sub(base, radius);

        auto base_ub = add::call(Ax0, constants);
        base_ub = add::call(base_ub, UB_final);
        auto final_UB = add::call(base_ub, radius);

        record_value("Final_LB", final_LB, -1);
        record_value("Final_UB", final_UB, -1);

        return std::make_pair(final_LB, final_UB);
    }

    std::pair<shark::span<u64>, shark::span<u64>> compute_worst_bound(
        shark::span<u64>& diff_vec, int true_label, int target_label) {

        compute_first_layer_bounds();

        for(int layer_idx = 1; layer_idx < num_layers - 1; ++layer_idx) {
            compute_middle_layer_bounds(layer_idx);
        }

        return compute_final_diff_bounds(diff_vec, true_label, target_label);
    }
};

// ==================== Main ====================

int main(int argc, char **argv) {
    init::from_args(argc, argv);

    int input_dim = 784;
    int output_dim = 10;
    std::vector<int> layer_dims = {784, 20, 20, 10};
    int num_layers = 3;
//    std::vector<int> layer_dims = {784,1024,1024,1024,10};
//    int num_layers = 4;


    std::string weights_file = "shark_crown_ml/crown_mpc_data/mnist_3layer_relu_20_best/weights/weights.dat";
    std::string input_file = "shark_crown_ml/crown_mpc_data/mnist_3layer_relu_20_best/images/0.bin";
    float eps = 0.09;
    int true_label = 7;
    int target_label = 6;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--weights=") == 0) weights_file = arg.substr(10);
        else if (arg.find("--input=") == 0) input_file = arg.substr(8);
        else if (arg.find("--eps=") == 0) eps = std::stof(arg.substr(6));
        else if (arg.find("--true_label=") == 0) true_label = std::stoi(arg.substr(13));
        else if (arg.find("--target_label=") == 0) target_label = std::stoi(arg.substr(15));
        else if (arg == "--debug") enable_debug = true;
    }

    std::cout << "\n============================================" << std::endl;
    std::cout << "CROWN MPC - Newton-Refined Reciprocal" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "eps = " << eps << ", true_label = " << true_label << ", target_label = " << target_label << std::endl;

    // 分配内存
    shark::span<u64> x0(input_dim);
    shark::span<u64> eps_share(1);
    shark::span<u64> epsilon_share(1);
    shark::span<u64> one_share(1);
    shark::span<u64> two_share(1);  // 新增
    shark::span<u64> diff_vec(output_dim);
    shark::span<u64> ones_input(input_dim);

    std::vector<shark::span<u64>> weights(num_layers);
    std::vector<shark::span<u64>> biases(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        int in_dim = layer_dims[i];
        int out_dim = layer_dims[i + 1];
        weights[i] = shark::span<u64>(out_dim * in_dim);
        biases[i] = shark::span<u64>(out_dim);
    }

    // CLIENT 初始化
    if (party == CLIENT) {
        Loader input_loader(input_file);
        input_loader.load(x0, f);

        eps_share[0] = float_to_fixed(eps);
        epsilon_share[0] = float_to_fixed(0.000001);
        one_share[0] = SCALAR_ONE;
        two_share[0] = float_to_fixed(2.0);  // 新增

        for (int i = 0; i < input_dim; ++i) ones_input[i] = SCALAR_ONE;

        for (int i = 0; i < output_dim; ++i) diff_vec[i] = 0;
        diff_vec[true_label] = float_to_fixed(1.0);
        diff_vec[target_label] = float_to_fixed(-1.0);
    }

    // SERVER 初始化
    if (party == SERVER) {
        Loader weights_loader(weights_file);
        for (int i = 0; i < num_layers; ++i) {
            weights_loader.load(weights[i], f);
            weights_loader.load(biases[i], f);
        }
    }

    // 秘密共享
    shark::utils::start_timer("input");
    shark::utils::start_timer("End_to_end_time");
    input::call(x0, CLIENT);
    input::call(eps_share, CLIENT);
    input::call(epsilon_share, CLIENT);
    input::call(one_share, CLIENT);
    input::call(two_share, CLIENT);  // 新增
    input::call(diff_vec, CLIENT);
    input::call(ones_input, CLIENT);

    for (int i = 0; i < num_layers; ++i) {
        input::call(weights[i], SERVER);
        input::call(biases[i], SERVER);
    }

    if (party != DEALER) peer->sync();
    shark::utils::stop_timer("input");

    // CROWN 计算
    shark::utils::start_timer("crown_mpc");

    CROWNComputer crown(input_dim);
    crown.set_input(x0, eps_share, epsilon_share, one_share, two_share, ones_input);

    for (int i = 0; i < num_layers; ++i) {
        int in_dim = layer_dims[i];
        int out_dim = layer_dims[i + 1];
        crown.add_layer(weights[i], biases[i], in_dim, out_dim);
    }

    auto [final_LB, final_UB] = crown.compute_worst_bound(diff_vec, true_label, target_label);

    shark::utils::stop_timer("crown_mpc");
    shark::utils::stop_timer("End_to_end_time");

    // 输出调试信息
    if (enable_debug) {
        for (auto& record : debug_records) {
            output::call(record.data);
        }
    }

    // 输出最终结果
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

        double lb_val = fixed_to_float(final_LB[0]);
        double ub_val = fixed_to_float(final_UB[0]);

        std::cout << "\n============================================" << std::endl;
        std::cout << "Final Results" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "MPC LB: " << std::fixed << std::setprecision(6) << lb_val << std::endl;
        std::cout << "MPC UB: " << std::fixed << std::setprecision(6) << ub_val << std::endl;
//        std::cout << "Python Reference: LB = -3198.92, UB = 3998.16 (eps=2)" << std::endl;
        std::cout << "============================================" << std::endl;
    }

    finalize::call();
    shark::utils::print_all_timers();

    return 0;
}