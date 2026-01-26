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

// [FIX 1] 精度 f=26
const int f = 26;
const u64 SCALAR_ONE = 1ULL << f;

// ==================== Snapshot Debug System ====================

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

void record_value(const std::string& name, shark::span<u64>& data, int layer_idx = -1) {
    debug_records.emplace_back(name, data, layer_idx);
}

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

// [FIX 2] Alpha Clamping: 使用共享的 1.0
shark::span<u64> secure_clamp_upper_1(shark::span<u64>& x, shark::span<u64>& one_share) {
    size_t size = x.size();
    auto ones = broadcast_scalar(one_share, size);
    auto one_minus_x = secure_sub(ones, x);
    auto relu_part = relu::call(one_minus_x);
    return secure_sub(ones, relu_part);
}

// 计算 Alpha (含截断)
shark::span<u64> compute_alpha_secure(shark::span<u64>& U, shark::span<u64>& L, shark::span<u64>& epsilon_share, shark::span<u64>& one_share) {
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
    alpha = ars::call(alpha, f);

    return secure_clamp_upper_1(alpha, one_share);
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
    shark::span<u64> ones_input; // 全1向量
    int input_dim;
    int num_layers;

    CROWNComputer(int input_dim_) : input_dim(input_dim_), num_layers(0) { }

    // [FIXED] 正确的参数赋值，避免 main 中手动赋值
    void set_input(shark::span<u64>& x0_, shark::span<u64>& eps_, shark::span<u64>& epsilon_, shark::span<u64>& one_, shark::span<u64>& ones_input_) {
        x0 = x0_;
        eps_share = eps_;
        epsilon_share = epsilon_;
        one_share = one_;
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

        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share, one_share);
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

        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share, one_share);
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
            record_value("Final_backprop_const_add_layer" + std::to_string(layer_idx), const_add, -1);
            record_value("Final_backprop_constants_cumul_layer" + std::to_string(layer_idx), constants, -1);

            auto lb_corr_layer = compute_lb_correction_vec(A_new_vec, curr_prev_bounds.LB, curr_layer.input_dim);
            auto ub_corr_layer = compute_ub_correction_vec(A_new_vec, curr_prev_bounds.LB, curr_layer.input_dim);
            record_value("Final_backprop_lb_corr_layer" + std::to_string(layer_idx), lb_corr_layer, -1);
            record_value("Final_backprop_ub_corr_layer" + std::to_string(layer_idx), ub_corr_layer, -1);

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

    // =========================================================
    // Modified Configuration based on Python Log
    // =========================================================
    int input_dim = 10;
    int output_dim = 10;
    std::vector<int> layer_dims = {10, 12, 12, 12, 12, 10};
    int num_layers = 5;

    float eps = 0.09;
    int true_label = 1;   // Predicted class from log
    int target_label = 0; // Attack target from log

    // =========================================================
    // Hardcoded Data from Python Log
    // =========================================================

    // Custom Input (size 10)
    std::vector<float> input_raw = {
        0.13, -0.45, -0.19, -0.27, 0.43, -0.21, 0.25, 1.01, 0.63, -0.22
    };

    // Layer 1 (12x10) - Flattened
    std::vector<float> w1_raw = {
         0.25, -0.07,  0.32,  0.76, -0.12, -0.12,  0.79,  0.38, -0.23,  0.27,
        -0.23, -0.23,  0.12, -0.96, -0.86, -0.28, -0.51,  0.16, -0.45, -0.71,
         0.73, -0.11,  0.03, -0.71, -0.27,  0.06, -0.58,  0.19, -0.3,  -0.15,
        -0.3,   0.93, -0.01, -0.53,  0.41, -0.61,  0.1,  -0.98, -0.66,  0.1,
         0.37,  0.09, -0.06, -0.15, -0.74, -0.36, -0.23,  0.53,  0.17, -0.88,
         0.16, -0.19, -0.34,  0.31,  0.52,  0.47, -0.42, -0.15,  0.17,  0.49,
        -0.24, -0.09, -0.55, -0.6,   0.41,  0.68, -0.04,  0.5,   0.18, -0.32,
         0.18,  0.77, -0.02,  0.78, -1.31,  0.41,  0.04, -0.15,  0.05, -0.99,
        -0.11,  0.18,  0.74, -0.26, -0.4,  -0.25,  0.46,  0.16, -0.26,  0.26,
         0.05,  0.48, -0.35, -0.16, -0.2,  -0.73,  0.15,  0.13,  0.,   -0.12,
        -0.71, -0.21, -0.17, -0.4,  -0.08,  0.2,   0.94,  0.09,  0.13, -0.04,
        -0.96, -0.01,  0.03,  1.23, -0.1,   0.15, -0.02, -0.58,  0.57,  0.38
    };
    std::vector<float> b1_raw = {0.08, -0.09, 0.14, -0.14, 0.06, 0.22, -0.1, -0.06, 0.01, -0.05, -0.16, 0.01};

    // Layer 2 (12x12) - Flattened
    std::vector<float> w2_raw = {
        -0.53,  0.24, -0.46,  0.77, -0.39, -0.16,  0.41, -0.62,  0.11,  0.65, -0.8,   0.09,
         0.13,  0.39, -0.62, -0.66,  0.26,  0.15,  0.13,  0.17, -0.34,  0.12,  0.15, -0.36,
         0.93,  0.24, -0.6,   0.33, -0.49,  0.39,  0.58, -0.41,  0.48,  0.21,  0.41,  0.95,
        -0.12, -0.38, -0.44, -0.41, -0.04,  0.17,  0.14,  0.41,  0.01,  0.73, -0.13,  1.36,
         0.31, -0.43, -0.54,  0.24, -0.11,  0.36,  0.24, -0.04, -0.42, -0.76, -0.22,  0.43,
         0.11, -0.62,  0.09,  0.19, -0.44,  0.08,  0.03, -0.57,  0.18,  0.28,  0.54,  0.53,
        -0.69, -0.47,  0.26,  0.26,  0.26,  1.93,  0.29,  0.57,  0.48,  0.33, -0.16,  0.38,
        -0.39, -0.12, -0.24,  0.04,  1.16, -0.93,  0.34, -0.81, -0.24,  0.54,  0.03, -0.54,
        -0.36,  0.34, -0.37,  0.11,  0.02, -0.33,  1.07,  0.32, -1.01,  0.09, -0.33,  0.43,
        -0.4,  -0.06,  0.25,  0.43, -0.6,  -0.17, -0.24, -0.33,  0.88,  0.2,  -0.63,  0.46,
         1.06,  0.52, -0.76, -0.24,  0.63, -0.35,  0.22,  0.39, -0.46, -0.03, -1.62, -0.51,
        -0.13, -0.62,  0.82, -0.72, -0.22,  0.07,  0.72, -0.72,  0.58,  0.01, -0.49,  0.23
    };
    std::vector<float> b2_raw = {0.02, -0.06, 0.01, -0.04, 0.01, 0.07, 0.16, -0.12, 0.21, -0.2, -0.02, 0.06};

    // Layer 3 (12x12) - Flattened
    std::vector<float> w3_raw = {
         0.14, -0.31, -0.1,  -0.25, -0.29,  0.42,  0.18, -0.35,  0.45,  0.15,  0.41,  0.31,
        -0.41, -0.28,  0.37,  0.31, -0.01,  0.06,  0.64, -0.3,   0.27, -0.1,  -0.11,  0.55,
         0.41,  0.41,  0.65,  0.01,  0.34, -0.16,  0.16, -0.07,  0.05,  0.3,  -0.41,  1.05,
        -0.5,  -0.61,  0.58,  0.4,   0.31,  0.31, -0.01, -0.45,  0.04, -0.34,  0.49, -0.07,
        -0.41, -0.16,  0.21, -0.28, -0.41,  0.12,  0.12, -0.25, -0.24,  0.12, -0.72, -0.7,
        -0.36, -0.11,  0.16,  0.74,  0.43, -0.08, -0.01, -0.5,  -0.01, -0.14,  0.16, -0.41,
         0.26,  0.77, -0.05,  0.2,   0.35, -0.2,   0.11,  0.01,  0.05, -0.39,  0.01,  0.25,
         0.73,  0.48,  1.08, -0.38,  0.44,  0.09,  1.09, -0.4,  -0.42, -0.3,  -1.06, -0.26,
        -0.38,  0.08,  0.17,  0.94,  0.48, -0.29, -0.45,  0.25, -0.66,  0.92,  0.59, -0.23,
        -0.86,  0.68, -0.06,  0.62, -0.8,  -0.3,   0.,    0.02, -0.23,  0.31, -0.53, -0.07,
         0.06,  0.26,  0.36, -0.56, -0.77,  0.64,  0.17, -0.37,  0.78,  0.06,  0.59,  0.03,
         1.03,  0.88, -0.12,  0.49,  0.32,  0.68, -0.48,  0.34,  0.53, -0.88, -0.59, -1.02
    };
    std::vector<float> b3_raw = {-0.03, 0.07, 0.15, 0.01, 0.16, -0.14, -0.17, -0.01, 0.04, -0., -0.21, -0.01};

    // Layer 4 (12x12) - Flattened
    std::vector<float> w4_raw = {
        -0.65,  0.33,  0.18, -0.47, -0.26, -0.53, -0.03,  0.48, -0.49,  0.25, -0.27, -0.4,
        -0.05, -0.52, -0.28, -0.6,   0.98,  0.02, -0.35,  0.11, -0.06, -0.11,  0.31,  0.38,
        -0.27, -0.29, -0.14, -1.15, -0.76,  0.68,  0.82, -0.12,  0.29,  0.16,  1.54,  0.56,
        -0.06, -0.48, -0.8,   0.1,  -0.38, -0.71, -0.32, -0.54,  0.84,  0.44, -0.,   0.74,
         0.04, -0.43,  0.76,  0.27, -0.52, -0.1,  -0.44, -0.69,  0.46,  0.95, -0.7,   0.28,
        -0.33, -0.24, -0.3,  -0.43,  0.02, -0.42,  0.14, -0.03, -0.12, -0.45, -0.29,  0.38,
         0.25, -0.49,  0.05,  0.38, -0.83,  0.27, -0.33,  0.29, -0.38, -0.9,  -0.81,  0.02,
         0.13, -0.45,  0.32, -0.83, -0.03, -0.61, -0.33,  0.02, -0.43, -0.19,  0.5,  -0.29,
         0.42, -0.56,  0.26,  0.72, -1.24, -0.4,   0.29, -0.1,   0.19, -0.3,   0.04, -0.08,
         0.58,  0.13,  0.17, -0.21, -0.24, -0.22,  0.2,  -0.21,  0.14,  1.04,  0.44, -0.16,
         0.6,  -0.2,  -1.02, -0.5,  -0.94, -0.18,  0.01,  0.84,  0.16, -0.11,  0.41, -1.11,
         0.12,  0.39, -0.74,  0.57,  0.17, -0.21,  0.32,  1.14,  0.09,  0.12, -0.23, -0.42
    };
    std::vector<float> b4_raw = {0.08, -0.09, 0.01, -0.05, 0.05, 0.03, 0.1, -0.05, -0.03, -0.1, -0.04, 0.04};

    // Layer 5 (10x12) - Flattened
    std::vector<float> w5_raw = {
         0.38, -0.46,  0.43,  0.68,  0.21,  0.94, -0.39, -0.62, -0.89,  0.75,  0.33, -0.03,
         0.14, -0.56,  1.22,  0.06,  0.05,  0.36,  0.24,  0.11, -0.4,   0.24,  0.94,  0.67,
         0.8,  -0.26, -0.49, -0.06,  0.03,  0.55, -0.85,  0.76, -0.08, -0.21, -0.51, -0.83,
         0.41,  0.04, -0.64, -0.65, -0.17,  0.83, -0.13, -0.75, -0.12, -0.14, -1.35, -0.03,
        -0.12,  0.35,  0.92,  0.56, -0.13, -0.55,  1.29,  0.03,  0.01, -0.01,  0.1,  -0.07,
        -0.29, -0.27, -0.02, -0.27, -0.36,  0.05, -0.13,  0.75, -1.33,  0.55,  0.62, -1.04,
        -0.17, -0.19, -0.7,  -0.39, -0.56,  0.88,  0.47,  0.64,  0.36, -0.56, -0.26,  0.24,
        -0.61,  0.36, -0.12, -0.19,  0.36,  0.22, -0.18,  0.58, -0.54,  0.31,  0.3,  -0.15,
         0.16, -0.63,  0.46, -0.09, -0.26,  0.52, -0.35, -0.7,  -0.78,  0.3,  -0.64,  0.88,
        -1.04,  0.85,  0.11, -0.05, -0.27,  0.2,  -0.02,  0.55,  0.06,  0.08, -0.18, -0.03
    };
    std::vector<float> b5_raw = {0.03, -0.17, -0.13, 0.07, 0.02, -0.02, 0., 0.03, -0.05, -0.08};

    // Store weights and biases in vectors of vectors
    std::vector<std::vector<float>> all_weights_raw = {w1_raw, w2_raw, w3_raw, w4_raw, w5_raw};
    std::vector<std::vector<float>> all_biases_raw = {b1_raw, b2_raw, b3_raw, b4_raw, b5_raw};

    // =========================================================
    // Prepare MPC Data Structures
    // =========================================================

    shark::span<u64> x0(input_dim);
    shark::span<u64> eps_share(1);
    shark::span<u64> epsilon_share(1);
    shark::span<u64> one_share(1);
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

    if (party == CLIENT) {
        // Load Input
        for (int i = 0; i < input_dim; ++i) {
            x0[i] = float_to_fixed(input_raw[i]);
            ones_input[i] = SCALAR_ONE;
        }

        eps_share[0] = float_to_fixed(eps);
        epsilon_share[0] = float_to_fixed(0.000001);
        one_share[0] = SCALAR_ONE;

        // Set diff_vec for linear bounds (Predicted - Target)
        // Note: Python log shows LB for f_c - f_j, where c=predicted, j=target
        for (int i = 0; i < output_dim; ++i) diff_vec[i] = 0;
        diff_vec[true_label] = float_to_fixed(1.0);   // +1 for correct class
        diff_vec[target_label] = float_to_fixed(-1.0); // -1 for attack target
    }

    if (party == SERVER) {
        // Load Weights and Biases
        for (int l = 0; l < num_layers; ++l) {
            // Load Weights
            for (size_t k = 0; k < all_weights_raw[l].size(); ++k) {
                weights[l][k] = float_to_fixed(all_weights_raw[l][k]);
            }
            // Load Biases
            for (size_t k = 0; k < all_biases_raw[l].size(); ++k) {
                biases[l][k] = float_to_fixed(all_biases_raw[l][k]);
            }
        }
    }

    input::call(x0, CLIENT);
    input::call(eps_share, CLIENT);
    input::call(epsilon_share, CLIENT);
    input::call(one_share, CLIENT);
    input::call(diff_vec, CLIENT);
    input::call(ones_input, CLIENT);

    for (int i = 0; i < num_layers; ++i) {
        input::call(weights[i], SERVER);
        input::call(biases[i], SERVER);
    }

    if (party != DEALER) peer->sync();

    CROWNComputer crown(input_dim);
    crown.set_input(x0, eps_share, epsilon_share, one_share, ones_input);

    for (int i = 0; i < num_layers; ++i) {
        int in_dim = layer_dims[i];
        int out_dim = layer_dims[i + 1];
        crown.add_layer(weights[i], biases[i], in_dim, out_dim);
    }

    auto [final_LB, final_UB] = crown.compute_worst_bound(diff_vec, true_label, target_label);

    std::cout << "\n============================================" << std::endl;
    std::cout << "Converting all intermediate values to plaintext..." << std::endl;
    std::cout << "============================================" << std::endl;

    for (auto& record : debug_records) {
        output::call(record.data);
    }

    if (party != DEALER) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "CROWN MPC - Custom Hardcoded Input (Plaintext)" << std::endl;
        std::cout << "============================================" << std::endl;

        for (auto& record : debug_records) {
            std::cout << "\n" << record.name << " (size=" << record.data.size() << "):" << std::endl;
            std::cout << "  [";
            int print_count = std::min((int)record.data.size(), 20);
            for (int i = 0; i < print_count; ++i) {
                double val = fixed_to_float(record.data[i]);
                std::cout << std::fixed << std::setprecision(6) << val;
                if (i < print_count - 1) std::cout << ", ";
            }
            if ((int)record.data.size() > 20) std::cout << ", ...";
            std::cout << "]" << std::endl;
        }

        double final_lb_val = fixed_to_float(debug_records[debug_records.size()-2].data[0]);
        double final_ub_val = fixed_to_float(debug_records[debug_records.size()-1].data[0]);

        std::cout << "\n============================================" << std::endl;
        std::cout << "Final Results (Target: " << true_label << " - " << target_label << ", Eps: " << eps << ")" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << "MPC LB: " << std::fixed << std::setprecision(6) << final_lb_val << std::endl;
        std::cout << "MPC UB: " << std::fixed << std::setprecision(6) << final_ub_val << std::endl;
        std::cout << "Expected Python LB: ~ -1.8169" << std::endl;
        std::cout << "============================================" << std::endl;
    }

    finalize::call();

    return 0;
}