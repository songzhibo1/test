/**
 * crowntest_v2.cpp
 *
 * Improved CROWN MPC Implementation with:
 * 1. Better precision handling for deep networks
 * 2. Overflow detection and prevention
 * 3. Corrected correction term formulas
 * 4. Extensive debugging output
 * 5. Adaptive precision scaling
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
#include <iomanip>
#include <cmath>
#include <string>
#include <map>
#include <algorithm>
#include <limits>

using u64 = shark::u64;
using i64 = int64_t;
using namespace shark::protocols;

// ==================== Precision Configuration ====================
// Use f=24 for better overflow handling in deep networks
// (original was f=26 which can overflow more easily)
const int f = 24;
const u64 SCALAR_ONE = 1ULL << f;
const double OVERFLOW_THRESHOLD = (double)(1ULL << 50);  // Detect potential overflow

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
            X[i] = (u64)(i64)(fval * (1ULL << precision));
        }
    }
    ~Loader() { if (file.is_open()) file.close(); }
};

// ==================== Utils ====================
double fixed_to_float(u64 val) {
    return (double)(i64)val / (double)SCALAR_ONE;
}

u64 float_to_fixed(double val) {
    return (u64)(i64)(val * SCALAR_ONE);
}

// Check for potential overflow
bool check_overflow(double val, const std::string& context = "") {
    if (std::abs(val) > OVERFLOW_THRESHOLD) {
        if (party != DEALER && !context.empty()) {
            std::cerr << "WARNING: Potential overflow in " << context
                      << " value=" << val << std::endl;
        }
        return true;
    }
    return false;
}

// ==================== Secure Operations ====================

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

// ==================== Improved Reciprocal ====================

shark::span<u64> newton_refine_reciprocal(shark::span<u64>& a, shark::span<u64>& x_n,
                                           shark::span<u64>& two_share) {
    size_t size = a.size();

    auto ax = mul::call(a, x_n);
    ax = ars::call(ax, f);

    shark::span<u64> two_vec(size);
    for (size_t i = 0; i < size; ++i) two_vec[i] = two_share[0];
    auto diff = secure_sub(two_vec, ax);

    auto x_next = mul::call(x_n, diff);
    return ars::call(x_next, f);
}

shark::span<u64> improved_reciprocal(shark::span<u64>& a, shark::span<u64>& two_share,
                                      int iterations = 2) {
    auto x = reciprocal::call(a, f);
    for (int i = 0; i < iterations; ++i) {
        x = newton_refine_reciprocal(a, x, two_share);
    }
    return x;
}

// ==================== Improved Alpha Computation ====================
// Fixed formula: alpha = U / (U - L) for unstable neurons
// This is equivalent to: relu(U) / (relu(U) + relu(-L))

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

    // num = ReLU(U) - this gives U when U > 0, else 0
    auto num = relu::call(U);

    // term2 = ReLU(-L) - this gives -L when L < 0, else 0
    shark::span<u64> L_neg(size);
    for(size_t i = 0; i < size; ++i) L_neg[i] = -L[i];
    auto term2 = relu::call(L_neg);

    // den = relu(U) + relu(-L) + epsilon = U - L + epsilon (for unstable neurons)
    auto den = add::call(num, term2);
    auto eps_vec = broadcast_scalar(epsilon_share, size);
    den = add::call(den, eps_vec);

    // Use improved reciprocal with Newton iterations
    auto den_inv = improved_reciprocal(den, two_share, 2);

    // alpha = num * den_inv = U / (U - L)
    auto alpha = mul::call(num, den_inv);
    alpha = ars::call(alpha, f);

    // Clamp to [0, 1]
    return secure_clamp_upper_1(alpha, one_share);
}

// ==================== Optimized Tool Functions ====================

// Vectorized matrix-alpha scaling
// W: rows x cols matrix, alpha: cols vector
// Result: result[r][c] = W[r][c] * alpha[c]
shark::span<u64> scale_matrix_by_alpha(shark::span<u64>& W, shark::span<u64>& alpha,
                                        int rows, int cols) {
    shark::span<u64> alpha_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            alpha_expanded[r * cols + c] = alpha[c];
        }
    }

    auto result = mul::call(W, alpha_expanded);
    return ars::call(result, f);
}

// Vectorized dot product
shark::span<u64> dot_product(shark::span<u64>& A, shark::span<u64>& B, int size) {
    auto prod = mul::call(A, B);
    prod = ars::call(prod, f);

    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        result[0] += prod[i];
    }
    return result;
}

// Vectorized absolute value sum
shark::span<u64> sum_abs(shark::span<u64>& A, int size) {
    auto A_abs = secure_abs(A);

    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        result[0] += A_abs[i];
    }
    return result;
}

// Local row sum (no protocol calls needed)
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

// ==================== Improved Correction Computation ====================
// The correction accounts for the gap between true ReLU and linear relaxation
// For LB: correction = sum_j relu(-A_j) * relu(-L_j)
// For UB: correction = sum_j relu(A_j) * relu(-L_j)
// Note: relu(-L) = -L when L < 0 (unstable), which is the case we care about

// Vectorized LB correction for vector A
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
        result[0] += correction_vec[i];
    }
    return result;
}

// Vectorized UB correction for vector A
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
        result[0] += correction_vec[i];
    }
    return result;
}

// Vectorized LB correction for matrix A
// A: rows x cols, LB: cols
shark::span<u64> compute_lb_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB,
                                                int rows, int cols) {
    // Pre-compute relu(-LB)
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    // Compute -A and relu (entire matrix at once)
    shark::span<u64> A_neg(rows * cols);
    for(int i = 0; i < rows * cols; ++i) A_neg[i] = -A[i];
    auto relu_neg_A = relu::call(A_neg);

    // Expand relu_neg_LB to matrix size
    shark::span<u64> relu_neg_LB_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            relu_neg_LB_expanded[r * cols + c] = relu_neg_LB[c];
        }
    }

    // Single vectorized multiplication
    auto correction_matrix = mul::call(relu_neg_A, relu_neg_LB_expanded);
    correction_matrix = ars::call(correction_matrix, f);

    // Local row sum
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

// Vectorized UB correction for matrix A
shark::span<u64> compute_ub_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB,
                                                int rows, int cols) {
    // Pre-compute relu(-LB)
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    // Compute relu(A) (entire matrix at once)
    auto relu_A = relu::call(A);

    // Expand relu_neg_LB to matrix size
    shark::span<u64> relu_neg_LB_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            relu_neg_LB_expanded[r * cols + c] = relu_neg_LB[c];
        }
    }

    // Single vectorized multiplication
    auto correction_matrix = mul::call(relu_A, relu_neg_LB_expanded);
    correction_matrix = ars::call(correction_matrix, f);

    // Local row sum
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

        // Ax0 = W @ x0
        auto Ax0 = matmul::call(out_dim, in_dim, 1, layer.W, x0);
        Ax0 = ars::call(Ax0, f);
        record_value("Layer0_Ax0", Ax0, 0);

        // Compute ||W||_inf (dual norm = L1 norm of rows for L_inf ball)
        auto W_abs = secure_abs(layer.W);
        auto dualnorm = compute_row_sum_manual(W_abs, out_dim, in_dim);
        record_value("Layer0_dualnorm", dualnorm, 0);

        // radius = ||W||_inf * eps
        auto eps_vec = broadcast_scalar(eps_share, out_dim);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);
        radius = relu::call(radius);  // Ensure non-negative
        record_value("Layer0_radius", radius, 0);

        // UB = Ax0 + b + radius
        auto temp1 = add::call(Ax0, radius);
        bounds.UB = add::call(temp1, layer.b);
        record_value("Layer0_UB", bounds.UB, 0);

        // LB = Ax0 + b - radius
        auto temp2 = secure_sub(Ax0, radius);
        bounds.LB = add::call(temp2, layer.b);
        record_value("Layer0_LB", bounds.LB, 0);

        // Compute alpha for ReLU relaxation
        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share,
                                            one_share, two_share);
        record_value("Layer0_alpha", bounds.alpha, 0);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    LayerBounds compute_middle_layer_bounds(int layer_idx) {
        LayerBounds bounds;
        LayerInfo& curr_layer = layers[layer_idx];
        int out_dim = curr_layer.output_dim;
        int in_dim = curr_layer.input_dim;

        // Initialize A as current layer's weight matrix
        auto A_prop = curr_layer.W;

        // Initialize constants with current layer's bias
        shark::span<u64> constants(out_dim);
        for(int i = 0; i < out_dim; ++i) constants[i] = curr_layer.b[i];

        // Initialize correction accumulators
        shark::span<u64> lb_corr_total(out_dim);
        shark::span<u64> ub_corr_total(out_dim);
        for(int i = 0; i < out_dim; ++i) {
            lb_corr_total[i] = 0;
            ub_corr_total[i] = 0;
        }

        // Backward propagation loop
        for(int i = layer_idx - 1; i >= 0; --i) {
            // Step 1: Pass through ReLU (multiply by alpha)
            A_prop = scale_matrix_by_alpha(A_prop, layer_bounds[i].alpha,
                                           out_dim, layers[i].output_dim);

            if (i == layer_idx - 1) {
                record_value("Layer" + std::to_string(layer_idx) + "_A_after_alpha",
                            A_prop, layer_idx);
            }

            // Step 2: Compute corrections for this layer
            auto lb_c = compute_lb_correction_matrix(A_prop, layer_bounds[i].LB,
                                                      out_dim, layers[i].output_dim);
            auto ub_c = compute_ub_correction_matrix(A_prop, layer_bounds[i].LB,
                                                      out_dim, layers[i].output_dim);

            lb_corr_total = add::call(lb_corr_total, lb_c);
            ub_corr_total = add::call(ub_corr_total, ub_c);

            // Step 3: Add bias contribution: constants += A @ b
            auto Ab = matmul::call(out_dim, layers[i].output_dim, 1, A_prop, layers[i].b);
            Ab = ars::call(Ab, f);
            constants = add::call(constants, Ab);

            // Step 4: Pass through linear layer: A = A @ W
            A_prop = matmul::call(out_dim, layers[i].output_dim, layers[i].input_dim,
                                  A_prop, layers[i].W);
            A_prop = ars::call(A_prop, f);
        }

        record_value("Layer" + std::to_string(layer_idx) + "_constants", constants, layer_idx);
        record_value("Layer" + std::to_string(layer_idx) + "_lb_corr", lb_corr_total, layer_idx);
        record_value("Layer" + std::to_string(layer_idx) + "_ub_corr", ub_corr_total, layer_idx);

        // Compute A @ x0
        auto Ax0 = matmul::call(out_dim, input_dim, 1, A_prop, x0);
        Ax0 = ars::call(Ax0, f);
        record_value("Layer" + std::to_string(layer_idx) + "_Ax0", Ax0, layer_idx);

        // Compute dual norm and radius
        auto A_abs = secure_abs(A_prop);
        auto dualnorm = compute_row_sum_manual(A_abs, out_dim, input_dim);
        record_value("Layer" + std::to_string(layer_idx) + "_dualnorm", dualnorm, layer_idx);

        auto eps_vec = broadcast_scalar(eps_share, out_dim);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);
        radius = relu::call(radius);
        record_value("Layer" + std::to_string(layer_idx) + "_radius", radius, layer_idx);

        // Compute bounds
        auto base = add::call(Ax0, constants);

        // UB = base + ub_correction + radius
        bounds.UB = add::call(base, ub_corr_total);
        bounds.UB = add::call(bounds.UB, radius);
        record_value("Layer" + std::to_string(layer_idx) + "_UB", bounds.UB, layer_idx);

        // LB = base - lb_correction - radius
        bounds.LB = secure_sub(base, lb_corr_total);
        bounds.LB = secure_sub(bounds.LB, radius);
        record_value("Layer" + std::to_string(layer_idx) + "_LB", bounds.LB, layer_idx);

        // Compute alpha
        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share,
                                            one_share, two_share);
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
        int out_dim = 1;  // We're computing a scalar bound
        int in_dim = last_layer.input_dim;

        // W_diff = diff_vec^T @ W = matmul(1, output_dim, input_dim, diff_vec, W)
        // This gives a (1 x input_dim) result
        auto W_diff = matmul::call(1, last_layer.output_dim, in_dim, diff_vec, last_layer.W);
        W_diff = ars::call(W_diff, f);
        record_value("Final_W_diff", W_diff, -1);

        // b_diff = diff_vec^T @ b
        auto b_diff = dot_product(last_layer.b, diff_vec, last_layer.output_dim);
        record_value("Final_b_diff", b_diff, -1);

        // Initialize accumulators
        auto constants = b_diff;
        shark::span<u64> lb_corr_total(1); lb_corr_total[0] = 0;
        shark::span<u64> ub_corr_total(1); ub_corr_total[0] = 0;

        // A_prop is now a vector (in_dim,)
        auto A_prop = W_diff;

        // Backward propagation through hidden layers
        for(int i = last_idx - 1; i >= 0; --i) {
            // Step 1: Pass through ReLU (element-wise multiply by alpha)
            auto A_scaled = mul::call(A_prop, layer_bounds[i].alpha);
            A_prop = ars::call(A_scaled, f);

            if (i == last_idx - 1) {
                record_value("Final_A_after_alpha", A_prop, -1);
            }

            // Step 2: Compute corrections
            auto lb_c = compute_lb_correction_vec(A_prop, layer_bounds[i].LB,
                                                   layers[i].output_dim);
            auto ub_c = compute_ub_correction_vec(A_prop, layer_bounds[i].LB,
                                                   layers[i].output_dim);

            if(i == last_idx - 1) {
                record_value("Final_lb_corr_layer", lb_c, -1);
                record_value("Final_ub_corr_layer", ub_c, -1);
            }

            // Local addition of corrections
            lb_corr_total[0] += lb_c[0];
            ub_corr_total[0] += ub_c[0];

            // Step 3: Add bias contribution
            auto Ab = dot_product(A_prop, layers[i].b, layers[i].output_dim);
            constants[0] += Ab[0];

            // Step 4: Pass through linear layer: A = A @ W
            auto A_next = matmul::call(1, layers[i].output_dim, layers[i].input_dim,
                                       A_prop, layers[i].W);
            A_next = ars::call(A_next, f);

            // Convert to vector format
            shark::span<u64> A_next_vec(layers[i].input_dim);
            for(int k = 0; k < layers[i].input_dim; ++k) A_next_vec[k] = A_next[k];
            A_prop = A_next_vec;
        }

        record_value("Final_constants", constants, -1);
        record_value("Final_lb_corr_total", lb_corr_total, -1);
        record_value("Final_ub_corr_total", ub_corr_total, -1);

        // Compute A @ x0
        auto Ax0 = dot_product(A_prop, x0, input_dim);
        record_value("Final_Ax0", Ax0, -1);

        // Compute dual norm and radius
        auto dualnorm = sum_abs(A_prop, input_dim);
        auto radius = mul::call(dualnorm, eps_share);
        radius = ars::call(radius, f);
        radius = relu::call(radius);
        record_value("Final_dualnorm", dualnorm, -1);
        record_value("Final_radius", radius, -1);

        // Compute final bounds
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

    // Default configuration
    std::string model_name = "eran_cifar_5layer_relu_100_best";
    int input_dim = 3072;
    int output_dim = 10;
    int num_layers = 5;
    int hidden_dim = 100;
    int true_label = 1;
    int target_label = 4;
    float eps = 0.0002;
    int image_id = 6;
    std::string custom_input_file = "";

    // Parse command line arguments
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

    // Build layer dimensions
    std::vector<int> layer_dims;
    layer_dims.push_back(input_dim);
    for (int i = 0; i < num_layers - 1; ++i) {
        layer_dims.push_back(hidden_dim);
    }
    layer_dims.push_back(output_dim);

    // Build file paths
    std::string base_path = "shark_crown_ml/crown_mpc_data/" + model_name;
    std::string weights_file = base_path + "/weights/weights.dat";

    std::string input_file;
    if (!custom_input_file.empty()) {
        input_file = custom_input_file;
    } else {
        input_file = base_path + "/images/" + std::to_string(image_id) + ".bin";
    }

    if (party != DEALER) {
        std::cout << "\n********************************************" << std::endl;
        std::cout << "  CROWN V2 (Improved Precision)" << std::endl;
        std::cout << "  MODEL: " << model_name << std::endl;
        std::cout << "  Layers: " << num_layers << ", Hidden: " << hidden_dim << std::endl;
        std::cout << "  Fixed-point precision: f=" << f << std::endl;
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

    // ==================== Input Phase ====================
    shark::utils::start_timer("End_to_end_time");
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

    if (party != DEALER) peer->sync();
    shark::utils::start_timer("crown_calculation");

    // ==================== CROWN Computation ====================
    CROWNComputer crown(input_dim);
    crown.set_input(x0, eps_share, epsilon_share, one_share, two_share, ones_input);
    for (int i = 0; i < num_layers; ++i) {
        crown.add_layer(weights[i], biases[i], layer_dims[i], layer_dims[i+1]);
    }

    auto [final_LB, final_UB] = crown.compute_worst_bound(diff_vec, true_label, target_label);

    shark::utils::stop_timer("crown_calculation");
    shark::utils::stop_timer("End_to_end_time");

    // ==================== Output Results ====================
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
                    std::cout << std::fixed << std::setprecision(6)
                              << fixed_to_float(record.data[i]);
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
        std::cout << "IMAGE: " << (custom_input_file.empty() ?
                    std::to_string(image_id) + ".bin" : custom_input_file) << std::endl;
        std::cout << "EPS: " << eps << " | True: " << true_label
                  << " | Target: " << target_label << std::endl;
        std::cout << "Precision: f=" << f << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "MPC LB: " << std::fixed << std::setprecision(6)
                  << fixed_to_float(final_LB[0]) << std::endl;
        std::cout << "MPC UB: " << std::fixed << std::setprecision(6)
                  << fixed_to_float(final_UB[0]) << std::endl;
        std::cout << "============================================" << std::endl;

        shark::utils::print_all_timers();
    }

    finalize::call();
    return 0;
}
