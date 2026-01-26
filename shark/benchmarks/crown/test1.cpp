// Auto-generated CROWN MPC code
// Network structure: 2 -> 4 -> 3 -> 2
// Generated for eps = 0.1

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
#include <iomanip>
#include <cmath>

using u64 = shark::u64;
using namespace shark::protocols;

const int f = 24;
const u64 SCALAR_ONE = 1ULL << f;

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

int main(int argc, char **argv) {
    init::from_args(argc, argv);

    // Network dimensions
    const int dim0 = 2;
    const int dim1 = 4;
    const int dim2 = 3;
    const int dim3 = 2;
    const int max_dim = 4;

    // Buffers
    shark::span<u64> x0(dim0);
    shark::span<u64> eps_share(1);
    shark::span<u64> epsilon_share(1);
    shark::span<u64> diff_vec(dim3);
    shark::span<u64> ones_global(max_dim);

    // Weights and biases (row-major storage)
    shark::span<u64> W1(dim1 * dim0);  // (4, 2)
    shark::span<u64> b1(dim1);
    shark::span<u64> W2(dim2 * dim1);  // (3, 4)
    shark::span<u64> b2(dim2);
    shark::span<u64> W3(dim3 * dim2);  // (2, 3)
    shark::span<u64> b3(dim3);

    // --- CLIENT Initialization ---
    if (party == CLIENT) {
        x0[0] = float_to_fixed(0.13);
        x0[1] = float_to_fixed(-0.45);
        eps_share[0] = float_to_fixed(0.1);
        epsilon_share[0] = float_to_fixed(0.000001);
        for(int i = 0; i < max_dim; ++i) ones_global[i] = SCALAR_ONE;
        diff_vec[0] = float_to_fixed(1.0);
        diff_vec[1] = float_to_fixed(-1.0);
    }

    // --- SERVER Initialization ---
    if (party == SERVER) {
        // W1: (4, 2)
        W1[0*2+0] = float_to_fixed(1.0);
        W1[0*2+1] = float_to_fixed(2.0);
        W1[1*2+0] = float_to_fixed(-1.0);
        W1[1*2+1] = float_to_fixed(1.0);
        W1[2*2+0] = float_to_fixed(2.0);
        W1[2*2+1] = float_to_fixed(-1.0);
        W1[3*2+0] = float_to_fixed(0.5);
        W1[3*2+1] = float_to_fixed(1.5);
        // b1
        b1[0] = float_to_fixed(0.1);
        b1[1] = float_to_fixed(0.0);
        b1[2] = float_to_fixed(-0.1);
        b1[3] = float_to_fixed(0.2);

        // W2: (3, 4)
        W2[0*4+0] = float_to_fixed(1.0);
        W2[0*4+1] = float_to_fixed(-1.0);
        W2[0*4+2] = float_to_fixed(0.5);
        W2[0*4+3] = float_to_fixed(0.5);
        W2[1*4+0] = float_to_fixed(0.5);
        W2[1*4+1] = float_to_fixed(1.0);
        W2[1*4+2] = float_to_fixed(-0.5);
        W2[1*4+3] = float_to_fixed(1.0);
        W2[2*4+0] = float_to_fixed(-0.5);
        W2[2*4+1] = float_to_fixed(0.5);
        W2[2*4+2] = float_to_fixed(1.0);
        W2[2*4+3] = float_to_fixed(-1.0);
        // b2
        b2[0] = float_to_fixed(0.0);
        b2[1] = float_to_fixed(0.1);
        b2[2] = float_to_fixed(-0.1);

        // W3: (2, 3)
        W3[0*3+0] = float_to_fixed(1.0);
        W3[0*3+1] = float_to_fixed(-1.0);
        W3[0*3+2] = float_to_fixed(0.5);
        W3[1*3+0] = float_to_fixed(-0.5);
        W3[1*3+1] = float_to_fixed(1.0);
        W3[1*3+2] = float_to_fixed(1.0);
        // b3
        b3[0] = float_to_fixed(0.0);
        b3[1] = float_to_fixed(0.0);

        for(int i = 0; i < max_dim; ++i) ones_global[i] = 0;
    }

    // --- Input Phase ---
    shark::utils::start_timer("input");
    input::call(x0, CLIENT);
    input::call(eps_share, CLIENT);
    input::call(epsilon_share, CLIENT);
    input::call(diff_vec, CLIENT);
    input::call(ones_global, CLIENT);
    input::call(W1, SERVER);
    input::call(b1, SERVER);
    input::call(W2, SERVER);
    input::call(b2, SERVER);
    input::call(W3, SERVER);
    input::call(b3, SERVER);
    shark::utils::stop_timer("input");

    if (party != DEALER) peer->sync();

    // ==================== Computation Phase ====================
    shark::utils::start_timer("crown_mpc");

    shark::span<u64> ones_dim0(dim0);
    for(int i = 0; i < dim0; ++i) ones_dim0[i] = ones_global[i];

    // ===== Layer 1: IBP =====
    auto Ax0_1 = matmul::call(dim1, dim0, 1, W1, x0);
    Ax0_1 = ars::call(Ax0_1, f);

    auto W1_abs = secure_abs(W1);
    auto dualnorm_1 = matmul::call(dim1, dim0, 1, W1_abs, ones_dim0);
    dualnorm_1 = ars::call(dualnorm_1, f);

    auto eps_vec_1 = broadcast_scalar(eps_share, dim1);
    auto radius_1 = mul::call(dualnorm_1, eps_vec_1);
    radius_1 = ars::call(radius_1, f);

    auto temp1 = add::call(Ax0_1, radius_1);
    auto UB1 = add::call(temp1, b1);
    auto temp2 = secure_sub(Ax0_1, radius_1);
    auto LB1 = add::call(temp2, b1);

    auto alpha_1 = compute_alpha_secure(UB1, LB1, epsilon_share);

    // ===== Layer 2: CROWN =====
    auto A_2 = scale_matrix_by_alpha(W2, alpha_1, dim2, dim1);

    shark::span<u64> constants_2(dim2);
    for(int i = 0; i < dim2; ++i) constants_2[i] = b2[i];
    auto Ab1 = matmul::call(dim2, dim1, 1, A_2, b1);
    Ab1 = ars::call(Ab1, f);
    constants_2 = add::call(constants_2, Ab1);

    auto A_2_prop = matmul::call(dim2, dim1, dim0, A_2, W1);
    A_2_prop = ars::call(A_2_prop, f);

    auto Ax0_2 = matmul::call(dim2, dim0, 1, A_2_prop, x0);
    Ax0_2 = ars::call(Ax0_2, f);

    auto A2_abs = secure_abs(A_2_prop);
    auto dualnorm_2 = matmul::call(dim2, dim0, 1, A2_abs, ones_dim0);
    dualnorm_2 = ars::call(dualnorm_2, f);

    auto eps_vec_2 = broadcast_scalar(eps_share, dim2);
    auto radius_2 = mul::call(dualnorm_2, eps_vec_2);
    radius_2 = ars::call(radius_2, f);

    auto base_2 = add::call(Ax0_2, constants_2);
    auto UB2 = add::call(base_2, radius_2);
    auto LB2 = secure_sub(base_2, radius_2);

    auto alpha_2 = compute_alpha_secure(UB2, LB2, epsilon_share);

    // ===== Final Layer: Differential Verification =====
    shark::span<u64> W_diff(dim2);
    for(int i = 0; i < dim2; ++i) {
        shark::span<u64> sum(1);
        sum[0] = 0;
        for(int j = 0; j < dim3; ++j) {
            shark::span<u64> w_elem(1), d_elem(1);
            w_elem[0] = W3[j * dim2 + i];
            d_elem[0] = diff_vec[j];
            auto prod = mul::call(w_elem, d_elem);
            prod = ars::call(prod, f);
            sum = add::call(sum, prod);
        }
        W_diff[i] = sum[0];
    }

    shark::span<u64> b_diff(1);
    b_diff[0] = 0;
    for(int j = 0; j < dim3; ++j) {
        shark::span<u64> b_elem(1), d_elem(1);
        b_elem[0] = b3[j];
        d_elem[0] = diff_vec[j];
        auto prod = mul::call(b_elem, d_elem);
        prod = ars::call(prod, f);
        b_diff = add::call(b_diff, prod);
    }

    shark::span<u64> A_final_layer(dim2);
    for(int i = 0; i < dim2; ++i) {
        shark::span<u64> w_elem(1), a_elem(1);
        w_elem[0] = W_diff[i];
        a_elem[0] = alpha_2[i];
        auto prod = mul::call(w_elem, a_elem);
        prod = ars::call(prod, f);
        A_final_layer[i] = prod[0];
    }

    auto constants_final = dot_product(A_final_layer, b2, dim2);
    constants_final = add::call(constants_final, b_diff);

    // Backpropagate A to input layer
    auto A_prop = scale_matrix_by_alpha(W2, alpha_1, dim2, dim1);
    auto A_current = matmul::call(1, dim2, dim1, A_final_layer, A_prop);
    A_current = ars::call(A_current, f);

    // Add constants from layer 1
    auto const_add_1 = dot_product(A_current, b1, dim1);
    constants_final = add::call(constants_final, const_add_1);

    A_current = matmul::call(1, dim1, dim0, A_current, W1);
    A_current = ars::call(A_current, f);

    // Final bound computation
    auto Ax0_final = dot_product(A_current, x0, dim0);
    auto dualnorm_final = sum_abs(A_current, dim0);
    auto radius_final = mul::call(dualnorm_final, eps_share);
    radius_final = ars::call(radius_final, f);

    auto base_final = add::call(Ax0_final, constants_final);
    auto LB_final = secure_sub(base_final, radius_final);

    shark::utils::stop_timer("crown_mpc");

    // ==================== Output Phase ====================
    output::call(LB_final);

    if (party != DEALER) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "CROWN MPC Result" << std::endl;
        std::cout << "============================================" << std::endl;
        double lb_val = fixed_to_float(LB_final[0]);
        std::cout << "LB = " << std::fixed << std::setprecision(6) << lb_val << std::endl;
        std::cout << "Robust: " << (lb_val > 0 ? "YES" : "NO") << std::endl;
        std::cout << "============================================" << std::endl;
    }

    finalize::call();
    shark::utils::print_all_timers();

    return 0;
}