#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/reciprocal.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/ars.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

using u64 = shark::u64;
using namespace shark::protocols;

const int f = 26;
const u64 SCALAR_ONE = 1ULL << f;

u64 float_to_fixed(double val) {
    return (u64)(int64_t)(val * SCALAR_ONE);
}

double fixed_to_float(u64 val) {
    return (double)(int64_t)val / (double)SCALAR_ONE;
}

// ==================== Newton Refinement Protocol ====================

shark::span<u64> secure_sub(shark::span<u64> A, shark::span<u64> B) {
    shark::span<u64> B_neg(B.size());
    for(size_t i = 0; i < B.size(); ++i) B_neg[i] = -B[i];
    return add::call(A, B_neg);
}

// 牛顿迭代: x_{n+1} = x_n * (2 - a * x_n)
shark::span<u64> newton_step(shark::span<u64> a, shark::span<u64> x_n, shark::span<u64> two_share) {
    // 1. a * x_n
    auto ax = mul::call(a, x_n);
    auto ax_scaled = ars::call(ax, f);

    // 2. 2 - (a * x_n)
    shark::span<u64> two_vec(a.size());
    for(size_t i = 0; i < a.size(); ++i) two_vec[i] = two_share[0];
    auto diff = secure_sub(two_vec, ax_scaled);

    // 3. x_n * (2 - a * x_n)
    auto x_next = mul::call(x_n, diff);
    return ars::call(x_next, f);
}

// ==================== Main Test ====================

int main(int argc, char **argv) {
    init::from_args(argc, argv);

    // 测试用例：根据你提供的 Layer 0 神经元数据
    struct TestCase {
        double num;
        double den;
        double python_alpha;
    };

    std::vector<TestCase> cases = {
        {154.909244, 303.747335, 0.509994}, // Neuron 0
        {171.819000, 331.877625, 0.517718}, // Neuron 1 (估算)
        {10.0, 50.0, 0.2}                   // 小值基准
    };

    int n = cases.size();
    shark::span<u64> mpc_nums(n), mpc_dens(n), mpc_two(1);

    if (party == CLIENT) {
        for (int i = 0; i < n; ++i) {
            mpc_nums[i] = float_to_fixed(cases[i].num);
            mpc_dens[i] = float_to_fixed(cases[i].den);
        }
        mpc_two[0] = float_to_fixed(2.0);
    }

    // 输入共享
    input::call(mpc_nums, CLIENT);
    input::call(mpc_dens, CLIENT);
    input::call(mpc_two, CLIENT);
    if (party != DEALER) peer->sync();

    // --- 方法 1: 原始 Reciprocal ---
    auto inv_orig = reciprocal::call(mpc_dens, f);
    auto alpha_orig_prod = mul::call(mpc_nums, inv_orig);
    auto alpha_orig = ars::call(alpha_orig_prod, f);

    // --- 方法 2: 改进 Reciprocal (1次牛顿迭代) ---
    auto inv_newton_1 = newton_step(mpc_dens, inv_orig, mpc_two);
    auto alpha_newton_1_prod = mul::call(mpc_nums, inv_newton_1);
    auto alpha_newton_1 = ars::call(alpha_newton_1_prod, f);

    // --- 方法 3: 改进 Reciprocal (2次牛顿迭代) ---
    auto inv_newton_2 = newton_step(mpc_dens, inv_newton_1, mpc_two);
    auto alpha_newton_2_prod = mul::call(mpc_nums, inv_newton_2);
    auto alpha_newton_2 = ars::call(alpha_newton_2_prod, f);

    // 输出结果
    output::call(alpha_orig);
    output::call(alpha_newton_1);
    output::call(alpha_newton_2);

    if (party != DEALER) {
        std::cout << "\n" << std::string(110, '=') << std::endl;
        std::cout << std::setw(10) << "ID"
                  << std::setw(15) << "Python"
                  << std::setw(18) << "MPC Original"
                  << std::setw(18) << "Newton (1-it)"
                  << std::setw(18) << "Newton (2-it)"
                  << std::setw(15) << "Best Error" << std::endl;
        std::cout << std::string(110, '-') << std::endl;

        for (int i = 0; i < n; ++i) {
            double py = cases[i].python_alpha;
            double v_orig = fixed_to_float(alpha_orig[i]);
            double v_n1 = fixed_to_float(alpha_newton_1[i]);
            double v_n2 = fixed_to_float(alpha_newton_2[i]);

            double error = std::abs(v_n2 - py);

            std::cout << std::setw(10) << i
                      << std::setprecision(6) << std::fixed
                      << std::setw(15) << py
                      << std::setw(18) << v_orig
                      << std::setw(18) << v_n1
                      << std::setw(18) << v_n2
                      << std::setw(15) << error << std::endl;
        }
        std::cout << std::string(110, '=') << std::endl;
        std::cout << "Note: If Newton (1-it) is close to Python, use 1 iteration to save communication costs." << std::endl;
    }

    finalize::call();
    return 0;
}