#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/reciprocal.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/ars.hpp>
#include <shark/utils/timer.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>

using u64 = shark::u64;
using namespace shark::protocols;

// 精度设置 (CROWN 默认 26)
const int f = 26;
const u64 SCALAR_ONE = 1ULL << f;

// --- 基础转换工具 ---
u64 float_to_fixed(double val) { return (u64)(int64_t)(val * SCALAR_ONE); }
double fixed_to_float(u64 val) { return (double)(int64_t)val / (double)SCALAR_ONE; }

// --- 牛顿迭代核心公式 ---
// x_{n+1} = x_n * (2 - a * x_n)
shark::span<u64> newton_step(shark::span<u64>& a, shark::span<u64>& x_n, shark::span<u64>& two_share) {
    auto ax = ars::call(mul::call(a, x_n), f);
    shark::span<u64> diff(a.size());
    for(size_t i = 0; i < a.size(); ++i) {
        diff[i] = two_share[0] - ax[i];
    }
    return ars::call(mul::call(x_n, diff), f);
}

// --- 误差分析汇总函数 ---
void analyze_result(std::string name, shark::span<u64>& mpc_res, const std::vector<double>& ref, int N) {
    double total_l1 = 0;
    double max_e = 0;
    for(int i = 0; i < N; ++i) {
        double mpc_val = fixed_to_float(mpc_res[i]);
        double err = std::abs(mpc_val - ref[i]);
        total_l1 += err;
        if(err > max_e) max_e = err;
    }
    std::cout << std::left << std::setw(15) << name
              << " | L1 Sum: " << std::scientific << std::setprecision(4) << total_l1
              << " | Max Err: " << std::fixed << std::setprecision(8) << max_e << std::endl;
}

int main(int argc, char **argv) {
    init::from_args(argc, argv);

    const int N = 100000;
    std::vector<double> true_inputs(N);
    std::vector<double> true_recips(N);
    shark::span<u64> mpc_inputs(N);
    shark::span<u64> mpc_two(1);

    // 1. CLIENT 生成对数分布的随机测试数据 (挑战数值稳定性)
    if (party == CLIENT) {
        std::mt19937 rng(12345);
        // 对数均匀分布：指数从 -3 (0.001) 到 4 (10000)
        std::uniform_real_distribution<double> dist_exp(-1.5, 3);

        std::cout << "Client: Generating " << N << " cross-magnitude inputs..." << std::endl;
        for(int i = 0; i < N; ++i) {
            true_inputs[i] = std::pow(10.0, dist_exp(rng));
            true_recips[i] = 1.0 / true_inputs[i];
            mpc_inputs[i] = float_to_fixed(true_inputs[i]);
        }
        mpc_two[0] = float_to_fixed(2.0);
    }

    // 2. 秘密共享
    input::call(mpc_inputs, CLIENT);
    input::call(mpc_two, CLIENT);
    if (party != DEALER) peer->sync();

    // ---------------------------------------------------------
    // 执行 Method 0: Original
    // ---------------------------------------------------------
    shark::utils::start_timer("0_Method_Original");
    auto res_orig = reciprocal::call(mpc_inputs, f);
    shark::utils::stop_timer("0_Method_Original");
    output::call(res_orig);

    // ---------------------------------------------------------
    // 执行 Method 1: Newton 1-Iteration
    // ---------------------------------------------------------
    shark::utils::start_timer("1_Method_Newton_1it");
    auto res_n1 = newton_step(mpc_inputs, res_orig, mpc_two); // 基于 Original 的结果迭代一次
    shark::utils::stop_timer("1_Method_Newton_1it");
    output::call(res_n1);

    // ---------------------------------------------------------
    // 执行 Method 2: Newton 2-Iterations
    // ---------------------------------------------------------
    shark::utils::start_timer("2_Method_Newton_2it");
    auto res_n2 = newton_step(mpc_inputs, res_n1, mpc_two); // 基于 N1 的结果再迭代一次
    shark::utils::stop_timer("2_Method_Newton_2it");
    output::call(res_n2);

    // ---------------------------------------------------------
    // 结果分析与抽样打印 (仅 CLIENT 端)
    // ---------------------------------------------------------
    if (party == CLIENT) {
        std::cout << "\n" << std::string(100, '=') << std::endl;
        std::cout << "MPC RECIPROCAL ULTRA STRESS TEST (N=100k, Magnitude: 10^-3 ~ 10^4)" << std::endl;
        std::cout << std::string(100, '=') << std::endl;

        analyze_result("Original", res_orig, true_recips, N);
        analyze_result("Newton-1it", res_n1, true_recips, N);
        analyze_result("Newton-2it", res_n2, true_recips, N);

        std::cout << "\n--- Sample Data Inspection (Comparison of True vs Newton-2it) ---" << std::endl;
        std::cout << std::left << std::setw(15) << "Input (x)"
                  << " | " << std::setw(15) << "True 1/x"
                  << " | " << std::setw(15) << "Newton-2it"
                  << " | " << "Relative Err" << std::endl;

        for(int i = 0; i < N; i += 20000) {
            double n2_val = fixed_to_float(res_n2[i]);
            double rel_err = std::abs(n2_val - true_recips[i]) / true_recips[i];

            std::cout << std::scientific << std::setprecision(4)
                      << std::setw(15) << true_inputs[i] << " | "
                      << std::setw(15) << true_recips[i] << " | "
                      << std::setw(15) << n2_val << " | "
                      << std::fixed << std::setprecision(2) << rel_err * 100 << "%" << std::endl;
        }
        std::cout << std::string(100, '=') << std::endl;
    }

    finalize::call();

    // 打印时间成本
    shark::utils::print_all_timers();

    return 0;
}