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

const int f = 26;
const u64 SCALAR_ONE = 1ULL << f;

u64 float_to_fixed(double val) {
    return (u64)(int64_t)(val * SCALAR_ONE);
}

double fixed_to_float(u64 val) {
    return (double)(int64_t)val / (double)SCALAR_ONE;
}

// ==================== Secure Operations ====================

shark::span<u64> secure_sub(shark::span<u64>& A, shark::span<u64>& B) {
    shark::span<u64> B_neg(B.size());
    for(size_t i = 0; i < B.size(); ++i) B_neg[i] = -B[i];
    return add::call(A, B_neg);
}

// 牛顿迭代: x_{n+1} = x_n * (2 - a * x_n)
shark::span<u64> newton_step(shark::span<u64>& a, shark::span<u64>& x_n, shark::span<u64>& two_share) {
    size_t size = a.size();

    // 1. ax = a * x_n
    auto ax = mul::call(a, x_n);
    ax = ars::call(ax, f);

    // 2. diff = 2 - ax
    shark::span<u64> two_vec(size);
    for(size_t i = 0; i < size; ++i) two_vec[i] = two_share[0];
    auto diff = secure_sub(two_vec, ax);

    // 3. x_next = x_n * diff
    auto x_next = mul::call(x_n, diff);
    return ars::call(x_next, f);
}

// 改进的倒数：原始 + N次牛顿迭代
shark::span<u64> improved_reciprocal(shark::span<u64>& a, shark::span<u64>& two_share, int iterations) {
    auto x = reciprocal::call(a, f);
    for(int i = 0; i < iterations; ++i) {
        x = newton_step(a, x, two_share);
    }
    return x;
}

// ==================== Test Case Generator ====================

struct TestCase {
    std::string category;
    double value;
    double true_reciprocal;
};

std::vector<TestCase> generate_test_cases() {
    std::vector<TestCase> cases;

    // 1. 小整数
    cases.push_back({"Small Int", 2.0, 0.5});
    cases.push_back({"Small Int", 4.0, 0.25});
    cases.push_back({"Small Int", 5.0, 0.2});
    cases.push_back({"Small Int", 8.0, 0.125});
    cases.push_back({"Small Int", 10.0, 0.1});

    // 2. 中等值 (CROWN中常见的范围)
    cases.push_back({"Medium", 50.0, 0.02});
    cases.push_back({"Medium", 100.0, 0.01});
    cases.push_back({"Medium", 200.0, 0.005});
    cases.push_back({"Medium", 500.0, 0.002});

    // 3. 大值 (CROWN alpha计算中的分母范围)
    cases.push_back({"Large", 303.747335, 1.0/303.747335});  // 来自你的Layer0
    cases.push_back({"Large", 331.877625, 1.0/331.877625});
    cases.push_back({"Large", 1000.0, 0.001});
    cases.push_back({"Large", 2000.0, 0.0005});
    cases.push_back({"Large", 5000.0, 0.0002});

    // 4. 非整数值
    cases.push_back({"Decimal", 1.5, 1.0/1.5});
    cases.push_back({"Decimal", 2.7, 1.0/2.7});
    cases.push_back({"Decimal", 3.14159, 1.0/3.14159});
    cases.push_back({"Decimal", 12.345, 1.0/12.345});

    // 5. 接近1的值
    cases.push_back({"Near 1", 1.0, 1.0});
    cases.push_back({"Near 1", 1.1, 1.0/1.1});
    cases.push_back({"Near 1", 0.9, 1.0/0.9});
    cases.push_back({"Near 1", 1.01, 1.0/1.01});

    // 6. 小于1的值
    cases.push_back({"< 1", 0.5, 2.0});
    cases.push_back({"< 1", 0.25, 4.0});
    cases.push_back({"< 1", 0.1, 10.0});
    cases.push_back({"< 1", 0.01, 100.0});

    // 7. 随机值 (模拟真实CROWN场景)
    std::mt19937 rng(42);  // 固定种子保证可重复
    std::uniform_real_distribution<double> dist_small(1.0, 100.0);
    std::uniform_real_distribution<double> dist_large(100.0, 1000.0);

    for(int i = 0; i < 5; ++i) {
        double v = dist_small(rng);
        cases.push_back({"Random S", v, 1.0/v});
    }
    for(int i = 0; i < 5; ++i) {
        double v = dist_large(rng);
        cases.push_back({"Random L", v, 1.0/v});
    }

    return cases;
}

// ==================== Main ====================

int main(int argc, char **argv) {
    init::from_args(argc, argv);

    auto cases = generate_test_cases();
    int n = cases.size();

    // 分配内存
    shark::span<u64> mpc_values(n);
    shark::span<u64> mpc_two(1);

    // 初始化（双方都初始化避免未定义行为）
    for(int i = 0; i < n; ++i) mpc_values[i] = 0;
    mpc_two[0] = 0;

    // CLIENT 设置真实值
    if (party == CLIENT) {
        for(int i = 0; i < n; ++i) {
            mpc_values[i] = float_to_fixed(cases[i].value);
        }
        mpc_two[0] = float_to_fixed(2.0);
    }

    // 秘密共享
    input::call(mpc_values, CLIENT);
    input::call(mpc_two, CLIENT);
    if (party != DEALER) peer->sync();

    // ==================== 计算各种倒数 ====================

    // 方法0: 原始 reciprocal
    shark::utils::start_timer("original_reciprocal");
    auto inv_original = reciprocal::call(mpc_values, f);
    shark::utils::stop_timer("original_reciprocal");

    // 方法1: 1次牛顿迭代
    shark::utils::start_timer("newton_1_iter");
    auto inv_newton1 = improved_reciprocal(mpc_values, mpc_two, 1);
    shark::utils::stop_timer("newton_1_iter");

    // 方法2: 2次牛顿迭代
    shark::utils::start_timer("newton_2_iter");
    auto inv_newton2 = improved_reciprocal(mpc_values, mpc_two, 2);
    shark::utils::stop_timer("newton_2_iter");

    // 方法3: 3次牛顿迭代
    shark::utils::start_timer("newton_3_iter");
    auto inv_newton3 = improved_reciprocal(mpc_values, mpc_two, 3);
    shark::utils::stop_timer("newton_3_iter");

    // ==================== 输出结果 ====================

    output::call(inv_original);
    output::call(inv_newton1);
    output::call(inv_newton2);
    output::call(inv_newton3);

    if (party != DEALER) {
        // 统计信息
        double sum_err_orig = 0, sum_err_n1 = 0, sum_err_n2 = 0, sum_err_n3 = 0;
        double max_err_orig = 0, max_err_n1 = 0, max_err_n2 = 0, max_err_n3 = 0;

        // 打印表头
        std::cout << "\n" << std::string(140, '=') << std::endl;
        std::cout << "Reciprocal Precision Comparison Test (f=" << f << ")" << std::endl;
        std::cout << std::string(140, '=') << std::endl;

        std::cout << std::left
                  << std::setw(12) << "Category"
                  << std::setw(14) << "Input"
                  << std::setw(14) << "True 1/x"
                  << std::setw(14) << "Original"
                  << std::setw(14) << "Newton-1"
                  << std::setw(14) << "Newton-2"
                  << std::setw(14) << "Newton-3"
                  << std::setw(12) << "Orig Err"
                  << std::setw(12) << "N1 Err"
                  << std::setw(12) << "Best"
                  << std::endl;
        std::cout << std::string(140, '-') << std::endl;

        for(int i = 0; i < n; ++i) {
            double true_val = cases[i].true_reciprocal;
            double v_orig = fixed_to_float(inv_original[i]);
            double v_n1 = fixed_to_float(inv_newton1[i]);
            double v_n2 = fixed_to_float(inv_newton2[i]);
            double v_n3 = fixed_to_float(inv_newton3[i]);

            double err_orig = std::abs(v_orig - true_val);
            double err_n1 = std::abs(v_n1 - true_val);
            double err_n2 = std::abs(v_n2 - true_val);
            double err_n3 = std::abs(v_n3 - true_val);

            // 累计误差
            sum_err_orig += err_orig;
            sum_err_n1 += err_n1;
            sum_err_n2 += err_n2;
            sum_err_n3 += err_n3;

            max_err_orig = std::max(max_err_orig, err_orig);
            max_err_n1 = std::max(max_err_n1, err_n1);
            max_err_n2 = std::max(max_err_n2, err_n2);
            max_err_n3 = std::max(max_err_n3, err_n3);

            // 找最佳方法
            std::string best = "Orig";
            double min_err = err_orig;
            if(err_n1 < min_err) { min_err = err_n1; best = "N1"; }
            if(err_n2 < min_err) { min_err = err_n2; best = "N2"; }
            if(err_n3 < min_err) { min_err = err_n3; best = "N3"; }

            std::cout << std::left << std::fixed
                      << std::setw(12) << cases[i].category
                      << std::setw(14) << std::setprecision(4) << cases[i].value
                      << std::setw(14) << std::setprecision(6) << true_val
                      << std::setw(14) << std::setprecision(6) << v_orig
                      << std::setw(14) << std::setprecision(6) << v_n1
                      << std::setw(14) << std::setprecision(6) << v_n2
                      << std::setw(14) << std::setprecision(6) << v_n3
                      << std::setw(12) << std::setprecision(8) << err_orig
                      << std::setw(12) << std::setprecision(8) << err_n1
                      << std::setw(12) << best
                      << std::endl;
        }

        // 汇总统计
        std::cout << std::string(140, '=') << std::endl;
        std::cout << "\nSummary Statistics:" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        std::cout << std::left << std::setw(20) << "Method"
                  << std::setw(20) << "Avg Error"
                  << std::setw(20) << "Max Error" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        std::cout << std::setw(20) << "Original"
                  << std::setw(20) << std::scientific << std::setprecision(4) << sum_err_orig / n
                  << std::setw(20) << max_err_orig << std::endl;

        std::cout << std::setw(20) << "Newton (1 iter)"
                  << std::setw(20) << std::scientific << std::setprecision(4) << sum_err_n1 / n
                  << std::setw(20) << max_err_n1 << std::endl;

        std::cout << std::setw(20) << "Newton (2 iter)"
                  << std::setw(20) << std::scientific << std::setprecision(4) << sum_err_n2 / n
                  << std::setw(20) << max_err_n2 << std::endl;

        std::cout << std::setw(20) << "Newton (3 iter)"
                  << std::setw(20) << std::scientific << std::setprecision(4) << sum_err_n3 / n
                  << std::setw(20) << max_err_n3 << std::endl;

        std::cout << std::string(60, '-') << std::endl;

        // 改进率
        double improve_n1 = (sum_err_orig - sum_err_n1) / sum_err_orig * 100;
        double improve_n2 = (sum_err_orig - sum_err_n2) / sum_err_orig * 100;
        double improve_n3 = (sum_err_orig - sum_err_n3) / sum_err_orig * 100;

        std::cout << "\nImprovement over Original:" << std::endl;
        std::cout << "  Newton-1: " << std::fixed << std::setprecision(2) << improve_n1 << "%" << std::endl;
        std::cout << "  Newton-2: " << std::fixed << std::setprecision(2) << improve_n2 << "%" << std::endl;
        std::cout << "  Newton-3: " << std::fixed << std::setprecision(2) << improve_n3 << "%" << std::endl;

        std::cout << "\n" << std::string(140, '=') << std::endl;
    }

    finalize::call();
    shark::utils::print_all_timers();

    return 0;
}