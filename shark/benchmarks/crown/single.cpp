// Debug test for Newton iteration with non-exact value
// 调试非精确值的牛顿迭代

#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/reciprocal.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/add.hpp>
#include <shark/protocols/ars.hpp>

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

int main(int argc, char **argv) {
    init::from_args(argc, argv);

    // 测试 x = 0.37 (有误差的情况)
    shark::span<u64> x(1);
    shark::span<u64> two(1);

    if (party == CLIENT) {
        x[0] = float_to_fixed(0.37);     // x = 0.37
        two[0] = float_to_fixed(2.0);    // 2.0
    }

    if (party == SERVER) {
        x[0] = 0;
        two[0] = 0;
    }

    input::call(x, CLIENT);
    input::call(two, CLIENT);

    if (party != DEALER) peer->sync();

    // ==================== 逐步调试 ====================

    // Step 1: y0 = reciprocal(x)
    auto y0 = reciprocal::call(x, f);

    // Step 2: xy0 = x * y0
    auto xy0_raw = mul::call(x, y0);
    auto xy0 = ars::call(xy0_raw, f);

    // Step 3: diff = 2 - xy0
    shark::span<u64> xy0_neg(1);
    xy0_neg[0] = -xy0[0];
    auto diff = add::call(two, xy0_neg);

    // Step 4: y1 = y0 * diff
    auto y0_times_diff_raw = mul::call(y0, diff);
    auto y1 = ars::call(y0_times_diff_raw, f);

    // Verification
    auto verify_y0 = mul::call(x, y0);
    verify_y0 = ars::call(verify_y0, f);

    auto verify_y1 = mul::call(x, y1);
    verify_y1 = ars::call(verify_y1, f);

    // ==================== 输出 ====================
    shark::span<u64> output_data(12);
    output_data[0] = x[0];
    output_data[1] = two[0];
    output_data[2] = y0[0];
    output_data[3] = xy0_raw[0];
    output_data[4] = xy0[0];
    output_data[5] = xy0_neg[0];
    output_data[6] = diff[0];
    output_data[7] = y0_times_diff_raw[0];
    output_data[8] = y1[0];
    output_data[9] = verify_y0[0];
    output_data[10] = verify_y1[0];

    // 计算期望值
    double expected_1_over_x = 1.0 / 0.37;
    output_data[11] = float_to_fixed(expected_1_over_x);

    output::call(output_data);

    if (party != DEALER) {
        double expected = 1.0 / 0.37;  // 2.7027...

        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Newton Iteration Debug (x = 0.37, expected 1/x = " << expected << ")" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        std::cout << std::fixed << std::setprecision(6);

        std::cout << "\n--- Input Values ---" << std::endl;
        std::cout << "x = " << fixed_to_float(output_data[0]) << std::endl;
        std::cout << "two = " << fixed_to_float(output_data[1]) << std::endl;

        std::cout << "\n--- Step 1: y0 = reciprocal(x) ---" << std::endl;
        double y0_val = fixed_to_float(output_data[2]);
        std::cout << "y0 = " << y0_val << " (expected: " << expected << ")" << std::endl;
        std::cout << "y0 error = " << (y0_val - expected) << std::endl;

        std::cout << "\n--- Step 2: xy0 = x * y0 ---" << std::endl;
        std::cout << "xy0_raw (before ars) = " << fixed_to_float(output_data[3]) << std::endl;
        double xy0_val = fixed_to_float(output_data[4]);
        std::cout << "xy0 (after ars) = " << xy0_val << " (expected: ~1.0)" << std::endl;

        std::cout << "\n--- Step 3: diff = 2 - xy0 ---" << std::endl;
        double neg_xy0 = fixed_to_float(output_data[5]);
        std::cout << "-xy0 = " << neg_xy0 << " (expected: " << -xy0_val << ")" << std::endl;
        double diff_val = fixed_to_float(output_data[6]);
        std::cout << "diff = 2 + (-xy0) = " << diff_val << " (expected: " << (2.0 - xy0_val) << ")" << std::endl;

        std::cout << "\n--- Step 4: y1 = y0 * diff ---" << std::endl;
        std::cout << "y0 * diff (before ars) = " << fixed_to_float(output_data[7]) << std::endl;
        double y1_val = fixed_to_float(output_data[8]);
        std::cout << "y1 (after ars) = " << y1_val << std::endl;
        std::cout << "Expected y1 = y0 * (2 - xy0) = " << (y0_val * (2.0 - xy0_val)) << std::endl;

        std::cout << "\n--- Verification ---" << std::endl;
        std::cout << "x * y0 = " << fixed_to_float(output_data[9]) << " (expected: ~1.0)" << std::endl;
        std::cout << "x * y1 = " << fixed_to_float(output_data[10]) << " (expected: ~1.0)" << std::endl;

        std::cout << "\n--- Raw Values (hex) for debugging ---" << std::endl;
        std::cout << "x raw = 0x" << std::hex << output_data[0] << std::dec
                  << " (" << output_data[0] << ")" << std::endl;
        std::cout << "y0 raw = 0x" << std::hex << output_data[2] << std::dec
                  << " (" << output_data[2] << ")" << std::endl;
        std::cout << "xy0 raw = 0x" << std::hex << output_data[4] << std::dec
                  << " (" << output_data[4] << ")" << std::endl;
        std::cout << "-xy0 raw = 0x" << std::hex << output_data[5] << std::dec
                  << " (" << (int64_t)output_data[5] << ")" << std::endl;
        std::cout << "diff raw = 0x" << std::hex << output_data[6] << std::dec
                  << " (" << (int64_t)output_data[6] << ")" << std::endl;
        std::cout << "y1 raw = 0x" << std::hex << output_data[8] << std::dec
                  << " (" << (int64_t)output_data[8] << ")" << std::endl;

        std::cout << "\n--- Analysis ---" << std::endl;
        if (y1_val < 0) {
            std::cout << "!!! y1 is NEGATIVE - something went wrong !!!" << std::endl;
            std::cout << "Checking intermediate values:" << std::endl;
            std::cout << "  y0 = " << y0_val << " (positive? " << (y0_val > 0 ? "YES" : "NO") << ")" << std::endl;
            std::cout << "  diff = " << diff_val << " (positive? " << (diff_val > 0 ? "YES" : "NO") << ")" << std::endl;
            std::cout << "  y0 * diff should be positive!" << std::endl;
        } else {
            std::cout << "y1 is positive - Newton iteration worked!" << std::endl;
            std::cout << "y1 error = " << (y1_val - expected) << std::endl;
            std::cout << "Improvement: " << std::abs(y0_val - expected) / std::abs(y1_val - expected) << "x" << std::endl;
        }

        std::cout << "\n" << std::string(70, '=') << std::endl;
    }

    finalize::call();
    return 0;
}