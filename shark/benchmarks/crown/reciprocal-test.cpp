#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/reciprocal.hpp>
#include <shark/protocols/mul.hpp>
#include <shark/protocols/ars.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

using u64 = shark::u64;
using namespace shark::protocols;

// 测试不同精度
const int f = 26;  // 与你的 CROWN 代码一致
const u64 SCALAR_ONE = 1ULL << f;

u64 float_to_fixed(double val) {
    return (u64)(int64_t)(val * SCALAR_ONE);
}

double fixed_to_float(u64 val) {
    return (double)(int64_t)val / (double)SCALAR_ONE;
}

int main(int argc, char **argv) {
    init::from_args(argc, argv);

    // 测试值：来自 CROWN Layer 0 alpha 计算
    // den = ReLU(UB) + ReLU(-LB) + epsilon
    // 对于神经元 0: UB=154.909244, LB=-148.838090
    // den = 154.909244 + 148.838090 + 1e-6 = 303.747335

    std::vector<double> test_values = {
        303.747335,    // Layer 0, neuron 0 的分母
        331.877625,    // Layer 0, neuron 1 的分母 (171.819 + 160.058)
        366.593570,    // Layer 0, neuron 2 的分母
        1.0,           // 基准测试
        10.0,
        100.0,
        500.0,
        1000.0,
        0.5,
        0.1,
        0.01
    };

    std::vector<shark::span<u64>> inputs(test_values.size());
    std::vector<shark::span<u64>> outputs(test_values.size());
    std::vector<shark::span<u64>> verify(test_values.size());  // x * (1/x) 应该 = 1

    for (size_t i = 0; i < test_values.size(); ++i) {
        inputs[i] = shark::span<u64>(1);
        outputs[i] = shark::span<u64>(1);
        verify[i] = shark::span<u64>(1);
    }

    // CLIENT 初始化输入
    if (party == CLIENT) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "Reciprocal Precision Test (f=" << f << ")" << std::endl;
        std::cout << "============================================" << std::endl;

        for (size_t i = 0; i < test_values.size(); ++i) {
            inputs[i][0] = float_to_fixed(test_values[i]);
            std::cout << "Input " << i << ": " << test_values[i]
                      << " -> fixed: " << inputs[i][0] << std::endl;
        }
    }

    // 秘密共享输入
    for (size_t i = 0; i < test_values.size(); ++i) {
        input::call(inputs[i], CLIENT);
    }

    if (party != DEALER) peer->sync();

    // 计算倒数
    for (size_t i = 0; i < test_values.size(); ++i) {
        outputs[i] = reciprocal::call(inputs[i], f);

        // 验证: x * (1/x) 应该 = 1
        auto product = mul::call(inputs[i], outputs[i]);
        verify[i] = ars::call(product, f);
    }

    // 输出结果
    for (size_t i = 0; i < test_values.size(); ++i) {
        output::call(outputs[i]);
        output::call(verify[i]);
    }

    if (party != DEALER) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "Results" << std::endl;
        std::cout << "============================================" << std::endl;
        std::cout << std::setw(15) << "Input"
                  << std::setw(20) << "Expected 1/x"
                  << std::setw(20) << "MPC 1/x"
                  << std::setw(15) << "Error"
                  << std::setw(15) << "x*(1/x)"
                  << std::setw(15) << "Error from 1"
                  << std::endl;
        std::cout << std::string(100, '-') << std::endl;

        for (size_t i = 0; i < test_values.size(); ++i) {
            double x = test_values[i];
            double expected = 1.0 / x;
            double mpc_result = fixed_to_float(outputs[i][0]);
            double error = std::abs(mpc_result - expected);
            double rel_error = error / expected * 100.0;

            double verify_result = fixed_to_float(verify[i][0]);
            double verify_error = std::abs(verify_result - 1.0);

            std::cout << std::fixed << std::setprecision(6)
                      << std::setw(15) << x
                      << std::setw(20) << expected
                      << std::setw(20) << mpc_result
                      << std::setw(15) << error
                      << std::setw(15) << verify_result
                      << std::setw(15) << verify_error
                      << std::endl;
        }

        std::cout << "\n============================================" << std::endl;
        std::cout << "Alpha Calculation Test" << std::endl;
        std::cout << "============================================" << std::endl;

        // 专门测试 alpha 计算
        // alpha = num / den = ReLU(UB) / (ReLU(UB) + ReLU(-LB) + eps)
        // 对于神经元 0: num = 154.909244, den = 303.747335
        double num = 154.909244;
        double den = 303.747335;
        double expected_alpha = num / den;

        // MPC 方式计算
        double mpc_inv = fixed_to_float(outputs[0][0]);  // 1/303.747335
        double mpc_alpha = num * mpc_inv;

        std::cout << "Neuron 0:" << std::endl;
        std::cout << "  num = " << num << std::endl;
        std::cout << "  den = " << den << std::endl;
        std::cout << "  Expected alpha = " << expected_alpha << std::endl;
        std::cout << "  MPC 1/den = " << mpc_inv << std::endl;
        std::cout << "  MPC alpha (num * 1/den) = " << mpc_alpha << std::endl;
        std::cout << "  Python alpha = 0.509994" << std::endl;
        std::cout << "  Your MPC alpha = 0.512929" << std::endl;
        std::cout << "  Error = " << std::abs(mpc_alpha - expected_alpha) << std::endl;
    }

    finalize::call();

    return 0;
}