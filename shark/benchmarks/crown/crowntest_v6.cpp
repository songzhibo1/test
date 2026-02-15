/**
 * crowntest_debug_fixed.cpp
 * * 修改说明:
 * 1. 引入了 debug_records 系统和 record_value 函数。
 * 2. 将所有辅助函数(secure_sub, secure_abs等)的参数改为按值传递 (shark::span<u64>)，
 * 以修复 "cannot bind non-const lvalue reference" 编译错误。
 * 3. 在 CROWN 计算流程中插入了 record_value 调用。
 * 4. 在 main 函数末尾添加了 debug 输出循环。
 * 5. 保持了原有的 f=26, i128 累加, matmul_ars/mul_ars 融合协议逻辑不变。
 */

#include <shark/protocols/init.hpp>
#include <shark/protocols/finalize.hpp>
#include <shark/protocols/input.hpp>
#include <shark/protocols/output.hpp>
#include <shark/protocols/relu.hpp>
#include <shark/protocols/matmul.hpp>
#include <shark/protocols/matmul_ars.hpp>
#include <shark/protocols/mul_ars.hpp>
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
#include <algorithm>
#include <map>

using u64 = shark::u64;
using i64 = int64_t;
using i128 = __int128;
using namespace shark::protocols;

// ==================== 精度配置 (保持不变) ====================
const int f = 26;
const u64 SCALAR_ONE = 1ULL << f;

// ==================== Debug System (新增) ====================

struct DebugRecord {
    std::string name;
    std::vector<u64> data; // 使用 vector 持久化存储数据
    int layer_idx;

    DebugRecord(const std::string& n, shark::span<u64> d, int layer = -1)
        : name(n), layer_idx(layer) {
        data.resize(d.size());
        for (size_t i = 0; i < d.size(); ++i) {
            data[i] = d[i];
        }
    }
};

std::vector<DebugRecord> debug_records;
bool enable_debug = false;

// 记录函数
void record_value(const std::string& name, shark::span<u64> data, int layer_idx = -1) {
    if (enable_debug) {
        debug_records.emplace_back(name, data, layer_idx);
    }
}

// ==================== Utils ====================

class Loader {
    std::ifstream file;
public:
    Loader(const std::string &fname) {
        file.open(fname, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Failed to open: " << fname << std::endl;
            std::exit(1);
        }
    }
    void load(shark::span<u64> &X, int precision) {
        for (size_t i = 0; i < X.size(); i++) {
            float fval;
            file.read((char *)&fval, sizeof(float));
            X[i] = (u64)(i64)(fval * (1ULL << precision));
        }
    }
    ~Loader() { if (file.is_open()) file.close(); }
};

double fixed_to_float(u64 val) {
    return (double)(i64)val / (double)SCALAR_ONE;
}

u64 float_to_fixed(double val) {
    return (u64)(i64)(val * SCALAR_ONE);
}

// ==================== 基础安全操作 (参数改为值传递以修复编译错误) ====================

shark::span<u64> secure_sub(shark::span<u64> A, shark::span<u64> B) {
    shark::span<u64> B_neg(B.size());
    for(size_t i = 0; i < B.size(); ++i) B_neg[i] = -B[i];
    return add::call(A, B_neg);
}

shark::span<u64> secure_abs(shark::span<u64> W) {
    shark::span<u64> W_neg(W.size());
    for(size_t i = 0; i < W.size(); ++i) W_neg[i] = -W[i];
    auto pos = relu::call(W);
    auto neg = relu::call(W_neg);
    return add::call(pos, neg);
}

shark::span<u64> broadcast_scalar(shark::span<u64> scalar, int size) {
    shark::span<u64> vec(size);
    u64 val = scalar[0];
    for(int i = 0; i < size; ++i) vec[i] = val;
    return vec;
}

// ==================== 高精度 Reciprocal (参数改为值传递) ====================

shark::span<u64> newton_refine(shark::span<u64> a, shark::span<u64> x_n,
                               shark::span<u64> two_share) {
    size_t size = a.size();
    auto ax = mul_ars::call(f, a, x_n);
    shark::span<u64> two_vec = broadcast_scalar(two_share, size); // 这里修改为调用 broadcast
    auto diff = secure_sub(two_vec, ax);
    return mul_ars::call(f, x_n, diff);
}

shark::span<u64> high_precision_reciprocal(shark::span<u64> a,
                                            shark::span<u64> two_share) {
    auto x = reciprocal::call(a, f);
    for (int i = 0; i < 4; ++i) {
        x = newton_refine(a, x, two_share);
    }
    return x;
}

// ==================== Alpha 计算 (参数改为值传递) ====================

shark::span<u64> secure_clamp_01(shark::span<u64> x, shark::span<u64> one_share) {
    size_t size = x.size();
    // First clamp to [0, infinity): max(x, 0)
    auto clamped_low = relu::call(x);
    // Then clamp to [0, 1]: min(max(x, 0), 1) = 1 - relu(1 - max(x, 0))
    auto ones = broadcast_scalar(one_share, size);
    auto one_minus_x = secure_sub(ones, clamped_low);
    auto relu_part = relu::call(one_minus_x);
    return secure_sub(ones, relu_part);
}

shark::span<u64> compute_alpha(shark::span<u64> UB, shark::span<u64> LB,
                                shark::span<u64> epsilon_share,
                                shark::span<u64> one_share,
                                shark::span<u64> two_share) {
    size_t size = UB.size();

    auto num = relu::call(UB);

    shark::span<u64> LB_neg(size);
    for(size_t i = 0; i < size; ++i) LB_neg[i] = -LB[i];
    auto term2 = relu::call(LB_neg);

    auto den = add::call(num, term2);
    auto eps_vec = broadcast_scalar(epsilon_share, size);
    den = add::call(den, eps_vec);

    auto den_inv = high_precision_reciprocal(den, two_share);

    auto alpha = mul_ars::call(f, num, den_inv);

    return secure_clamp_01(alpha, one_share);
}

// ==================== i128 高精度累加操作 (参数改为值传递) ====================

shark::span<u64> row_sum_abs_i128(shark::span<u64> W_abs, int rows, int cols) {
    shark::span<u64> result(rows);
    for (int r = 0; r < rows; ++r) {
        i128 sum = 0;
        for (int c = 0; c < cols; ++c) {
            sum += (i64)W_abs[r * cols + c];
        }
        result[r] = (u64)(i64)sum;
    }
    return result;
}

shark::span<u64> scale_by_alpha(shark::span<u64> W, shark::span<u64> alpha,
                                 int rows, int cols) {
    shark::span<u64> alpha_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            alpha_expanded[r * cols + c] = alpha[c];
        }
    }
    return mul_ars::call(f, W, alpha_expanded);
}

shark::span<u64> dot_product_i128(shark::span<u64> A, shark::span<u64> B, int size) {
    auto prod = mul_ars::call(f, A, B);

    shark::span<u64> result(1);
    i128 sum = 0;
    for(int i = 0; i < size; ++i) {
        sum += (i64)prod[i];
    }
    result[0] = (u64)(i64)sum;
    return result;
}

shark::span<u64> sum_abs_i128(shark::span<u64> A, int size) {
    auto A_abs = secure_abs(A);
    shark::span<u64> result(1);
    i128 sum = 0;
    for(int i = 0; i < size; ++i) {
        sum += (i64)A_abs[i];
    }
    result[0] = (u64)(i64)sum;
    return result;
}

// ==================== Correction 计算 (i128) (参数改为值传递) ====================

shark::span<u64> lb_correction_vec_i128(shark::span<u64> A, shark::span<u64> LB, int size) {
    shark::span<u64> A_neg(size);
    for(int i = 0; i < size; ++i) A_neg[i] = -A[i];
    auto relu_neg_A = relu::call(A_neg);

    shark::span<u64> LB_neg(size);
    for(int i = 0; i < size; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    auto corr = mul_ars::call(f, relu_neg_A, relu_neg_LB);

    shark::span<u64> result(1);
    i128 sum = 0;
    for(int i = 0; i < size; ++i) sum += (i64)corr[i];
    result[0] = (u64)(i64)sum;
    return result;
}

shark::span<u64> ub_correction_vec_i128(shark::span<u64> A, shark::span<u64> LB, int size) {
    auto relu_A = relu::call(A);

    shark::span<u64> LB_neg(size);
    for(int i = 0; i < size; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    auto corr = mul_ars::call(f, relu_A, relu_neg_LB);

    shark::span<u64> result(1);
    i128 sum = 0;
    for(int i = 0; i < size; ++i) sum += (i64)corr[i];
    result[0] = (u64)(i64)sum;
    return result;
}

shark::span<u64> lb_correction_matrix_i128(shark::span<u64> A, shark::span<u64> LB,
                                            int rows, int cols) {
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    shark::span<u64> A_neg(rows * cols);
    for(int i = 0; i < rows * cols; ++i) A_neg[i] = -A[i];
    auto relu_neg_A = relu::call(A_neg);

    shark::span<u64> relu_neg_LB_exp(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            relu_neg_LB_exp[r * cols + c] = relu_neg_LB[c];
        }
    }

    auto corr_mat = mul_ars::call(f, relu_neg_A, relu_neg_LB_exp);

    shark::span<u64> corr(rows);
    for(int r = 0; r < rows; ++r) {
        i128 sum = 0;
        for(int c = 0; c < cols; ++c) {
            sum += (i64)corr_mat[r * cols + c];
        }
        corr[r] = (u64)(i64)sum;
    }
    return corr;
}

shark::span<u64> ub_correction_matrix_i128(shark::span<u64> A, shark::span<u64> LB,
                                            int rows, int cols) {
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);

    auto relu_A = relu::call(A);

    shark::span<u64> relu_neg_LB_exp(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            relu_neg_LB_exp[r * cols + c] = relu_neg_LB[c];
        }
    }

    auto corr_mat = mul_ars::call(f, relu_A, relu_neg_LB_exp);

    shark::span<u64> corr(rows);
    for(int r = 0; r < rows; ++r) {
        i128 sum = 0;
        for(int c = 0; c < cols; ++c) {
            sum += (i64)corr_mat[r * cols + c];
        }
        corr[r] = (u64)(i64)sum;
    }
    return corr;
}

// ==================== CROWN 算法 (已植入 record_value) ====================

struct LayerInfo {
    shark::span<u64> W;
    shark::span<u64> b;
    int in_dim;
    int out_dim;
};

struct LayerBounds {
    shark::span<u64> UB;
    shark::span<u64> LB;
    shark::span<u64> alpha;
};

class CROWNComputer {
public:
    std::vector<LayerInfo> layers;
    std::vector<LayerBounds> bounds;

    shark::span<u64> x0;
    shark::span<u64> eps_share;
    shark::span<u64> epsilon_share;
    shark::span<u64> one_share;
    shark::span<u64> two_share;
    int input_dim;
    int num_layers;

    CROWNComputer(int in_dim) : input_dim(in_dim), num_layers(0) {}

    // 这里的参数可以是引用，因为 set_input 被调用时传入的是变量
    void set_input(shark::span<u64>& x0_, shark::span<u64>& eps_,
                   shark::span<u64>& epsilon_, shark::span<u64>& one_,
                   shark::span<u64>& two_) {
        x0 = x0_; eps_share = eps_; epsilon_share = epsilon_;
        one_share = one_; two_share = two_;
    }

    void add_layer(shark::span<u64>& W, shark::span<u64>& b, int in_dim, int out_dim) {
        LayerInfo layer;
        layer.W = W; layer.b = b;
        layer.in_dim = in_dim; layer.out_dim = out_dim;
        layers.push_back(layer);
        num_layers++;
    }

    LayerBounds compute_layer0_bounds() {
        LayerBounds bnd;
        auto& L = layers[0];

        // 使用 fused matmul_ars 协议
        auto Wx0 = matmul_ars::call(L.out_dim, L.in_dim, 1, f, L.W, x0);
        record_value("Layer0_Wx0", Wx0, 0); // DEBUG

        auto W_abs = secure_abs(L.W);
        auto dual_norm = row_sum_abs_i128(W_abs, L.out_dim, L.in_dim);

        auto eps_vec = broadcast_scalar(eps_share, L.out_dim);
        auto radius = mul_ars::call(f, dual_norm, eps_vec);
        radius = relu::call(radius);
        record_value("Layer0_radius", radius, 0); // DEBUG

        auto tmp = add::call(Wx0, radius);
        bnd.UB = add::call(tmp, L.b);
        record_value("Layer0_UB", bnd.UB, 0); // DEBUG

        tmp = secure_sub(Wx0, radius);
        bnd.LB = add::call(tmp, L.b);
        record_value("Layer0_LB", bnd.LB, 0); // DEBUG

        bnd.alpha = compute_alpha(bnd.UB, bnd.LB, epsilon_share, one_share, two_share);
        record_value("Layer0_alpha", bnd.alpha, 0); // DEBUG

        bounds.push_back(bnd);
        return bnd;
    }

    LayerBounds compute_layer_bounds(int layer_idx) {
        LayerBounds bnd;
        auto& L = layers[layer_idx];

        auto A = L.W;

        shark::span<u64> constants(L.out_dim);
        for(int i = 0; i < L.out_dim; ++i) constants[i] = L.b[i];

        shark::span<u64> lb_corr(L.out_dim);
        shark::span<u64> ub_corr(L.out_dim);
        for(int i = 0; i < L.out_dim; ++i) {
            lb_corr[i] = 0;
            ub_corr[i] = 0;
        }

        for(int i = layer_idx - 1; i >= 0; --i) {
            auto& prev_L = layers[i];
            auto& prev_bnd = bounds[i];

            A = scale_by_alpha(A, prev_bnd.alpha, L.out_dim, prev_L.out_dim);

            auto lb_c = lb_correction_matrix_i128(A, prev_bnd.LB, L.out_dim, prev_L.out_dim);
            auto ub_c = ub_correction_matrix_i128(A, prev_bnd.LB, L.out_dim, prev_L.out_dim);
            lb_corr = add::call(lb_corr, lb_c);
            ub_corr = add::call(ub_corr, ub_c);

            // 使用 fused matmul_ars 协议
            auto Ab = matmul_ars::call(L.out_dim, prev_L.out_dim, 1, f, A, prev_L.b);
            constants = add::call(constants, Ab);

            // 使用 fused matmul_ars 协议
            A = matmul_ars::call(L.out_dim, prev_L.out_dim, prev_L.in_dim, f, A, prev_L.W);
        }

        // 使用 fused matmul_ars 协议
        auto Ax0 = matmul_ars::call(L.out_dim, input_dim, 1, f, A, x0);
        record_value("Layer" + std::to_string(layer_idx) + "_Ax0", Ax0, layer_idx); // DEBUG

        auto A_abs = secure_abs(A);
        auto dual_norm = row_sum_abs_i128(A_abs, L.out_dim, input_dim);

        auto eps_vec = broadcast_scalar(eps_share, L.out_dim);
        auto radius = mul_ars::call(f, dual_norm, eps_vec);
        radius = relu::call(radius);

        auto base = add::call(Ax0, constants);

        bnd.UB = add::call(base, ub_corr);
        bnd.UB = add::call(bnd.UB, radius);
        record_value("Layer" + std::to_string(layer_idx) + "_UB", bnd.UB, layer_idx); // DEBUG

        bnd.LB = secure_sub(base, lb_corr);
        bnd.LB = secure_sub(bnd.LB, radius);
        record_value("Layer" + std::to_string(layer_idx) + "_LB", bnd.LB, layer_idx); // DEBUG

        bnd.alpha = compute_alpha(bnd.UB, bnd.LB, epsilon_share, one_share, two_share);
        record_value("Layer" + std::to_string(layer_idx) + "_alpha", bnd.alpha, layer_idx); // DEBUG

        bounds.push_back(bnd);
        return bnd;
    }

    std::pair<shark::span<u64>, shark::span<u64>>
    compute_final_bounds(shark::span<u64>& diff_vec) {
        int last_idx = num_layers - 1;
        auto& last_L = layers[last_idx];

        // 使用 fused matmul_ars 协议
        auto W_diff = matmul_ars::call(1, last_L.out_dim, last_L.in_dim, f, diff_vec, last_L.W);

        auto b_diff = dot_product_i128(last_L.b, diff_vec, last_L.out_dim);

        // Use i128 for accumulation to prevent overflow
        i128 constants_acc = (i64)b_diff[0];
        i128 lb_corr_acc = 0;
        i128 ub_corr_acc = 0;

        auto A = W_diff;

        for(int i = last_idx - 1; i >= 0; --i) {
            auto& prev_L = layers[i];
            auto& prev_bnd = bounds[i];

            A = mul_ars::call(f, A, prev_bnd.alpha);

            auto lb_c = lb_correction_vec_i128(A, prev_bnd.LB, prev_L.out_dim);
            auto ub_c = ub_correction_vec_i128(A, prev_bnd.LB, prev_L.out_dim);
            lb_corr_acc += (i64)lb_c[0];
            ub_corr_acc += (i64)ub_c[0];

            auto Ab = dot_product_i128(A, prev_L.b, prev_L.out_dim);
            constants_acc += (i64)Ab[0];

            // 使用 fused matmul_ars 协议
            auto A_next = matmul_ars::call(1, prev_L.out_dim, prev_L.in_dim, f, A, prev_L.W);
            shark::span<u64> A_vec(prev_L.in_dim);
            for(int k = 0; k < prev_L.in_dim; ++k) A_vec[k] = A_next[k];
            A = A_vec;
        }

        // Convert i128 accumulators back to u64 spans
        shark::span<u64> constants(1);
        constants[0] = (u64)(i64)constants_acc;
        shark::span<u64> lb_corr(1);
        lb_corr[0] = (u64)(i64)lb_corr_acc;
        shark::span<u64> ub_corr(1);
        ub_corr[0] = (u64)(i64)ub_corr_acc;

        auto Ax0 = dot_product_i128(A, x0, input_dim);
        record_value("Final_Ax0", Ax0); // DEBUG

        auto dual_norm = sum_abs_i128(A, input_dim);
        auto radius = mul_ars::call(f, dual_norm, eps_share);
        radius = relu::call(radius);
        record_value("Final_radius", radius); // DEBUG

        auto base = add::call(Ax0, constants);

        auto final_LB = secure_sub(base, lb_corr);
        final_LB = secure_sub(final_LB, radius);

        auto final_UB = add::call(base, ub_corr);
        final_UB = add::call(final_UB, radius);

        // 使用 make_pair 明确构造，避免初始化列表推导错误
        return std::make_pair(final_LB, final_UB);
    }

    std::pair<shark::span<u64>, shark::span<u64>>
    compute_worst_bound(shark::span<u64>& diff_vec) {
        compute_layer0_bounds();
        for(int i = 1; i < num_layers - 1; ++i) {
            compute_layer_bounds(i);
        }
        return compute_final_bounds(diff_vec);
    }
};

// ==================== Main ====================
int main(int argc, char **argv) {
    init::from_args(argc, argv);

//    std::string model_name = "eran_cifar_5layer_relu_100_best";
//    int input_dim = 3072, output_dim = 10, num_layers = 5, hidden_dim = 100;
//    int true_label = 1, target_label = 4, image_id = 6;
//    float eps = 0.002;
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
        if (arg.find("--model=") == 0) model_name = arg.substr(8);
        else if (arg.find("--num_layers=") == 0) num_layers = std::stoi(arg.substr(13));
        else if (arg.find("--hidden_dim=") == 0) hidden_dim = std::stoi(arg.substr(13));
        else if (arg.find("--input_dim=") == 0) input_dim = std::stoi(arg.substr(12));
        else if (arg.find("--output_dim=") == 0) output_dim = std::stoi(arg.substr(13));
        else if (arg.find("--eps=") == 0) eps = std::stof(arg.substr(6));
        else if (arg.find("--true_label=") == 0) true_label = std::stoi(arg.substr(13));
        else if (arg.find("--target_label=") == 0) target_label = std::stoi(arg.substr(15));
        else if (arg.find("--image_id=") == 0) image_id = std::stoi(arg.substr(11));
        else if (arg.find("--input_file=") == 0) custom_input_file = arg.substr(13);
        else if (arg == "--debug") enable_debug = true;
    }

    std::vector<int> layer_dims;
    layer_dims.push_back(input_dim);
    for (int i = 0; i < num_layers - 1; ++i) layer_dims.push_back(hidden_dim);
    layer_dims.push_back(output_dim);

    std::string base_path = "shark_crown_ml/crown_mpc_data/" + model_name;
    std::string weights_file = base_path + "/weights/weights.dat";
    std::string input_file = custom_input_file.empty() ?
                             base_path + "/images/" + std::to_string(image_id) + ".bin" :
                             custom_input_file;

    if (party != DEALER) {
        std::cout << "\n********************************************" << std::endl;
        std::cout << "  CROWN V6 (High Precision: f=26 + matmul_ars + mul_ars)" << std::endl;
        std::cout << "  DEBUG MODE: " << (enable_debug ? "ON" : "OFF") << std::endl;
        std::cout << "  MODEL: " << model_name << std::endl;
        std::cout << "  Precision: f=" << f << " (~7-8 decimal digits)" << std::endl;
        std::cout << "********************************************" << std::endl;
    }

    // Input phase
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
    shark::span<u64> diff_vec(output_dim), x0(input_dim), eps_share(1);

    if (party == CLIENT) {
        Loader input_loader(input_file);
        input_loader.load(x0, f);
        epsilon_share[0] = float_to_fixed(0.000001);
        one_share[0] = SCALAR_ONE;
        two_share[0] = float_to_fixed(2.0);
        eps_share[0] = float_to_fixed(eps);
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

    shark::utils::stop_timer("input");

    if (party != DEALER) peer->sync();
    shark::utils::start_timer("crown_calculation");

    // CROWN computation
    CROWNComputer crown(input_dim);
    crown.set_input(x0, eps_share, epsilon_share, one_share, two_share);
    for (int i = 0; i < num_layers; ++i) {
        crown.add_layer(weights[i], biases[i], layer_dims[i], layer_dims[i+1]);
    }

    auto [final_LB, final_UB] = crown.compute_worst_bound(diff_vec);

    shark::utils::stop_timer("crown_calculation");
    shark::utils::stop_timer("End_to_end_time");

    // ==================== Output Results ====================
    // NOTE: matmul_ars and mul_ars protocols already reconstruct internally,
    // so all intermediate values and final results are already plaintext on evaluator side.
    // We should NOT call output::call on these values, as it would incorrectly
    // subtract the dealer's random values from the already-reconstructed plaintext.
    //
    // For debug_records: evaluator has plaintext, dealer has random garbage - no output::call needed
    // For final_LB/UB: evaluator has plaintext - no output::call needed

    if (party != DEALER) {
        // 打印 Debug 信息
        if (enable_debug) {
            std::cout << "\n============================================" << std::endl;
            std::cout << "Debug Output" << std::endl;
            std::cout << "============================================" << std::endl;
            for (auto& record : debug_records) {
                std::cout << record.name << " (L" << record.layer_idx << "): [";
                int print_count = std::min((int)record.data.size(), 10);
                for (int i = 0; i < print_count; ++i) {
                    std::cout << std::fixed << std::setprecision(6) << fixed_to_float(record.data[i]);
                    if (i < print_count - 1) std::cout << ", ";
                }
                if ((int)record.data.size() > 10) std::cout << ", ...";
                std::cout << "]" << std::endl;
            }
        }

        std::cout << "\n============================================" << std::endl;
        std::cout << "MODEL: " << model_name << std::endl;
        std::cout << "EPS: " << eps << " | True: " << true_label << " | Target: " << target_label << std::endl;
        std::cout << "Precision: f=" << f << " with matmul_ars + mul_ars fusion" << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        std::cout << "MPC LB: " << std::fixed << std::setprecision(6) << fixed_to_float(final_LB[0]) << std::endl;
        std::cout << "MPC UB: " << std::fixed << std::setprecision(6) << fixed_to_float(final_UB[0]) << std::endl;
        std::cout << "============================================" << std::endl;

        shark::utils::print_all_timers();
    }

    finalize::call();
    return 0;
}
