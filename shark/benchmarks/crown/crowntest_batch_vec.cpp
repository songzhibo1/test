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
#include <sstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <algorithm>

using u64 = shark::u64;
using namespace shark::protocols;

// 精度设置
const int f = 26;
const u64 SCALAR_ONE = 1ULL << f;

// ==================== Batch Config ====================
struct ImageConfig {
    int image_id;
    int true_label;
    int target_label;
};

std::vector<ImageConfig> load_batch_config(const std::string& config_file) {
    std::vector<ImageConfig> configs;
    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open batch config file: " << config_file << std::endl;
        return configs;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        ImageConfig cfg;
        char comma;
        if (iss >> cfg.image_id >> comma >> cfg.true_label >> comma >> cfg.target_label) {
            configs.push_back(cfg);
        }
    }
    return configs;
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
        size_t size = X.size();
        for (size_t i = 0; i < size; i++) {
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

// 向量化的减法: A - B (element-wise)
shark::span<u64> secure_sub(shark::span<u64>& A, shark::span<u64>& B) {
    shark::span<u64> B_neg(B.size());
    for(size_t i = 0; i < B.size(); ++i) B_neg[i] = -B[i];
    return add::call(A, B_neg);
}

// 向量化的绝对值: |W|
shark::span<u64> secure_abs(shark::span<u64>& W) {
    shark::span<u64> W_neg(W.size());
    for(size_t i = 0; i < W.size(); ++i) W_neg[i] = -W[i];
    auto pos = relu::call(W);
    auto neg = relu::call(W_neg);
    return add::call(pos, neg);
}

// ==================== 向量化CROWN计算器 ====================
// 同时处理B张图片，减少协议调用次数

class VectorizedCROWN {
public:
    int input_dim;
    int output_dim;
    int num_layers;
    int batch_size;  // B = 图片数量
    std::vector<int> layer_dims;

    // 权重 (共享，不依赖batch)
    std::vector<shark::span<u64>> weights;  // W[l]: (out_dim[l] * in_dim[l])
    std::vector<shark::span<u64>> biases;   // b[l]: (out_dim[l])

    // 批量输入数据: 每个元素存储所有图片的数据
    // X0: (input_dim * B) - 所有图片的输入，按列存储
    // diff_vecs: (output_dim * B) - 所有图片的diff向量
    shark::span<u64> X0;
    shark::span<u64> diff_vecs;

    // 共享常量
    shark::span<u64> eps_share;
    shark::span<u64> epsilon_share;
    shark::span<u64> one_share;
    shark::span<u64> two_share;

    // 批量边界: (hidden_dim * B) 存储所有图片的边界
    std::vector<shark::span<u64>> layer_UB;
    std::vector<shark::span<u64>> layer_LB;
    std::vector<shark::span<u64>> layer_alpha;

    VectorizedCROWN(int input_dim_, int output_dim_, int num_layers_, int batch_size_,
                    std::vector<int>& layer_dims_)
        : input_dim(input_dim_), output_dim(output_dim_),
          num_layers(num_layers_), batch_size(batch_size_),
          layer_dims(layer_dims_) {}

    void set_weights(std::vector<shark::span<u64>>& W, std::vector<shark::span<u64>>& b) {
        weights = W;
        biases = b;
    }

    void set_constants(shark::span<u64>& eps, shark::span<u64>& epsilon,
                       shark::span<u64>& one, shark::span<u64>& two) {
        eps_share = eps;
        epsilon_share = epsilon;
        one_share = one;
        two_share = two;
    }

    void set_batch_input(shark::span<u64>& x0_batch, shark::span<u64>& diff_batch) {
        X0 = x0_batch;
        diff_vecs = diff_batch;
    }

    // 广播bias到batch: b (out_dim) -> b_batch (out_dim * B)
    shark::span<u64> broadcast_bias(shark::span<u64>& b, int out_dim) {
        shark::span<u64> b_batch(out_dim * batch_size);
        for (int img = 0; img < batch_size; ++img) {
            for (int i = 0; i < out_dim; ++i) {
                b_batch[img * out_dim + i] = b[i];
            }
        }
        return b_batch;
    }

    // 广播标量到batch: s -> (size * B)
    shark::span<u64> broadcast_scalar_batch(shark::span<u64>& scalar, int size) {
        shark::span<u64> result(size * batch_size);
        u64 val = scalar[0];
        for (int i = 0; i < size * batch_size; ++i) {
            result[i] = val;
        }
        return result;
    }

    // 计算行和 (针对批量数据)
    // A_abs: (rows * cols * B), 结果: (rows * B)
    shark::span<u64> compute_row_sum_batch(shark::span<u64>& A_abs, int rows, int cols) {
        shark::span<u64> result(rows * batch_size);
        for (int img = 0; img < batch_size; ++img) {
            for (int row = 0; row < rows; ++row) {
                u64 sum = 0;
                for (int col = 0; col < cols; ++col) {
                    // A_abs 按 (row, col, img) 或 (img, row, col) 存储
                    // 这里假设 W 是共享的，所以 W_abs 只有 (rows * cols)
                    sum += A_abs[row * cols + col];
                }
                result[img * rows + row] = sum;
            }
        }
        return result;
    }

    // 改进的reciprocal with Newton refinement
    shark::span<u64> improved_reciprocal(shark::span<u64>& a, int iterations = 1) {
        auto x = reciprocal::call(a, f);
        for (int iter = 0; iter < iterations; ++iter) {
            // x_next = x * (2 - a*x)
            auto ax = mul::call(a, x);
            ax = ars::call(ax, f);
            shark::span<u64> two_vec(a.size());
            for (size_t i = 0; i < a.size(); ++i) two_vec[i] = two_share[0];
            auto diff = secure_sub(two_vec, ax);
            x = mul::call(x, diff);
            x = ars::call(x, f);
        }
        return x;
    }

    // 计算alpha (批量): UB, LB 都是 (hidden_dim * B)
    shark::span<u64> compute_alpha_batch(shark::span<u64>& UB, shark::span<u64>& LB, int hidden_dim) {
        size_t size = hidden_dim * batch_size;

        // num = relu(UB)
        auto num = relu::call(UB);

        // den = relu(UB) + relu(-LB) + epsilon
        shark::span<u64> LB_neg(size);
        for (size_t i = 0; i < size; ++i) LB_neg[i] = -LB[i];
        auto relu_neg_LB = relu::call(LB_neg);
        auto den = add::call(num, relu_neg_LB);

        // 加 epsilon
        auto eps_vec = broadcast_scalar_batch(epsilon_share, hidden_dim);
        den = add::call(den, eps_vec);

        // alpha = num / den
        auto den_inv = improved_reciprocal(den, 3);
        auto alpha = mul::call(num, den_inv);
        alpha = ars::call(alpha, f);

        // clamp to [0, 1]
        auto ones = broadcast_scalar_batch(one_share, hidden_dim);
        auto one_minus_alpha = secure_sub(ones, alpha);
        auto relu_part = relu::call(one_minus_alpha);
        alpha = secure_sub(ones, relu_part);

        return alpha;
    }

    // 第一层边界计算 (批量)
    void compute_first_layer_bounds_batch() {
        int out_dim = layer_dims[1];
        int in_dim = layer_dims[0];

        // Ax0_batch: W @ X0, where X0 is (in_dim, B)
        // matmul(M, K, N, A, B) computes A(M×K) @ B(K×N) -> C(M×N)
        // W: (out_dim × in_dim), X0: (in_dim × B) -> result: (out_dim × B)
        auto Ax0 = matmul::call(out_dim, in_dim, batch_size, weights[0], X0);
        Ax0 = ars::call(Ax0, f);

        // 广播bias
        auto b_batch = broadcast_bias(biases[0], out_dim);

        // 计算 ||W||_∞ (行和的绝对值) - 权重是共享的，所以只计算一次
        auto W_abs = secure_abs(weights[0]);
        shark::span<u64> dualnorm(out_dim);
        for (int row = 0; row < out_dim; ++row) {
            u64 sum = 0;
            for (int col = 0; col < in_dim; ++col) {
                sum += W_abs[row * in_dim + col];
            }
            dualnorm[row] = sum;
        }

        // 广播dualnorm到batch: (out_dim) -> (out_dim * B)
        shark::span<u64> dualnorm_batch(out_dim * batch_size);
        for (int img = 0; img < batch_size; ++img) {
            for (int i = 0; i < out_dim; ++i) {
                dualnorm_batch[img * out_dim + i] = dualnorm[i];
            }
        }

        // radius = dualnorm * eps
        auto eps_vec = broadcast_scalar_batch(eps_share, out_dim);
        auto radius = mul::call(dualnorm_batch, eps_vec);
        radius = ars::call(radius, f);
        radius = relu::call(radius);  // 确保非负

        // UB = Ax0 + b + radius
        // LB = Ax0 + b - radius
        auto temp = add::call(Ax0, b_batch);
        layer_UB.push_back(add::call(temp, radius));
        layer_LB.push_back(secure_sub(temp, radius));

        // 计算 alpha
        layer_alpha.push_back(compute_alpha_batch(layer_UB[0], layer_LB[0], out_dim));
    }

    // 中间层边界计算 (批量) - 简化版本
    void compute_middle_layer_bounds_batch(int layer_idx) {
        int out_dim = layer_dims[layer_idx + 1];
        int prev_dim = layer_dims[layer_idx];

        // 这里使用简化的bound propagation
        // 完整版本需要追踪A矩阵通过所有层的传播

        // 获取前一层的alpha
        auto& prev_alpha = layer_alpha[layer_idx - 1];

        // A_scaled = W[layer_idx] * diag(alpha)
        // 对于批量处理，alpha是 (prev_dim * B)
        // 我们需要 scale W 的每一列 by 对应的 alpha
        shark::span<u64> W_scaled(out_dim * prev_dim * batch_size);
        for (int img = 0; img < batch_size; ++img) {
            for (int row = 0; row < out_dim; ++row) {
                for (int col = 0; col < prev_dim; ++col) {
                    W_scaled[(img * out_dim + row) * prev_dim + col] =
                        weights[layer_idx][row * prev_dim + col];
                }
            }
        }

        // 使用批量matmul计算 W @ (alpha * prev_output)
        // 简化：直接使用前一层的bounds作为输入
        auto& prev_LB = layer_LB[layer_idx - 1];
        auto& prev_UB = layer_UB[layer_idx - 1];

        // 中点作为输入估计
        auto mid = add::call(prev_LB, prev_UB);
        for (size_t i = 0; i < mid.size(); ++i) mid[i] >>= 1;

        // 计算 W @ mid
        auto Ax = matmul::call(out_dim, prev_dim, batch_size, weights[layer_idx], mid);
        Ax = ars::call(Ax, f);

        // 加 bias
        auto b_batch = broadcast_bias(biases[layer_idx], out_dim);
        auto base = add::call(Ax, b_batch);

        // 计算 radius (使用权重的行范数)
        auto W_abs = secure_abs(weights[layer_idx]);
        shark::span<u64> dualnorm(out_dim);
        for (int row = 0; row < out_dim; ++row) {
            u64 sum = 0;
            for (int col = 0; col < prev_dim; ++col) {
                sum += W_abs[row * prev_dim + col];
            }
            dualnorm[row] = sum;
        }

        // 计算前一层的范围大小
        auto range = secure_sub(prev_UB, prev_LB);
        shark::span<u64> max_range(batch_size);
        for (int img = 0; img < batch_size; ++img) {
            u64 max_r = 0;
            for (int i = 0; i < prev_dim; ++i) {
                if (range[img * prev_dim + i] > max_r) {
                    max_r = range[img * prev_dim + i];
                }
            }
            max_range[img] = max_r;
        }

        // 广播到 (out_dim * B)
        shark::span<u64> radius(out_dim * batch_size);
        for (int img = 0; img < batch_size; ++img) {
            for (int i = 0; i < out_dim; ++i) {
                // radius = dualnorm * max_range / 2 (简化估计)
                radius[img * out_dim + i] = (dualnorm[i] * max_range[img]) >> (f + 1);
            }
        }

        layer_UB.push_back(add::call(base, radius));
        layer_LB.push_back(secure_sub(base, radius));
        layer_alpha.push_back(compute_alpha_batch(layer_UB[layer_idx], layer_LB[layer_idx], out_dim));
    }

    // 计算最终的 diff bounds (批量)
    std::pair<shark::span<u64>, shark::span<u64>> compute_final_bounds_batch() {
        // 清除之前的bounds
        layer_UB.clear();
        layer_LB.clear();
        layer_alpha.clear();

        // 计算所有层的bounds
        compute_first_layer_bounds_batch();
        for (int l = 1; l < num_layers - 1; ++l) {
            compute_middle_layer_bounds_batch(l);
        }

        // 最后一层: 计算 diff = output[true] - output[target]
        int last_layer = num_layers - 1;
        int prev_dim = layer_dims[last_layer];
        int out_dim = layer_dims[last_layer + 1];  // = output_dim

        // 使用前一层的中点
        auto& prev_LB = layer_LB[last_layer - 1];
        auto& prev_UB = layer_UB[last_layer - 1];
        auto mid = add::call(prev_LB, prev_UB);
        for (size_t i = 0; i < mid.size(); ++i) mid[i] >>= 1;

        // output = W @ mid + b
        auto output = matmul::call(out_dim, prev_dim, batch_size, weights[last_layer], mid);
        output = ars::call(output, f);
        auto b_batch = broadcast_bias(biases[last_layer], out_dim);
        output = add::call(output, b_batch);

        // 计算 diff = output * diff_vec (点积)
        // diff_vecs: (output_dim * B), output: (output_dim * B)
        auto diff_prod = mul::call(output, diff_vecs);
        diff_prod = ars::call(diff_prod, f);

        // 对每张图片求和得到最终的 diff 值
        shark::span<u64> final_diff(batch_size);
        for (int img = 0; img < batch_size; ++img) {
            u64 sum = 0;
            for (int i = 0; i < out_dim; ++i) {
                sum += diff_prod[img * out_dim + i];
            }
            final_diff[img] = sum;
        }

        // 计算 radius 的影响
        auto W_abs = secure_abs(weights[last_layer]);
        auto range = secure_sub(prev_UB, prev_LB);

        shark::span<u64> final_radius(batch_size);
        for (int img = 0; img < batch_size; ++img) {
            u64 rad = 0;
            for (int i = 0; i < out_dim; ++i) {
                u64 w_row_sum = 0;
                for (int j = 0; j < prev_dim; ++j) {
                    w_row_sum += W_abs[i * prev_dim + j];
                }
                // 这里简化处理
                u64 max_range = 0;
                for (int j = 0; j < prev_dim; ++j) {
                    if (range[img * prev_dim + j] > max_range) {
                        max_range = range[img * prev_dim + j];
                    }
                }
                rad += (w_row_sum * max_range) >> f;
            }
            final_radius[img] = rad >> 1;
        }

        // LB = diff - radius, UB = diff + radius
        shark::span<u64> final_LB(batch_size);
        shark::span<u64> final_UB(batch_size);
        for (int img = 0; img < batch_size; ++img) {
            final_LB[img] = final_diff[img] - final_radius[img];
            final_UB[img] = final_diff[img] + final_radius[img];
        }

        return std::make_pair(final_LB, final_UB);
    }
};

// ==================== Main ====================
int main(int argc, char **argv) {
    init::from_args(argc, argv);

    // 默认配置
    std::string model_name = "eran_cifar_5layer_relu_100_best";
    int input_dim = 3072;
    int output_dim = 10;
    int num_layers = 5;
    int hidden_dim = 100;
    float eps = 0.002;
    std::string batch_config_file = "";

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.find("--model=") == 0) model_name = arg.substr(8);
        else if (arg.find("--num_layers=") == 0) num_layers = std::stoi(arg.substr(13));
        else if (arg.find("--hidden_dim=") == 0) hidden_dim = std::stoi(arg.substr(13));
        else if (arg.find("--input_dim=") == 0) input_dim = std::stoi(arg.substr(12));
        else if (arg.find("--output_dim=") == 0) output_dim = std::stoi(arg.substr(13));
        else if (arg.find("--eps=") == 0) eps = std::stof(arg.substr(6));
        else if (arg.find("--batch_config=") == 0) batch_config_file = arg.substr(15);
    }

    // 加载批量配置
    std::vector<ImageConfig> batch_configs = load_batch_config(batch_config_file);
    if (batch_configs.empty()) {
        std::cerr << "No valid batch configs found. Use --batch_config=<file>" << std::endl;
        finalize::call();
        return 1;
    }

    int batch_size = batch_configs.size();

    // 构建 layer_dims
    std::vector<int> layer_dims;
    layer_dims.push_back(input_dim);
    for (int i = 0; i < num_layers - 1; ++i) {
        layer_dims.push_back(hidden_dim);
    }
    layer_dims.push_back(output_dim);

    std::string base_path = "shark_crown_ml/crown_mpc_data/" + model_name;
    std::string weights_file = base_path + "/weights/weights.dat";

    if (party != DEALER) {
        std::cout << "\n********************************************" << std::endl;
        std::cout << "  VECTORIZED BATCH MODE: " << batch_size << " images" << std::endl;
        std::cout << "  MODEL: " << model_name << std::endl;
        std::cout << "  Layers: " << num_layers << ", Hidden: " << hidden_dim << std::endl;
        std::cout << "  EPS: " << eps << std::endl;
        std::cout << "********************************************" << std::endl;
    }

    // ==================== 加载权重 ====================
    shark::utils::start_timer("total_time");
    shark::utils::start_timer("weights_input");

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

    // ==================== 加载所有图片 (批量) ====================
    // X0_batch: (input_dim * batch_size) - 所有图片打包在一起
    // diff_batch: (output_dim * batch_size) - 所有diff向量打包
    shark::span<u64> X0_batch(input_dim * batch_size);
    shark::span<u64> diff_batch(output_dim * batch_size);

    if (party == CLIENT) {
        for (int img = 0; img < batch_size; ++img) {
            std::string input_file = base_path + "/images/" + std::to_string(batch_configs[img].image_id) + ".bin";
            Loader input_loader(input_file);

            // 加载到对应位置
            shark::span<u64> x0_temp(input_dim);
            input_loader.load(x0_temp, f);
            for (int i = 0; i < input_dim; ++i) {
                X0_batch[img * input_dim + i] = x0_temp[i];
            }

            // 创建 diff_vec
            for (int i = 0; i < output_dim; ++i) {
                diff_batch[img * output_dim + i] = 0;
            }
            diff_batch[img * output_dim + batch_configs[img].true_label] = float_to_fixed(1.0);
            diff_batch[img * output_dim + batch_configs[img].target_label] = float_to_fixed(-1.0);
        }
    }

    // 一次性共享所有图片数据
    input::call(X0_batch, CLIENT);
    input::call(diff_batch, CLIENT);

    // 共享常量
    shark::span<u64> epsilon_share(1), one_share(1), two_share(1), eps_share(1);
    if (party == CLIENT) {
        epsilon_share[0] = float_to_fixed(0.000001);
        one_share[0] = SCALAR_ONE;
        two_share[0] = float_to_fixed(2.0);
        eps_share[0] = float_to_fixed(eps);
    }
    input::call(epsilon_share, CLIENT);
    input::call(one_share, CLIENT);
    input::call(two_share, CLIENT);
    input::call(eps_share, CLIENT);

    shark::utils::stop_timer("weights_input");

    if (party != DEALER) peer->sync();

    // ==================== 向量化CROWN计算 ====================
    shark::utils::start_timer("crown_computation");

    VectorizedCROWN crown(input_dim, output_dim, num_layers, batch_size, layer_dims);
    crown.set_weights(weights, biases);
    crown.set_constants(eps_share, epsilon_share, one_share, two_share);
    crown.set_batch_input(X0_batch, diff_batch);

    auto [final_LB, final_UB] = crown.compute_final_bounds_batch();

    // 输出结果
    output::call(final_LB);
    output::call(final_UB);

    shark::utils::stop_timer("crown_computation");
    shark::utils::stop_timer("total_time");

    // ==================== 输出汇总 ====================
    if (party != DEALER) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "VECTORIZED BATCH RESULTS" << std::endl;
        std::cout << "============================================" << std::endl;

        int verified_count = 0;
        for (int i = 0; i < batch_size; ++i) {
            double lb = fixed_to_float(final_LB[i]);
            double ub = fixed_to_float(final_UB[i]);
            bool verified = lb > 0;
            if (verified) verified_count++;

            std::cout << "[" << i + 1 << "/" << batch_size << "] "
                      << "id=" << batch_configs[i].image_id
                      << ", true=" << batch_configs[i].true_label
                      << ", target=" << batch_configs[i].target_label
                      << " => LB=" << std::fixed << std::setprecision(4) << lb
                      << ", UB=" << ub
                      << (verified ? " [VERIFIED]" : " [NOT VERIFIED]")
                      << std::endl;
        }

        std::cout << "\nTotal images: " << batch_size << std::endl;
        std::cout << "Verified: " << verified_count << " ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * verified_count / batch_size) << "%)" << std::endl;
        std::cout << "============================================" << std::endl;

        shark::utils::print_all_timers();
    }

    finalize::call();
    return 0;
}
