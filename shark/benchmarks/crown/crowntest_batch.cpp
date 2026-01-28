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
#include <map>
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
        if (line.empty() || line[0] == '#') continue;  // 跳过空行和注释

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
    size_t file_size;
    size_t bytes_read;
public:
    Loader(const std::string &fname) : filename(fname), bytes_read(0) {
        file.open(fname, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "[ERROR] Failed to open file: " << fname << std::endl;
            std::cerr.flush();
            std::exit(1);
        }
        file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::cout << "[DEBUG] Opened file: " << fname << " (size: " << file_size << " bytes)" << std::endl;
        std::cout.flush();
    }

    void load(shark::span<u64> &X, int precision) {
        size_t size = X.size();
        size_t bytes_needed = size * sizeof(float);
        if (bytes_read + bytes_needed > file_size) {
            std::cerr << "[ERROR] File too small! Need " << (bytes_read + bytes_needed)
                      << " bytes, but file has only " << file_size << " bytes" << std::endl;
            std::cerr.flush();
            std::exit(1);
        }
        for (size_t i = 0; i < size; i++) {
            float fval;
            file.read((char *)&fval, sizeof(float));
            if (!file.good()) {
                std::cerr << "[ERROR] Read failed at position " << bytes_read << std::endl;
                std::cerr.flush();
                std::exit(1);
            }
            X[i] = (u64)(int64_t)(fval * (1ULL << precision));
        }
        bytes_read += bytes_needed;
    }

    size_t get_file_size() const { return file_size; }
    size_t get_bytes_read() const { return bytes_read; }

    ~Loader() { if (file.is_open()) file.close(); }
};

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

// ==================== 改进的 Reciprocal ====================

shark::span<u64> newton_refine_reciprocal(shark::span<u64>& a, shark::span<u64>& x_n, shark::span<u64>& two_share) {
    size_t size = a.size();
    auto ax = mul::call(a, x_n);
    ax = ars::call(ax, f);
    shark::span<u64> two_vec(size);
    for (size_t i = 0; i < size; ++i) two_vec[i] = two_share[0];
    auto diff = secure_sub(two_vec, ax);
    auto x_next = mul::call(x_n, diff);
    return ars::call(x_next, f);
}

shark::span<u64> improved_reciprocal(shark::span<u64>& a, shark::span<u64>& two_share, int iterations = 1) {
    auto x = reciprocal::call(a, f);
    for (int i = 0; i < iterations; ++i) {
        x = newton_refine_reciprocal(a, x, two_share);
    }
    return x;
}

// ==================== Alpha 计算 ====================

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
    auto num = relu::call(U);
    shark::span<u64> L_neg(size);
    for(size_t i = 0; i < size; ++i) L_neg[i] = -L[i];
    auto term2 = relu::call(L_neg);
    auto den = add::call(num, term2);
    auto eps_vec = broadcast_scalar(epsilon_share, size);
    den = add::call(den, eps_vec);
    auto den_inv = improved_reciprocal(den, two_share, 3);
    auto alpha = mul::call(num, den_inv);
    alpha = ars::call(alpha, f);
    return secure_clamp_upper_1(alpha, one_share);
}

// ==================== 优化的工具函数 ====================

shark::span<u64> scale_matrix_by_alpha(shark::span<u64>& W, shark::span<u64>& alpha, int rows, int cols) {
    shark::span<u64> alpha_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            alpha_expanded[r * cols + c] = alpha[c];
        }
    }
    auto result = mul::call(W, alpha_expanded);
    return ars::call(result, f);
}

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

shark::span<u64> sum_abs(shark::span<u64>& A, int size) {
    auto A_abs = secure_abs(A);
    shark::span<u64> result(1);
    result[0] = 0;
    for(int i = 0; i < size; ++i) {
        result[0] += A_abs[i];
    }
    return result;
}

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
        result[0] += correction_vec[i];
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
        result[0] += correction_vec[i];
    }
    return result;
}

shark::span<u64> compute_lb_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB, int rows, int cols) {
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);
    shark::span<u64> A_neg(rows * cols);
    for(int i = 0; i < rows * cols; ++i) A_neg[i] = -A[i];
    auto relu_neg_A = relu::call(A_neg);
    shark::span<u64> relu_neg_LB_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            relu_neg_LB_expanded[r * cols + c] = relu_neg_LB[c];
        }
    }
    auto correction_matrix = mul::call(relu_neg_A, relu_neg_LB_expanded);
    correction_matrix = ars::call(correction_matrix, f);
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

shark::span<u64> compute_ub_correction_matrix(shark::span<u64>& A, shark::span<u64>& LB, int rows, int cols) {
    shark::span<u64> LB_neg(cols);
    for(int i = 0; i < cols; ++i) LB_neg[i] = -LB[i];
    auto relu_neg_LB = relu::call(LB_neg);
    auto relu_A = relu::call(A);
    shark::span<u64> relu_neg_LB_expanded(rows * cols);
    for(int r = 0; r < rows; ++r) {
        for(int c = 0; c < cols; ++c) {
            relu_neg_LB_expanded[r * cols + c] = relu_neg_LB[c];
        }
    }
    auto correction_matrix = mul::call(relu_A, relu_neg_LB_expanded);
    correction_matrix = ars::call(correction_matrix, f);
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

    void reset_bounds() {
        layer_bounds.clear();
    }

    LayerBounds compute_first_layer_bounds() {
        LayerBounds bounds;
        LayerInfo& layer = layers[0];
        int out_dim = layer.output_dim;
        int in_dim = layer.input_dim;

        auto Ax0 = matmul::call(out_dim, in_dim, 1, layer.W, x0);
        Ax0 = ars::call(Ax0, f);

        auto W_abs = secure_abs(layer.W);
        auto dualnorm = compute_row_sum_manual(W_abs, out_dim, in_dim);

        auto eps_vec = broadcast_scalar(eps_share, out_dim);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);
        radius = relu::call(radius);

        auto temp1 = add::call(Ax0, radius);
        bounds.UB = add::call(temp1, layer.b);

        auto temp2 = secure_sub(Ax0, radius);
        bounds.LB = add::call(temp2, layer.b);

        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share, one_share, two_share);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    LayerBounds compute_middle_layer_bounds(int layer_idx) {
        LayerBounds bounds;
        LayerInfo& curr_layer = layers[layer_idx];
        int out_dim = curr_layer.output_dim;

        auto A_prop = curr_layer.W;

        shark::span<u64> constants(out_dim);
        for(int i = 0; i < out_dim; ++i) constants[i] = curr_layer.b[i];

        shark::span<u64> lb_corr_total(out_dim);
        shark::span<u64> ub_corr_total(out_dim);
        for(int i = 0; i < out_dim; ++i) {
            lb_corr_total[i] = 0;
            ub_corr_total[i] = 0;
        }

        for(int i = layer_idx - 1; i >= 0; --i) {
            A_prop = scale_matrix_by_alpha(A_prop, layer_bounds[i].alpha, out_dim, layers[i].output_dim);

            auto lb_c = compute_lb_correction_matrix(A_prop, layer_bounds[i].LB, out_dim, layers[i].output_dim);
            auto ub_c = compute_ub_correction_matrix(A_prop, layer_bounds[i].LB, out_dim, layers[i].output_dim);

            lb_corr_total = add::call(lb_corr_total, lb_c);
            ub_corr_total = add::call(ub_corr_total, ub_c);

            auto Ab = matmul::call(out_dim, layers[i].output_dim, 1, A_prop, layers[i].b);
            Ab = ars::call(Ab, f);
            constants = add::call(constants, Ab);

            A_prop = matmul::call(out_dim, layers[i].output_dim, layers[i].input_dim, A_prop, layers[i].W);
            A_prop = ars::call(A_prop, f);
        }

        auto Ax0 = matmul::call(out_dim, input_dim, 1, A_prop, x0);
        Ax0 = ars::call(Ax0, f);

        auto A_abs = secure_abs(A_prop);
        auto dualnorm = compute_row_sum_manual(A_abs, out_dim, input_dim);

        auto eps_vec = broadcast_scalar(eps_share, out_dim);
        auto radius = mul::call(dualnorm, eps_vec);
        radius = ars::call(radius, f);
        radius = relu::call(radius);

        auto base = add::call(Ax0, constants);

        bounds.UB = add::call(base, ub_corr_total);
        bounds.UB = add::call(bounds.UB, radius);

        bounds.LB = secure_sub(base, lb_corr_total);
        bounds.LB = secure_sub(bounds.LB, radius);

        bounds.alpha = compute_alpha_secure(bounds.UB, bounds.LB, epsilon_share, one_share, two_share);

        layer_bounds.push_back(bounds);
        return bounds;
    }

    std::pair<shark::span<u64>, shark::span<u64>> compute_worst_bound(
        shark::span<u64>& diff_vec, int true_label, int target_label) {

        reset_bounds();
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
        int in_dim = last_layer.input_dim;

        auto W_diff = matmul::call(1, last_layer.output_dim, in_dim, diff_vec, last_layer.W);
        W_diff = ars::call(W_diff, f);

        auto b_diff = dot_product(last_layer.b, diff_vec, last_layer.output_dim);

        auto constants = b_diff;
        shark::span<u64> lb_corr_total(1); lb_corr_total[0] = 0;
        shark::span<u64> ub_corr_total(1); ub_corr_total[0] = 0;

        auto A_prop = W_diff;

        for(int i = last_idx - 1; i >= 0; --i) {
            auto A_scaled = mul::call(A_prop, layer_bounds[i].alpha);
            A_prop = ars::call(A_scaled, f);

            auto lb_c = compute_lb_correction_vec(A_prop, layer_bounds[i].LB, layers[i].output_dim);
            auto ub_c = compute_ub_correction_vec(A_prop, layer_bounds[i].LB, layers[i].output_dim);

            lb_corr_total[0] += lb_c[0];
            ub_corr_total[0] += ub_c[0];

            auto Ab = dot_product(A_prop, layers[i].b, layers[i].output_dim);
            constants[0] += Ab[0];

            auto A_next = matmul::call(1, layers[i].output_dim, layers[i].input_dim, A_prop, layers[i].W);
            A_next = ars::call(A_next, f);

            shark::span<u64> A_next_vec(layers[i].input_dim);
            for(int k=0; k<layers[i].input_dim; ++k) A_next_vec[k] = A_next[k];
            A_prop = A_next_vec;
        }

        auto Ax0 = dot_product(A_prop, x0, input_dim);
        auto dualnorm = sum_abs(A_prop, input_dim);
        auto radius = mul::call(dualnorm, eps_share);
        radius = ars::call(radius, f);
        radius = relu::call(radius);

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
        else if (arg.find("--batch_config=") == 0) {
            batch_config_file = arg.substr(15);
        }
    }

    // 加载批量配置
    std::vector<ImageConfig> batch_configs = load_batch_config(batch_config_file);
    if (batch_configs.empty()) {
        std::cerr << "No valid batch configs found. Use --batch_config=<file>" << std::endl;
        std::cerr << "Config format: image_id,true_label,target_label (one per line)" << std::endl;
        finalize::call();
        return 1;
    }

    int num_images = batch_configs.size();

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
        std::cout << "  BATCH MODE: " << num_images << " images" << std::endl;
        std::cout << "  MODEL: " << model_name << std::endl;
        std::cout << "  Layers: " << num_layers << ", Hidden: " << hidden_dim << std::endl;
        std::cout << "  Layer dims: [";
        for (size_t i = 0; i < layer_dims.size(); ++i) {
            std::cout << layer_dims[i];
            if (i < layer_dims.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  Weights file: " << weights_file << std::endl;
        std::cout << "  EPS: " << eps << std::endl;
        std::cout << "********************************************" << std::endl;
        std::cout.flush();
    }

    // ==================== 权重加载 (只做一次) ====================
    if (party != DEALER) {
        std::cout << "[DEBUG] Starting weight loading phase..." << std::endl;
        std::cout.flush();
    }

    shark::utils::start_timer("total_time");
    shark::utils::start_timer("weights_input");

    // Calculate expected weights file size
    size_t expected_floats = 0;
    for (int i = 0; i < num_layers; ++i) {
        expected_floats += layer_dims[i + 1] * layer_dims[i];  // weights
        expected_floats += layer_dims[i + 1];  // biases
    }
    if (party != DEALER) {
        std::cout << "[DEBUG] Expected weights file size: " << (expected_floats * sizeof(float)) << " bytes (" << expected_floats << " floats)" << std::endl;
        std::cout.flush();
    }

    if (party != DEALER) {
        std::cout << "[DEBUG] Allocating weight arrays..." << std::endl;
        std::cout.flush();
    }

    std::vector<shark::span<u64>> weights(num_layers);
    std::vector<shark::span<u64>> biases(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        int w_size = layer_dims[i + 1] * layer_dims[i];
        int b_size = layer_dims[i + 1];
        if (party != DEALER) {
            std::cout << "[DEBUG]   Layer " << i << ": W[" << layer_dims[i] << "x" << layer_dims[i+1] << "]=" << w_size << ", b=" << b_size << std::endl;
            std::cout.flush();
        }
        weights[i] = shark::span<u64>(w_size);
        biases[i] = shark::span<u64>(b_size);
    }

    if (party != DEALER) {
        std::cout << "[DEBUG] Weight arrays allocated successfully" << std::endl;
        std::cout.flush();
    }

    if (party == SERVER) {
        std::cout << "[DEBUG] SERVER: Loading weights from file..." << std::endl;
        std::cout.flush();
        Loader weights_loader(weights_file);
        for (int i = 0; i < num_layers; ++i) {
            std::cout << "[DEBUG] SERVER: Loading layer " << i << " weights (size=" << weights[i].size() << ")..." << std::endl;
            std::cout.flush();
            weights_loader.load(weights[i], f);
            std::cout << "[DEBUG] SERVER: Loading layer " << i << " biases (size=" << biases[i].size() << ")..." << std::endl;
            std::cout.flush();
            weights_loader.load(biases[i], f);
        }
        std::cout << "[DEBUG] SERVER: Weights loaded. Total bytes read: " << weights_loader.get_bytes_read() << std::endl;
        std::cout.flush();
    }

    if (party != DEALER) {
        std::cout << "[DEBUG] Starting input::call for weights (party=" << party << ")..." << std::endl;
        std::cout.flush();
    }

    for (int i = 0; i < num_layers; ++i) {
        if (party != DEALER) {
            std::cout << "[DEBUG] input::call weights[" << i << "] (size=" << weights[i].size() << ")..." << std::endl;
            std::cout.flush();
        }
        input::call(weights[i], SERVER);
        if (party != DEALER) {
            std::cout << "[DEBUG] input::call biases[" << i << "] (size=" << biases[i].size() << ")..." << std::endl;
            std::cout.flush();
        }
        input::call(biases[i], SERVER);
    }

    if (party != DEALER) {
        std::cout << "[DEBUG] All weight input::call completed" << std::endl;
        std::cout.flush();
    }

    // 预分配共享常量
    if (party != DEALER) {
        std::cout << "[DEBUG] Allocating shared constants..." << std::endl;
        std::cout.flush();
    }

    shark::span<u64> epsilon_share(1), one_share(1), two_share(1);
    shark::span<u64> eps_share(1), ones_input(input_dim);

    if (party == CLIENT) {
        std::cout << "[DEBUG] CLIENT: Initializing constants..." << std::endl;
        std::cout.flush();
        epsilon_share[0] = float_to_fixed(0.000001);
        one_share[0] = SCALAR_ONE;
        two_share[0] = float_to_fixed(2.0);
        eps_share[0] = float_to_fixed(eps);
        for (int i = 0; i < input_dim; ++i) ones_input[i] = SCALAR_ONE;
        std::cout << "[DEBUG] CLIENT: Constants initialized" << std::endl;
        std::cout.flush();
    }

    if (party != DEALER) {
        std::cout << "[DEBUG] input::call for shared constants..." << std::endl;
        std::cout.flush();
    }

    input::call(epsilon_share, CLIENT);
    input::call(one_share, CLIENT);
    input::call(two_share, CLIENT);
    input::call(eps_share, CLIENT);
    input::call(ones_input, CLIENT);

    if (party != DEALER) {
        std::cout << "[DEBUG] Shared constants input complete" << std::endl;
        std::cout.flush();
    }

    shark::utils::stop_timer("weights_input");

    if (party != DEALER) {
        std::cout << "[DEBUG] Syncing with peer..." << std::endl;
        std::cout.flush();
        peer->sync();
        std::cout << "[DEBUG] Sync complete" << std::endl;
        std::cout.flush();
    }

    // ==================== 批量处理图片 ====================
    shark::utils::start_timer("batch_computation");

    // 存储结果
    std::vector<double> all_LB(num_images);
    std::vector<double> all_UB(num_images);

    for (int img_idx = 0; img_idx < num_images; ++img_idx) {
        ImageConfig& cfg = batch_configs[img_idx];

        std::string timer_name = "image_" + std::to_string(img_idx);
        shark::utils::start_timer(timer_name);

        // 加载当前图片
        std::string input_file = base_path + "/images/" + std::to_string(cfg.image_id) + ".bin";

        shark::span<u64> x0(input_dim);
        shark::span<u64> diff_vec(output_dim);

        if (party == CLIENT) {
            Loader input_loader(input_file);
            input_loader.load(x0, f);
            for (int i = 0; i < output_dim; ++i) diff_vec[i] = 0;
            diff_vec[cfg.true_label] = float_to_fixed(1.0);
            diff_vec[cfg.target_label] = float_to_fixed(-1.0);
        }

        input::call(x0, CLIENT);
        input::call(diff_vec, CLIENT);

        // 创建 CROWN 计算器
        CROWNComputer crown(input_dim);
        crown.set_input(x0, eps_share, epsilon_share, one_share, two_share, ones_input);
        for (int i = 0; i < num_layers; ++i) {
            crown.add_layer(weights[i], biases[i], layer_dims[i], layer_dims[i+1]);
        }

        // 计算边界
        auto [final_LB, final_UB] = crown.compute_worst_bound(diff_vec, cfg.true_label, cfg.target_label);

        // 揭示结果
        output::call(final_LB);
        output::call(final_UB);

        all_LB[img_idx] = fixed_to_float(final_LB[0]);
        all_UB[img_idx] = fixed_to_float(final_UB[0]);

        shark::utils::stop_timer(timer_name);

        // 打印进度
        if (party != DEALER) {
            std::cout << "[" << img_idx + 1 << "/" << num_images << "] "
                      << "id=" << cfg.image_id
                      << ", true=" << cfg.true_label
                      << ", target=" << cfg.target_label
                      << " => LB=" << std::fixed << std::setprecision(4) << all_LB[img_idx]
                      << ", UB=" << all_UB[img_idx]
                      << (all_LB[img_idx] > 0 ? " [VERIFIED]" : " [NOT VERIFIED]")
                      << std::endl;
        }
    }

    shark::utils::stop_timer("batch_computation");
    shark::utils::stop_timer("total_time");

    // ==================== 输出汇总 ====================
    if (party != DEALER) {
        std::cout << "\n============================================" << std::endl;
        std::cout << "BATCH RESULTS SUMMARY" << std::endl;
        std::cout << "============================================" << std::endl;

        int verified_count = 0;
        for (int i = 0; i < num_images; ++i) {
            if (all_LB[i] > 0) verified_count++;
        }

        std::cout << "Total images: " << num_images << std::endl;
        std::cout << "Verified: " << verified_count << " ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * verified_count / num_images) << "%)" << std::endl;
        std::cout << "============================================" << std::endl;

        shark::utils::print_all_timers();
    }

    finalize::call();
    return 0;
}
