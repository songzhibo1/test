#include "crown_mlp_file.h"
#include "crown_plaintext.h"
#include "crown_ops.h"
#include "crown_standard.h"
#include "defines.h"
#include "globals.h"
#include "library_fixed_uniform.h"
#include "OT/ot-phase-metrics.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int party = SERVER;
std::string address = "127.0.0.1";
int port = 32000;
int num_threads = 1;
int32_t bitlength = 41;
int scale_bits = 12;

namespace {

using PlainLayer = sci::crown::plain::DenseLayerPublic;

std::string g_mlp_path;
std::string g_x0_bin_path;
double g_eps_plain = 0.05;
int g_target_label = -1;

intType encode_scalar(double v, int sf) {
  const int64_t q = static_cast<int64_t>(std::llround(std::ldexp(v, sf)));
  return funcSSCons(q);
}

intType encode_scalar_client(double v, int sf) {
  const int64_t q = static_cast<int64_t>(std::llround(std::ldexp(v, sf)));
  return (party == CLIENT) ? q : 0;
}

sci::crown::ShareVec encode_vec_client(const std::vector<double> &v, int sf) {
  sci::crown::ShareVec out(v.size(), 0);
  for (size_t i = 0; i < v.size(); ++i) {
    const int64_t q = static_cast<int64_t>(std::llround(std::ldexp(v[i], sf)));
    out[i] = (party == CLIENT) ? q : 0;
  }
  return out;
}

sci::crown::ShareVec encode_vec_public(const std::vector<double> &v, int sf) {
  sci::crown::ShareVec out(v.size(), 0);
  for (size_t i = 0; i < v.size(); ++i) {
    out[i] = encode_scalar(v[i], sf);
  }
  return out;
}

sci::crown::ShareMat encode_mat_public(const std::vector<std::vector<double>> &M,
                                       int sf) {
  if (M.empty()) {
    return {};
  }
  sci::crown::ShareMat out(M.size() * M[0].size(), 0);
  for (size_t i = 0; i < M.size(); ++i) {
    for (size_t j = 0; j < M[0].size(); ++j) {
      out[i * M[0].size() + j] = encode_scalar(M[i][j], sf);
    }
  }
  return out;
}

void read_x0_f64_bin(const std::string &path, size_t n, std::vector<double> *out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Failed to open x0 bin: " + path);
  }
  out->resize(n);
  for (size_t i = 0; i < n; ++i) {
    double v = 0.0;
    f.read(reinterpret_cast<char *>(&v), sizeof(v));
    if (!f) {
      throw std::runtime_error("Short read in x0 bin (expected " + std::to_string(n) +
                               " float64 values)");
    }
    (*out)[i] = v;
  }
  char extra = 0;
  if (f.read(&extra, 1)) {
    throw std::runtime_error("x0 bin has trailing bytes after " + std::to_string(n) +
                             " float64 values");
  }
}

std::vector<PlainLayer> sample_mini_network_public() {
  return {
      {4,
       8,
       {{0.5, -0.3, 0.8, 0.1},
        {-0.4, 0.7, -0.2, 0.6},
        {0.2, 0.9, -0.5, 0.3},
        {-0.6, -0.1, 0.4, -0.8},
        {0.7, 0.2, 0.1, -0.4},
        {-0.3, 0.5, 0.6, -0.2},
        {0.1, -0.7, 0.3, 0.9},
        {-0.5, 0.4, -0.1, 0.2}},
       {0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2}},
      {8,
       6,
       {{0.3, -0.5, 0.1, 0.7, -0.2, 0.4, -0.6, 0.8},
        {-0.1, 0.6, -0.3, 0.2, 0.5, -0.7, 0.4, -0.3},
        {0.8, -0.2, 0.5, -0.4, 0.3, 0.1, -0.5, 0.6},
        {-0.4, 0.3, 0.7, -0.1, -0.6, 0.2, 0.8, -0.5},
        {0.2, 0.5, -0.4, 0.6, -0.3, 0.8, -0.1, 0.4},
        {-0.7, 0.1, 0.3, -0.5, 0.4, -0.2, 0.6, -0.8}},
       {0.2, -0.1, 0.3, -0.2, 0.1, -0.3}},
      {6,
       3,
       {{0.6, -0.4, 0.2, 0.5, -0.3, 0.7},
        {-0.2, 0.8, -0.5, 0.3, 0.6, -0.4},
        {0.4, -0.1, 0.7, -0.6, 0.2, 0.5}},
       {0.1, -0.1, 0.0}},
  };
}

std::vector<sci::crown::DenseLayerShares>
to_secure_network_public_shared(const std::vector<PlainLayer> &net, int sf) {
  std::vector<sci::crown::DenseLayerShares> out;
  out.reserve(net.size());
  for (const auto &layer : net) {
    sci::crown::DenseLayerShares s;
    s.in_dim = layer.in_dim;
    s.out_dim = layer.out_dim;
    s.W = encode_mat_public(layer.W, sf);
    s.b = encode_vec_public(layer.b, sf);
    out.push_back(std::move(s));
  }
  return out;
}

double total_comm_bytes() {
  uint64_t total = 0;
  for (int i = 0; i < num_threads; ++i) {
    total += iopackArr[i]->get_comm();
  }
  return static_cast<double>(total);
}

bool env_flag_enabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int env_int_value(const char *name, int default_value) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }
  try {
    return std::stoi(value);
  } catch (const std::exception &) {
    return default_value;
  }
}

void configure_split_ot_precompute(int batch_size) {
  sci::ot_phase_metrics_enabled = batch_size > 0;
  sci::ResetOTPhaseMetrics();
  if (batch_size <= 0) {
    return;
  }
  for (int i = 0; i < num_threads; ++i) {
    sci::OTPack *pack = otpackArr[i];
    if (pack == nullptr) {
      continue;
    }
    pack->iknp_straight->set_precomp_batch_size(batch_size);
    pack->iknp_reversed->set_precomp_batch_size(batch_size);
    for (int k = 0; k < KKOT_TYPES; ++k) {
      pack->kkot[k]->set_precomp_batch_size(batch_size);
    }
  }
}

void parse_args(int argc, char **argv) {
  ArgMapping amap;
  amap.arg("r", party, "Role: SERVER=1, CLIENT=2");
  amap.arg("ip", address, "Server IP");
  amap.arg("p", port, "Base TCP port");
  amap.arg("nt", num_threads, "Number of SCI threads");
  amap.arg("bl", bitlength, "Ring/field bitlength");
  amap.arg("sf", scale_bits, "Fixed-point fractional bits");
  amap.arg("eps", g_eps_plain, "Perturbation radius in plaintext units");
  amap.arg("target", g_target_label,
           "Target label for targeted verification; default picks the first other label");
  amap.arg("mlp", g_mlp_path,
           "Packed MLP from export_crown_mlp_bin (.crownmlp); empty = built-in sample");
  amap.arg("x0bin", g_x0_bin_path,
           "float64 little-endian x0 (in_dim values); only with -mlp; default zeros");
  amap.parse(argc, argv);
}

void validate_args() {
  if (party != SERVER && party != CLIENT) {
    throw std::invalid_argument("Invalid -r value. Use 1 (SERVER) or 2 (CLIENT).");
  }
  if (num_threads <= 0) {
    throw std::invalid_argument("Invalid -nt value. Must be > 0.");
  }
  if (scale_bits < 0) {
    throw std::invalid_argument("Invalid -sf value. Must be >= 0.");
  }
  if (g_eps_plain < 0.0) {
    throw std::invalid_argument("Invalid -eps value. Must be >= 0.");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    parse_args(argc, argv);
    validate_args();

    const std::vector<PlainLayer> plain_net =
        g_mlp_path.empty() ? sample_mini_network_public()
                           : sci::crown::io::load_crown_mlp_file(g_mlp_path);

    std::vector<double> x0_plain;
    if (g_mlp_path.empty()) {
      x0_plain = {0.5, -0.3, 0.8, 0.2};
    } else {
      x0_plain.assign(static_cast<size_t>(plain_net[0].in_dim), 0.0);
      if (!g_x0_bin_path.empty()) {
        read_x0_f64_bin(g_x0_bin_path, x0_plain.size(), &x0_plain);
      }
    }
    const double eps_plain = g_eps_plain;

    const int pred = sci::crown::plain::PredictLabel(plain_net, x0_plain);
    const int output_dim = plain_net.empty() ? 0 : plain_net.back().out_dim;
    const int target =
        (g_target_label >= 0) ? g_target_label : (pred == 0 ? 1 : 0);
    if (target < 0 || target >= output_dim || target == pred) {
      throw std::invalid_argument(
          "Invalid target label. Must be a different valid output class.");
    }
    const bool untargeted = false;

    sci::crown::CrownConfig cfg;
    cfg.bitlength = bitlength;
    cfg.scale_bits = scale_bits;
    cfg.num_threads = num_threads;
    cfg.reveal_party = CLIENT;

    if (!g_mlp_path.empty() && party == cfg.reveal_party) {
      std::cout << "[mlp] layers=" << plain_net.size() << " in_dim=" << plain_net[0].in_dim
                << " pred=" << pred << "\n";
    }

    const sci::crown::plain::CrownResultPlain plain_res =
        sci::crown::plain::CrownStandardBounds(plain_net, x0_plain, eps_plain, pred,
                                               target, untargeted, false);

    std::vector<sci::crown::DenseLayerShares> secure_net =
        to_secure_network_public_shared(plain_net, cfg.scale_bits);
    sci::crown::ShareVec x0_share = encode_vec_client(x0_plain, cfg.scale_bits);
    const intType eps_share = encode_scalar_client(eps_plain, cfg.scale_bits);

    const int split_ot_precomp_batch =
        env_int_value("CROWN_OT_PRECOMP_BATCH", 0);

    const auto offline_t0 = std::chrono::high_resolution_clock::now();
    StartComputation();
    configure_split_ot_precompute(split_ot_precomp_batch);
    const auto offline_t1 = std::chrono::high_resolution_clock::now();
    const double offline_comm_bytes = total_comm_bytes();

    const double online_comm_start = total_comm_bytes();
    const auto online_t0 = std::chrono::high_resolution_clock::now();
    sci::crown::CrownResultShares secure_shares = sci::crown::CrownStandardBounds2PC(
        secure_net, x0_share, eps_share, pred, target, untargeted, cfg, true);

    sci::crown::CrownResultPublic secure_pub =
        sci::crown::RevealCrownResult(secure_shares, cfg);
    const auto online_t1 = std::chrono::high_resolution_clock::now();
    const double online_comm_end = total_comm_bytes();
    EndComputation();

    const double offline_time_ms =
        std::chrono::duration<double, std::milli>(offline_t1 - offline_t0).count();
    const double online_time_ms =
        std::chrono::duration<double, std::milli>(online_t1 - online_t0).count();
    const double total_time_ms = offline_time_ms + online_time_ms;
    const double online_comm_bytes = online_comm_end - online_comm_start;
    const double total_comm_bytes = offline_comm_bytes + online_comm_bytes;
    const double split_ot_preprocess_time_ms =
        static_cast<double>(sci::ot_phase_preprocess_time_us.load()) / 1000.0;
    const double split_ot_preprocess_comm_bytes =
        static_cast<double>(sci::ot_phase_preprocess_comm_bytes.load());
    const double paper_offline_time_ms =
        offline_time_ms + split_ot_preprocess_time_ms;
    const double paper_online_time_ms =
        std::max(0.0, online_time_ms - split_ot_preprocess_time_ms);
    const double paper_offline_comm_bytes =
        offline_comm_bytes + split_ot_preprocess_comm_bytes;
    const double paper_online_comm_bytes =
        std::max(0.0, online_comm_bytes - split_ot_preprocess_comm_bytes);
    const bool paper_accounting_complete =
        split_ot_precomp_batch > 0 && sci::ot_phase_fallback_calls.load() == 0;
    const char *paper_accounting_model =
        split_ot_precomp_batch > 0 ? "paper_split_ot_experimental"
                                   : "phase_perf_no_split_preprocessing";

    if (env_flag_enabled("CROWN_PHASE_PERF") || split_ot_precomp_batch > 0) {
      std::cout << "[phase_perf] party=" << party
                << " offline_time_ms=" << offline_time_ms
                << " online_time_ms=" << online_time_ms
                << " total_time_ms=" << total_time_ms
                << " offline_comm_bytes=" << offline_comm_bytes
                << " online_comm_bytes=" << online_comm_bytes
                << " total_comm_bytes=" << total_comm_bytes
                << " paper_offline_time_ms=" << paper_offline_time_ms
                << " paper_online_time_ms=" << paper_online_time_ms
                << " paper_total_time_ms=" << total_time_ms
                << " paper_offline_comm_bytes=" << paper_offline_comm_bytes
                << " paper_online_comm_bytes=" << paper_online_comm_bytes
                << " paper_total_comm_bytes=" << total_comm_bytes
                << " split_ot_precomp_batch=" << split_ot_precomp_batch
                << " split_ot_preprocess_calls="
                << sci::ot_phase_preprocess_calls.load()
                << " split_ot_preprocess_ots="
                << sci::ot_phase_preprocess_ots.load()
                << " split_ot_fallback_calls="
                << sci::ot_phase_fallback_calls.load()
                << " split_ot_fallback_ots="
                << sci::ot_phase_fallback_ots.load()
                << " paper_accounting_complete="
                << (paper_accounting_complete ? 1 : 0)
                << " paper_accounting_model=" << paper_accounting_model
                << "\n";
    }

    if (party == cfg.reveal_party) {
      std::cout << "[mini] plain gap=" << plain_res.gap << "\n";
      std::cout << "[mini] sci gap=" << secure_pub.gap << "\n";
      std::cout << "[mini] abs err gap=" << std::abs(plain_res.gap - secure_pub.gap)
                << "\n";
      std::cout << "[perf] time_ms=" << total_time_ms
                << " comm_bytes=" << total_comm_bytes
                << " offline_time_ms=" << offline_time_ms
                << " online_time_ms=" << online_time_ms
                << " total_time_ms=" << total_time_ms
                << " offline_comm_bytes=" << offline_comm_bytes
                << " online_comm_bytes=" << online_comm_bytes
                << " total_comm_bytes=" << total_comm_bytes
                << " accounting_model=per_sample_setup"
                << " perf_schema_version=1"
                << "\n";
      if (split_ot_precomp_batch > 0) {
        std::cout << "[perf] paper_offline_time_ms=" << paper_offline_time_ms
                  << " paper_online_time_ms=" << paper_online_time_ms
                  << " paper_offline_comm_bytes=" << paper_offline_comm_bytes
                  << " paper_online_comm_bytes=" << paper_online_comm_bytes
                  << " paper_total_time_ms=" << total_time_ms
                  << " paper_total_comm_bytes=" << total_comm_bytes
                  << " split_ot_precomp_batch=" << split_ot_precomp_batch
                  << " split_ot_preprocess_calls="
                  << sci::ot_phase_preprocess_calls.load()
                  << " split_ot_preprocess_ots="
                  << sci::ot_phase_preprocess_ots.load()
                  << " split_ot_fallback_calls="
                  << sci::ot_phase_fallback_calls.load()
                  << " split_ot_fallback_ots="
                  << sci::ot_phase_fallback_ots.load()
                  << " paper_accounting_complete="
                  << (sci::ot_phase_fallback_calls.load() == 0 ? 1 : 0)
                  << " paper_accounting_model=paper_split_ot_experimental"
                  << "\n";
      }

    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "CROWN compare run failed: " << e.what() << std::endl;
    return 1;
  }
}
