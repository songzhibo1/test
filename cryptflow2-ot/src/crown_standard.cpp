#include "crown_standard.h"

#include "crown_ops.h"
#include "crown_standard_kernel.h"
#include "globals.h"
#include "library_fixed_uniform.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sci::crown {

namespace {

using DVec = std::vector<double>;
using DMat = std::vector<DVec>;

static bool runtime_uses_he_field(int bitlength) {
  if (bitlength == 64) {
    return false;
  }
  const uint64_t ring_mod = (1ULL << bitlength);
  return prime_mod != ring_mod;
}

static void ensure_runtime_protocols_ready(bool use_he) {
  if (relu == nullptr) {
    throw std::runtime_error(
        "CrownStandardBounds2PC: relu protocol is not initialized");
  }
  if (aux == nullptr) {
    throw std::runtime_error(
        "CrownStandardBounds2PC: aux protocol is not initialized");
  }
  if (math == nullptr) {
    throw std::runtime_error(
        "CrownStandardBounds2PC: math protocol is not initialized");
  }
  if (truncation == nullptr) {
    throw std::runtime_error(
        "CrownStandardBounds2PC: truncation protocol is not initialized");
  }
  if (xt == nullptr) {
    throw std::runtime_error(
        "CrownStandardBounds2PC: XT protocol is not initialized");
  }
  (void)use_he;
}

static DenseLayerShares make_margin_layer_targeted(const DenseLayerShares &last,
                                                   int pred, int target) {
  if (pred < 0 || pred >= last.out_dim || target < 0 || target >= last.out_dim ||
      pred == target) {
    throw std::invalid_argument("make_margin_layer_targeted: invalid labels");
  }
  DenseLayerShares out;
  out.in_dim = last.in_dim;
  out.out_dim = 1;
  out.W.resize((size_t)out.in_dim);
  out.b.resize(1);

  for (int k = 0; k < last.in_dim; ++k) {
    intType wp = last.W[(size_t)pred * last.in_dim + (size_t)k];
    intType wt = last.W[(size_t)target * last.in_dim + (size_t)k];
    out.W[(size_t)k] = SecretSub(wp, wt);
  }
  out.b[0] = SecretSub(last.b[(size_t)pred], last.b[(size_t)target]);
  return out;
}

static DenseLayerShares make_margin_layer_untargeted(const DenseLayerShares &last,
                                                     int pred) {
  if (pred < 0 || pred >= last.out_dim) {
    throw std::invalid_argument("make_margin_layer_untargeted: invalid pred label");
  }
  DenseLayerShares out;
  out.in_dim = last.in_dim;
  out.out_dim = last.out_dim - 1;
  out.W.resize((size_t)out.out_dim * out.in_dim);
  out.b.resize((size_t)out.out_dim);

  int row = 0;
  for (int j = 0; j < last.out_dim; ++j) {
    if (j == pred) {
      continue;
    }
    for (int k = 0; k < last.in_dim; ++k) {
      intType wp = last.W[(size_t)pred * last.in_dim + (size_t)k];
      intType wj = last.W[(size_t)j * last.in_dim + (size_t)k];
      out.W[(size_t)row * out.in_dim + (size_t)k] = SecretSub(wp, wj);
    }
    out.b[(size_t)row] = SecretSub(last.b[(size_t)pred], last.b[(size_t)j]);
    row++;
  }
  return out;
}

static std::vector<ShareVec> prepend_input_bound(const ShareVec &input,
                                                 const std::vector<ShareVec> &rest) {
  std::vector<ShareVec> out;
  out.reserve(rest.size() + 1);
  out.push_back(input);
  out.insert(out.end(), rest.begin(), rest.end());
  return out;
}

} // namespace

CrownResultShares CrownStandardBounds2PC(
    const std::vector<DenseLayerShares> &net, const ShareVec &x0, intType eps,
    int pred_label, int target_label, bool untargeted, const CrownConfig &cfg,
    bool model_is_A) {
  if (net.size() < 2) {
    throw std::invalid_argument("CrownStandardBounds2PC: need at least 2 layers");
  }
  if (cfg.bitlength <= 0 || cfg.bitlength > 63) {
    throw std::invalid_argument(
        "CrownStandardBounds2PC: bitlength must be in [1, 63]");
  }
  if (cfg.scale_bits < 0) {
    throw std::invalid_argument(
        "CrownStandardBounds2PC: scale_bits must be non-negative");
  }
  // math->div uses hadamard_product with output bw = bitlength + scale_bits.
  // SCI LinearOT backend only supports bw <= 64, so fail early here.
  if (cfg.bitlength + cfg.scale_bits > 64) {
    throw std::invalid_argument(
        "CrownStandardBounds2PC: unsupported (bitlength + scale_bits > 64). "
        "Got bitlength=" +
        std::to_string(cfg.bitlength) +
        " scale_bits=" + std::to_string(cfg.scale_bits));
  }
  if (!runtime_uses_he_field(cfg.bitlength)) {
    // In SCI MathFunctions::div (compute_msnzb=true), OT path uses:
    // bw_tmp1 = bw_out + (bw_dn - 1 - s_dn) = 2*bitlength - 1 - scale_bits.
    // LinearOT requires all intermediate bitwidths <= 64.
    const int min_scale_for_ot_div = 2 * cfg.bitlength - 65;
    if (min_scale_for_ot_div > 0 && cfg.scale_bits < min_scale_for_ot_div) {
      throw std::invalid_argument(
          "CrownStandardBounds2PC: OT backend unsupported for this (bitlength, "
          "scale_bits) in div path. Require scale_bits >= " +
          std::to_string(min_scale_for_ot_div) + " for bitlength=" +
          std::to_string(cfg.bitlength) + ", but got scale_bits=" +
          std::to_string(cfg.scale_bits) +
          ". Example: bitlength=41 needs scale_bits>=17 (e.g. 18).");
    }
  }
  if (runtime_uses_he_field(cfg.bitlength) &&
      (cfg.bitlength < 32 || cfg.bitlength > 41)) {
    throw std::invalid_argument(
        "CrownStandardBounds2PC: SCI-HE requires bitlength in [32, 41]. "
        "Got bitlength=" +
        std::to_string(cfg.bitlength) +
        ". Use SCI-OT for larger bitlength.");
  }

  bitlength = cfg.bitlength;
  num_threads = cfg.num_threads;
  const bool use_he = runtime_uses_he_field(cfg.bitlength);
  ensure_runtime_protocols_ready(use_he);

  std::vector<ShareVec> ubs;
  std::vector<ShareVec> lbs;
  ShareVec ub0(x0.size(), 0), lb0(x0.size(), 0);
  for (size_t i = 0; i < x0.size(); ++i) {
    ub0[i] = SecretAdd(x0[i], eps);
    lb0[i] = SecretSub(x0[i], eps);
  }
  ubs.push_back(ub0);
  lbs.push_back(lb0);

  std::vector<ShareVec> pre_ub;
  std::vector<ShareVec> pre_lb;
  std::vector<ShareVec> state_pos;
  std::vector<ShareVec> state_neg;

  for (size_t num = 0; num < net.size() - 1; ++num) {
    if (num == 0) {
      ShareVec u, l;
      FirstLayerBoundsStandard(net[0], x0, eps, u, l, cfg.scale_bits, model_is_A);
      pre_ub.push_back(u);
      pre_lb.push_back(l);
      ShareVec mpos, mneg, munstable;
      ReLUStateMasks(u, l, mpos, mneg, munstable);
      state_pos.push_back(mpos);
      state_neg.push_back(mneg);
    } else {
      std::vector<DenseLayerShares> prefix(net.begin(), net.begin() + (num + 1));
      const std::vector<ShareVec> us2 = prepend_input_bound(ubs[0], pre_ub);
      const std::vector<ShareVec> ls2 = prepend_input_bound(lbs[0], pre_lb);

      ShareVec u, l;
      StandardRelaxBoundKernel(prefix, us2, ls2, state_pos, state_neg, x0, eps,
                               u, l, cfg.scale_bits, model_is_A);
      pre_ub.push_back(u);
      pre_lb.push_back(l);
      ShareVec mpos, mneg, munstable;
      ReLUStateMasks(u, l, mpos, mneg, munstable);
      state_pos.push_back(mpos);
      state_neg.push_back(mneg);
    }
  }

  std::vector<DenseLayerShares> final_layers(net.begin(), net.end() - 1);
  if (untargeted) {
    final_layers.push_back(make_margin_layer_untargeted(net.back(), pred_label));
  } else {
    final_layers.push_back(
        make_margin_layer_targeted(net.back(), pred_label, target_label));
  }

  const std::vector<ShareVec> usf = prepend_input_bound(ubs[0], pre_ub);
  const std::vector<ShareVec> lsf = prepend_input_bound(lbs[0], pre_lb);

  ShareVec uf, lf;
  StandardRelaxBoundKernel(final_layers, usf, lsf, state_pos, state_neg, x0,
                           eps, uf, lf, cfg.scale_bits, model_is_A);

  CrownResultShares out;
  out.lb = lf;
  out.ub = uf;
  if (untargeted) {
    int idx = 0;
    for (size_t i = 1; i < lf.size(); ++i) {
      // Secret min with mask: approximate with reveal-free heuristic based on sign only.
      // We keep v1 simple and deterministic by selecting first negative if any.
      ShareVec cmp;
      SecretLTZeroMask({SecretSub(lf[i], lf[idx])}, cmp);
      ShareVec chosen;
      SecretSelect(cmp, {lf[i]}, {lf[idx]}, chosen);
      lf[idx] = chosen[0];
    }
    out.gap = lf[idx];
  } else {
    out.gap = lf.empty() ? 0 : lf[0];
  }
  return out;
}

CrownResultPublic RevealCrownResult(const CrownResultShares &shares,
                                    const CrownConfig &cfg) {
  CrownResultPublic out;
  RevealToParty(shares.lb, out.lb, cfg.scale_bits, cfg.reveal_party);
  RevealToParty(shares.ub, out.ub, cfg.scale_bits, cfg.reveal_party);
  RevealToParty(shares.gap, out.gap, cfg.scale_bits, cfg.reveal_party);

  if (party == cfg.reveal_party) {
    out.certified = (out.gap > 0.0);
  } else {
    out.certified = false;
  }
  return out;
}

} // namespace sci::crown
