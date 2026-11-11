#include "crown_standard_kernel.h"

#include "crown_ops.h"
#include "globals.h"
#include "library_fixed_uniform.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace sci::crown {

namespace {

bool crown_div_compute_msnzb_enabled() {
  static int cached = -1;
  if (cached == -1) {
    const char *e = std::getenv("CROWN_DIV_COMPUTE_MSNZB");
    cached = (e == nullptr || e[0] == '\0' || e[0] != '0') ? 1 : 0;
  }
  return cached != 0;
}

bool crown_div_per_element_enabled() {
  static int cached = -1;
  if (cached == -1) {
    const char *e = std::getenv("CROWN_DIV_PER_ELEMENT");
    if (e != nullptr && e[0] != '\0') {
      cached = (e[0] != '0') ? 1 : 0;
    } else {
#ifdef SCI_OT
      cached = 0;
#else
      // Keep the conservative HE/non-OT behavior unless explicitly overridden.
      cached = 1;
#endif
    }
  }
  return cached != 0;
}

} // namespace

static ShareVec make_zero_vec(size_t n) { return ShareVec(n, 0); }

static intType one_share() { return funcSSCons(int64_t(1)); }

static intType one_scaled(int sf) { return funcSSCons(int64_t(1) << sf); }

static intType mul_public_const(intType x, int64_t c) {
#ifdef SCI_OT
  return x * static_cast<intType>(c);
#else
  const uint64_t p = static_cast<uint64_t>(prime_mod);
  const uint64_t xm = static_cast<uint64_t>(x) % p;
  const uint64_t cm = static_cast<uint64_t>(c % static_cast<int64_t>(p));
  const unsigned __int128 prod =
      static_cast<unsigned __int128>(xm) * static_cast<unsigned __int128>(cm);
  return static_cast<intType>(static_cast<uint64_t>(prod % p));
#endif
}

static ShareVec mul_public_const_vec(const ShareVec &in, int64_t c) {
  ShareVec out(in.size(), 0);
  for (size_t i = 0; i < in.size(); ++i) {
    out[i] = mul_public_const(in[i], c);
  }
  return out;
}

static bool runtime_uses_he_field() {
  if (bitlength == 64) {
    return false;
  }
  const uint64_t ring_mod = (1ULL << bitlength);
  return prime_mod != ring_mod;
}

static void ensure_same_size(const ShareVec &a, const ShareVec &b,
                             const char *ctx) {
  if (a.size() != b.size()) {
    throw std::invalid_argument(std::string(ctx) + ": size mismatch");
  }
}

static ShareVec tile_by_rows(const ShareVec &v, int rows) {
  if (rows < 0) {
    throw std::invalid_argument("tile_by_rows: negative row count");
  }
  ShareVec out(static_cast<size_t>(rows) * v.size(), 0);
  for (int r = 0; r < rows; ++r) {
    const size_t row_base = static_cast<size_t>(r) * v.size();
    for (size_t k = 0; k < v.size(); ++k) {
      out[row_base + k] = v[k];
    }
  }
  return out;
}

static intType SecretDot(const ShareVec &a, const ShareVec &b, int sf) {
  ensure_same_size(a, b, "SecretDot");
  ShareVec prod;
  SecretElemMul(a, b, prod, sf);
  intType s = 0;
  for (size_t i = 0; i < prod.size(); ++i) {
    s = SecretAdd(s, prod[i]);
  }
  return s;
}

static ShareVec SecretDivVec(const ShareVec &num, const ShareVec &den,
                             const ShareVec &unstable_mask, int sf) {
  ensure_same_size(num, den, "SecretDivVec");
  ensure_same_size(num, unstable_mask, "SecretDivVec(mask)");

  const size_t n = num.size();
  ShareVec out(n, 0), den_safe(n, 0), one_u(n, one_scaled(sf)), inv_mask(n, 0),
      tmp(n, 0);

  intType one = one_share();
  for (size_t i = 0; i < n; ++i) {
    inv_mask[i] = SecretSub(one, unstable_mask[i]);
  }
  SecretElemMul(inv_mask, one_u, tmp, 0);
  SecretElemAdd(den, tmp, den_safe);

  if (math == nullptr) {
    throw std::runtime_error(
        "SecretDivVec: math protocol is not initialized");
  }
  const bool signed_nm = true;
  const bool compute_msnzb = crown_div_compute_msnzb_enabled();
  const bool per_elem = crown_div_per_element_enabled();
  if (per_elem) {
    for (size_t i = 0; i < n; ++i) {
      uint64_t num_i = static_cast<uint64_t>(num[i]);
      uint64_t den_i = static_cast<uint64_t>(den_safe[i]);
      uint64_t out_i = 0;
      math->div(1, &num_i, &den_i, &out_i, bitlength, bitlength, bitlength,
                sf, sf, sf, signed_nm, compute_msnzb);
      out[i] = static_cast<intType>(out_i);
    }
  } else {
    math->div((int32_t)n, (uint64_t *)num.data(), (uint64_t *)den_safe.data(),
              (uint64_t *)out.data(), bitlength, bitlength, bitlength, sf, sf,
              sf, signed_nm, compute_msnzb);
  }
  if (runtime_uses_he_field()) {
    const int64_t delta = static_cast<int64_t>((1ULL << bitlength) - prime_mod);
    if (delta <= 0 || delta >= static_cast<int64_t>(prime_mod)) {
      throw std::runtime_error("SecretDivVec: invalid HE delta for field-safe conversion");
    }
    const intType delta_share = funcSSCons(delta);
    for (size_t i = 0; i < n; ++i) {
      out[i] = SecretSub(out[i], delta_share);
    }
  }
  return out;
}

static ShareMat matmul_rows(const ShareMat &A, int a_rows, int a_cols,
                            const ShareMat &B, int b_rows, int b_cols, int sf,
                            bool model_is_A) {
  if (a_cols != b_rows) {
    throw std::invalid_argument("matmul_rows: invalid shape");
  }
  ShareMat C;
  SecretMatMul(a_rows, a_cols, b_cols, A, B, C, sf, model_is_A);
  return C;
}

void FirstLayerBoundsStandard(const DenseLayerShares &layer, const ShareVec &x0,
                              intType eps, ShareVec &ub, ShareVec &lb, int sf,
                              bool model_is_A) {
  ShareVec ax0;
  SecretMatVecAddBias(layer.out_dim, layer.in_dim, layer.W, x0,
                      make_zero_vec((size_t)layer.out_dim), ax0, sf,
                      model_is_A);

  ShareVec l1;
  SecretRowL1Norm(layer.W, layer.out_dim, layer.in_dim, l1, sf);
  ShareVec eps_vec((size_t)layer.out_dim, eps);
  ShareVec d;
  SecretElemMul(l1, eps_vec, d, sf);

  ub.resize((size_t)layer.out_dim);
  lb.resize((size_t)layer.out_dim);
  for (int i = 0; i < layer.out_dim; ++i) {
    ub[(size_t)i] = SecretAdd(SecretAdd(ax0[(size_t)i], d[(size_t)i]),
                              layer.b[(size_t)i]);
    lb[(size_t)i] = SecretAdd(SecretSub(ax0[(size_t)i], d[(size_t)i]),
                              layer.b[(size_t)i]);
  }
}

void ReLUStateMasks(const ShareVec &ub, const ShareVec &lb, ShareVec &m_pos,
                    ShareVec &m_neg, ShareVec &m_unstable) {
  SecretGTZeroMask(lb, m_pos);
  SecretLEZeroMask(ub, m_neg);

  ensure_same_size(m_pos, m_neg, "ReLUStateMasks");
  m_unstable.resize(m_pos.size());
  intType one = one_share();
  for (size_t i = 0; i < m_pos.size(); ++i) {
    m_unstable[i] = SecretSub(SecretSub(one, m_pos[i]), m_neg[i]);
  }
  // Clamp negative unstable entries (e.g., contradictory boundary masks) to 0.
  ShareVec m_unstable_clamped;
  SecretReLU(m_unstable, m_unstable_clamped, 0);
  m_unstable.swap(m_unstable_clamped);
}

void StandardRelaxBoundKernel(
    const std::vector<DenseLayerShares> &layers,
    const std::vector<ShareVec> &ubs,
    const std::vector<ShareVec> &lbs,
    const std::vector<ShareVec> &state_pos_masks,
    const std::vector<ShareVec> &state_neg_masks, const ShareVec &x0,
    intType eps, ShareVec &uf, ShareVec &lf, int sf, bool model_is_A) {
  const int nl = (int)layers.size();
  if (nl < 1) {
    throw std::invalid_argument("StandardRelaxBoundKernel: empty layers");
  }

  const DenseLayerShares &last = layers.back();
  const int no = last.out_dim;
  const int nl_n = last.in_dim;
  (void)nl_n;

  intType one = one_share();

  ShareVec c = last.b;
  uf = make_zero_vec((size_t)no);
  lf = make_zero_vec((size_t)no);

  ShareMat a = last.W;

  for (int i = nl - 1; i >= 1; --i) {
    const int nc = layers[(size_t)(i - 1)].out_dim;

    ShareVec pos = state_pos_masks[(size_t)(i - 1)];
    ShareVec neg = state_neg_masks[(size_t)(i - 1)];
    ShareVec unstable(pos.size());
    for (size_t k = 0; k < pos.size(); ++k) {
      unstable[k] = SecretSub(SecretSub(one, pos[k]), neg[k]);
    }

    ShareVec den_i, base_i;
    SecretElemSub(ubs[(size_t)i], lbs[(size_t)i], den_i);
    base_i = SecretDivVec(ubs[(size_t)i], den_i, unstable, sf);

    ShareVec alpha_i, pos_part_i, unstable_part_i;
    pos_part_i = mul_public_const_vec(pos, int64_t(1) << sf);
    SecretElemMul(unstable, base_i, unstable_part_i, 0);
    SecretElemAdd(pos_part_i, unstable_part_i, alpha_i);

    ShareVec alpha_tiled(static_cast<size_t>(no * nc), 0);
    for (int j = 0; j < no; ++j) {
      const size_t row_base = static_cast<size_t>(j) * static_cast<size_t>(nc);
      for (int k = 0; k < nc; ++k) {
        alpha_tiled[row_base + static_cast<size_t>(k)] =
            alpha_i[static_cast<size_t>(k)];
      }
    }
    ShareMat a_scaled;
    SecretElemMul(a, alpha_tiled, a_scaled, sf);
    a.swap(a_scaled);

    ShareVec a_pos_flat, a_neg_flat;
    SecretGTZeroMask(a, a_pos_flat);
    SecretLTZeroMask(a, a_neg_flat);

    ShareVec unstable_tiled = tile_by_rows(unstable, no);
    ShareVec lb_tiled = tile_by_rows(lbs[(size_t)i], no);
    ShareVec bias_tiled = tile_by_rows(layers[(size_t)(i - 1)].b, no);

    ShareVec cond_lu_flat, cond_ll_flat;
    SecretElemMul(unstable_tiled, a_pos_flat, cond_lu_flat, 0);
    SecretElemMul(unstable_tiled, a_neg_flat, cond_ll_flat, 0);

    ShareVec lu_flat, ll_flat;
    SecretSelectZeroB(cond_lu_flat, lb_tiled, lu_flat);
    SecretSelectZeroB(cond_ll_flat, lb_tiled, ll_flat);

    ShareVec u_prod, l_prod, c_prod;
    SecretElemMul(a, lu_flat, u_prod, sf);
    SecretElemMul(a, ll_flat, l_prod, sf);
    SecretElemMul(a, bias_tiled, c_prod, sf);

    for (int j = 0; j < no; ++j) {
      intType u_term = 0;
      intType l_term = 0;
      intType c_add = 0;
      const size_t row_base = static_cast<size_t>(j) * static_cast<size_t>(nc);
      for (int k = 0; k < nc; ++k) {
        const size_t idx = row_base + static_cast<size_t>(k);
        u_term = SecretAdd(u_term, u_prod[idx]);
        l_term = SecretAdd(l_term, l_prod[idx]);
        c_add = SecretAdd(c_add, c_prod[idx]);
      }
      uf[static_cast<size_t>(j)] = SecretSub(uf[static_cast<size_t>(j)], u_term);
      lf[static_cast<size_t>(j)] = SecretSub(lf[static_cast<size_t>(j)], l_term);
      c[static_cast<size_t>(j)] = SecretAdd(c[static_cast<size_t>(j)], c_add);
    }

    const DenseLayerShares &prev = layers[(size_t)(i - 1)];
    // Plain kernel uses a[j][m] * W[m][k] where W is stored row-major as
    // [out_dim, in_dim]. So we must multiply by prev.W directly (no transpose).
    // In reverse pass we multiply (...) * W, i.e. model weights are operand B.
    // So the model-owner flag for this matmul is the inverse of global
    // "forward model_is_A" convention.
    const bool matmul_model_is_A = !model_is_A;
    a = matmul_rows(a, no, prev.out_dim, prev.W, prev.out_dim, prev.in_dim, sf,
                    matmul_model_is_A);

  }

  const int input_dim = layers.front().in_dim;
  ShareVec x0_tiled(static_cast<size_t>(no * input_dim), 0);
  for (int j = 0; j < no; ++j) {
    const size_t row_base =
        static_cast<size_t>(j) * static_cast<size_t>(input_dim);
    for (int k = 0; k < input_dim; ++k) {
      x0_tiled[row_base + static_cast<size_t>(k)] =
          x0[static_cast<size_t>(k)];
    }
  }

  ShareVec ax0_terms;
  SecretElemMul(a, x0_tiled, ax0_terms, sf);
  ShareVec ax0_vec(static_cast<size_t>(no), 0);
  for (int j = 0; j < no; ++j) {
    intType ax0 = 0;
    const size_t row_base =
        static_cast<size_t>(j) * static_cast<size_t>(input_dim);
    for (int k = 0; k < input_dim; ++k) {
      ax0 = SecretAdd(ax0, ax0_terms[row_base + static_cast<size_t>(k)]);
    }
    ax0_vec[static_cast<size_t>(j)] = ax0;
  }

  ShareVec a_l1;
  SecretRowL1Norm(a, no, input_dim, a_l1, sf);

  ShareVec eps_vec(static_cast<size_t>(no), eps);
  ShareVec dv;
  SecretElemMul(a_l1, eps_vec, dv, sf);

  for (int j = 0; j < no; ++j) {
    const size_t idx = static_cast<size_t>(j);
    const intType center = SecretAdd(c[idx], ax0_vec[idx]);

    uf[idx] = SecretAdd(uf[idx], SecretAdd(center, dv[idx]));
    lf[idx] = SecretAdd(lf[idx], SecretSub(center, dv[idx]));
  }
}

} // namespace sci::crown
