#include "crown_ops.h"

#include "globals.h"
#include "library_fixed_uniform.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace sci::crown {

static intType one_share() { return funcSSCons(int64_t(1)); }
static intType mul_public_const_share(intType x, int64_t c) {
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

static int32_t bitwidth_for_signed_const(int64_t c) {
  uint64_t mag = (c < 0) ? static_cast<uint64_t>(-c) : static_cast<uint64_t>(c);
  int32_t bits = 1;
  while (mag > 0) {
    ++bits;
    mag >>= 1;
  }
  return bits;
}

static uint64_t mask_for_bw(int32_t bw) {
  return (bw >= 64) ? ~uint64_t(0) : ((uint64_t(1) << bw) - 1);
}

static intType mul_public_const_share_bw(intType x, int64_t c, int32_t bw) {
#ifdef SCI_OT
  const uint64_t mask = mask_for_bw(bw);
  const uint64_t xm = static_cast<uint64_t>(x) & mask;
  const uint64_t cm =
      (bw >= 64)
          ? static_cast<uint64_t>(c)
          : (static_cast<uint64_t>(c) & mask);
  const unsigned __int128 prod =
      static_cast<unsigned __int128>(xm) * static_cast<unsigned __int128>(cm);
  return static_cast<intType>(static_cast<uint64_t>(prod) & mask);
#else
  return mul_public_const_share(x, c);
#endif
}

static bool runtime_uses_he_field() {
  if (bitlength == 64) {
    return false;
  }
  const uint64_t ring_mod = (1ULL << bitlength);
  return prime_mod != ring_mod;
}

static bool enforce_mask_binary_check() {
  static int cached = -1;
  if (cached == -1) {
    const char *v = std::getenv("CROWN_ENFORCE_MASK_BINARY");
    cached = (v == nullptr || std::atoi(v) != 0) ? 1 : 0;
  }
  return cached == 1;
}

static void assert_mask_binary_or_throw(const ShareVec &mask,
                                        const char *name) {
  if (!runtime_uses_he_field() || !enforce_mask_binary_check() || mask.empty()) {
    return;
  }
  std::vector<signedIntType> rec(mask.size(), 0);
  funcReconstruct2PCCons(rec.data(), mask.data(), (int)mask.size());
  if (party != CLIENT) {
    return;
  }

  size_t bad_idx = std::numeric_limits<size_t>::max();
  signedIntType bad_v = 0;
  for (size_t i = 0; i < rec.size(); ++i) {
    if (!(rec[i] == 0 || rec[i] == 1)) {
      bad_idx = i;
      bad_v = rec[i];
      break;
    }
  }
  if (bad_idx != std::numeric_limits<size_t>::max()) {
    throw std::runtime_error(std::string("mask binary check failed for ") +
                             name + " at idx=" + std::to_string(bad_idx) +
                             " value=" + std::to_string(bad_v));
  }
}

static void check_same_size(const ShareVec &a, const ShareVec &b,
                            const char *ctx) {
  if (a.size() != b.size()) {
    throw std::invalid_argument(std::string(ctx) + ": size mismatch");
  }
}

void SecretMatMul(int m, int k, int n, const ShareMat &A, const ShareMat &B,
                  ShareMat &C, int sf, bool model_is_A) {
  if (A.size() != static_cast<size_t>(m * k) ||
      B.size() != static_cast<size_t>(k * n)) {
    throw std::invalid_argument("SecretMatMul: invalid matrix shape");
  }
  C.assign(static_cast<size_t>(m * n), 0);
  const bool matmul_model_is_A = model_is_A;
  MatMul2D(m, k, n, A.data(), B.data(), C.data(), matmul_model_is_A);
  if (sf > 0) {
    ScaleDown(m * n, C.data(), sf);
  }
}

void SecretMatVecAddBias(int out_dim, int in_dim, const ShareMat &W,
                         const ShareVec &x, const ShareVec &b, ShareVec &y,
                         int sf, bool model_is_A) {
  if (W.size() != static_cast<size_t>(out_dim * in_dim) ||
      x.size() != static_cast<size_t>(in_dim) ||
      b.size() != static_cast<size_t>(out_dim)) {
    throw std::invalid_argument("SecretMatVecAddBias: invalid shape");
  }
  y.assign(out_dim, 0);

  if (runtime_uses_he_field() && model_is_A) {
    // HE linear backend is most stable when the model is operand B.
    // Compute y_row = x_row * W^T, where W is [out_dim, in_dim].
    ShareMat w_t(static_cast<size_t>(in_dim * out_dim), 0);
    for (int o = 0; o < out_dim; ++o) {
      for (int i = 0; i < in_dim; ++i) {
        w_t[(size_t)i * out_dim + (size_t)o] =
            W[(size_t)o * in_dim + (size_t)i];
      }
    }
    ShareMat x_row(x.begin(), x.end()); // [1, in_dim]
    ShareMat out_row;
    SecretMatMul(1, in_dim, out_dim, x_row, w_t, out_row, sf, false);
    for (int i = 0; i < out_dim; ++i) {
      y[(size_t)i] = SecretAdd(out_row[(size_t)i], b[(size_t)i]);
    }
    return;
  }

  ShareMat x_col(x.begin(), x.end()); // [in_dim, 1]
  ShareMat out_col;
  SecretMatMul(out_dim, in_dim, 1, W, x_col, out_col, sf, model_is_A);
  for (int i = 0; i < out_dim; ++i) {
    y[(size_t)i] = SecretAdd(out_col[(size_t)i], b[(size_t)i]);
  }
}

void SecretElemAdd(const ShareVec &a, const ShareVec &b, ShareVec &out) {
  check_same_size(a, b, "SecretElemAdd");
  out.resize(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = SecretAdd(a[i], b[i]);
  }
}

void SecretElemSub(const ShareVec &a, const ShareVec &b, ShareVec &out) {
  check_same_size(a, b, "SecretElemSub");
  out.resize(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    out[i] = SecretSub(a[i], b[i]);
  }
}

void SecretElemMul(const ShareVec &a, const ShareVec &b, ShareVec &out,
                   int sf) {
  check_same_size(a, b, "SecretElemMul");
  ShareVec a_copy;
  ShareVec b_copy;
  const ShareVec *a_src = &a;
  const ShareVec *b_src = &b;
  if (&out == &a) {
    a_copy = a;
    a_src = &a_copy;
  }
  if (&out == &b) {
    b_copy = b;
    b_src = &b_copy;
  }

  out.assign(a.size(), 0);
  ElemWiseSecretSharedVectorMult((int32_t)a.size(),
                                 const_cast<intType *>(a_src->data()),
                                 const_cast<intType *>(b_src->data()),
                                 out.data());
  if (sf > 0) {
    ScaleDown((int32_t)out.size(), out.data(), sf);
  }
}

void SecretElemMul_secret_times_public_model(const ShareVec &secret,
                                             const ShareVec &model_pub_server,
                                             ShareVec &out, int sf) {
  check_same_size(secret, model_pub_server, "SecretElemMul_secret_times_public_model");
  ShareVec sec_copy;
  ShareVec mod_copy;
  const ShareVec *sec = &secret;
  const ShareVec *mod = &model_pub_server;
  if (&out == &secret) {
    sec_copy = secret;
    sec = &sec_copy;
  }
  if (&out == &model_pub_server) {
    mod_copy = model_pub_server;
    mod = &mod_copy;
  }

  out.assign(secret.size(), 0);
  // SCI: inArr = fully secret, multArrVec = model (nonzero only on SERVER).
  ElemWiseActModelVectorMult((int32_t)secret.size(),
                             const_cast<intType *>(sec->data()),
                             const_cast<intType *>(mod->data()), out.data());
  if (sf > 0) {
    ScaleDown((int32_t)out.size(), out.data(), sf);
  }
}

void SecretScalarMul(const ShareVec &a, intType s, ShareVec &out, int sf) {
  ShareVec scalar(a.size(), s);
  SecretElemMul(a, scalar, out, sf);
}

void SecretElemMul_public_constant(const ShareVec &a, int64_t public_q,
                                   ShareVec &out, int sf) {
  if (runtime_uses_he_field()) {
    SecretElemMul_public_constant_unscaled(a, public_q, out);
    if (sf > 0) {
      ScaleDown((int32_t)out.size(), out.data(), sf);
    }
    return;
  }

  const int32_t prod_bw = bitlength + bitwidth_for_signed_const(public_q);
  if (prod_bw <= sf || prod_bw > 64) {
    throw std::runtime_error(
        "SecretElemMul_public_constant: unsupported product bitwidth");
  }
  if (xt == nullptr || truncation == nullptr) {
    throw std::runtime_error(
        "SecretElemMul_public_constant: XT/truncation protocol not initialized");
  }

  ShareVec a_copy;
  const ShareVec *a_src = &a;
  if (&out == &a) {
    a_copy = a;
    a_src = &a_copy;
  }

  std::vector<uint64_t> ext(a_src->size(), 0);
  std::vector<uint64_t> prod(a_src->size(), 0);
  std::vector<uint64_t> trunc_out(a_src->size(), 0);
  xt->s_extend((int32_t)a_src->size(), reinterpret_cast<uint64_t *>(
                                           const_cast<intType *>(a_src->data())),
               ext.data(), bitlength, prod_bw);
  for (size_t i = 0; i < ext.size(); ++i) {
    prod[i] = static_cast<uint64_t>(
        mul_public_const_share_bw(static_cast<intType>(ext[i]), public_q, prod_bw));
  }
  truncation->truncate((int32_t)prod.size(), prod.data(), trunc_out.data(), sf,
                       prod_bw, true);

  out.assign(a.size(), 0);
  const uint64_t out_mask = mask_for_bw(bitlength);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<intType>(trunc_out[i] & out_mask);
  }
}

void SecretElemMul_public_constant_unscaled(const ShareVec &a, int64_t public_q,
                                            ShareVec &out) {
  ShareVec a_copy;
  const ShareVec *a_src = &a;
  if (&out == &a) {
    a_copy = a;
    a_src = &a_copy;
  }

  out.assign(a.size(), 0);
  for (size_t i = 0; i < a_src->size(); ++i) {
    out[i] = mul_public_const_share((*a_src)[i], public_q);
  }
}

void SecretReLU(const ShareVec &x, ShareVec &y, int sf) {
  y.assign(x.size(), 0);
  Relu((int32_t)x.size(), const_cast<intType *>(x.data()), y.data(), sf, false);
}

void SecretDReLU(const ShareVec &x, ShareVec &drelu_bit) {
  if (relu == nullptr) {
    throw std::runtime_error("SecretDReLU: relu protocol is not initialized");
  }
  if (aux == nullptr) {
    throw std::runtime_error("SecretDReLU: aux protocol is not initialized");
  }
  std::vector<uint8_t> drelu_bool(x.size(), 0);
  ShareVec scratch(x.size(), 0);
  relu->relu(scratch.data(), const_cast<intType *>(x.data()), (int)x.size(),
             drelu_bool.data(), true);
  drelu_bit.assign(x.size(), 0);
  aux->B2A(drelu_bool.data(), (uint64_t *)drelu_bit.data(),
           (int32_t)x.size(), bitlength);
  if (runtime_uses_he_field()) {
    // HE path: aux->B2A runs over ring semantics internally.
    // Convert ring-style boolean arithmetic shares into field-safe {0,1} shares.
    // Observed encoding is affine around delta = (2^l - p): {delta, delta+1}.
    const int64_t p = static_cast<int64_t>(prime_mod);
    const int64_t delta = static_cast<int64_t>((1ULL << bitlength) - prime_mod);
    if (delta <= 0 || delta >= p) {
      throw std::runtime_error("SecretDReLU: invalid HE delta for field-safe conversion");
    }
    const intType delta_share = funcSSCons(delta);
    for (size_t i = 0; i < drelu_bit.size(); ++i) {
      drelu_bit[i] = SecretSub(drelu_bit[i], delta_share);
    }
  }
}

void SecretGTZeroMask(const ShareVec &x, ShareVec &mask01) {
  ShareVec le_zero;
  SecretDReLU(x, le_zero); // SCI DReLU returns 1 when x <= 0.
  mask01.resize(x.size());
  intType one = one_share();
  for (size_t i = 0; i < x.size(); ++i) {
    mask01[i] = SecretSub(one, le_zero[i]);
  }
#ifdef VERIFY_MASK_BINARY
  assert_mask_binary_or_throw(mask01, "GTZero");
#endif
}

void SecretLTZeroMask(const ShareVec &x, ShareVec &mask01) {
  ShareVec neg_x(x.size(), 0);
  for (size_t i = 0; i < x.size(); ++i) {
    neg_x[i] = SecretSub(0, x[i]);
  }
  SecretGTZeroMask(neg_x, mask01);
}

void SecretLEZeroMask(const ShareVec &x, ShareVec &mask01) {
  SecretDReLU(x, mask01);
#ifdef VERIFY_MASK_BINARY
  assert_mask_binary_or_throw(mask01, "LEZero");
#endif
}

void SecretSelectZeroB(const ShareVec &mask01, const ShareVec &a,
                       ShareVec &out) {
  // Optimized path: out = mask01 * a + (1 - mask01) * 0 = mask01 * a.
  // Only requires 1 multiplication instead of 2.
  SecretElemMul(mask01, a, out, 0);
}

void SecretSelectZeroA(const ShareVec &mask01, const ShareVec &b,
                       ShareVec &out) {
  // Optimized path: out = mask01 * 0 + (1 - mask01) * b = (1 - mask01) * b.
  // Only requires 1 multiplication instead of 2.
  ShareVec one_minus_mask(mask01.size(), 0);
  intType one = one_share();
  for (size_t i = 0; i < mask01.size(); ++i) {
    one_minus_mask[i] = SecretSub(one, mask01[i]);
  }
  SecretElemMul(one_minus_mask, b, out, 0);
}

void SecretSelect(const ShareVec &mask01, const ShareVec &a, const ShareVec &b,
                  ShareVec &out) {
  check_same_size(mask01, a, "SecretSelect(mask,a)");
  check_same_size(mask01, b, "SecretSelect(mask,b)");

  ShareVec left, right, one_minus_mask(mask01.size(), 0);
  intType one = one_share();
  for (size_t i = 0; i < mask01.size(); ++i) {
    one_minus_mask[i] = SecretSub(one, mask01[i]);
  }

  // mask is {0,1} with scale 0, values have current data scale => no trunc.
  SecretElemMul(mask01, a, left, 0);
  SecretElemMul(one_minus_mask, b, right, 0);
  SecretElemAdd(left, right, out);
}

void SecretRowL1Norm(const ShareMat &W, int rows, int cols, ShareVec &l1,
                     int sf) {
  if (W.size() != static_cast<size_t>(rows * cols)) {
    throw std::invalid_argument("SecretRowL1Norm: invalid shape");
  }
  l1.assign(rows, 0);

  (void)sf;
  ShareVec neg_w(W.size(), 0);
  for (size_t i = 0; i < W.size(); ++i) {
    neg_w[i] = SecretSub(0, W[i]);
  }

  ShareVec pos_mask, abs_w;
  SecretGTZeroMask(W, pos_mask);
  SecretSelect(pos_mask, W, neg_w, abs_w);

  for (int r = 0; r < rows; ++r) {
    intType sum = 0;
    const size_t row_base = static_cast<size_t>(r) * cols;
    for (int c = 0; c < cols; ++c) {
      sum = SecretAdd(sum, abs_w[row_base + static_cast<size_t>(c)]);
    }
    l1[static_cast<size_t>(r)] = sum;
  }
}

void RevealToParty(const ShareVec &in, std::vector<double> &out, int sf,
                   int reveal_party) {
  if (reveal_party != CLIENT) {
    throw std::invalid_argument("RevealToParty(vec): only CLIENT supported");
  }

  out.assign(in.size(), 0.0);
  std::vector<signedIntType> rec(in.size(), 0);
  funcReconstruct2PCCons(rec.data(), in.data(), (int)in.size());

  if (party == CLIENT) {
    const double scale = std::ldexp(1.0, sf);
    for (size_t i = 0; i < rec.size(); ++i) {
      out[i] = static_cast<double>(rec[i]) / scale;
    }
  }
}

void RevealToParty(const intType &in, double &out, int sf, int reveal_party) {
  if (reveal_party != CLIENT) {
    throw std::invalid_argument("RevealToParty(scalar): only CLIENT supported");
  }
  signedIntType v = 0;
  intType in_buf[1] = {in};
  signedIntType out_buf[1] = {0};
  funcReconstruct2PCCons(out_buf, in_buf, 1);
  v = out_buf[0];
  if (party == CLIENT) {
    const double scale = std::ldexp(1.0, sf);
    out = static_cast<double>(v) / scale;
  } else {
    out = 0.0;
  }
}

} // namespace sci::crown
