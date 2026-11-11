#include "crown_plaintext.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sci::crown::plain {

namespace {

double dot(const Vec &a, const Vec &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("dot: size mismatch");
  }
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    s += a[i] * b[i];
  }
  return s;
}

Vec matvec(const Mat &M, const Vec &x) {
  Vec y(M.size(), 0.0);
  for (size_t i = 0; i < M.size(); ++i) {
    y[i] = dot(M[i], x);
  }
  return y;
}

double l1_norm(const Vec &w) {
  double s = 0.0;
  for (double v : w) {
    s += std::abs(v);
  }
  return s;
}

std::pair<Vec, Vec> first_layer_bounds(const DenseLayerPublic &layer,
                                       const Vec &x0, double eps) {
  Vec ub(layer.out_dim, 0.0), lb(layer.out_dim, 0.0);
  Vec ax0 = matvec(layer.W, x0);
  for (int j = 0; j < layer.out_dim; ++j) {
    const double d = eps * l1_norm(layer.W[(size_t)j]);
    ub[(size_t)j] = ax0[(size_t)j] + d + layer.b[(size_t)j];
    lb[(size_t)j] = ax0[(size_t)j] - d + layer.b[(size_t)j];
  }
  return {ub, lb};
}

std::vector<int> relu_state_from_bounds(const Vec &ub, const Vec &lb) {
  if (ub.size() != lb.size()) {
    throw std::invalid_argument("relu_state_from_bounds: size mismatch");
  }
  std::vector<int> state(ub.size(), 0); // 1: active, -1: inactive, 0: unstable
  for (size_t i = 0; i < ub.size(); ++i) {
    if (std::max(0.0, ub[i]) == 0.0) {
      state[i] = -1;
    } else if (std::max(0.0, lb[i]) > 0.0) {
      state[i] = 1;
    }
  }
  return state;
}

DenseLayerPublic make_margin_layer_targeted(const DenseLayerPublic &last,
                                            int pred, int target) {
  if (pred < 0 || pred >= last.out_dim || target < 0 || target >= last.out_dim ||
      pred == target) {
    throw std::invalid_argument("make_margin_layer_targeted: invalid labels");
  }

  DenseLayerPublic out;
  out.in_dim = last.in_dim;
  out.out_dim = 1;
  out.W.assign(1, Vec((size_t)last.in_dim, 0.0));
  out.b.assign(1, 0.0);

  for (int k = 0; k < last.in_dim; ++k) {
    out.W[0][(size_t)k] =
        last.W[(size_t)pred][(size_t)k] - last.W[(size_t)target][(size_t)k];
  }
  out.b[0] = last.b[(size_t)pred] - last.b[(size_t)target];
  return out;
}

DenseLayerPublic make_margin_layer_untargeted(const DenseLayerPublic &last,
                                              int pred) {
  if (pred < 0 || pred >= last.out_dim) {
    throw std::invalid_argument("make_margin_layer_untargeted: invalid pred");
  }

  DenseLayerPublic out;
  out.in_dim = last.in_dim;
  out.out_dim = last.out_dim - 1;
  out.W.assign((size_t)out.out_dim, Vec((size_t)out.in_dim, 0.0));
  out.b.assign((size_t)out.out_dim, 0.0);

  int row = 0;
  for (int j = 0; j < last.out_dim; ++j) {
    if (j == pred) {
      continue;
    }
    for (int k = 0; k < out.in_dim; ++k) {
      out.W[(size_t)row][(size_t)k] =
          last.W[(size_t)pred][(size_t)k] - last.W[(size_t)j][(size_t)k];
    }
    out.b[(size_t)row] = last.b[(size_t)pred] - last.b[(size_t)j];
    ++row;
  }
  return out;
}

std::pair<Vec, Vec> standard_relax_bound_kernel(
    const std::vector<DenseLayerPublic> &layers,
    const std::vector<Vec> &ubs, // includes input bounds at index 0
    const std::vector<Vec> &lbs,
    const std::vector<std::vector<int>> &states, const Vec &x0, double eps) {
  const int nl = static_cast<int>(layers.size());
  if (nl < 1) {
    throw std::invalid_argument("standard_relax_bound_kernel: empty layers");
  }
  if (nl < 2) {
    throw std::invalid_argument(
        "standard_relax_bound_kernel: requires at least 2 layers");
  }

  const DenseLayerPublic &last = layers.back();
  const size_t no = (size_t)last.out_dim;

  Vec c = last.b;
  Vec uf(no, 0.0), lf(no, 0.0);
  Mat a = last.W;

  for (int i = nl - 1; i >= 1; --i) {
    const std::vector<int> &st = states[(size_t)(i - 1)];
    const size_t nc = st.size();
    const Vec &ubi = ubs[(size_t)i];
    const Vec &lbi = lbs[(size_t)i];

    Vec alpha(nc, 0.0);
    Vec alpha_base(nc, 0.0);
    for (size_t k = 0; k < nc; ++k) {
      if (st[k] == 0) {
        const double u = ubi[k];
        const double l = lbi[k];
        alpha_base[k] = (u - l > 1e-12) ? (u / (u - l)) : 0.0;
        alpha[k] = alpha_base[k];
      } else {
        alpha[k] = std::max(0, st[k]);
      }
    }
    for (size_t j = 0; j < no; ++j) {
      for (size_t k = 0; k < nc; ++k) {
        a[j][k] *= alpha[k];
      }
    }

    for (size_t j = 0; j < no; ++j) {
      Vec lu(nc, 0.0), ll(nc, 0.0);
      for (size_t k = 0; k < nc; ++k) {
        if (st[k] == 0 && a[j][k] > 0.0) {
          lu[k] = lbi[k];
        }
        if (st[k] == 0 && a[j][k] < 0.0) {
          ll[k] = lbi[k];
        }
      }
      uf[j] -= dot(a[j], lu);
      lf[j] -= dot(a[j], ll);
    }

    for (size_t j = 0; j < no; ++j) {
      c[j] += dot(a[j], layers[(size_t)(i - 1)].b);
    }

    const DenseLayerPublic &prev = layers[(size_t)(i - 1)];
    Mat na(no, Vec((size_t)prev.in_dim, 0.0));
    for (size_t j = 0; j < no; ++j) {
      for (int k = 0; k < prev.in_dim; ++k) {
        double s = 0.0;
        for (size_t m = 0; m < nc; ++m) {
          s += a[j][m] * prev.W[m][(size_t)k];
        }
        na[j][(size_t)k] = s;
      }
    }
    a = std::move(na);
  }

  for (size_t j = 0; j < no; ++j) {
    const double ax0 = dot(a[j], x0);
    const double dv = eps * l1_norm(a[j]);
    uf[j] += c[j] + ax0 + dv;
    lf[j] += c[j] + ax0 - dv;
  }
  return {uf, lf};
}

} // namespace

int PredictLabel(const std::vector<DenseLayerPublic> &net, const Vec &x0) {
  if (net.empty()) {
    throw std::invalid_argument("PredictLabel: empty network");
  }
  Vec a = x0;
  for (size_t l = 0; l + 1 < net.size(); ++l) {
    Vec pre = matvec(net[l].W, a);
    for (size_t i = 0; i < pre.size(); ++i) {
      pre[i] = std::max(0.0, pre[i] + net[l].b[i]);
    }
    a = std::move(pre);
  }

  Vec logits = matvec(net.back().W, a);
  for (size_t i = 0; i < logits.size(); ++i) {
    logits[i] += net.back().b[i];
  }
  return (int)std::distance(logits.begin(),
                            std::max_element(logits.begin(), logits.end()));
}

CrownResultPlain CrownStandardBounds(const std::vector<DenseLayerPublic> &net,
                                     const Vec &x0, double eps, int pred_label,
                                     int target_label, bool untargeted,
                                     bool collect_hidden_bounds) {
  if (net.size() < 2) {
    throw std::invalid_argument("CrownStandardBounds: need at least 2 layers");
  }

  Vec ub0(x0.size(), 0.0), lb0(x0.size(), 0.0);
  for (size_t i = 0; i < x0.size(); ++i) {
    ub0[i] = x0[i] + eps;
    lb0[i] = x0[i] - eps;
  }

  std::vector<Vec> ubs = {ub0};
  std::vector<Vec> lbs = {lb0};
  std::vector<Vec> pre_ub, pre_lb;
  std::vector<std::vector<int>> states;

  for (size_t num = 0; num < net.size() - 1; ++num) {
    if (num == 0) {
      auto [u, l] = first_layer_bounds(net[0], x0, eps);
      pre_ub.push_back(u);
      pre_lb.push_back(l);
      states.push_back(relu_state_from_bounds(u, l));
    } else {
      std::vector<DenseLayerPublic> prefix(net.begin(), net.begin() + (num + 1));
      std::vector<Vec> us2 = {ubs[0]};
      std::vector<Vec> ls2 = {lbs[0]};
      us2.insert(us2.end(), pre_ub.begin(), pre_ub.end());
      ls2.insert(ls2.end(), pre_lb.begin(), pre_lb.end());

      auto [u, l] =
          standard_relax_bound_kernel(prefix, us2, ls2, states, x0, eps);
      pre_ub.push_back(u);
      pre_lb.push_back(l);
      states.push_back(relu_state_from_bounds(u, l));
    }
  }

  std::vector<DenseLayerPublic> final_layers(net.begin(), net.end() - 1);
  if (untargeted) {
    final_layers.push_back(make_margin_layer_untargeted(net.back(), pred_label));
  } else {
    final_layers.push_back(
        make_margin_layer_targeted(net.back(), pred_label, target_label));
  }

  std::vector<Vec> usf = {ubs[0]};
  std::vector<Vec> lsf = {lbs[0]};
  usf.insert(usf.end(), pre_ub.begin(), pre_ub.end());
  lsf.insert(lsf.end(), pre_lb.begin(), pre_lb.end());

  auto [uf, lf] =
      standard_relax_bound_kernel(final_layers, usf, lsf, states, x0, eps);

  CrownResultPlain out;
  out.lb = lf;
  out.ub = uf;
  if (untargeted) {
    double g = std::numeric_limits<double>::infinity();
    for (double v : lf) {
      g = std::min(g, v);
    }
    out.gap = std::isfinite(g) ? g : 0.0;
  } else {
    out.gap = lf.empty() ? 0.0 : lf[0];
  }
  out.certified = (out.gap > 0.0);
  if (collect_hidden_bounds) {
    out.pre_relu_lb = pre_lb;
    out.pre_relu_ub = pre_ub;
  }
  return out;
}

} // namespace sci::crown::plain
