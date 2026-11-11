#ifndef SCI_VERIFICATION_CROWN_PLAINTEXT_H__
#define SCI_VERIFICATION_CROWN_PLAINTEXT_H__

#include <vector>

namespace sci::crown::plain {

using Vec = std::vector<double>;
using Mat = std::vector<Vec>;

struct DenseLayerPublic {
  int in_dim = 0;
  int out_dim = 0;
  Mat W; // [out_dim][in_dim]
  Vec b; // [out_dim]
};

struct CrownResultPlain {
  Vec lb;
  Vec ub;
  double gap = 0.0;
  bool certified = false;
  std::vector<Vec> pre_relu_lb;
  std::vector<Vec> pre_relu_ub;
};

int PredictLabel(const std::vector<DenseLayerPublic> &net, const Vec &x0);

CrownResultPlain CrownStandardBounds(
    const std::vector<DenseLayerPublic> &net, const Vec &x0, double eps,
    int pred_label, int target_label, bool untargeted,
    bool collect_hidden_bounds = true);

} // namespace sci::crown::plain

#endif
