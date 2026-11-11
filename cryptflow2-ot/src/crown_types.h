#ifndef SCI_VERIFICATION_CROWN_TYPES_H__
#define SCI_VERIFICATION_CROWN_TYPES_H__

#include "defines_uniform.h"

#include <cstdint>
#include <vector>

namespace sci::crown {

using ShareVec = std::vector<intType>;
using ShareMat = std::vector<intType>; // row-major [rows * cols]

struct DenseLayerShares {
  int in_dim;
  int out_dim;
  ShareMat W; // [out_dim, in_dim]
  ShareVec b; // [out_dim]
};

struct CrownConfig {
  int bitlength;
  int scale_bits;
  int num_threads;
  int reveal_party;
};

struct CrownResultShares {
  ShareVec lb;
  ShareVec ub;
  intType gap;
};

struct CrownResultPublic {
  std::vector<double> lb;
  std::vector<double> ub;
  double gap;
  bool certified;
};

} // namespace sci::crown

#endif
