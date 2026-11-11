#ifndef SCI_VERIFICATION_CROWN_STANDARD_H__
#define SCI_VERIFICATION_CROWN_STANDARD_H__

#include "crown_types.h"

namespace sci::crown {

CrownResultShares CrownStandardBounds2PC(
    const std::vector<DenseLayerShares> &net, const ShareVec &x0, intType eps,
    int pred_label, int target_label, bool untargeted, const CrownConfig &cfg,
    bool model_is_A = true);

CrownResultPublic RevealCrownResult(const CrownResultShares &shares,
                                    const CrownConfig &cfg);

} // namespace sci::crown

#endif
