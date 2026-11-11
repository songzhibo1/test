#ifndef SCI_VERIFICATION_CROWN_STANDARD_KERNEL_H__
#define SCI_VERIFICATION_CROWN_STANDARD_KERNEL_H__

#include "crown_types.h"

namespace sci::crown {

void FirstLayerBoundsStandard(const DenseLayerShares &layer, const ShareVec &x0,
                              intType eps, ShareVec &ub, ShareVec &lb, int sf,
                              bool model_is_A);

void ReLUStateMasks(const ShareVec &ub, const ShareVec &lb, ShareVec &m_pos,
                    ShareVec &m_neg, ShareVec &m_unstable);

void StandardRelaxBoundKernel(
    const std::vector<DenseLayerShares> &layers,
    const std::vector<ShareVec> &ubs, // include input bounds at index 0
    const std::vector<ShareVec> &lbs,
    const std::vector<ShareVec> &state_pos_masks,
    const std::vector<ShareVec> &state_neg_masks, const ShareVec &x0,
    intType eps, ShareVec &uf, ShareVec &lf, int sf, bool model_is_A);

} // namespace sci::crown

#endif
