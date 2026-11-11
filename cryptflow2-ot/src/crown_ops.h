#ifndef SCI_VERIFICATION_CROWN_OPS_H__
#define SCI_VERIFICATION_CROWN_OPS_H__

#include "crown_types.h"

namespace sci::crown {

void SecretMatMul(int m, int k, int n, const ShareMat &A, const ShareMat &B,
                  ShareMat &C, int sf, bool model_is_A);

void SecretMatVecAddBias(int out_dim, int in_dim, const ShareMat &W,
                         const ShareVec &x, const ShareVec &b, ShareVec &y,
                         int sf, bool model_is_A);

void SecretElemAdd(const ShareVec &a, const ShareVec &b, ShareVec &out);
void SecretElemSub(const ShareVec &a, const ShareVec &b, ShareVec &out);
void SecretElemMul(const ShareVec &a, const ShareVec &b, ShareVec &out, int sf);
/// Elementwise ``secret * model`` where ``model`` is public on SERVER (client
/// shares zero), matching SCI ``ElemWiseActModelVectorMult`` / ``Alice_has_A``.
void SecretElemMul_secret_times_public_model(const ShareVec &secret,
                                              const ShareVec &model_pub_server,
                                              ShareVec &out, int sf);
void SecretScalarMul(const ShareVec &a, intType s, ShareVec &out, int sf);
void SecretElemMul_public_constant(const ShareVec &a, int64_t public_q,
                                   ShareVec &out, int sf);
void SecretElemMul_public_constant_unscaled(const ShareVec &a, int64_t public_q,
                                            ShareVec &out);

void SecretReLU(const ShareVec &x, ShareVec &y, int sf);
void SecretDReLU(const ShareVec &x, ShareVec &drelu_bit);

void SecretGTZeroMask(const ShareVec &x, ShareVec &mask01);
void SecretLTZeroMask(const ShareVec &x, ShareVec &mask01);
void SecretLEZeroMask(const ShareVec &x, ShareVec &mask01);

void SecretSelect(const ShareVec &mask01, const ShareVec &a, const ShareVec &b,
                  ShareVec &out);
void SecretSelectZeroA(const ShareVec &mask01, const ShareVec &b,
                       ShareVec &out);
void SecretSelectZeroB(const ShareVec &mask01, const ShareVec &a,
                       ShareVec &out);

void SecretRowL1Norm(const ShareMat &W, int rows, int cols, ShareVec &l1,
                     int sf);

void RevealToParty(const ShareVec &in, std::vector<double> &out, int sf,
                   int reveal_party);
void RevealToParty(const intType &in, double &out, int sf, int reveal_party);

} // namespace sci::crown

#endif
