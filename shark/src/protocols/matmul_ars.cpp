#include <shark/protocols/matmul_ars.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>
#include <shark/utils/eigen.hpp>

using namespace shark::matrix;

// Type definitions for signed 128-bit and 64-bit integers
using i128 = __int128;
using i64 = int64_t;

namespace shark {
    namespace protocols {
        // Forward declaration of batchCheckArithmBuffer from common.cpp
        extern std::vector<u128> batchCheckArithmBuffer;

        namespace matmul_ars {

            // ============================================================
            // Malicious security version (with MAC verification)
            // ============================================================

            void gen(u64 a, u64 b, u64 c, int f, const shark::span<u64> &r_X, const shark::span<u64> &r_Y, shark::span<u64> &r_Z)
            {
                always_assert(r_X.size() == a * b);
                always_assert(r_Y.size() == b * c);
                always_assert(r_Z.size() == a * c);

                randomize(r_Z);

                // Use u128 for intermediate computation to avoid overflow
                auto mat_r_X = getMat(a, b, r_X).cast<u128>();
                auto mat_r_Y = getMat(b, c, r_Y).cast<u128>();

                shark::span<u128> r_C_128(a * c);
                auto mat_r_C_128 = getMat(a, c, r_C_128);

                // Compute (r_X @ r_Y) in u128
                mat_r_C_128 = mat_r_X * mat_r_Y;

                // Right shift by f, then add r_Z
                shark::span<u64> r_C(a * c);
                for (u64 i = 0; i < a * c; ++i) {
                    // Arithmetic right shift on the matmul result
                    i128 val = (i128)r_C_128[i];
                    val >>= f;
                    r_C[i] = (u64)(i64)val + r_Z[i];
                }

                send_authenticated_ashare(r_X);
                send_authenticated_ashare(r_Y);
                send_authenticated_ashare(r_C);
            }

            void eval(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                shark::utils::start_timer("key_read");
                auto [r_X, r_X_tag] = recv_authenticated_ashare(a * b);
                auto [r_Y, r_Y_tag] = recv_authenticated_ashare(b * c);
                auto [r_Z, r_Z_tag] = recv_authenticated_ashare(a * c);
                shark::utils::stop_timer("key_read");

                auto mat_X = getMat(a, b, X).cast<u128>();
                auto mat_Y = getMat(b, c, Y).cast<u128>();

                shark::span<u128> Z_share(a * c);
                shark::span<u128> Z_tag(a * c);
                auto mat_Z_share = getMat(a, c, Z_share);
                auto mat_Z_tag = getMat(a, c, Z_tag);

                auto mat_r_X = getMat(a, b, r_X);
                auto mat_r_X_tag = getMat(a, b, r_X_tag);
                auto mat_r_Y = getMat(b, c, r_Y);
                auto mat_r_Y_tag = getMat(b, c, r_Y_tag);
                auto mat_r_Z = getMat(a, c, r_Z);
                auto mat_r_Z_tag = getMat(a, c, r_Z_tag);

                // Z_matmul = r_Z + X @ Y - r_X @ Y - X @ r_Y
                // This is the standard Beaver triple based multiplication
                mat_Z_share = mat_r_Z + (mat_X * u128(party) - mat_r_X) * mat_Y;
                mat_Z_share -= mat_X * mat_r_Y;

                mat_Z_tag = mat_r_Z_tag + (mat_X * ring_key - mat_r_X_tag) * mat_Y;
                mat_Z_tag -= mat_X * mat_r_Y_tag;

                // Custom reconstruct with arithmetic right shift
                // This is the key difference from regular matmul:
                // We shift the u128 result BEFORE converting to u64
                shark::span<u128> tmp(a * c);

                peer->send_array(Z_share);
                peer->recv_array(tmp);

                for (u64 i = 0; i < Z_share.size(); i++) {
                    Z_share[i] += tmp[i];
                }

                // MAC verification uses original (non-shifted) value
                for (u64 i = 0; i < Z_share.size(); i++) {
                    u128 z = Z_share[i] * ring_key - Z_tag[i];
                    batchCheckArithmBuffer.push_back(z);

                    // CRITICAL: Arithmetic right shift BEFORE taking low 64 bits
                    // This prevents overflow when f is large
                    i128 signed_val = (i128)Z_share[i];
                    signed_val >>= f;
                    Z[i] = (u64)(i64)signed_val;
                }
            }

            // ============================================================
            // Semi-honest version (no MAC verification)
            // ============================================================

            void gen_sh(u64 a, u64 b, u64 c, int f, const shark::span<u64> &r_X, const shark::span<u64> &r_Y, shark::span<u64> &r_Z)
            {
                always_assert(r_X.size() == a * b);
                always_assert(r_Y.size() == b * c);
                always_assert(r_Z.size() == a * c);

                randomize(r_Z);

                // Use u128 for intermediate computation
                auto mat_r_X = getMat(a, b, r_X).cast<u128>();
                auto mat_r_Y = getMat(b, c, r_Y).cast<u128>();

                shark::span<u128> r_C_128(a * c);
                auto mat_r_C_128 = getMat(a, c, r_C_128);

                mat_r_C_128 = mat_r_X * mat_r_Y;

                shark::span<u64> r_C(a * c);
                for (u64 i = 0; i < a * c; ++i) {
                    i128 val = (i128)r_C_128[i];
                    val >>= f;
                    r_C[i] = (u64)(i64)val + r_Z[i];
                }

                send_sh_ashare(r_X);
                send_sh_ashare(r_Y);
                send_sh_ashare(r_C);
            }

            void eval_sh(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                shark::utils::start_timer("key_read");
                auto r_X = recv_sh_ashare(a * b);
                auto r_Y = recv_sh_ashare(b * c);
                auto r_Z = recv_sh_ashare(a * c);
                shark::utils::stop_timer("key_read");

                // Use u128 for computation
                auto mat_X = getMat(a, b, X).cast<u128>();
                auto mat_Y = getMat(b, c, Y).cast<u128>();

                shark::span<u128> Z_share_128(a * c);
                auto mat_Z_share = getMat(a, c, Z_share_128);

                auto mat_r_X = getMat(a, b, r_X).cast<u128>();
                auto mat_r_Y = getMat(b, c, r_Y).cast<u128>();
                auto mat_r_Z = getMat(a, c, r_Z).cast<u128>();

                // Z = r_Z + X @ Y - r_X @ Y - X @ r_Y
                mat_Z_share = mat_r_Z + (mat_X * u128(party) - mat_r_X) * mat_Y;
                mat_Z_share -= mat_X * mat_r_Y;

                // Custom reconstruct with right shift
                shark::span<u128> tmp(a * c);
                shark::span<u64> Z_share_64(a * c);

                // Convert to u64 for sending (take low bits for shares)
                for (u64 i = 0; i < a * c; ++i) {
                    Z_share_64[i] = (u64)Z_share_128[i];
                }

                shark::span<u64> tmp_64(a * c);
                peer->send_array(Z_share_64);
                peer->recv_array(tmp_64);

                for (u64 i = 0; i < Z_share_128.size(); i++) {
                    // Reconstruct in u128
                    Z_share_128[i] += (u128)tmp_64[i];

                    // Arithmetic right shift before converting to u64
                    i128 signed_val = (i128)Z_share_128[i];
                    signed_val >>= f;
                    Z[i] = (u64)(i64)signed_val;
                }
            }

            // ============================================================
            // Unified interface (selects based on semi_honest_mode)
            // ============================================================

            void call(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                if (party == DEALER)
                {
                    if (semi_honest_mode)
                        gen_sh(a, b, c, f, X, Y, Z);
                    else
                        gen(a, b, c, f, X, Y, Z);
                }
                else
                {
                    if (semi_honest_mode)
                        eval_sh(a, b, c, f, X, Y, Z);
                    else
                        eval(a, b, c, f, X, Y, Z);
                }
            }

            shark::span<u64> call(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                shark::span<u64> Z(a * c);
                call(a, b, c, f, X, Y, Z);
                return Z;
            }

            shark::span<u64> emul(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                shark::span<u64> Z(a * c);

                // Use u128 for computation to avoid overflow
                auto X_mat = getMat(a, b, X).cast<u128>();
                auto Y_mat = getMat(b, c, Y).cast<u128>();

                shark::span<u128> Z_128(a * c);
                auto Z_mat = getMat(a, c, Z_128);
                Z_mat = X_mat * Y_mat;

                // Arithmetic right shift
                for (u64 i = 0; i < a * c; ++i) {
                    i128 val = (i128)Z_128[i];
                    val >>= f;
                    Z[i] = (u64)(i64)val;
                }

                return Z;
            }
        }
    }
}
