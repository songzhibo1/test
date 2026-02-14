#include <shark/protocols/mul_ars.hpp>
#include <shark/protocols/common.hpp>
#include <shark/utils/assert.hpp>
#include <shark/utils/timer.hpp>

// Type definitions for signed 128-bit and 64-bit integers
using i128 = __int128;
using i64 = int64_t;

namespace shark {
    namespace protocols {
        // Forward declaration of batchCheckArithmBuffer from common.cpp
        extern std::vector<u128> batchCheckArithmBuffer;

        namespace mul_ars {

            // ============================================================
            // Malicious security version (with MAC verification)
            // ============================================================

            void gen(int f, const shark::span<u64> &r_X, const shark::span<u64> &r_Y, shark::span<u64> &r_Z)
            {
                // NOTE: parameter f is unused in gen - it's only used in eval
                // gen must be IDENTICAL to original mul gen to preserve Beaver triple correctness
                (void)f;

                u64 n = r_X.size();
                always_assert(r_Y.size() == n);
                always_assert(r_Z.size() == n);

                randomize(r_Z);

                // r_C = r_X * r_Y + r_Z (NO SHIFT! Same as original mul gen)
                shark::span<u64> r_C(n);
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    r_C[i] = r_X[i] * r_Y[i] + r_Z[i];
                }

                send_authenticated_ashare(r_X);
                send_authenticated_ashare(r_Y);
                send_authenticated_ashare(r_C);
            }

            void eval(int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                u64 n = X.size();
                always_assert(Y.size() == n);
                always_assert(Z.size() == n);

                shark::utils::start_timer("key_read");
                auto [r_X, r_X_tag] = recv_authenticated_ashare(n);
                auto [r_Y, r_Y_tag] = recv_authenticated_ashare(n);
                auto [r_Z, r_Z_tag] = recv_authenticated_ashare(n);
                shark::utils::stop_timer("key_read");

                shark::span<u128> Z_share(n);
                shark::span<u128> Z_tag(n);

                // Z = r_Z + X * Y - r_X * Y - X * r_Y
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    Z_share[i] = r_Z[i] + (u128(party) * X[i] - r_X[i]) * Y[i] - r_Y[i] * X[i];
                    Z_tag[i] = r_Z_tag[i] + (ring_key * X[i] - r_X_tag[i]) * Y[i] - r_Y_tag[i] * X[i];
                }

                // Custom reconstruct with arithmetic right shift
                shark::span<u128> tmp(n);

                peer->send_array(Z_share);
                peer->recv_array(tmp);

                for (u64 i = 0; i < n; i++) {
                    Z_share[i] += tmp[i];
                }

                // MAC verification on original value, then shift before output
                for (u64 i = 0; i < n; i++) {
                    u128 z = Z_share[i] * ring_key - Z_tag[i];
                    batchCheckArithmBuffer.push_back(z);

                    // Arithmetic right shift BEFORE taking low 64 bits
                    i128 signed_val = (i128)Z_share[i];
                    signed_val >>= f;
                    Z[i] = (u64)(i64)signed_val;
                }
            }

            // ============================================================
            // Semi-honest version (no MAC verification)
            // ============================================================

            void gen_sh(int f, const shark::span<u64> &r_X, const shark::span<u64> &r_Y, shark::span<u64> &r_Z)
            {
                // NOTE: parameter f is unused in gen - it's only used in eval
                (void)f;

                u64 n = r_X.size();
                always_assert(r_Y.size() == n);
                always_assert(r_Z.size() == n);

                randomize(r_Z);

                // CRITICAL: r_C must be computed in u128 to avoid overflow!
                // mul_ars performs computation in u128, so Beaver triple must also be in u128.
                shark::span<u128> r_C(n);
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    // r_C = r_X * r_Y + r_Z (in u128 to avoid overflow)
                    r_C[i] = (u128)r_X[i] * (u128)r_Y[i] + (u128)r_Z[i];
                }

                send_sh_ashare(r_X);
                send_sh_ashare(r_Y);
                send_sh_ashare_u128(r_C);  // u128 share to preserve full precision
            }

            void eval_sh(int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                u64 n = X.size();
                always_assert(Y.size() == n);
                always_assert(Z.size() == n);

                shark::utils::start_timer("key_read");
                auto r_X = recv_sh_ashare(n);
                auto r_Y = recv_sh_ashare(n);
                // Receive r_C as u128 share to maintain Beaver triple consistency with gen_sh
                auto r_C = recv_sh_ashare_u128(n);
                shark::utils::stop_timer("key_read");

                shark::span<u128> Z_share_128(n);

                // Z = r_C + X * Y - r_X * Y - X * r_Y (r_C = r_X * r_Y + r_Z in gen)
                // Use u128 for intermediate computation to avoid overflow
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    Z_share_128[i] = r_C[i] + ((u128)X[i] * u128(party) - (u128)r_X[i]) * (u128)Y[i] - (u128)r_Y[i] * (u128)X[i];
                }

                // Reconstruct in u128 to preserve high bits, then apply arithmetic right shift
                shark::span<u128> tmp(n);

                peer->send_array(Z_share_128);
                peer->recv_array(tmp);

                for (u64 i = 0; i < n; i++) {
                    Z_share_128[i] += tmp[i];

                    // Arithmetic right shift after reconstruction
                    i128 signed_val = (i128)Z_share_128[i];
                    signed_val >>= f;
                    Z[i] = (u64)(i64)signed_val;
                }
            }

            // ============================================================
            // Unified interface (selects based on semi_honest_mode)
            // ============================================================

            void call(int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z)
            {
                if (party == DEALER)
                {
                    if (semi_honest_mode)
                        gen_sh(f, X, Y, Z);
                    else
                        gen(f, X, Y, Z);
                }
                else
                {
                    if (semi_honest_mode)
                        eval_sh(f, X, Y, Z);
                    else
                        eval(f, X, Y, Z);
                }
            }

            shark::span<u64> call(int f, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                shark::span<u64> Z(X.size());
                call(f, X, Y, Z);
                return Z;
            }

            shark::span<u64> emul(int f, const shark::span<u64> &X, const shark::span<u64> &Y)
            {
                u64 n = X.size();
                shark::span<u64> Z(n);
                #pragma omp parallel for
                for (u64 i = 0; i < n; i++)
                {
                    i128 prod = (i128)(i64)X[i] * (i64)Y[i];
                    prod >>= f;
                    Z[i] = (u64)(i64)prod;
                }
                return Z;
            }
        }
    }
}
