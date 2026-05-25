#include <shark/protocols/input.hpp>
#include <shark/protocols/common.hpp>

namespace shark
{
    namespace protocols
    {
        namespace output
        {
            // ============================================================
            // Malicious security version (with MAC verification)
            // ============================================================

            void gen(const shark::span<u64> &r_X)
            {
                send_authenticated_ashare(r_X);
            }

            void gen(const shark::span<u8> &r_X)
            {
                send_authenticated_bshare(r_X);
            }

            void eval(shark::span<u64> &X)
            {
                auto [r_X, r_X_tag] = recv_authenticated_ashare(X.size());
                batch_check();
                auto r = authenticated_reconstruct(r_X, r_X_tag);
                batch_check();

                for (u64 i = 0; i < X.size(); i++)
                {
                    X[i] -= r[i];
                }
            }

            void eval(shark::span<u8> &X)
            {
                auto r_bdoz = recv_authenticated_bshare(X.size());
                batch_check();
                auto r = authenticated_reconstruct(r_bdoz);
                batch_check();

                for (u64 i = 0; i < X.size(); i++)
                {
                    X[i] ^= r[i];
                }
            }

            // ============================================================
            // Semi-honest version (no MAC verification)
            // ============================================================

            void gen_sh(const shark::span<u64> &r_X)
            {
                send_sh_ashare(r_X);
            }

            void gen_sh(const shark::span<u8> &r_X)
            {
                send_sh_bshare(r_X);
            }

            void eval_sh(shark::span<u64> &X)
            {
                auto r_X = recv_sh_ashare(X.size());
                // No batch_check in semi-honest mode
                auto r = sh_reconstruct(r_X);
                // No batch_check in semi-honest mode

                for (u64 i = 0; i < X.size(); i++)
                {
                    X[i] -= r[i];
                }
            }

            void eval_sh(shark::span<u8> &X)
            {
                auto r_X = recv_sh_bshare(X.size());
                // No batch_check in semi-honest mode
                auto r = sh_reconstruct(r_X);
                // No batch_check in semi-honest mode

                for (u64 i = 0; i < X.size(); i++)
                {
                    X[i] ^= r[i];
                }
            }

            // ============================================================
            // Unified interface (selects based on semi_honest_mode)
            // ============================================================

            void call(shark::span<u64> &X)
            {
                if (party == DEALER)
                {
                    if (semi_honest_mode)
                        gen_sh(X);
                    else
                        gen(X);
                }
                else
                {
                    if (semi_honest_mode)
                        eval_sh(X);
                    else
                        eval(X);
                }
            }

            void call(shark::span<u8> &X)
            {
                if (party == DEALER)
                {
                    if (semi_honest_mode)
                        gen_sh(X);
                    else
                        gen(X);
                }
                else
                {
                    if (semi_honest_mode)
                        eval_sh(X);
                    else
                        eval(X);
                }
            }
        }
    }
}
