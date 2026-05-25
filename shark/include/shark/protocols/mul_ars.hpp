#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/span.hpp>

namespace shark {
    namespace protocols {
        namespace mul_ars {
            // Fused element-wise multiplication + arithmetic right shift
            // Computes: Z[i] = (X[i] * Y[i]) >> f
            // Key benefit: intermediate result stays in u128, avoiding overflow
            // This allows using higher f values (e.g., f=26) without overflow

            void gen(int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void eval(int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void call(int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            shark::span<u64> call(int f, const shark::span<u64> &X, const shark::span<u64> &Y);

            // Emulation (local computation, no MPC)
            shark::span<u64> emul(int f, const shark::span<u64> &X, const shark::span<u64> &Y);
        }
    }
}
