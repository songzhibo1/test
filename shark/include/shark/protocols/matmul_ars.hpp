#pragma once

#include <shark/types/u64.hpp>
#include <shark/types/span.hpp>

namespace shark {
    namespace protocols {
        namespace matmul_ars {
            // Fused matmul + arithmetic right shift
            // Computes: Z = (X @ Y) >> f
            // Key benefit: intermediate result stays in u128, avoiding overflow
            // This allows using higher f values (e.g., f=30) without overflow

            void gen(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void eval(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            void call(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y, shark::span<u64> &Z);
            shark::span<u64> call(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y);

            // Emulation (local computation, no MPC)
            shark::span<u64> emul(u64 a, u64 b, u64 c, int f, const shark::span<u64> &X, const shark::span<u64> &Y);
        }
    }
}
