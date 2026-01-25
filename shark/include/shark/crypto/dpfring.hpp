#pragma once

#include <tuple>
#include <vector>
#include <shark/types/span.hpp>
#include <shark/types/u128.hpp>
#include <shark/utils/assert.hpp>

namespace shark
{
    namespace crypto
    {
        struct DPFRingKey
        {
            shark::span<block> k;
            u128 g_ring;
            u128 g_tag;

            DPFRingKey(const shark::span<block> &k, u128 g_ring, u128 g_tag)
                : k(k), g_ring(g_ring), g_tag(g_tag)
            {
            }

            DPFRingKey(shark::span<block> &&k, u128 g_ring, u128 g_tag)
                : k(std::move(k)), g_ring(g_ring), g_tag(g_tag)
            {
            }

            // move constructor
            DPFRingKey(DPFRingKey &&other) : k(std::move(other.k)), g_ring(other.g_ring), g_tag(other.g_tag)
            {
            }

            // move assignment
            DPFRingKey &operator=(DPFRingKey &&other)
            {
                k = std::move(other.k);
                g_ring = other.g_ring;
                g_tag = other.g_tag;
                return *this;
            }

            DPFRingKey() = default;
        };


        std::pair<DPFRingKey, DPFRingKey> dpfring_gen(int bin, const u64 alpha);
        std::tuple<u128, u128> dpfring_evalall_reduce(int party, const DPFRingKey &key, const std::vector<u64> &lut, u64 lut_offset);

        // ============================================================
        // Semi-honest version (no MAC tags)
        // ============================================================

        struct DPFRingKeySH
        {
            shark::span<block> k;
            u64 g_ring;

            DPFRingKeySH(const shark::span<block> &k, u64 g_ring)
                : k(k), g_ring(g_ring) {}

            DPFRingKeySH(shark::span<block> &&k, u64 g_ring)
                : k(std::move(k)), g_ring(g_ring) {}

            DPFRingKeySH(DPFRingKeySH &&other)
                : k(std::move(other.k)), g_ring(other.g_ring) {}

            DPFRingKeySH &operator=(DPFRingKeySH &&other)
            {
                k = std::move(other.k);
                g_ring = other.g_ring;
                return *this;
            }

            DPFRingKeySH() = default;
        };

        std::pair<DPFRingKeySH, DPFRingKeySH> dpfring_gen_sh(int bin, const u64 alpha);
        u64 dpfring_evalall_reduce_sh(int party, const DPFRingKeySH &key, const std::vector<u64> &lut, u64 lut_offset);
    }
}