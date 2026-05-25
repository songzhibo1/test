#pragma once

#include <tuple>
#include <shark/types/span.hpp>
#include <shark/types/u128.hpp>
#include <shark/utils/assert.hpp>

namespace shark
{
    namespace crypto
    {
        struct DCFBitKey
        {
            shark::span<block> k;
            shark::span<u8> v_bit;
            shark::span<u64> v_tag_1;
            // shark::span<u64> v_tag_2;
            u8 g_bit;
            u64 g_tag_1;
            u64 g_tag_2;

            DCFBitKey(const shark::span<block> &k, const shark::span<u8> &v_bit, const shark::span<u64> &v_tag_1,  u8 g_bit, u64 g_tag_1, u64 g_tag_2)
                : k(k), v_bit(v_bit), v_tag_1(v_tag_1), g_bit(g_bit), g_tag_1(g_tag_1), g_tag_2(g_tag_2)
            {
                int bin = k.size() - 1;
                always_assert(bin == v_bit.size());
                always_assert(bin == v_tag_1.size());
            }

            DCFBitKey(shark::span<block> &&k, shark::span<u8> &&v_bit, shark::span<u64> &&v_tag_1, u8 g_bit, u64 g_tag_1, u64 g_tag_2)
                : k(std::move(k)), v_bit(std::move(v_bit)), v_tag_1(std::move(v_tag_1)), g_bit(g_bit), g_tag_1(g_tag_1), g_tag_2(g_tag_2)
            {
                int bin = this->k.size() - 1;
                always_assert(bin == this->v_bit.size());
                always_assert(bin == this->v_tag_1.size());
            }

            // move constructor
            DCFBitKey(DCFBitKey &&other) : k(std::move(other.k)), v_bit(std::move(other.v_bit)), v_tag_1(std::move(other.v_tag_1)), g_bit(other.g_bit), g_tag_1(other.g_tag_1), g_tag_2(other.g_tag_2)
            {
            }

            // move assignment
            DCFBitKey &operator=(DCFBitKey &&other)
            {
                k = std::move(other.k);
                v_bit = std::move(other.v_bit);
                v_tag_1 = std::move(other.v_tag_1);
                g_bit = other.g_bit;
                g_tag_1 = other.g_tag_1;
                g_tag_2 = other.g_tag_2;
                return *this;
            }

            DCFBitKey() = default;
        };


        std::pair<DCFBitKey, DCFBitKey> dcfbit_gen(int bin, const u64 alpha, const bool greaterThan = false);
        std::tuple<u8, u64> dcfbit_eval(const DCFBitKey &key, const u64 &x, const bool greaterThan = false);

        // ============================================================
        // Semi-honest version of DCFBit (no MAC tags)
        // ============================================================

        /// Semi-honest DCF key structure - no authentication tags
        struct DCFBitKeySH
        {
            shark::span<block> k;       // correction words
            shark::span<u8> v_bit;      // v correction bits (no tags)
            u8 g_bit;                   // final correction bit

            DCFBitKeySH(const shark::span<block> &k, const shark::span<u8> &v_bit, u8 g_bit)
                : k(k), v_bit(v_bit), g_bit(g_bit)
            {
                int bin = k.size() - 1;
                always_assert(bin == v_bit.size());
            }

            DCFBitKeySH(shark::span<block> &&k, shark::span<u8> &&v_bit, u8 g_bit)
                : k(std::move(k)), v_bit(std::move(v_bit)), g_bit(g_bit)
            {
                int bin = this->k.size() - 1;
                always_assert(bin == this->v_bit.size());
            }

            // move constructor
            DCFBitKeySH(DCFBitKeySH &&other)
                : k(std::move(other.k)), v_bit(std::move(other.v_bit)), g_bit(other.g_bit)
            {
            }

            // move assignment
            DCFBitKeySH &operator=(DCFBitKeySH &&other)
            {
                k = std::move(other.k);
                v_bit = std::move(other.v_bit);
                g_bit = other.g_bit;
                return *this;
            }

            DCFBitKeySH() = default;
        };

        /// Semi-honest DCF generation - returns keys without MAC tags
        std::pair<DCFBitKeySH, DCFBitKeySH> dcfbit_gen_sh(int bin, const u64 alpha, const bool greaterThan = false);

        /// Semi-honest DCF evaluation - returns only the bit, no MAC tag
        u8 dcfbit_eval_sh(const DCFBitKeySH &key, const u64 &x, const bool greaterThan = false);
    }
}