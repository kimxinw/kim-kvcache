#include "heteropage_kv/reference/kv_reference.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace kimkvcache {
namespace {

[[nodiscard]] std::uint32_t bitCastToUint(float value) noexcept
{
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be binary32");
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

[[nodiscard]] float bitCastToFloat(std::uint32_t bits) noexcept
{
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

} // namespace

KvScalar floatToKvScalar(float value) noexcept
{
    std::uint32_t const bits = bitCastToUint(value);
    std::uint32_t const sign = (bits >> 16U) & 0x8000U;
    std::uint32_t exponent = (bits >> 23U) & 0xFFU;
    std::uint32_t mantissa = bits & 0x7FFFFFU;

    if (exponent == 0xFFU) {
        if (mantissa == 0) {
            return static_cast<KvScalar>(sign | 0x7C00U);
        }

        return static_cast<KvScalar>(sign | 0x7E00U);
    }

    std::int32_t half_exponent =
        static_cast<std::int32_t>(exponent) - 127 + 15;

    if (half_exponent >= 31) {
        return static_cast<KvScalar>(sign | 0x7C00U);
    }

    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return static_cast<KvScalar>(sign);
        }

        mantissa |= 0x800000U;
        std::uint32_t const shift =
            static_cast<std::uint32_t>(14 - half_exponent);
        std::uint32_t half_mantissa = mantissa >> shift;
        std::uint32_t const remainder_mask = (1U << shift) - 1U;
        std::uint32_t const remainder = mantissa & remainder_mask;
        std::uint32_t const halfway = 1U << (shift - 1U);

        if (remainder > halfway
            || (remainder == halfway && (half_mantissa & 1U) != 0)) {
            ++half_mantissa;
        }

        return static_cast<KvScalar>(sign | half_mantissa);
    }

    std::uint32_t half_mantissa = mantissa >> 13U;
    std::uint32_t const remainder = mantissa & 0x1FFFU;

    if (remainder > 0x1000U
        || (remainder == 0x1000U && (half_mantissa & 1U) != 0)) {
        ++half_mantissa;

        if (half_mantissa == 0x400U) {
            half_mantissa = 0;
            ++half_exponent;

            if (half_exponent >= 31) {
                return static_cast<KvScalar>(sign | 0x7C00U);
            }
        }
    }

    return static_cast<KvScalar>(
        sign
        | (static_cast<std::uint32_t>(half_exponent) << 10U)
        | half_mantissa
    );
}

float kvScalarToFloat(KvScalar value) noexcept
{
    std::uint32_t const sign =
        (static_cast<std::uint32_t>(value) & 0x8000U) << 16U;
    std::uint32_t exponent =
        (static_cast<std::uint32_t>(value) >> 10U) & 0x1FU;
    std::uint32_t mantissa =
        static_cast<std::uint32_t>(value) & 0x3FFU;
    std::uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            std::int32_t normalized_exponent = -14;

            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1U;
                --normalized_exponent;
            }

            mantissa &= 0x3FFU;
            bits = sign
                | (static_cast<std::uint32_t>(normalized_exponent + 127)
                    << 23U)
                | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign
            | ((exponent + 112U) << 23U)
            | (mantissa << 13U);
    }

    return bitCastToFloat(bits);
}

bool referenceAttention(
    KvLayout const& layout,
    std::vector<KvScalar> const& contiguous_kv,
    std::uint32_t token_count,
    std::vector<float> const& query,
    std::vector<float>& output)
{
    std::size_t kv_elements = 0;
    std::size_t const query_elements =
        static_cast<std::size_t>(layout.layer_count)
        * layout.kv_head_count
        * layout.head_dimension;

    if (token_count == 0
        || !layout.elementsForTokens(token_count, kv_elements)
        || contiguous_kv.size() != kv_elements
        || query.size() != query_elements) {
        return false;
    }

    output.assign(query_elements, 0.0F);
    std::vector<float> scores(token_count, 0.0F);
    float const scale = 1.0F
        / std::sqrt(static_cast<float>(layout.head_dimension));

    for (std::uint32_t layer = 0;
         layer < layout.layer_count;
         ++layer) {
        for (std::uint32_t head = 0;
             head < layout.kv_head_count;
             ++head) {
            std::size_t const query_base =
                (static_cast<std::size_t>(layer)
                    * layout.kv_head_count + head)
                * layout.head_dimension;
            float maximum = -std::numeric_limits<float>::infinity();

            for (std::uint32_t token = 0;
                 token < token_count;
                 ++token) {
                float dot = 0.0F;

                for (std::uint32_t dimension = 0;
                     dimension < layout.head_dimension;
                     ++dimension) {
                    std::size_t const key_offset = layout.offset(
                        layer,
                        KvComponent::Key,
                        token,
                        head,
                        dimension,
                        token_count
                    );
                    dot += query[query_base + dimension]
                        * kvScalarToFloat(contiguous_kv[key_offset]);
                }

                scores[token] = dot * scale;
                maximum = std::max(maximum, scores[token]);
            }

            float denominator = 0.0F;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }

            if (!(denominator > 0.0F) || !std::isfinite(denominator)) {
                return false;
            }

            for (std::uint32_t dimension = 0;
                 dimension < layout.head_dimension;
                 ++dimension) {
                float weighted_value = 0.0F;

                for (std::uint32_t token = 0;
                     token < token_count;
                     ++token) {
                    std::size_t const value_offset = layout.offset(
                        layer,
                        KvComponent::Value,
                        token,
                        head,
                        dimension,
                        token_count
                    );
                    weighted_value += scores[token]
                        * kvScalarToFloat(contiguous_kv[value_offset]);
                }

                output[query_base + dimension] =
                    weighted_value / denominator;
            }
        }
    }

    return true;
}

} // namespace kimkvcache
