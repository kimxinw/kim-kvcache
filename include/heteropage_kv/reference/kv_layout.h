#pragma once

#include "heteropage_kv/core/page_state.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace kimkvcache {

// FP16 的原始 16-bit 编码。Host API 不依赖 CUDA 的 __half 类型。
using KvScalar = std::uint16_t;

enum class KvComponent : std::uint8_t {
    Key = 0,
    Value = 1,
};

struct KvLayout final {
    std::uint32_t layer_count{0};
    std::uint32_t kv_head_count{0};
    std::uint32_t head_dimension{0};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return layer_count != 0
            && kv_head_count != 0
            && head_dimension != 0;
    }

    [[nodiscard]] bool elementsPerToken(std::size_t& result) const noexcept
    {
        if (!valid()) {
            return false;
        }

        std::size_t value = layer_count;
        if (!checkedMultiply(value, 2, value)
            || !checkedMultiply(value, kv_head_count, value)
            || !checkedMultiply(value, head_dimension, value)) {
            return false;
        }

        result = value;
        return true;
    }

    [[nodiscard]] bool elementsForTokens(
        std::uint32_t token_count,
        std::size_t& result) const noexcept
    {
        std::size_t per_token = 0;
        return elementsPerToken(per_token)
            && checkedMultiply(per_token, token_count, result);
    }

    [[nodiscard]] bool bytesForTokens(
        std::uint32_t token_count,
        std::size_t& result) const noexcept
    {
        std::size_t elements = 0;
        return elementsForTokens(token_count, elements)
            && checkedMultiply(elements, sizeof(KvScalar), result);
    }

    [[nodiscard]] bool pageElements(
        PageKind kind,
        std::size_t& result) const noexcept
    {
        return elementsForTokens(pageTokenCapacity(kind), result);
    }

    // [layer][K/V][token][head][dim]
    [[nodiscard]] std::size_t offset(
        std::uint32_t layer,
        KvComponent component,
        std::uint32_t token,
        std::uint32_t head,
        std::uint32_t dimension,
        std::uint32_t token_capacity) const noexcept
    {
        return (((static_cast<std::size_t>(layer) * 2
                    + static_cast<std::uint8_t>(component))
                    * token_capacity
                    + token)
                    * kv_head_count
                    + head)
                    * head_dimension
            + dimension;
    }

private:
    [[nodiscard]] static bool checkedMultiply(
        std::size_t lhs,
        std::size_t rhs,
        std::size_t& result) noexcept
    {
        if (lhs != 0
            && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
            return false;
        }

        result = lhs * rhs;
        return true;
    }
};

} // namespace kimkvcache
