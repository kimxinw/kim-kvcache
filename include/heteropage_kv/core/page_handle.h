#pragma once

#include "heteropage_kv/core/page_types.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace kimkvcache {
struct PageHandle final {
    static constexpr std::uint32_t kInvalidSlot =
        std::numeric_limits<std::uint32_t>::max();

    static constexpr std::uint32_t kInvalidGeneration = 0;

    PageKind kind {PageKind::Micro};
    std::uint32_t slot{kInvalidSlot};
    std::uint32_t generation{kInvalidGeneration};

    [[nodiscard]] static constexpr PageHandle invalid() noexcept{
        return {};
    }

    // 只检查 Handle 自身编码，不访问 PagePool。
    [[nodiscard]] constexpr bool isStructurallyValid() const noexcept
    {
        return isKnownPageKind(kind) &&
            slot != kInvalidSlot &&
            generation != kInvalidGeneration;
    }

    friend constexpr bool operator==(
        PageHandle lhs,
        PageHandle rhs) noexcept
    {
        return lhs.kind == rhs.kind &&
            lhs.slot == rhs.slot &&
            lhs.generation == rhs.generation;
    }

    friend constexpr bool operator!=(
        PageHandle lhs,
        PageHandle rhs) noexcept
    {
        return !(lhs == rhs);
    }
};

static_assert(
    std::is_trivially_copyable_v<PageHandle>,
    "PageHandle must remain trivially copyable"
);

static_assert(
    std::is_standard_layout_v<PageHandle>,
    "PageHandle must remain standard-layout"
);

}//namespace kimkvcache
