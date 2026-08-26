#pragma once

#include "heteropage_kv/core/page_handle.h"
#include "heteropage_kv/core/page_state.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace kimkvcache {

using RequestId = std::uint64_t;
using PromotionId = std::uint64_t;

constexpr RequestId kInvalidRequestId = 0;
constexpr PromotionId kInvalidPromotionId = 0;
constexpr std::size_t kPromotionSourcePageCount = 8;

struct PromotionPrepareResult final {
    KvCacheError error{KvCacheError::None};
    PromotionId promotion_id{kInvalidPromotionId};
    RequestId request_id{kInvalidRequestId};
    std::uint32_t logical_token_begin{0};
    std::array<PageHandle, kPromotionSourcePageCount> source_handles{};
    PageHandle target_handle{PageHandle::invalid()};

    [[nodiscard]] constexpr bool ok() const noexcept
    {
        return error == KvCacheError::None;
    }

    [[nodiscard]] explicit constexpr operator bool() const noexcept
    {
        return ok();
    }
};

struct PromotionTransactionSnapshot final {
    PromotionId promotion_id{kInvalidPromotionId};
    RequestId request_id{kInvalidRequestId};
    std::uint64_t prepared_table_version{0};
    std::uint32_t logical_token_begin{0};
    std::array<PageHandle, kPromotionSourcePageCount> source_handles{};
    PageHandle target_handle{PageHandle::invalid()};
};

} // namespace kimkvcache
