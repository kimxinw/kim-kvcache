#pragma once

#include "kim-kv/core/page_types.h"

#include <cstdint>
#include <string_view>

namespace kimkvcache {
    constexpr std::uint16_t kMicroPageTokenCapacity = 8;
    constexpr std::uint16_t kExtentPageTokenCapacity = 64;

    [[nodiscard]] constexpr std::uint16_t pageTokenCapacity(
        PageKind kind) noexcept
    {
        switch (kind) {
        case PageKind::Micro:
            return kMicroPageTokenCapacity;
        case PageKind::Extent:
            return kExtentPageTokenCapacity;
        }

        return 0;
    }

    enum class PageState : std::uint8_t {
      Free,
      Mutable,
      Sealed,
      CopyTarget,
      Retiring,
    };

    [[nodiscard]] constexpr std::string_view toString(
        PageState state) noexcept
    {
        switch (state) {
        case PageState::Free:
            return "free";
        case PageState::Mutable:
            return "mutable";
        case PageState::Sealed:
            return "sealed";
        case PageState::CopyTarget:
            return "copy_target";
        case PageState::Retiring:
            return "retiring";
        }

        return "unknown";
    }

    enum class KvCacheError : std::uint8_t {
        None,
        InvalidArgument,
        RequestNotFound,
        RequestAlreadyExists,
        ResourceExhausted,
        InvalidState,
        PromotionNotEligible,
        PromotionNotFound,
        PromotionConflict,
        InternalInvariantViolation,
    };

    [[nodiscard]] constexpr std::string_view toString(
        KvCacheError error) noexcept
    {
        switch (error) {
        case KvCacheError::None:
            return "none";
        case KvCacheError::InvalidArgument:
            return "invalid_argument";
        case KvCacheError::RequestNotFound:
            return "request_not_found";
        case KvCacheError::RequestAlreadyExists:
            return "request_already_exists";
        case KvCacheError::ResourceExhausted:
            return "resource_exhausted";
        case KvCacheError::InvalidState:
            return "invalid_state";
        case KvCacheError::PromotionNotEligible:
            return "promotion_not_eligible";
        case KvCacheError::PromotionNotFound:
            return "promotion_not_found";
        case KvCacheError::PromotionConflict:
            return "promotion_conflict";
        case KvCacheError::InternalInvariantViolation:
            return "internal_invariant_violation";
        }

        return "unknown";
    }

}//namespace kimkvcache
