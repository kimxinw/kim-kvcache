#pragma once

#include <cstdint>
#include <string_view>

namespace kimkvcache{

enum class PageKind : uint8_t {
    Micro,
    Extent,
};

[[nodiscard]] constexpr bool isKnownPageKind(PageKind kind) noexcept{
    switch (kind)
    {
    case PageKind::Micro:
    case PageKind::Extent:
        /* code */
        return true;
    }
    return false;
}

[[nodiscard]] constexpr std::string_view toString(PageKind kind) noexcept{
    switch (kind) {
    case PageKind::Micro:
        return "micro";
    case PageKind::Extent:
        return "extent";
    }

    return "unknown";
}

enum class PagePoolError : std::uint8_t {
    None,
    Exhausted,
    InvalidHandle,
    WrongKind,
    OutOfRange,
    StaleGeneration,
    AlreadyFree,
  };

  [[nodiscard]] constexpr std::string_view toString(
      PagePoolError error) noexcept
  {
    switch (error) {
    case PagePoolError::None:
        return "none";
    case PagePoolError::Exhausted:
        return "exhausted";
    case PagePoolError::InvalidHandle:
        return "invalid_handle";
    case PagePoolError::WrongKind:
        return "wrong_kind";
    case PagePoolError::OutOfRange:
        return "out_of_range";
    case PagePoolError::StaleGeneration:
        return "stale_generation";
    case PagePoolError::AlreadyFree:
        return "already_free";
    }

    return "unknown";
  }

}//namespace kimkvcache
