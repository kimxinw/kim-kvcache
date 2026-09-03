#pragma once

#include "kim-kv/core/page_state.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace kimkvcache {

// Opaque stream handle keeps public headers independent of CUDA SDK headers.
// nullptr denotes the CUDA default stream.
using CudaStream = void*;

struct CudaStorageSnapshot final {
    std::uint32_t micro_capacity{0};
    std::uint32_t extent_capacity{0};
    std::uint16_t micro_page_tokens{kMicroPageTokenCapacity};
    std::uint16_t extent_page_tokens{kExtentPageTokenCapacity};
    std::size_t micro_reserved_bytes{0};
    std::size_t extent_reserved_bytes{0};

    [[nodiscard]] constexpr std::size_t totalReservedBytes() const noexcept
    {
        return micro_reserved_bytes + extent_reserved_bytes;
    }
};

enum class CudaError : std::uint8_t {
    None,
    InvalidArgument,
    RuntimeUnavailable,
    AllocationFailed,
    SubmissionFailed,
    NotReady,
    ExecutionFailed,
    InternalError,
};

[[nodiscard]] constexpr std::string_view toString(CudaError error) noexcept
{
    switch (error) {
    case CudaError::None:
        return "none";
    case CudaError::InvalidArgument:
        return "invalid_argument";
    case CudaError::RuntimeUnavailable:
        return "runtime_unavailable";
    case CudaError::AllocationFailed:
        return "allocation_failed";
    case CudaError::SubmissionFailed:
        return "submission_failed";
    case CudaError::NotReady:
        return "not_ready";
    case CudaError::ExecutionFailed:
        return "execution_failed";
    case CudaError::InternalError:
        return "internal_error";
    }

    return "unknown";
}

struct CudaStatus final {
    CudaError error{CudaError::None};
    int native_error{0};

    [[nodiscard]] constexpr bool ok() const noexcept
    {
        return error == CudaError::None;
    }

    [[nodiscard]] explicit constexpr operator bool() const noexcept
    {
        return ok();
    }
};

enum class CudaFailurePoint : std::uint8_t {
    Submission,
    Completion,
};

} // namespace kimkvcache
