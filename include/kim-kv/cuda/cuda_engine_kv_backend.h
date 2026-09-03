#pragma once

#include "kim-kv/cuda/cuda_status.h"
#include "kim-kv/engine/engine_kv.h"

#include <cstdint>
#include <memory>

namespace kimkvcache {

[[nodiscard]] std::unique_ptr<EngineKvBackend>
createHeterogeneousCudaEngineKvBackend(
    EngineKvConfig config,
    std::uint32_t micro_page_capacity,
    std::uint32_t extent_page_capacity
);

[[nodiscard]] std::unique_ptr<EngineKvBackend>
createFixedCudaEngineKvBackend(
    EngineKvConfig config,
    std::uint16_t tokens_per_page,
    std::uint32_t page_capacity
);

// Deterministic one-shot fault hook used by Engine error-path contracts.
[[nodiscard]] bool injectCudaEngineFailureOnce(
    EngineKvBackend& backend,
    CudaFailurePoint point
) noexcept;

} // namespace kimkvcache
