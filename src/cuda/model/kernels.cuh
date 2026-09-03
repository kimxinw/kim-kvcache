#pragma once

#include "kim-kv/core/kv_layout.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace kimkvcache::cuda_model_detail {

void launchEmbedding(
    KvScalar const* weights,
    std::uint32_t token_id,
    KvScalar* output,
    std::uint32_t hidden_size,
    cudaStream_t stream
) noexcept;

void launchRmsNorm(
    KvScalar const* input,
    KvScalar const* weights,
    KvScalar* output,
    std::uint32_t hidden_size,
    float epsilon,
    cudaStream_t stream
) noexcept;

void launchRope(
    KvScalar* query,
    KvScalar* key,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dimension,
    std::uint32_t position,
    float theta,
    cudaStream_t stream
) noexcept;

void launchResidualAdd(
    KvScalar const* residual,
    KvScalar const* update,
    KvScalar* output,
    std::uint32_t element_count,
    cudaStream_t stream
) noexcept;

void launchSwiGlu(
    KvScalar const* gate,
    KvScalar const* up,
    KvScalar* output,
    std::uint32_t element_count,
    cudaStream_t stream
) noexcept;

void launchArgmax(
    float const* logits,
    std::uint32_t vocabulary_size,
    std::uint32_t* output_token,
    cudaStream_t stream
) noexcept;

} // namespace kimkvcache::cuda_model_detail
