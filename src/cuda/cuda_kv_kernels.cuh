#pragma once

#include "heteropage_kv/cuda/cuda_kv_storage.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace kimkvcache::cuda_detail {

inline constexpr unsigned int kThreadsPerBlock = 256;

struct DeviceLayout final {
    std::uint32_t layers;
    std::uint32_t heads;
    std::uint32_t dimensions;
};

struct DeviceBlockDescriptor final {
    KvScalar const* data;
    std::uint32_t logical_token_begin;
    std::uint16_t valid_tokens;
    std::uint16_t token_capacity;
};

struct PromotionSlots final {
    std::uint32_t slots[kPromotionSourcePageCount];
};

void launchCopyPageTokens(
    KvScalar const* source,
    std::uint32_t source_capacity,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t token_count,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream
);

void launchAppendTokens(
    KvScalar const* source,
    std::uint32_t source_token_capacity,
    std::uint32_t source_token_begin,
    KvScalar* target,
    std::uint32_t target_token_capacity,
    std::uint32_t target_token_begin,
    std::uint32_t segment_token_count,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream
);

void launchPromotion(
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    PromotionSlots source_slots,
    KvScalar* target,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream
);

void launchGather(
    DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar* output,
    std::uint32_t total_tokens,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream
);

void launchAttentionScores(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* query,
    float* scores,
    DeviceLayout layout,
    cudaStream_t stream
);

void launchAttentionOutput(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* scores,
    float* output,
    DeviceLayout layout,
    cudaStream_t stream
);

} // namespace kimkvcache::cuda_detail
