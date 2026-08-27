#include "cuda_kv_kernels.cuh"

#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kimkvcache::cuda_detail {
namespace {
__device__ void decodeElement(
    std::size_t index,
    DeviceLayout layout,
    std::uint32_t token_count,
    std::uint32_t& layer,
    std::uint32_t& component,
    std::uint32_t& token,
    std::uint32_t& head,
    std::uint32_t& dimension)
{
    dimension = static_cast<std::uint32_t>(index % layout.dimensions);
    index /= layout.dimensions;
    head = static_cast<std::uint32_t>(index % layout.heads);
    index /= layout.heads;
    token = static_cast<std::uint32_t>(index % token_count);
    index /= token_count;
    component = static_cast<std::uint32_t>(index % 2);
    index /= 2;
    layer = static_cast<std::uint32_t>(index);
}

__device__ std::size_t tensorOffset(
    DeviceLayout layout,
    std::uint32_t layer,
    std::uint32_t component,
    std::uint32_t token,
    std::uint32_t head,
    std::uint32_t dimension,
    std::uint32_t token_capacity)
{
    return (((static_cast<std::size_t>(layer) * 2 + component)
                * token_capacity + token)
                * layout.heads + head)
                * layout.dimensions
        + dimension;
}

__global__ void copyPageTokensKernel(
    KvScalar const* source,
    std::uint32_t source_capacity,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t token_count,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= element_count) {
        return;
    }

    std::uint32_t layer = 0;
    std::uint32_t component = 0;
    std::uint32_t token = 0;
    std::uint32_t head = 0;
    std::uint32_t dimension = 0;
    decodeElement(
        index,
        layout,
        token_count,
        layer,
        component,
        token,
        head,
        dimension
    );

    target[tensorOffset(
        layout,
        layer,
        component,
        token,
        head,
        dimension,
        target_capacity
    )] = source[tensorOffset(
        layout,
        layer,
        component,
        token,
        head,
        dimension,
        source_capacity
    )];
}

__global__ void appendTokensKernel(
    KvScalar const* source,
    std::uint32_t source_token_capacity,
    std::uint32_t source_token_begin,
    KvScalar* target,
    std::uint32_t target_token_capacity,
    std::uint32_t target_token_begin,
    std::uint32_t segment_token_count,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= element_count) {
        return;
    }

    std::uint32_t layer = 0;
    std::uint32_t component = 0;
    std::uint32_t token = 0;
    std::uint32_t head = 0;
    std::uint32_t dimension = 0;
    decodeElement(
        index,
        layout,
        segment_token_count,
        layer,
        component,
        token,
        head,
        dimension
    );

    target[tensorOffset(
        layout,
        layer,
        component,
        target_token_begin + token,
        head,
        dimension,
        target_token_capacity
    )] = source[tensorOffset(
        layout,
        layer,
        component,
        source_token_begin + token,
        head,
        dimension,
        source_token_capacity
    )];
}

__global__ void promotionKernel(
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    PromotionSlots source_slots,
    KvScalar* target,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= element_count) {
        return;
    }

    std::uint32_t layer = 0;
    std::uint32_t component = 0;
    std::uint32_t token = 0;
    std::uint32_t head = 0;
    std::uint32_t dimension = 0;
    decodeElement(
        index,
        layout,
        kExtentPageTokenCapacity,
        layer,
        component,
        token,
        head,
        dimension
    );

    std::uint32_t const source_page = token / kMicroPageTokenCapacity;
    std::uint32_t const source_token = token % kMicroPageTokenCapacity;
    KvScalar const* source = micro_pool
        + static_cast<std::size_t>(source_slots.slots[source_page])
            * micro_page_elements;

    target[tensorOffset(
        layout,
        layer,
        component,
        token,
        head,
        dimension,
        kExtentPageTokenCapacity
    )] = source[tensorOffset(
        layout,
        layer,
        component,
        source_token,
        head,
        dimension,
        kMicroPageTokenCapacity
    )];
}

__global__ void gatherKernel(
    DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar* output,
    std::uint32_t total_tokens,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= element_count) {
        return;
    }

    std::uint32_t layer = 0;
    std::uint32_t component = 0;
    std::uint32_t token = 0;
    std::uint32_t head = 0;
    std::uint32_t dimension = 0;
    decodeElement(
        index,
        layout,
        total_tokens,
        layer,
        component,
        token,
        head,
        dimension
    );

    DeviceBlockDescriptor descriptor{};
    bool found = false;

    for (std::uint32_t descriptor_index = 0;
         descriptor_index < descriptor_count;
         ++descriptor_index) {
        DeviceBlockDescriptor const candidate =
            descriptors[descriptor_index];
        std::uint32_t const end = candidate.logical_token_begin
            + candidate.valid_tokens;

        if (token >= candidate.logical_token_begin && token < end) {
            descriptor = candidate;
            found = true;
            break;
        }
    }

    if (!found) {
        return;
    }

    std::uint32_t const page_token = token - descriptor.logical_token_begin;
    output[index] = descriptor.data[tensorOffset(
        layout,
        layer,
        component,
        page_token,
        head,
        dimension,
        descriptor.token_capacity
    )];
}

__global__ void attentionScoresKernel(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* query,
    float* scores,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    std::uint32_t const layer_head = blockIdx.x;
    std::uint32_t const layer = layer_head / layout.heads;
    std::uint32_t const head = layer_head % layout.heads;
    std::size_t const query_base =
        static_cast<std::size_t>(layer_head) * layout.dimensions;
    float const scale = rsqrtf(static_cast<float>(layout.dimensions));

    for (std::uint32_t token = 0; token < token_count; ++token) {
        float partial = 0.0F;

        for (std::uint32_t dimension = threadIdx.x;
             dimension < layout.dimensions;
             dimension += blockDim.x) {
            std::size_t const key_offset = tensorOffset(
                layout,
                layer,
                0,
                token,
                head,
                dimension,
                token_count
            );
            __half const key =
                reinterpret_cast<__half const*>(contiguous_kv)[key_offset];
            partial += query[query_base + dimension] * __half2float(key);
        }

        reduction[threadIdx.x] = partial;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2;
             stride != 0;
             stride /= 2) {
            if (threadIdx.x < stride) {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            scores[static_cast<std::size_t>(layer_head) * token_count
                + token] = reduction[0] * scale;
        }
        __syncthreads();
    }
}

__global__ void attentionOutputKernel(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* scores,
    float* output,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    std::uint32_t const layer_head = blockIdx.x;
    std::uint32_t const layer = layer_head / layout.heads;
    std::uint32_t const head = layer_head % layout.heads;
    float const* block_scores = scores
        + static_cast<std::size_t>(layer_head) * token_count;

    float local_maximum = -FLT_MAX;
    for (std::uint32_t token = threadIdx.x;
         token < token_count;
         token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, block_scores[token]);
    }

    reduction[threadIdx.x] = local_maximum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x],
                reduction[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }

    float const maximum = reduction[0];
    // Every thread must cache the maximum before the shared reduction buffer
    // is reused for the softmax sum. Without this barrier, faster threads can
    // overwrite reduction[0] while slower threads are still reading it.
    __syncthreads();
    float local_sum = 0.0F;
    for (std::uint32_t token = threadIdx.x;
         token < token_count;
         token += blockDim.x) {
        local_sum += expf(block_scores[token] - maximum);
    }

    reduction[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float const denominator = reduction[0];
    std::size_t const output_base =
        static_cast<std::size_t>(layer_head) * layout.dimensions;

    for (std::uint32_t dimension = threadIdx.x;
         dimension < layout.dimensions;
         dimension += blockDim.x) {
        float weighted_value = 0.0F;

        for (std::uint32_t token = 0; token < token_count; ++token) {
            std::size_t const value_offset = tensorOffset(
                layout,
                layer,
                1,
                token,
                head,
                dimension,
                token_count
            );
            __half const value =
                reinterpret_cast<__half const*>(contiguous_kv)[value_offset];
            weighted_value += expf(block_scores[token] - maximum)
                * __half2float(value);
        }

        output[output_base + dimension] = weighted_value / denominator;
    }
}

[[nodiscard]] dim3 gridFor(std::size_t element_count) noexcept
{
    std::size_t const blocks =
        (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return dim3(static_cast<unsigned int>(blocks));
}

} // namespace

void launchCopyPageTokens(
    KvScalar const* source,
    std::uint32_t source_capacity,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t token_count,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream)
{
    copyPageTokensKernel<<<gridFor(element_count), kThreadsPerBlock, 0, stream>>>(
        source,
        source_capacity,
        target,
        target_capacity,
        token_count,
        layout,
        element_count
    );
}

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
    cudaStream_t stream)
{
    appendTokensKernel<<<gridFor(element_count), kThreadsPerBlock, 0, stream>>>(
        source,
        source_token_capacity,
        source_token_begin,
        target,
        target_token_capacity,
        target_token_begin,
        segment_token_count,
        layout,
        element_count
    );
}

void launchPromotion(
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    PromotionSlots source_slots,
    KvScalar* target,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream)
{
    promotionKernel<<<gridFor(element_count), kThreadsPerBlock, 0, stream>>>(
        micro_pool,
        micro_page_elements,
        source_slots,
        target,
        layout,
        element_count
    );
}

void launchGather(
    DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar* output,
    std::uint32_t total_tokens,
    DeviceLayout layout,
    std::size_t element_count,
    cudaStream_t stream)
{
    gatherKernel<<<gridFor(element_count), kThreadsPerBlock, 0, stream>>>(
        descriptors,
        descriptor_count,
        output,
        total_tokens,
        layout,
        element_count
    );
}

void launchAttentionScores(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* query,
    float* scores,
    DeviceLayout layout,
    cudaStream_t stream)
{
    std::uint32_t const blocks = layout.layers * layout.heads;
    std::size_t const shared_bytes = kThreadsPerBlock * sizeof(float);
    attentionScoresKernel<<<blocks, kThreadsPerBlock, shared_bytes, stream>>>(
        contiguous_kv,
        token_count,
        query,
        scores,
        layout
    );
}

void launchAttentionOutput(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* scores,
    float* output,
    DeviceLayout layout,
    cudaStream_t stream)
{
    std::uint32_t const blocks = layout.layers * layout.heads;
    std::size_t const shared_bytes = kThreadsPerBlock * sizeof(float);
    attentionOutputKernel<<<blocks, kThreadsPerBlock, shared_bytes, stream>>>(
        contiguous_kv,
        token_count,
        scores,
        output,
        layout
    );
}

} // namespace kimkvcache::cuda_detail
