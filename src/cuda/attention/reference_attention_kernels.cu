#include "../kv/kernels.cuh"

#include <cuda_fp16.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kimkvcache::cuda_detail {
namespace {

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
    // Preserve the barrier before reusing the reduction buffer.
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

} // namespace

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
