#include "kernels.cuh"

#include <cuda_fp16.h>

#include <cmath>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace kimkvcache::cuda_model_detail {
namespace {

constexpr std::uint32_t kThreads = 256;

__global__ void embeddingKernel(
    __half const* weights,
    std::uint32_t token_id,
    __half* output,
    std::uint32_t hidden_size)
{
    std::uint32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < hidden_size) {
        output[index] = weights[
            static_cast<std::size_t>(token_id) * hidden_size + index
        ];
    }
}

__global__ void rmsNormKernel(
    __half const* input,
    __half const* weights,
    __half* output,
    std::uint32_t hidden_size,
    float epsilon)
{
    __shared__ float reduction[kThreads];
    float sum = 0.0F;
    for (std::uint32_t index = threadIdx.x;
         index < hidden_size;
         index += blockDim.x) {
        float const value = __half2float(input[index]);
        sum += value * value;
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float const scale = rsqrtf(
        reduction[0] / static_cast<float>(hidden_size) + epsilon
    );
    for (std::uint32_t index = threadIdx.x;
         index < hidden_size;
         index += blockDim.x) {
        output[index] = __float2half(
            __half2float(input[index]) * scale
            * __half2float(weights[index])
        );
    }
}

__global__ void ropeKernel(
    __half* values,
    std::uint32_t head_count,
    std::uint32_t head_dimension,
    std::uint32_t position,
    float theta)
{
    std::uint32_t const half_dimension = head_dimension / 2;
    std::uint32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t const element_count = head_count * half_dimension;
    if (index >= element_count) {
        return;
    }
    std::uint32_t const head = index / half_dimension;
    std::uint32_t const dimension = index % half_dimension;
    std::size_t const first =
        static_cast<std::size_t>(head) * head_dimension + dimension;
    std::size_t const second = first + half_dimension;
    float const inverse_frequency = powf(
        theta,
        -2.0F * static_cast<float>(dimension)
            / static_cast<float>(head_dimension)
    );
    float const angle = static_cast<float>(position) * inverse_frequency;
    float const cosine = cosf(angle);
    float const sine = sinf(angle);
    float const first_value = __half2float(values[first]);
    float const second_value = __half2float(values[second]);
    values[first] = __float2half(first_value * cosine - second_value * sine);
    values[second] = __float2half(second_value * cosine + first_value * sine);
}

__global__ void residualAddKernel(
    __half const* residual,
    __half const* update,
    __half* output,
    std::uint32_t element_count)
{
    std::uint32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count) {
        output[index] = __hadd(residual[index], update[index]);
    }
}

__global__ void swiGluKernel(
    __half const* gate,
    __half const* up,
    __half* output,
    std::uint32_t element_count)
{
    std::uint32_t const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < element_count) {
        float const gate_value = __half2float(gate[index]);
        float const silu = gate_value / (1.0F + expf(-gate_value));
        output[index] = __float2half(silu * __half2float(up[index]));
    }
}

struct ArgmaxValue final {
    float value;
    std::uint32_t index;
};

__device__ ArgmaxValue better(
    ArgmaxValue left,
    ArgmaxValue right) noexcept
{
    return right.value > left.value
            || (right.value == left.value && right.index < left.index)
        ? right : left;
}

__global__ void argmaxKernel(
    float const* logits,
    std::uint32_t vocabulary_size,
    std::uint32_t* output_token)
{
    __shared__ ArgmaxValue reduction[kThreads];
    ArgmaxValue local{-FLT_MAX, 0};
    for (std::uint32_t index = threadIdx.x;
         index < vocabulary_size;
         index += blockDim.x) {
        local = better(local, ArgmaxValue{logits[index], index});
    }
    reduction[threadIdx.x] = local;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = better(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output_token = reduction[0].index;
    }
}

[[nodiscard]] constexpr dim3 blocks(std::uint32_t elements) noexcept
{
    return dim3{(elements + kThreads - 1) / kThreads};
}

} // namespace

void launchEmbedding(
    KvScalar const* weights,
    std::uint32_t token_id,
    KvScalar* output,
    std::uint32_t hidden_size,
    cudaStream_t stream) noexcept
{
    embeddingKernel<<<blocks(hidden_size), kThreads, 0, stream>>>(
        reinterpret_cast<__half const*>(weights),
        token_id,
        reinterpret_cast<__half*>(output),
        hidden_size
    );
}

void launchRmsNorm(
    KvScalar const* input,
    KvScalar const* weights,
    KvScalar* output,
    std::uint32_t hidden_size,
    float epsilon,
    cudaStream_t stream) noexcept
{
    rmsNormKernel<<<1, kThreads, 0, stream>>>(
        reinterpret_cast<__half const*>(input),
        reinterpret_cast<__half const*>(weights),
        reinterpret_cast<__half*>(output),
        hidden_size,
        epsilon
    );
}

void launchRope(
    KvScalar* query,
    KvScalar* key,
    std::uint32_t query_heads,
    std::uint32_t kv_heads,
    std::uint32_t head_dimension,
    std::uint32_t position,
    float theta,
    cudaStream_t stream) noexcept
{
    std::uint32_t const query_elements = query_heads * head_dimension / 2;
    std::uint32_t const key_elements = kv_heads * head_dimension / 2;
    ropeKernel<<<blocks(query_elements), kThreads, 0, stream>>>(
        reinterpret_cast<__half*>(query), query_heads, head_dimension,
        position, theta
    );
    ropeKernel<<<blocks(key_elements), kThreads, 0, stream>>>(
        reinterpret_cast<__half*>(key), kv_heads, head_dimension,
        position, theta
    );
}

void launchResidualAdd(
    KvScalar const* residual,
    KvScalar const* update,
    KvScalar* output,
    std::uint32_t element_count,
    cudaStream_t stream) noexcept
{
    residualAddKernel<<<blocks(element_count), kThreads, 0, stream>>>(
        reinterpret_cast<__half const*>(residual),
        reinterpret_cast<__half const*>(update),
        reinterpret_cast<__half*>(output),
        element_count
    );
}

void launchSwiGlu(
    KvScalar const* gate,
    KvScalar const* up,
    KvScalar* output,
    std::uint32_t element_count,
    cudaStream_t stream) noexcept
{
    swiGluKernel<<<blocks(element_count), kThreads, 0, stream>>>(
        reinterpret_cast<__half const*>(gate),
        reinterpret_cast<__half const*>(up),
        reinterpret_cast<__half*>(output),
        element_count
    );
}

void launchArgmax(
    float const* logits,
    std::uint32_t vocabulary_size,
    std::uint32_t* output_token,
    cudaStream_t stream) noexcept
{
    argmaxKernel<<<1, kThreads, 0, stream>>>(
        logits, vocabulary_size, output_token
    );
}

} // namespace kimkvcache::cuda_model_detail
