#pragma once

#include "kim-kv/cuda/cuda_model_runner.h"
#include "kim-kv/model/weight_manifest.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace kimkvcache::cuda_model_runner_detail {

struct WorkspaceLayout final {
    std::size_t hidden{0};
    std::size_t normalized{0};
    std::size_t query{0};
    std::size_t key{0};
    std::size_t value{0};
    std::size_t attention{0};
    std::size_t projection{0};
    std::size_t gate{0};
    std::size_t up{0};
    std::size_t activated{0};
    std::size_t logits{0};
    std::size_t greedy_token{0};
    std::size_t attention_scores{0};
    std::size_t bytes{0};
};

[[nodiscard]] CudaModelRunnerStatus failure(
    CudaModelRunnerError error,
    int native_code,
    std::string detail
);

[[nodiscard]] CudaModelRunnerStatus cudaFailure(
    cudaError_t error,
    std::string detail
);

[[nodiscard]] CudaModelRunnerStatus cublasFailure(
    cublasStatus_t status,
    std::string detail
);

[[nodiscard]] CudaModelRunnerStatus kvFailure(
    EngineKvStatus status,
    std::string detail
);

[[nodiscard]] bool makeWorkspaceLayout(
    TinyLlamaConfig const& config,
    WorkspaceLayout& layout
) noexcept;

} // namespace kimkvcache::cuda_model_runner_detail

namespace kimkvcache {

struct CudaTinyLlamaModelRunner::Impl final {
    WeightManifest manifest{};
    EngineKvBackend* backend{nullptr};
    cudaStream_t stream{nullptr};
    cublasHandle_t cublas{nullptr};
    std::uint8_t* device_weights{nullptr};
    std::uint8_t* device_workspace{nullptr};
    cuda_model_runner_detail::WorkspaceLayout workspace_layout{};

    ~Impl();

    [[nodiscard]] KvScalar const* weight(
        std::string const& name
    ) const noexcept;

    template <typename Value>
    [[nodiscard]] Value* workspace(std::size_t offset) const noexcept
    {
        return reinterpret_cast<Value*>(device_workspace + offset);
    }

    [[nodiscard]] CudaModelRunnerStatus checkLaunch(
        std::string const& operation
    ) const;

    [[nodiscard]] CudaModelRunnerStatus gemv(
        KvScalar const* matrix,
        KvScalar const* input,
        KvScalar* output,
        std::uint32_t rows,
        std::uint32_t columns,
        std::string const& operation
    ) const;

    [[nodiscard]] CudaModelRunnerStatus logitsGemv(
        KvScalar const* matrix,
        KvScalar const* input,
        float* output
    ) const;
};

} // namespace kimkvcache
