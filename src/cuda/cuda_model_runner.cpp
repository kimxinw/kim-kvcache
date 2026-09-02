#include "kim-kv/cuda/cuda_model_runner.h"

#include "cuda_model_kernels.cuh"
#include "kim-kv/model/weight_manifest.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kimkvcache {
namespace {

constexpr std::size_t kAlignment = 256;

[[nodiscard]] CudaModelRunnerStatus failure(
    CudaModelRunnerError error,
    int native_code,
    std::string detail)
{
    return CudaModelRunnerStatus{error, native_code, std::move(detail)};
}

[[nodiscard]] CudaModelRunnerStatus cudaFailure(
    cudaError_t error,
    std::string detail)
{
    CudaModelRunnerError const kind = error == cudaErrorMemoryAllocation
        ? CudaModelRunnerError::AllocationFailed
        : CudaModelRunnerError::SubmissionFailed;
    return failure(kind, static_cast<int>(error), std::move(detail));
}

[[nodiscard]] CudaModelRunnerStatus cublasFailure(
    cublasStatus_t status,
    std::string detail)
{
    return failure(
        CudaModelRunnerError::CublasFailed,
        static_cast<int>(status),
        std::move(detail)
    );
}

[[nodiscard]] CudaModelRunnerStatus kvFailure(
    EngineKvStatus status,
    std::string detail)
{
    CudaModelRunnerError kind = CudaModelRunnerError::KvBackendFailed;
    if (status.error == EngineKvError::SubmissionFailed) {
        kind = CudaModelRunnerError::SubmissionFailed;
    } else if (status.error == EngineKvError::ExecutionFailed) {
        kind = CudaModelRunnerError::ExecutionFailed;
    }
    detail += ": ";
    detail += toString(status.error);
    return failure(kind, static_cast<int>(status.error), std::move(detail));
}

[[nodiscard]] std::size_t aligned(std::size_t value) noexcept
{
    return (value + kAlignment - 1) & ~(kAlignment - 1);
}

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

[[nodiscard]] bool addRegion(
    std::size_t bytes,
    std::size_t& cursor,
    std::size_t& offset) noexcept
{
    cursor = aligned(cursor);
    offset = cursor;
    if (bytes > std::numeric_limits<std::size_t>::max() - cursor) {
        return false;
    }
    cursor += bytes;
    return true;
}

[[nodiscard]] bool makeWorkspaceLayout(
    TinyLlamaConfig const& config,
    WorkspaceLayout& layout) noexcept
{
    std::size_t cursor = 0;
    std::size_t const hidden_bytes =
        static_cast<std::size_t>(config.hidden_size) * sizeof(KvScalar);
    std::size_t const kv_bytes = static_cast<std::size_t>(
        config.kv_head_count * config.head_dimension
    ) * sizeof(KvScalar);
    std::size_t const intermediate_bytes =
        static_cast<std::size_t>(config.intermediate_size)
        * sizeof(KvScalar);
    std::size_t const logits_bytes =
        static_cast<std::size_t>(config.vocabulary_size) * sizeof(float);
    std::size_t const score_bytes =
        static_cast<std::size_t>(config.attention_head_count)
        * config.max_position_embeddings * sizeof(float);
    bool const valid = addRegion(hidden_bytes, cursor, layout.hidden)
        && addRegion(hidden_bytes, cursor, layout.normalized)
        && addRegion(hidden_bytes, cursor, layout.query)
        && addRegion(kv_bytes, cursor, layout.key)
        && addRegion(kv_bytes, cursor, layout.value)
        && addRegion(hidden_bytes, cursor, layout.attention)
        && addRegion(hidden_bytes, cursor, layout.projection)
        && addRegion(intermediate_bytes, cursor, layout.gate)
        && addRegion(intermediate_bytes, cursor, layout.up)
        && addRegion(intermediate_bytes, cursor, layout.activated)
        && addRegion(logits_bytes, cursor, layout.logits)
        && addRegion(sizeof(std::uint32_t), cursor, layout.greedy_token)
        && addRegion(score_bytes, cursor, layout.attention_scores);
    if (valid) {
        layout.bytes = aligned(cursor);
    }
    return valid && layout.bytes != 0;
}

} // namespace

struct CudaTinyLlamaModelRunner::Impl final {
    WeightManifest manifest{};
    EngineKvBackend* backend{nullptr};
    cudaStream_t stream{nullptr};
    cublasHandle_t cublas{nullptr};
    std::uint8_t* device_weights{nullptr};
    std::uint8_t* device_workspace{nullptr};
    WorkspaceLayout workspace_layout{};

    ~Impl()
    {
        if (stream != nullptr) {
            static_cast<void>(cudaStreamSynchronize(stream));
        }
        if (device_workspace != nullptr) {
            static_cast<void>(cudaFree(device_workspace));
        }
        if (device_weights != nullptr) {
            static_cast<void>(cudaFree(device_weights));
        }
        if (cublas != nullptr) {
            static_cast<void>(cublasDestroy(cublas));
        }
    }

    [[nodiscard]] KvScalar const* weight(std::string const& name) const noexcept
    {
        WeightTensorDescriptor const* tensor = manifest.find(name);
        return tensor == nullptr ? nullptr
            : reinterpret_cast<KvScalar const*>(
                device_weights + tensor->offset
            );
    }

    template <typename Value>
    [[nodiscard]] Value* workspace(std::size_t offset) const noexcept
    {
        return reinterpret_cast<Value*>(device_workspace + offset);
    }

    [[nodiscard]] CudaModelRunnerStatus checkLaunch(
        std::string const& operation) const
    {
        cudaError_t const error = cudaGetLastError();
        return error == cudaSuccess
            ? CudaModelRunnerStatus{}
            : cudaFailure(error, operation);
    }

    [[nodiscard]] CudaModelRunnerStatus gemv(
        KvScalar const* matrix,
        KvScalar const* input,
        KvScalar* output,
        std::uint32_t rows,
        std::uint32_t columns,
        std::string const& operation) const
    {
        float const alpha = 1.0F;
        float const beta = 0.0F;
        cublasStatus_t const status = cublasGemmEx(
            cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(rows),
            1,
            static_cast<int>(columns),
            &alpha,
            matrix,
            CUDA_R_16F,
            static_cast<int>(columns),
            input,
            CUDA_R_16F,
            static_cast<int>(columns),
            &beta,
            output,
            CUDA_R_16F,
            static_cast<int>(rows),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        return status == CUBLAS_STATUS_SUCCESS
            ? CudaModelRunnerStatus{}
            : cublasFailure(status, operation);
    }

    [[nodiscard]] CudaModelRunnerStatus logitsGemv(
        KvScalar const* matrix,
        KvScalar const* input,
        float* output) const
    {
        float const alpha = 1.0F;
        float const beta = 0.0F;
        TinyLlamaConfig const& config = manifest.config;
        cublasStatus_t const status = cublasGemmEx(
            cublas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(config.vocabulary_size),
            1,
            static_cast<int>(config.hidden_size),
            &alpha,
            matrix,
            CUDA_R_16F,
            static_cast<int>(config.hidden_size),
            input,
            CUDA_R_16F,
            static_cast<int>(config.hidden_size),
            &beta,
            output,
            CUDA_R_32F,
            static_cast<int>(config.vocabulary_size),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );
        return status == CUBLAS_STATUS_SUCCESS
            ? CudaModelRunnerStatus{}
            : cublasFailure(status, "lm_head");
    }
};

CudaTinyLlamaModelRunner::CudaTinyLlamaModelRunner(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl))
{
}

CudaTinyLlamaModelRunner::~CudaTinyLlamaModelRunner() = default;
CudaTinyLlamaModelRunner::CudaTinyLlamaModelRunner(
    CudaTinyLlamaModelRunner&&) noexcept = default;
CudaTinyLlamaModelRunner& CudaTinyLlamaModelRunner::operator=(
    CudaTinyLlamaModelRunner&&) noexcept = default;

TinyLlamaConfig CudaTinyLlamaModelRunner::config() const noexcept
{
    return impl_ != nullptr ? impl_->manifest.config : TinyLlamaConfig{};
}

std::uint64_t CudaTinyLlamaModelRunner::deviceWeightBytes() const noexcept
{
    return impl_ != nullptr ? impl_->manifest.data_bytes : 0;
}

std::uint64_t CudaTinyLlamaModelRunner::deviceWorkspaceBytes() const noexcept
{
    return impl_ != nullptr ? impl_->workspace_layout.bytes : 0;
}

ModelTokenResult CudaTinyLlamaModelRunner::forwardToken(
    RequestId request_id,
    std::uint32_t token_id,
    std::uint32_t expected_position,
    ModelRunnerDebugCapture* debug_capture,
    ModelRunnerOutputOptions output_options)
{
    ModelTokenResult result;
    if (impl_ == nullptr || impl_->backend == nullptr) {
        result.status = failure(
            CudaModelRunnerError::InvalidState, 0, "runner is empty"
        );
        return result;
    }
    TinyLlamaConfig const& config = impl_->manifest.config;
    if (request_id == kInvalidRequestId
        || token_id >= config.vocabulary_size
        || expected_position >= config.max_position_embeddings) {
        result.status = failure(
            CudaModelRunnerError::InvalidArgument,
            0,
            "request, token, or position is out of range"
        );
        return result;
    }

    std::unordered_map<std::uint32_t, std::size_t> capture_indices;
    try {
        if (output_options.copy_logits) {
            result.logits.resize(config.vocabulary_size);
        }
        if (debug_capture != nullptr) {
            debug_capture->embedding.resize(config.hidden_size);
            debug_capture->final_hidden_state.resize(config.hidden_size);
            debug_capture->layer_hidden_states.clear();
            debug_capture->layer_hidden_states.resize(
                debug_capture->layer_indices.size(),
                std::vector<KvScalar>(config.hidden_size)
            );
            for (std::size_t index = 0;
                 index < debug_capture->layer_indices.size();
                 ++index) {
                std::uint32_t const layer =
                    debug_capture->layer_indices[index];
                if (layer >= config.layer_count
                    || !capture_indices.emplace(layer, index).second) {
                    result.status = failure(
                        CudaModelRunnerError::InvalidArgument,
                        0,
                        "debug layer is duplicated or out of range"
                    );
                    return result;
                }
            }
        }
    } catch (std::bad_alloc const&) {
        result.status = failure(
            CudaModelRunnerError::AllocationFailed,
            0,
            "host output allocation failed"
        );
        return result;
    }

    TokenReserveResult reserved = impl_->backend->reserveToken(
        ReserveTokenRequest{
            request_id,
            expected_position,
            reinterpret_cast<EngineStream>(impl_->stream),
        }
    );
    if (!reserved.ok()) {
        result.status = kvFailure(reserved.status, "reserve token");
        return result;
    }

    WorkspaceLayout const& offsets = impl_->workspace_layout;
    KvScalar* const hidden = impl_->workspace<KvScalar>(offsets.hidden);
    KvScalar* const normalized =
        impl_->workspace<KvScalar>(offsets.normalized);
    KvScalar* const query = impl_->workspace<KvScalar>(offsets.query);
    KvScalar* const key = impl_->workspace<KvScalar>(offsets.key);
    KvScalar* const value = impl_->workspace<KvScalar>(offsets.value);
    KvScalar* const attention =
        impl_->workspace<KvScalar>(offsets.attention);
    KvScalar* const projection =
        impl_->workspace<KvScalar>(offsets.projection);
    KvScalar* const gate = impl_->workspace<KvScalar>(offsets.gate);
    KvScalar* const up = impl_->workspace<KvScalar>(offsets.up);
    KvScalar* const activated =
        impl_->workspace<KvScalar>(offsets.activated);
    float* const logits = impl_->workspace<float>(offsets.logits);
    std::uint32_t* const greedy_token =
        impl_->workspace<std::uint32_t>(offsets.greedy_token);
    float* const attention_scores =
        impl_->workspace<float>(offsets.attention_scores);

    cuda_model_detail::launchEmbedding(
        impl_->weight("model.embed_tokens.weight"),
        token_id,
        hidden,
        config.hidden_size,
        impl_->stream
    );
    result.status = impl_->checkLaunch("embedding lookup");
    if (!result.status.ok()) {
        return result;
    }
    if (debug_capture != nullptr) {
        cudaError_t const copied = cudaMemcpyAsync(
            debug_capture->embedding.data(),
            hidden,
            config.hidden_size * sizeof(KvScalar),
            cudaMemcpyDeviceToHost,
            impl_->stream
        );
        if (copied != cudaSuccess) {
            result.status = cudaFailure(copied, "capture embedding");
            return result;
        }
    }

    std::uint32_t const kv_size =
        config.kv_head_count * config.head_dimension;
    float const attention_scale =
        1.0F / std::sqrt(static_cast<float>(config.head_dimension));
    std::size_t const score_bytes =
        static_cast<std::size_t>(config.attention_head_count)
        * config.max_position_embeddings * sizeof(float);
    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        std::string const prefix = "model.layers."
            + std::to_string(layer) + ".";
        cuda_model_detail::launchRmsNorm(
            hidden,
            impl_->weight(prefix + "input_layernorm.weight"),
            normalized,
            config.hidden_size,
            config.rms_norm_epsilon,
            impl_->stream
        );
        result.status = impl_->checkLaunch("input rmsnorm");
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "self_attn.q_proj.weight"),
            normalized, query, config.hidden_size, config.hidden_size,
            "q projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "self_attn.k_proj.weight"),
            normalized, key, kv_size, config.hidden_size, "k projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "self_attn.v_proj.weight"),
            normalized, value, kv_size, config.hidden_size, "v projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchRope(
            query,
            key,
            config.attention_head_count,
            config.kv_head_count,
            config.head_dimension,
            expected_position,
            config.rope_theta,
            impl_->stream
        );
        result.status = impl_->checkLaunch("rope");
        if (!result.status.ok()) {
            return result;
        }

        EngineKvStatus const write = reserved.transaction.writeLayer(
            LayerKvWrite{layer, key, value}
        );
        if (!write.ok()) {
            result.status = kvFailure(write, "write layer kv");
            return result;
        }
        EngineKvStatus const attended = reserved.transaction.attendLayer(
            PagedDecodeRequest{
                layer,
                query,
                attention,
                attention_scores,
                score_bytes,
                attention_scale,
            }
        );
        if (!attended.ok()) {
            result.status = kvFailure(attended, "paged decode attention");
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "self_attn.o_proj.weight"),
            attention, projection, config.hidden_size, config.hidden_size,
            "attention output projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchResidualAdd(
            hidden, projection, hidden, config.hidden_size, impl_->stream
        );
        result.status = impl_->checkLaunch("attention residual");
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchRmsNorm(
            hidden,
            impl_->weight(prefix + "post_attention_layernorm.weight"),
            normalized,
            config.hidden_size,
            config.rms_norm_epsilon,
            impl_->stream
        );
        result.status = impl_->checkLaunch("post attention rmsnorm");
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "mlp.gate_proj.weight"),
            normalized, gate, config.intermediate_size, config.hidden_size,
            "mlp gate projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "mlp.up_proj.weight"),
            normalized, up, config.intermediate_size, config.hidden_size,
            "mlp up projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchSwiGlu(
            gate, up, activated, config.intermediate_size, impl_->stream
        );
        result.status = impl_->checkLaunch("swiglu");
        if (!result.status.ok()) {
            return result;
        }
        result.status = impl_->gemv(
            impl_->weight(prefix + "mlp.down_proj.weight"),
            activated, projection, config.hidden_size,
            config.intermediate_size, "mlp down projection"
        );
        if (!result.status.ok()) {
            return result;
        }
        cuda_model_detail::launchResidualAdd(
            hidden, projection, hidden, config.hidden_size, impl_->stream
        );
        result.status = impl_->checkLaunch("mlp residual");
        if (!result.status.ok()) {
            return result;
        }

        auto const capture = capture_indices.find(layer);
        if (capture != capture_indices.end()) {
            cudaError_t const copied = cudaMemcpyAsync(
                debug_capture->layer_hidden_states[capture->second].data(),
                hidden,
                config.hidden_size * sizeof(KvScalar),
                cudaMemcpyDeviceToHost,
                impl_->stream
            );
            if (copied != cudaSuccess) {
                result.status = cudaFailure(copied, "capture layer hidden");
                return result;
            }
        }
    }

    cuda_model_detail::launchRmsNorm(
        hidden,
        impl_->weight("model.norm.weight"),
        normalized,
        config.hidden_size,
        config.rms_norm_epsilon,
        impl_->stream
    );
    result.status = impl_->checkLaunch("final rmsnorm");
    if (!result.status.ok()) {
        return result;
    }
    result.status = impl_->logitsGemv(
        impl_->weight("lm_head.weight"), normalized, logits
    );
    if (!result.status.ok()) {
        return result;
    }
    cuda_model_detail::launchArgmax(
        logits, config.vocabulary_size, greedy_token, impl_->stream
    );
    result.status = impl_->checkLaunch("greedy argmax");
    if (!result.status.ok()) {
        return result;
    }

    cudaError_t copied = cudaSuccess;
    if (output_options.copy_logits) {
        copied = cudaMemcpyAsync(
            result.logits.data(),
            logits,
            result.logits.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            impl_->stream
        );
    }
    if (copied == cudaSuccess) {
        copied = cudaMemcpyAsync(
            &result.greedy_token_id,
            greedy_token,
            sizeof(result.greedy_token_id),
            cudaMemcpyDeviceToHost,
            impl_->stream
        );
    }
    if (copied == cudaSuccess && debug_capture != nullptr) {
        copied = cudaMemcpyAsync(
            debug_capture->final_hidden_state.data(),
            normalized,
            config.hidden_size * sizeof(KvScalar),
            cudaMemcpyDeviceToHost,
            impl_->stream
        );
    }
    if (copied != cudaSuccess) {
        result.status = cudaFailure(copied, "copy model output");
        return result;
    }

    EngineKvStatus const committed = reserved.transaction.commit();
    if (!committed.ok()) {
        result.status = kvFailure(committed, "commit model token");
        return result;
    }
    result.status = {};
    return result;
}

TinyLlamaConfig CudaTinyLlamaModelRunner::generationConfig() const noexcept
{
    return config();
}

GenerationStepResult CudaTinyLlamaModelRunner::generationForwardToken(
    RequestId request_id,
    std::uint32_t token_id,
    std::uint32_t expected_position)
{
    ModelTokenResult result = forwardToken(
        request_id,
        token_id,
        expected_position,
        nullptr,
        ModelRunnerOutputOptions{false}
    );
    return GenerationStepResult{
        result.ok(), result.greedy_token_id, std::move(result.status.detail),
    };
}

CudaModelRunnerCreateResult createCudaTinyLlamaModelRunner(
    std::string const& manifest_path,
    std::string const& data_path,
    EngineKvBackend& kv_backend,
    EngineStream stream)
{
    CudaModelRunnerCreateResult result;
    if (manifest_path.empty() || data_path.empty() || stream == nullptr) {
        result.status = failure(
            CudaModelRunnerError::InvalidArgument,
            0,
            "manifest, data path, and stream are required"
        );
        return result;
    }
    WeightManifestLoadResult loaded = loadWeightManifest(manifest_path);
    if (!loaded.ok()) {
        result.status = failure(
            CudaModelRunnerError::ManifestError,
            static_cast<int>(loaded.status.error),
            loaded.status.detail
        );
        return result;
    }
    WeightManifestStatus const data_status = validateWeightDataFile(
        loaded.manifest, data_path
    );
    if (!data_status.ok()) {
        result.status = failure(
            CudaModelRunnerError::DataFileError,
            static_cast<int>(data_status.error),
            data_status.detail
        );
        return result;
    }

    TinyLlamaConfig const config = loaded.manifest.config;
    EngineKvConfig const kv_config = kv_backend.config();
    std::size_t const required_attention_workspace =
        static_cast<std::size_t>(config.attention_head_count)
        * config.max_position_embeddings * sizeof(float);
    if (kv_config.kv_layout.layer_count != config.layer_count
        || kv_config.kv_layout.kv_head_count != config.kv_head_count
        || kv_config.kv_layout.head_dimension != config.head_dimension
        || kv_config.query_head_count != config.attention_head_count
        || kv_config.attention_workspace_bytes
            < required_attention_workspace) {
        result.status = failure(
            CudaModelRunnerError::InvalidArgument,
            0,
            "KV backend layout does not match model manifest"
        );
        return result;
    }

    std::unique_ptr<CudaTinyLlamaModelRunner::Impl> impl;
    try {
        impl = std::make_unique<CudaTinyLlamaModelRunner::Impl>();
        impl->manifest = std::move(loaded.manifest);
    } catch (std::bad_alloc const&) {
        result.status = failure(
            CudaModelRunnerError::AllocationFailed,
            0,
            "runner metadata allocation failed"
        );
        return result;
    }
    impl->backend = &kv_backend;
    impl->stream = reinterpret_cast<cudaStream_t>(stream);
    if (!makeWorkspaceLayout(config, impl->workspace_layout)) {
        result.status = failure(
            CudaModelRunnerError::AllocationFailed,
            0,
            "workspace size overflow"
        );
        return result;
    }

    cudaError_t cuda_status = cudaMalloc(
        reinterpret_cast<void**>(&impl->device_weights),
        static_cast<std::size_t>(impl->manifest.data_bytes)
    );
    if (cuda_status != cudaSuccess) {
        result.status = cudaFailure(cuda_status, "allocate model weights");
        return result;
    }
    cuda_status = cudaMalloc(
        reinterpret_cast<void**>(&impl->device_workspace),
        impl->workspace_layout.bytes
    );
    if (cuda_status != cudaSuccess) {
        result.status = cudaFailure(cuda_status, "allocate model workspace");
        return result;
    }

    std::ifstream weights(data_path, std::ios::binary);
    std::vector<std::uint8_t> staging;
    try {
        staging.resize(16 * 1024 * 1024);
    } catch (std::bad_alloc const&) {
        result.status = failure(
            CudaModelRunnerError::AllocationFailed,
            0,
            "weight staging buffer allocation failed"
        );
        return result;
    }
    for (WeightTensorDescriptor const& tensor : impl->manifest.tensors) {
        weights.clear();
        weights.seekg(static_cast<std::streamoff>(tensor.offset));
        std::uint64_t position = 0;
        while (position < tensor.byte_size) {
            std::size_t const bytes = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    tensor.byte_size - position, staging.size()
                )
            );
            weights.read(
                reinterpret_cast<char*>(staging.data()),
                static_cast<std::streamsize>(bytes)
            );
            if (weights.gcount() != static_cast<std::streamsize>(bytes)) {
                result.status = failure(
                    CudaModelRunnerError::DataFileError,
                    0,
                    "short read while uploading " + tensor.name
                );
                return result;
            }
            cuda_status = cudaMemcpy(
                impl->device_weights + tensor.offset + position,
                staging.data(),
                bytes,
                cudaMemcpyHostToDevice
            );
            if (cuda_status != cudaSuccess) {
                result.status = cudaFailure(
                    cuda_status, "upload " + tensor.name
                );
                return result;
            }
            position += bytes;
        }
    }

    cublasStatus_t cublas_status = cublasCreate(&impl->cublas);
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
        cublas_status = cublasSetStream(impl->cublas, impl->stream);
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS) {
        cublas_status = cublasSetAtomicsMode(
            impl->cublas, CUBLAS_ATOMICS_NOT_ALLOWED
        );
    }
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        result.status = cublasFailure(cublas_status, "initialize cuBLAS");
        return result;
    }

    result.runner = std::unique_ptr<CudaTinyLlamaModelRunner>(
        new CudaTinyLlamaModelRunner(std::move(impl))
    );
    result.status = {};
    return result;
}

} // namespace kimkvcache
