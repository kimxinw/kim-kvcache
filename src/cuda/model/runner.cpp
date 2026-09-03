#include "kernels.cuh"
#include "runner_internal.cuh"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kimkvcache {

using namespace cuda_model_runner_detail;

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


} // namespace kimkvcache
