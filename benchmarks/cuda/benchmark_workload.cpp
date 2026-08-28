#include "benchmark_workload.h"

#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <stdexcept>

namespace kimkvcache::benchmark::detail {

CudaBenchmarkCache::CudaBenchmarkCache(
    BenchmarkConfig const& config,
    std::uint16_t fixed_page_tokens)
{
    if (fixed_page_tokens == 0) {
        std::uint64_t const micro_capacity =
            (static_cast<std::uint64_t>(config.maximum_sequence_length) + 7)
                / 8
            + 32;
        std::uint64_t const extent_capacity =
            (static_cast<std::uint64_t>(config.maximum_sequence_length) + 63)
                / 64
            + 8;
        if (micro_capacity > std::numeric_limits<std::uint32_t>::max()
            || extent_capacity
                > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument(
                "CUDA benchmark pool capacity overflow"
            );
        }
        hetero_ = std::make_unique<CudaKvCache>(
            config.layout,
            static_cast<std::uint32_t>(micro_capacity),
            static_cast<std::uint32_t>(extent_capacity)
        );
        return;
    }

    std::uint64_t const page_capacity =
        (static_cast<std::uint64_t>(config.maximum_sequence_length)
            + fixed_page_tokens - 1)
            / fixed_page_tokens
        + 32;
    if (page_capacity > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "Fixed CUDA benchmark pool capacity overflow"
        );
    }
    fixed_ = std::make_unique<FixedCudaKvCache>(
        config.layout,
        fixed_page_tokens,
        static_cast<std::uint32_t>(page_capacity)
    );
}

bool CudaBenchmarkCache::fixed() const noexcept
{
    return fixed_ != nullptr;
}

bool CudaBenchmarkCache::supportsPromotion() const noexcept
{
    return hetero_ != nullptr;
}

CudaStatus CudaBenchmarkCache::status() const noexcept
{
    return fixed_ != nullptr ? fixed_->status() : hetero_->status();
}

KvCacheError CudaBenchmarkCache::createRequest(RequestId request_id)
{
    return fixed_ != nullptr
        ? fixed_->createRequest(request_id)
        : hetero_->createRequest(request_id);
}

KvCacheError CudaBenchmarkCache::sealTail(RequestId request_id)
{
    return fixed_ != nullptr
        ? fixed_->sealTail(request_id)
        : hetero_->sealTail(request_id);
}

KvCacheError CudaBenchmarkCache::forkRequest(
    RequestId source_request_id,
    RequestId child_request_id)
{
    return fixed_ != nullptr
        ? fixed_->forkRequest(source_request_id, child_request_id)
        : hetero_->forkRequest(source_request_id, child_request_id);
}

KvCacheError CudaBenchmarkCache::releaseRequest(RequestId request_id)
{
    return fixed_ != nullptr
        ? fixed_->releaseRequest(request_id)
        : hetero_->releaseRequest(request_id);
}

CudaKvOperationResult CudaBenchmarkCache::append(
    RequestId request_id,
    std::uint32_t token_count,
    KvScalar const* device_input,
    CudaStream stream)
{
    return fixed_ != nullptr
        ? fixed_->append(request_id, token_count, device_input, stream)
        : hetero_->append(request_id, token_count, device_input, stream);
}

CudaKvOperationResult CudaBenchmarkCache::promote(
    RequestId request_id,
    std::uint32_t logical_token_begin,
    CudaStream stream)
{
    if (hetero_ == nullptr) {
        return CudaKvOperationResult{KvCacheError::InvalidState, CudaStatus{}};
    }
    return hetero_->promote(request_id, logical_token_begin, stream);
}

CudaKvOperationResult CudaBenchmarkCache::gather(
    RequestId request_id,
    KvScalar* device_output,
    CudaStream stream)
{
    return fixed_ != nullptr
        ? fixed_->gather(request_id, device_output, stream)
        : hetero_->gather(request_id, device_output, stream);
}

CudaKvOperationResult CudaBenchmarkCache::referenceAttention(
    RequestId request_id,
    float const* device_query,
    float* device_output,
    CudaStream stream)
{
    return fixed_ != nullptr
        ? fixed_->referenceAttention(
            request_id,
            device_query,
            device_output,
            stream
        )
        : hetero_->referenceAttention(
            request_id,
            device_query,
            device_output,
            stream
        );
}

std::optional<BlockTable> CudaBenchmarkCache::blockTable(
    RequestId request_id) const
{
    return fixed_ != nullptr
        ? fixed_->blockTable(request_id)
        : hetero_->blockTable(request_id);
}

bool CudaBenchmarkCache::checkInvariants() const
{
    return fixed_ != nullptr
        ? fixed_->checkInvariants()
        : hetero_->checkInvariants();
}

bool CudaBenchmarkCache::resourcesReleased() const
{
    if (fixed_ != nullptr) {
        FixedPageManagerSnapshot const snapshot = fixed_->metadataSnapshot();
        return snapshot.request_count == 0
            && snapshot.pool.allocated_slots == 0;
    }
    KvCacheManagerSnapshot const snapshot = hetero_->metadataSnapshot();
    return snapshot.request_count == 0
        && snapshot.promotion_count == 0
        && snapshot.page_lease_count == 0
        && snapshot.micro_pool.allocated_slots == 0
        && snapshot.extent_pool.allocated_slots == 0;
}

void CudaBenchmarkCache::injectFailureOnce(
    CudaFailurePoint point) noexcept
{
    if (fixed_ != nullptr) {
        fixed_->injectFailureOnce(point);
    } else {
        hetero_->injectFailureOnce(point);
    }
}

WorkloadResult runCudaWorkload(
    WorkloadKind workload,
    BenchmarkConfig const& config,
    std::uint16_t fixed_page_tokens)
{
    if (fixed_page_tokens != 0 && workload == WorkloadKind::Fault) {
        throw std::invalid_argument(
            "Fixed CUDA benchmark does not support fault workload"
        );
    }
    CudaBenchmarkCache cache(config, fixed_page_tokens);
    if (!cache.status().ok()) {
        throw std::runtime_error("cannot initialize CUDA KV cache");
    }

    std::size_t maximum_elements = 0;
    if (!config.layout.elementsForTokens(
            config.maximum_sequence_length,
            maximum_elements)) {
        throw std::overflow_error("maximum CUDA input size overflow");
    }
    std::size_t attention_elements =
        static_cast<std::size_t>(config.layout.layer_count)
        * config.layout.kv_head_count
        * config.layout.head_dimension;

    DeviceBuffer<KvScalar> input(maximum_elements);
    DeviceBuffer<KvScalar> output(maximum_elements);
    DeviceBuffer<float> query(attention_elements);
    DeviceBuffer<float> attention(attention_elements);
    BenchmarkStream stream;
    GpuTimer timer;
    if (!input.ok() || !output.ok() || !query.ok() || !attention.ok()
        || !stream.ok() || !timer.ok()
        || !input.clear() || !output.clear()
        || !query.clear() || !attention.clear()) {
        throw std::runtime_error("cannot allocate CUDA benchmark resources");
    }

    WorkloadResult result{};
    result.workload = workload;
    RequestId next_request_id = 1;
    std::uint32_t const total_iterations =
        config.warmup_iterations + config.measured_iterations;

    auto const wall_begin = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0;
         iteration < total_iterations;
         ++iteration) {
        bool const measured = iteration >= config.warmup_iterations;
        std::uint32_t const measured_index = measured
            ? iteration - config.warmup_iterations
            : iteration;

        switch (workload) {
        case WorkloadKind::Short:
        case WorkloadKind::Mixed:
        case WorkloadKind::Adversarial:
        case WorkloadKind::Long:
            runIndependentIteration(
                cache,
                input,
                output,
                query,
                attention,
                stream,
                timer,
                result,
                config,
                workload,
                next_request_id++,
                measured_index,
                measured
            );
            break;
        case WorkloadKind::SharedPrompt:
            runSharedIteration(
                cache,
                input,
                output,
                stream,
                timer,
                result,
                config,
                next_request_id,
                measured_index,
                measured
            );
            break;
        case WorkloadKind::ForkCow:
            runForkCowIteration(
                cache,
                input,
                output,
                stream,
                timer,
                result,
                config,
                next_request_id,
                measured_index,
                measured
            );
            break;
        case WorkloadKind::Fault:
            runFaultIteration(
                cache,
                input,
                stream,
                timer,
                result,
                config,
                next_request_id,
                measured_index,
                measured
            );
            break;
        }
    }

    result.elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wall_begin
        ).count()
    );
    result.requests_per_second = result.elapsed_ns == 0
        ? 0.0
        : static_cast<double>(result.completed_requests) * 1.0e9
            / static_cast<double>(result.elapsed_ns);
    result.invariants_ok = cache.checkInvariants();
    result.resources_released = cache.resourcesReleased();

    constexpr std::array<OperationKind, 6> kCudaOperations{
        OperationKind::Append,
        OperationKind::CowAppend,
        OperationKind::Promote,
        OperationKind::PromotionRollback,
        OperationKind::Gather,
        OperationKind::Attention,
    };
    for (OperationKind const operation : kCudaOperations) {
        LatencySummary const summary = summarizeLatency(
            operation,
            result.samples
        );
        if (summary.sample_count != 0) {
            result.latency.push_back(summary);
        }
    }
    return result;
}

} // namespace kimkvcache::benchmark::detail
