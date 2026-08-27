#include "benchmark_workload.h"

#include <array>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace kimkvcache::benchmark::detail {

WorkloadResult runCudaWorkload(
    WorkloadKind workload,
    BenchmarkConfig const& config)
{
    std::uint64_t const micro_capacity_value =
        (static_cast<std::uint64_t>(config.maximum_sequence_length) + 7) / 8
        + 32;
    std::uint64_t const extent_capacity_value =
        (static_cast<std::uint64_t>(config.maximum_sequence_length) + 63) / 64
        + 8;
    if (micro_capacity_value > std::numeric_limits<std::uint32_t>::max()
        || extent_capacity_value
            > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA benchmark pool capacity overflow");
    }
    CudaKvCache cache(
        config.layout,
        static_cast<std::uint32_t>(micro_capacity_value),
        static_cast<std::uint32_t>(extent_capacity_value)
    );
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
    KvCacheManagerSnapshot const snapshot = cache.metadataSnapshot();
    result.resources_released = snapshot.request_count == 0
        && snapshot.promotion_count == 0
        && snapshot.page_lease_count == 0
        && snapshot.micro_pool.allocated_slots == 0
        && snapshot.extent_pool.allocated_slots == 0;

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
