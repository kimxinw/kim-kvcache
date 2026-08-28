#pragma once

#include "heteropage_kv/benchmark/benchmark.h"
#include "heteropage_kv/fixed/fixed_page_manager.h"
#include "heteropage_kv/runtime/kv_cache_manager.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

namespace kimkvcache::benchmark::cpu_detail {

using Clock = std::chrono::steady_clock;

[[nodiscard]] inline std::uint64_t durationNs(
    Clock::time_point begin) noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - begin
        ).count()
    );
}

template <typename Callable>
KvCacheError recordKvOperation(
    WorkloadResult& result,
    OperationKind operation,
    std::uint32_t token_count,
    std::uint64_t bytes,
    KvCacheError expected,
    Callable&& callable,
    std::string detail_prefix = {})
{
    Clock::time_point const begin = Clock::now();
    KvCacheError const actual = callable();
    bool const success = actual == expected;

    std::string detail = std::move(detail_prefix);
    if (!detail.empty()) {
        detail += ':';
    }
    detail += std::string(toString(actual));
    result.samples.push_back(OperationSample{
        operation,
        result.samples.size(),
        durationNs(begin),
        bytes,
        token_count,
        success,
        std::move(detail),
    });
    if (!success) {
        ++result.failed_operations;
    }
    return actual;
}

[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> poolCapacities(
    BenchmarkConfig const& config);

void finalizeWorkload(
    WorkloadResult& result,
    KvCacheManager const& manager,
    Clock::time_point begin);

// CPU 与固定页对照共用的环境信息收集。
[[nodiscard]] EnvironmentInfo benchmarkEnvironment(BenchmarkConfig const& config);

void attachTraceStatistics(
    WorkloadResult& result,
    WorkloadTrace const& trace);

void promoteEligibleRuns(
    KvCacheManager& manager,
    WorkloadResult& result,
    RequestId request_id,
    std::uint32_t token_count);

[[nodiscard]] WorkloadResult runIndependentRequests(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config);

[[nodiscard]] WorkloadResult runSharedPrompt(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config);

[[nodiscard]] WorkloadResult runForkCow(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config);

[[nodiscard]] WorkloadResult runFaultWorkload(
    WorkloadTrace const& trace);

[[nodiscard]] WorkloadResult runCpuWorkload(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config);

void finalizeFixedWorkload(
    WorkloadResult& result,
    FixedPageManager const& manager,
    Clock::time_point begin);

[[nodiscard]] CapacityResult runFixedCapacityProbe(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config,
    std::uint16_t tokens_per_page);

[[nodiscard]] WorkloadResult runFixedWorkload(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config,
    std::uint16_t tokens_per_page);

} // namespace kimkvcache::benchmark::cpu_detail
