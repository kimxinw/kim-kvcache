#include "cpu_benchmark_internal.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace kimkvcache::benchmark::cpu_detail {
namespace {

[[nodiscard]] std::string timestampUtc()
{
    std::time_t const now = std::time(nullptr);
    std::tm value{};
#if defined(_WIN32)
    gmtime_s(&value, &now);
#else
    gmtime_r(&now, &value);
#endif
    std::ostringstream output;
    output << std::put_time(&value, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

[[nodiscard]] std::string compilerName()
{
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
    return std::string("MSVC ") + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

} // namespace

EnvironmentInfo benchmarkEnvironment(BenchmarkConfig const& config)
{
    char const* hostname = std::getenv("HOSTNAME");
    return EnvironmentInfo{
        timestampUtc(),
        config.git_commit.empty() ? defaultGitCommit() : config.git_commit,
        hostname == nullptr ? "unknown" : hostname,
        compilerName(),
        "not_applicable",
        "not_applicable",
    };
}

namespace {

constexpr std::array<OperationKind, 10> kOperations{
    OperationKind::Create,
    OperationKind::Append,
    OperationKind::Seal,
    OperationKind::Fork,
    OperationKind::CowAppend,
    OperationKind::Promote,
    OperationKind::PromotionRollback,
    OperationKind::Gather,
    OperationKind::Attention,
    OperationKind::Release,
};

[[nodiscard]] constexpr std::uint64_t ceilDivide(
    std::uint64_t value,
    std::uint64_t divisor) noexcept
{
    return value == 0 ? 0 : 1 + (value - 1) / divisor;
}

[[nodiscard]] std::uint32_t checkedPoolCapacity(
    std::uint64_t value,
    char const* name)
{
    if (value == 0
        || value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string(name) + " capacity is outside uint32 range"
        );
    }
    return static_cast<std::uint32_t>(value);
}

} // namespace

std::pair<std::uint32_t, std::uint32_t> poolCapacities(
    BenchmarkConfig const& config)
{
    std::uint64_t const workers =
        std::max<std::uint64_t>(config.concurrency, 8) + 1;
    std::uint64_t const micro_per_request = ceilDivide(
        config.maximum_sequence_length,
        kMicroPageTokenCapacity
    );
    std::uint64_t const extent_per_request = ceilDivide(
        config.maximum_sequence_length,
        kExtentPageTokenCapacity
    );
    return {
        checkedPoolCapacity(workers * micro_per_request + 16, "micro"),
        checkedPoolCapacity(workers * extent_per_request + 8, "extent"),
    };
}

void finalizeWorkload(
    WorkloadResult& result,
    KvCacheManager const& manager,
    Clock::time_point begin)
{
    result.elapsed_ns = durationNs(begin);
    result.requests_per_second = result.elapsed_ns == 0
        ? 0.0
        : static_cast<double>(result.completed_requests) * 1.0e9
            / static_cast<double>(result.elapsed_ns);
    result.invariants_ok = manager.checkInvariants();
    KvCacheManagerSnapshot const snapshot = manager.snapshot();
    result.resources_released = snapshot.request_count == 0
        && snapshot.promotion_count == 0
        && snapshot.page_lease_count == 0
        && snapshot.micro_pool.allocated_slots == 0
        && snapshot.extent_pool.allocated_slots == 0;

    for (OperationKind const operation : kOperations) {
        LatencySummary summary = summarizeLatency(operation, result.samples);
        if (summary.sample_count != 0) {
            result.latency.push_back(summary);
        }
    }
}

void attachTraceStatistics(
    WorkloadResult& result,
    WorkloadTrace const& trace)
{
    result.trace_seed = trace.seed;
    result.trace_request_count = trace.sequence_lengths.size();
    if (trace.sequence_lengths.empty()) {
        return;
    }
    auto const bounds = std::minmax_element(
        trace.sequence_lengths.begin(),
        trace.sequence_lengths.end()
    );
    result.minimum_sequence_length = *bounds.first;
    result.maximum_sequence_length = *bounds.second;
    long double total = 0.0L;
    for (std::uint32_t const length : trace.sequence_lengths) {
        total += length;
    }
    result.mean_sequence_length = static_cast<double>(
        total / trace.sequence_lengths.size()
    );
}

void promoteEligibleRuns(
    KvCacheManager& manager,
    WorkloadResult& result,
    RequestId request_id,
    std::uint32_t token_count)
{
    std::uint32_t const promotable_tokens =
        token_count / kExtentPageTokenCapacity
        * kExtentPageTokenCapacity;
    for (std::uint32_t logical_begin = 0;
         logical_begin < promotable_tokens;
         logical_begin += kExtentPageTokenCapacity) {
        static_cast<void>(recordKvOperation(
            result,
            OperationKind::Promote,
            kExtentPageTokenCapacity,
            0,
            KvCacheError::None,
            [&manager, request_id, logical_begin]() {
                PromotionPrepareResult const prepared =
                    manager.preparePromotion(request_id, logical_begin);
                if (!prepared.ok()) {
                    return prepared.error;
                }
                return manager.commitPromotion(prepared.promotion_id);
            }
        ));
    }
}

WorkloadResult runCpuWorkload(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config)
{
    WorkloadResult result{};
    switch (trace.workload) {
    case WorkloadKind::Short:
    case WorkloadKind::Mixed:
    case WorkloadKind::Adversarial:
    case WorkloadKind::Long:
        result = runIndependentRequests(trace, config);
        break;
    case WorkloadKind::SharedPrompt:
        result = runSharedPrompt(trace, config);
        break;
    case WorkloadKind::ForkCow:
        result = runForkCow(trace, config);
        break;
    case WorkloadKind::Fault:
        result = runFaultWorkload(trace);
        break;
    }
    attachTraceStatistics(result, trace);
    return result;
}

} // namespace kimkvcache::benchmark::cpu_detail
