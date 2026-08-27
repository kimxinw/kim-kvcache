#include "heteropage_kv/benchmark/benchmark.h"

#include <algorithm>
#include <cmath>

namespace kimkvcache::benchmark {

LatencySummary summarizeLatency(
    OperationKind operation,
    std::vector<OperationSample> const& samples)
{
    std::vector<std::uint64_t> durations;
    std::uint64_t failures = 0;
    std::uint64_t total_bytes = 0;
    long double total_duration_ns = 0.0L;
    for (OperationSample const& sample : samples) {
        if (sample.operation != operation) {
            continue;
        }
        durations.push_back(sample.duration_ns);
        total_bytes += sample.bytes;
        total_duration_ns += sample.duration_ns;
        if (!sample.success) {
            ++failures;
        }
    }

    LatencySummary result{};
    result.operation = operation;
    result.sample_count = durations.size();
    result.failure_count = failures;
    if (durations.empty()) {
        return result;
    }

    std::sort(durations.begin(), durations.end());
    auto const percentile = [&durations](double quantile) {
        std::size_t const rank = static_cast<std::size_t>(
            std::ceil(quantile * static_cast<double>(durations.size()))
        );
        return durations[std::max<std::size_t>(rank, 1) - 1];
    };

    result.minimum_ns = durations.front();
    result.maximum_ns = durations.back();
    result.mean_ns = static_cast<std::uint64_t>(
        std::llround(total_duration_ns / durations.size())
    );
    result.p50_ns = percentile(0.50);
    result.p95_ns = percentile(0.95);
    result.p99_ns = percentile(0.99);
    result.total_bytes = total_bytes;
    result.effective_bandwidth_gbps = total_duration_ns == 0.0L
        ? 0.0
        : static_cast<double>(
            static_cast<long double>(total_bytes) / total_duration_ns
        );
    return result;
}

} // namespace kimkvcache::benchmark
