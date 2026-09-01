#include "kim-kv/benchmark/benchmark.h"

#include <algorithm>
#include <array>
#include <limits>

namespace kimkvcache::benchmark {
namespace {

constexpr std::array<WorkloadKind, 7> kWorkloads{
    WorkloadKind::Short,
    WorkloadKind::Mixed,
    WorkloadKind::Adversarial,
    WorkloadKind::Long,
    WorkloadKind::SharedPrompt,
    WorkloadKind::ForkCow,
    WorkloadKind::Fault,
};

constexpr std::array<BaselineKind, 7> kBaselines{
    BaselineKind::ContiguousMax,
    BaselineKind::Fixed8,
    BaselineKind::Fixed16,
    BaselineKind::Fixed32,
    BaselineKind::Fixed64,
    BaselineKind::HeteroWithoutPromotion,
    BaselineKind::HeteroWithPromotion,
};

} // namespace

bool BenchmarkReport::successful() const noexcept
{
    return !workloads.empty()
        && std::all_of(
            workloads.begin(),
            workloads.end(),
            [](WorkloadResult const& result) {
                return result.failed_operations == 0
                    && result.invariants_ok
                    && result.resources_released;
            }
        );
}

std::vector<WorkloadKind> allWorkloads()
{
    return {kWorkloads.begin(), kWorkloads.end()};
}

std::vector<BaselineKind> allBaselines()
{
    return {kBaselines.begin(), kBaselines.end()};
}

std::optional<WorkloadKind> parseWorkload(
    std::string_view value) noexcept
{
    for (WorkloadKind const workload : kWorkloads) {
        if (value == toString(workload)) {
            return workload;
        }
    }
    return std::nullopt;
}

bool validateConfig(
    BenchmarkConfig const& config,
    std::string& error) noexcept
{
    if (!config.layout.valid()) {
        error = "KV layout dimensions must be non-zero";
        return false;
    }
    std::size_t bytes_per_token = 0;
    if (!config.layout.bytesForTokens(1, bytes_per_token)) {
        error = "KV layout byte count overflows size_t";
        return false;
    }
    if (config.request_count == 0) {
        error = "request count must be non-zero";
        return false;
    }
    if (config.concurrency == 0) {
        error = "concurrency must be non-zero";
        return false;
    }
    if (config.maximum_sequence_length < 64) {
        error = "maximum sequence length must be at least 64";
        return false;
    }
    if (config.measured_iterations == 0) {
        error = "measured iteration count must be non-zero";
        return false;
    }
    if (config.warmup_iterations
        > std::numeric_limits<std::uint32_t>::max()
            - config.measured_iterations) {
        error = "warmup plus measured iterations overflows uint32";
        return false;
    }
    if (config.capacity_budget_bytes < bytes_per_token) {
        error = "capacity budget must hold at least one KV token";
        return false;
    }
    return true;
}

std::string defaultGitCommit()
{
#ifdef KIM_KV_GIT_COMMIT
    return KIM_KV_GIT_COMMIT;
#else
    return "unknown";
#endif
}

} // namespace kimkvcache::benchmark
