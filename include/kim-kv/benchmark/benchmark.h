#pragma once

#include "kim-kv/core/kv_layout.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace kimkvcache::benchmark {

enum class WorkloadKind : std::uint8_t {
    Short,
    Mixed,
    Adversarial,
    Long,
    SharedPrompt,
    ForkCow,
    Fault,
};

enum class BaselineKind : std::uint8_t {
    ContiguousMax,
    Fixed8,
    Fixed16,
    Fixed32,
    Fixed64,
    HeteroWithoutPromotion,
    HeteroWithPromotion,
};

enum class OperationKind : std::uint8_t {
    Create,
    Append,
    Seal,
    Fork,
    CowAppend,
    Promote,
    PromotionRollback,
    Gather,
    Attention,
    Release,
};

[[nodiscard]] constexpr std::string_view toString(
    WorkloadKind kind) noexcept
{
    switch (kind) {
    case WorkloadKind::Short:
        return "short";
    case WorkloadKind::Mixed:
        return "mixed";
    case WorkloadKind::Adversarial:
        return "adversarial";
    case WorkloadKind::Long:
        return "long";
    case WorkloadKind::SharedPrompt:
        return "shared_prompt";
    case WorkloadKind::ForkCow:
        return "fork_cow";
    case WorkloadKind::Fault:
        return "fault";
    }

    return "unknown";
}

[[nodiscard]] constexpr std::string_view toString(
    BaselineKind kind) noexcept
{
    switch (kind) {
    case BaselineKind::ContiguousMax:
        return "contiguous_max";
    case BaselineKind::Fixed8:
        return "fixed_8";
    case BaselineKind::Fixed16:
        return "fixed_16";
    case BaselineKind::Fixed32:
        return "fixed_32";
    case BaselineKind::Fixed64:
        return "fixed_64";
    case BaselineKind::HeteroWithoutPromotion:
        return "hetero_8_64_without_promotion";
    case BaselineKind::HeteroWithPromotion:
        return "hetero_8_64_with_promotion";
    }

    return "unknown";
}

[[nodiscard]] constexpr std::string_view toString(
    OperationKind kind) noexcept
{
    switch (kind) {
    case OperationKind::Create:
        return "create";
    case OperationKind::Append:
        return "append";
    case OperationKind::Seal:
        return "seal";
    case OperationKind::Fork:
        return "fork";
    case OperationKind::CowAppend:
        return "cow_append";
    case OperationKind::Promote:
        return "promote";
    case OperationKind::PromotionRollback:
        return "promotion_rollback";
    case OperationKind::Gather:
        return "gather";
    case OperationKind::Attention:
        return "reference_attention";
    case OperationKind::Release:
        return "release";
    }

    return "unknown";
}

struct BenchmarkConfig final {
    std::uint64_t seed{0x4B564341434845ULL};
    std::uint32_t request_count{10'000};
    std::uint32_t concurrency{8};
    std::uint32_t maximum_sequence_length{544};
    std::uint32_t warmup_iterations{3};
    std::uint32_t measured_iterations{20};
    std::uint64_t capacity_budget_bytes{12ULL * 1024ULL * 1024ULL * 1024ULL};
    KvLayout layout{22, 4, 64};
    std::string git_commit{};
};

struct WorkloadTrace final {
    WorkloadKind workload{WorkloadKind::Short};
    std::uint64_t seed{0};
    std::uint64_t reused_tokens{0};
    std::vector<std::uint32_t> sequence_lengths{};
};

struct CapacityResult final {
    WorkloadKind workload{WorkloadKind::Short};
    BaselineKind baseline{BaselineKind::Fixed8};
    std::uint64_t used_tokens{0};
    std::uint64_t reserved_tokens{0};
    std::uint64_t peak_reserved_tokens{0};
    std::uint64_t internal_fragmentation_tokens{0};
    std::uint64_t reused_tokens{0};
    std::uint64_t used_bytes{0};
    std::uint64_t reserved_bytes{0};
    std::uint64_t peak_reserved_bytes{0};
    std::uint64_t internal_fragmentation_bytes{0};
    std::uint64_t reused_bytes{0};
    std::uint64_t block_table_entries{0};
    std::uint64_t micro_pages{0};
    std::uint64_t extent_pages{0};
    std::uint64_t admitted_requests{0};
    double utilization{0.0};
};

struct OperationSample final {
    OperationKind operation{OperationKind::Create};
    std::uint64_t sample_index{0};
    std::uint64_t duration_ns{0};
    std::uint64_t bytes{0};
    std::uint32_t token_count{0};
    bool success{false};
    std::string detail{};
};

struct LatencySummary final {
    OperationKind operation{OperationKind::Create};
    std::uint64_t sample_count{0};
    std::uint64_t failure_count{0};
    std::uint64_t minimum_ns{0};
    std::uint64_t maximum_ns{0};
    std::uint64_t mean_ns{0};
    std::uint64_t p50_ns{0};
    std::uint64_t p95_ns{0};
    std::uint64_t p99_ns{0};
    std::uint64_t total_bytes{0};
    double effective_bandwidth_gbps{0.0};
};

struct WorkloadResult final {
    WorkloadKind workload{WorkloadKind::Short};
    std::uint64_t trace_seed{0};
    std::uint64_t trace_request_count{0};
    std::uint32_t minimum_sequence_length{0};
    std::uint32_t maximum_sequence_length{0};
    double mean_sequence_length{0.0};
    std::uint64_t elapsed_ns{0};
    std::uint64_t completed_requests{0};
    double requests_per_second{0.0};
    std::uint64_t failed_operations{0};
    bool invariants_ok{false};
    bool resources_released{false};
    std::vector<OperationSample> samples{};
    std::vector<LatencySummary> latency{};
};

struct EnvironmentInfo final {
    std::string timestamp_utc{};
    std::string git_commit{};
    std::string hostname{};
    std::string compiler{};
    std::string gpu_name{"not_applicable"};
    std::string cuda_runtime{"not_applicable"};
};

struct BenchmarkReport final {
    std::string suite{};
    BenchmarkConfig config{};
    EnvironmentInfo environment{};
    std::vector<CapacityResult> capacity{};
    std::vector<WorkloadResult> workloads{};

    [[nodiscard]] bool successful() const noexcept;
};

struct CommandLineOptions final {
    BenchmarkConfig config{};
    std::vector<WorkloadKind> workloads{};
    // 0 表示按 Hetero 双池运行时运行；非零表示启用固定页对照。
    std::uint16_t fixed_page_tokens{0};
    std::string output_directory{"benchmarks/results"};
    bool show_help{false};
};

[[nodiscard]] std::vector<WorkloadKind> allWorkloads();
[[nodiscard]] std::vector<BaselineKind> allBaselines();
[[nodiscard]] std::optional<WorkloadKind> parseWorkload(
    std::string_view value) noexcept;
[[nodiscard]] bool validateConfig(
    BenchmarkConfig const& config,
    std::string& error
) noexcept;

[[nodiscard]] WorkloadTrace generateWorkload(
    WorkloadKind workload,
    BenchmarkConfig const& config
);

[[nodiscard]] std::vector<CapacityResult> calculateCapacity(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config
);

[[nodiscard]] LatencySummary summarizeLatency(
    OperationKind operation,
    std::vector<OperationSample> const& samples
);

[[nodiscard]] BenchmarkReport runCpuBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads
);

// 仅在 KIM_KV_ENABLE_CUDA=ON 时提供实现。
[[nodiscard]] BenchmarkReport runCudaBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads
);

// K6 Fixed CUDA 对照入口：真实执行参数化固定页 Storage、Append/COW、
// Gather 与 Reference Attention。固定页没有 Promotion/Fault 语义。
[[nodiscard]] BenchmarkReport runCudaFixedBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads,
    std::uint16_t tokens_per_page
);

// K6 固定页运行时对照入口：以可执行 FixedPageManager 按 tokens_per_page
// 运行同一批 Workload Trace，并输出实测容量探针结果。Fault 需要
// Promotion 语义，固定页对照不接受该 Workload。
[[nodiscard]] BenchmarkReport runCpuFixedBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads,
    std::uint16_t tokens_per_page
);

[[nodiscard]] bool parseCommandLine(
    int argc,
    char const* const* argv,
    CommandLineOptions& options,
    std::string& error
);

[[nodiscard]] std::string benchmarkUsage(std::string_view program_name);
[[nodiscard]] std::string defaultGitCommit();

void writeJsonReport(
    BenchmarkReport const& report,
    std::string const& path
);

void writeCsvReport(
    BenchmarkReport const& report,
    std::string const& path
);

} // namespace kimkvcache::benchmark
