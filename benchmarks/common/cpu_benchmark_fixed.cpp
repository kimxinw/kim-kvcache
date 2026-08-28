#include "cpu_benchmark_internal.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kimkvcache::benchmark {
namespace cpu_detail {
namespace {

// 与 cpu_benchmark_support.cpp 的口径保持一致。
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

[[nodiscard]] BaselineKind baselineForTokens(std::uint16_t tokens_per_page)
{
    switch (tokens_per_page) {
    case 8:
        return BaselineKind::Fixed8;
    case 16:
        return BaselineKind::Fixed16;
    case 32:
        return BaselineKind::Fixed32;
    case 64:
        return BaselineKind::Fixed64;
    }

    throw std::invalid_argument(
        "fixed page tokens must be one of 8/16/32/64"
    );
}

} // namespace

std::pair<std::uint16_t, std::uint32_t> fixedPoolConfig(
    BenchmarkConfig const& config,
    std::uint16_t tokens_per_page)
{
    std::uint64_t const workers =
        std::max<std::uint64_t>(config.concurrency, 8) + 1;
    std::uint64_t const pages_per_request = ceilDivide(
        config.maximum_sequence_length,
        tokens_per_page
    );
    return {
        tokens_per_page,
        checkedPoolCapacity(
            workers * pages_per_request + 16,
            "fixed"
        ),
    };
}

void finalizeFixedWorkload(
    WorkloadResult& result,
    FixedPageManager const& manager,
    Clock::time_point begin)
{
    result.elapsed_ns = durationNs(begin);
    result.requests_per_second = result.elapsed_ns == 0
        ? 0.0
        : static_cast<double>(result.completed_requests) * 1.0e9
            / static_cast<double>(result.elapsed_ns);
    result.invariants_ok = manager.checkInvariants();
    FixedPageManagerSnapshot const snapshot = manager.snapshot();
    result.resources_released = snapshot.request_count == 0
        && snapshot.pool.allocated_slots == 0
        && snapshot.pool.capacityBalanced();

    for (OperationKind const operation : kOperations) {
        LatencySummary summary = summarizeLatency(operation, result.samples);
        if (summary.sample_count != 0) {
            result.latency.push_back(summary);
        }
    }
}

WorkloadResult runFixedIndependentRequests(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config,
    std::uint16_t tokens_per_page)
{
    auto const pool_config = fixedPoolConfig(config, tokens_per_page);
    FixedPageManager manager(pool_config.first, pool_config.second);
    WorkloadResult result{};
    result.workload = trace.workload;
    RequestId next_request_id = 1;
    Clock::time_point const workload_begin = Clock::now();

    for (std::size_t batch_begin = 0;
         batch_begin < trace.sequence_lengths.size();
         batch_begin += config.concurrency) {
        std::size_t const batch_end = std::min(
            trace.sequence_lengths.size(),
            batch_begin + config.concurrency
        );
        std::vector<RequestId> active;
        active.reserve(batch_end - batch_begin);

        for (std::size_t index = batch_begin;
             index < batch_end;
             ++index) {
            RequestId const request_id = next_request_id++;
            std::uint32_t const length = trace.sequence_lengths[index];
            KvCacheError const create_error = recordKvOperation(
                result,
                OperationKind::Create,
                0,
                0,
                KvCacheError::None,
                [&manager, request_id]() {
                    return manager.createRequest(request_id);
                }
            );
            if (create_error != KvCacheError::None) {
                continue;
            }

            active.push_back(request_id);

            static_cast<void>(recordKvOperation(
                result,
                OperationKind::Append,
                length,
                0,
                KvCacheError::None,
                [&manager, request_id, length]() {
                    return manager.append(request_id, length);
                }
            ));

        }

        // 与 Hetero Runner 保持相同的批次生命周期：本批请求全部完成
        // Create/Append 后再统一 Release，避免把 Fixed 对照退化为串行。
        for (RequestId const request_id : active) {
            KvCacheError const release_error = recordKvOperation(
                result,
                OperationKind::Release,
                0,
                0,
                KvCacheError::None,
                [&manager, request_id]() {
                    return manager.releaseRequest(request_id);
                }
            );
            if (release_error == KvCacheError::None) {
                ++result.completed_requests;
            }
        }
    }

    finalizeFixedWorkload(result, manager, workload_begin);
    return result;
}

WorkloadResult runFixedSharedPrompt(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config,
    std::uint16_t tokens_per_page)
{
    auto const pool_config = fixedPoolConfig(config, tokens_per_page);
    FixedPageManager manager(pool_config.first, pool_config.second);
    WorkloadResult result{};
    result.workload = trace.workload;
    RequestId next_request_id = 1;
    Clock::time_point const workload_begin = Clock::now();
    constexpr std::array<std::uint32_t, 3> kPrefixes{64, 128, 256};
    constexpr std::array<std::uint32_t, 3> kFanouts{2, 4, 8};

    std::size_t trace_index = 0;
    std::size_t group_index = 0;
    while (trace_index < trace.sequence_lengths.size()) {
        std::uint32_t const prefix = std::min(
            kPrefixes[group_index % kPrefixes.size()],
            config.maximum_sequence_length - 1
        );
        std::uint32_t const requested_fanout =
            kFanouts[group_index % kFanouts.size()];
        std::uint32_t const fanout = static_cast<std::uint32_t>(
            std::min<std::size_t>(
                requested_fanout,
                trace.sequence_lengths.size() - trace_index
            )
        );
        RequestId const parent = next_request_id++;

        KvCacheError const create_error = recordKvOperation(
            result,
            OperationKind::Create,
            0,
            0,
            KvCacheError::None,
            [&manager, parent]() { return manager.createRequest(parent); }
        );
        if (create_error != KvCacheError::None) {
            break;
        }

        static_cast<void>(recordKvOperation(
            result,
            OperationKind::Append,
            prefix,
            0,
            KvCacheError::None,
            [&manager, parent, prefix]() {
                return manager.append(parent, prefix);
            }
        ));

        std::vector<RequestId> children;
        children.reserve(fanout);
        for (std::uint32_t child_index = 0;
             child_index < fanout;
             ++child_index) {
            RequestId const child = next_request_id++;
            KvCacheError const fork_error = recordKvOperation(
                result,
                OperationKind::Fork,
                prefix,
                0,
                KvCacheError::None,
                [&manager, parent, child]() {
                    return manager.forkRequest(parent, child);
                }
            );
            if (fork_error != KvCacheError::None) {
                continue;
            }

            children.push_back(child);
            std::uint32_t const total =
                trace.sequence_lengths[trace_index + child_index];
            std::uint32_t const suffix = total > prefix
                ? total - prefix
                : 1;
            static_cast<void>(recordKvOperation(
                result,
                OperationKind::Append,
                suffix,
                0,
                KvCacheError::None,
                [&manager, child, suffix]() {
                    return manager.append(child, suffix);
                }
            ));
        }

        for (RequestId const child : children) {
            KvCacheError const release_error = recordKvOperation(
                result,
                OperationKind::Release,
                0,
                0,
                KvCacheError::None,
                [&manager, child]() {
                    return manager.releaseRequest(child);
                }
            );
            if (release_error == KvCacheError::None) {
                ++result.completed_requests;
            }
        }

        static_cast<void>(recordKvOperation(
            result,
            OperationKind::Release,
            0,
            0,
            KvCacheError::None,
            [&manager, parent]() { return manager.releaseRequest(parent); }
        ));

        trace_index += fanout;
        ++group_index;
    }

    finalizeFixedWorkload(result, manager, workload_begin);
    return result;
}

WorkloadResult runFixedForkCow(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config,
    std::uint16_t tokens_per_page)
{
    auto const pool_config = fixedPoolConfig(config, tokens_per_page);
    FixedPageManager manager(pool_config.first, pool_config.second);
    WorkloadResult result{};
    result.workload = trace.workload;
    RequestId next_request_id = 1;
    Clock::time_point const workload_begin = Clock::now();
    constexpr std::array<std::uint32_t, 5> kForkPoints{1, 7, 8, 9, 63};
    constexpr std::array<std::uint32_t, 3> kFanouts{2, 4, 8};

    std::size_t trace_index = 0;
    std::size_t group_index = 0;
    while (trace_index < trace.sequence_lengths.size()) {
        std::uint32_t const fork_point = std::min(
            kForkPoints[group_index % kForkPoints.size()],
            config.maximum_sequence_length - 1
        );
        std::uint32_t const requested_fanout =
            kFanouts[group_index % kFanouts.size()];
        std::uint32_t const fanout = static_cast<std::uint32_t>(
            std::min<std::size_t>(
                requested_fanout,
                trace.sequence_lengths.size() - trace_index
            )
        );
        RequestId const parent = next_request_id++;

        static_cast<void>(recordKvOperation(
            result,
            OperationKind::Create,
            0,
            0,
            KvCacheError::None,
            [&manager, parent]() { return manager.createRequest(parent); }
        ));
        static_cast<void>(recordKvOperation(
            result,
            OperationKind::Append,
            fork_point,
            0,
            KvCacheError::None,
            [&manager, parent, fork_point]() {
                return manager.append(parent, fork_point);
            }
        ));
        static_cast<void>(recordKvOperation(
            result,
            OperationKind::Seal,
            0,
            0,
            KvCacheError::None,
            [&manager, parent]() { return manager.sealTail(parent); }
        ));

        std::vector<RequestId> children;
        children.reserve(fanout);
        for (std::uint32_t child_index = 0;
             child_index < fanout;
             ++child_index) {
            RequestId const child = next_request_id++;
            KvCacheError const fork_error = recordKvOperation(
                result,
                OperationKind::Fork,
                fork_point,
                0,
                KvCacheError::None,
                [&manager, parent, child]() {
                    return manager.forkRequest(parent, child);
                }
            );
            if (fork_error != KvCacheError::None) {
                continue;
            }

            children.push_back(child);
            static_cast<void>(recordKvOperation(
                result,
                OperationKind::CowAppend,
                1,
                0,
                KvCacheError::None,
                [&manager, child]() { return manager.append(child, 1); }
            ));
        }

        for (RequestId const child : children) {
            KvCacheError const release_error = recordKvOperation(
                result,
                OperationKind::Release,
                0,
                0,
                KvCacheError::None,
                [&manager, child]() {
                    return manager.releaseRequest(child);
                }
            );
            if (release_error == KvCacheError::None) {
                ++result.completed_requests;
            }
        }
        static_cast<void>(recordKvOperation(
            result,
            OperationKind::Release,
            0,
            0,
            KvCacheError::None,
            [&manager, parent]() { return manager.releaseRequest(parent); }
        ));

        trace_index += fanout;
        ++group_index;
    }

    finalizeFixedWorkload(result, manager, workload_begin);
    return result;
}

WorkloadResult runFixedWorkload(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config,
    std::uint16_t tokens_per_page)
{
    WorkloadResult result{};
    switch (trace.workload) {
    case WorkloadKind::Short:
    case WorkloadKind::Mixed:
    case WorkloadKind::Adversarial:
    case WorkloadKind::Long:
        result = runFixedIndependentRequests(trace, config, tokens_per_page);
        break;
    case WorkloadKind::SharedPrompt:
        result = runFixedSharedPrompt(trace, config, tokens_per_page);
        break;
    case WorkloadKind::ForkCow:
        result = runFixedForkCow(trace, config, tokens_per_page);
        break;
    case WorkloadKind::Fault:
        throw std::invalid_argument(
            "fault workload requires promotion semantics"
        );
    }
    attachTraceStatistics(result, trace);
    return result;
}

CapacityResult runFixedCapacityProbe(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config,
    std::uint16_t tokens_per_page)
{
    CapacityResult probe{};
    probe.workload = trace.workload;
    probe.baseline = baselineForTokens(tokens_per_page);

    std::size_t bytes_per_token_size = 0;
    if (!config.layout.bytesForTokens(1, bytes_per_token_size)) {
        throw std::overflow_error("bytes per token overflow");
    }
    std::uint64_t const bytes_per_token = bytes_per_token_size;

    // 准入口径与解析公式一致：整请求原子准入，预算耗尽即停止后续 Admission。
    std::uint64_t const budget_pages =
        config.capacity_budget_bytes / bytes_per_token
        / tokens_per_page;

    if (budget_pages == 0) {
        throw std::invalid_argument(
            "capacity budget smaller than one fixed page"
        );
    }

    FixedPageManager manager(
        tokens_per_page,
        checkedPoolCapacity(budget_pages, "fixed")
    );

    RequestId next_request_id = 1;
    std::uint64_t allocated_pages = 0;

    for (std::uint32_t const length : trace.sequence_lengths) {
        std::uint64_t const needed_pages = ceilDivide(length, tokens_per_page);

        FixedPageManagerSnapshot const before = manager.snapshot();

        if (before.pool.free_slots < needed_pages) {
            break;
        }

        KvCacheError const create_error = manager.createRequest(
            next_request_id
        );

        if (create_error != KvCacheError::None) {
            throw std::runtime_error("capacity probe create failed");
        }

        KvCacheError const append_error = manager.append(
            next_request_id,
            length
        );

        if (append_error != KvCacheError::None) {
            throw std::runtime_error("capacity probe append failed");
        }

        FixedPageManagerSnapshot const after = manager.snapshot();
        auto const table = manager.blockTable(next_request_id);

        if (!table.has_value()) {
            throw std::runtime_error("capacity probe table missing");
        }

        probe.used_tokens += length;
        probe.block_table_entries += table->entries().size();
        allocated_pages +=
            after.successful_allocations - before.successful_allocations;
        probe.reserved_tokens = allocated_pages * tokens_per_page;
        probe.admitted_requests += 1;
        next_request_id += 1;
    }

    // CapacityResult 没有 fixed_pages 字段；Fixed 页数由
    // block_table_entries 表达，Micro/Extent 字段保持与解析公式一致为 0。
    probe.micro_pages = 0;
    probe.extent_pages = 0;
    // 探针只做单调 Admission，无释放，峰值等于终态预留。
    probe.peak_reserved_tokens = probe.reserved_tokens;
    probe.internal_fragmentation_tokens =
        probe.reserved_tokens - probe.used_tokens;

    auto const toBytes = [&](std::uint64_t tokens) {
        std::size_t converted = 0;
        if (tokens != 0
            && !config.layout.bytesForTokens(tokens, converted)) {
            throw std::overflow_error("capacity byte count overflow");
        }
        return static_cast<std::uint64_t>(converted);
    };

    probe.used_bytes = toBytes(probe.used_tokens);
    probe.reserved_bytes = toBytes(probe.reserved_tokens);
    probe.peak_reserved_bytes = toBytes(probe.peak_reserved_tokens);
    probe.internal_fragmentation_bytes = toBytes(
        probe.internal_fragmentation_tokens
    );
    probe.reused_tokens = trace.reused_tokens;
    probe.reused_bytes = toBytes(probe.reused_tokens);
    probe.utilization = probe.reserved_tokens == 0
        ? 0.0
        : static_cast<double>(probe.used_tokens)
            / static_cast<double>(probe.reserved_tokens);

    return probe;
}

} // namespace cpu_detail

BenchmarkReport runCpuFixedBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads,
    std::uint16_t tokens_per_page)
{
    std::string error;
    if (!validateConfig(config, error)) {
        throw std::invalid_argument(error);
    }
    if (workloads.empty()) {
        throw std::invalid_argument("at least one workload is required");
    }
    (void)cpu_detail::baselineForTokens(tokens_per_page);

    for (WorkloadKind const workload : workloads) {
        if (workload == WorkloadKind::Fault) {
            throw std::invalid_argument(
                "fault workload requires promotion semantics"
            );
        }
    }

    BenchmarkReport report{};
    report.suite =
        std::string("cpu_metadata_fixed_") + std::to_string(tokens_per_page);
    report.config = config;
    if (report.config.git_commit.empty()) {
        report.config.git_commit = defaultGitCommit();
    }
    report.environment = cpu_detail::benchmarkEnvironment(report.config);

    for (WorkloadKind const workload : workloads) {
        WorkloadTrace const trace = generateWorkload(workload, report.config);
        report.capacity.push_back(
            cpu_detail::runFixedCapacityProbe(trace, report.config, tokens_per_page)
        );
        report.workloads.push_back(
            cpu_detail::runFixedWorkload(trace, report.config, tokens_per_page)
        );
    }
    return report;
}

} // namespace kimkvcache::benchmark
