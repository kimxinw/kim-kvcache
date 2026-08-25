#include "heteropage_kv/benchmark.h"

#include "heteropage_kv/kv_cache_manager.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kimkvcache::benchmark {
namespace {

using Clock = std::chrono::steady_clock;

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

[[nodiscard]] std::uint32_t uniformLength(
    std::mt19937_64& generator,
    std::uint32_t minimum,
    std::uint32_t maximum)
{
    if (minimum >= maximum) {
        return maximum;
    }

    std::uniform_int_distribution<std::uint32_t> distribution(
        minimum,
        maximum
    );
    return distribution(generator);
}

[[nodiscard]] std::uint64_t durationNs(Clock::time_point begin) noexcept
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

[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> poolCapacities(
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

[[nodiscard]] WorkloadResult runIndependentRequests(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config)
{
    auto const capacities = poolCapacities(config);
    KvCacheManager manager(capacities.first, capacities.second);
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
        std::vector<std::pair<RequestId, std::uint32_t>> active;
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

            active.emplace_back(request_id, length);
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

        for (auto const& request : active) {
            promoteEligibleRuns(
                manager,
                result,
                request.first,
                request.second
            );
        }

        for (auto const& request : active) {
            KvCacheError const release_error = recordKvOperation(
                result,
                OperationKind::Release,
                0,
                0,
                KvCacheError::None,
                [&manager, request_id = request.first]() {
                    return manager.releaseRequest(request_id);
                }
            );
            if (release_error == KvCacheError::None) {
                ++result.completed_requests;
            }
        }
    }

    finalizeWorkload(result, manager, workload_begin);
    return result;
}

[[nodiscard]] WorkloadResult runSharedPrompt(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config)
{
    auto const capacities = poolCapacities(config);
    KvCacheManager manager(capacities.first, capacities.second);
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

    finalizeWorkload(result, manager, workload_begin);
    return result;
}

[[nodiscard]] WorkloadResult runForkCow(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config)
{
    auto const capacities = poolCapacities(config);
    KvCacheManager manager(capacities.first, capacities.second);
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

    finalizeWorkload(result, manager, workload_begin);
    return result;
}

[[nodiscard]] WorkloadResult runFaultWorkload(
    WorkloadTrace const& trace)
{
    KvCacheManager manager(8, 1);
    WorkloadResult result{};
    result.workload = trace.workload;
    RequestId next_request_id = 1;
    Clock::time_point const workload_begin = Clock::now();

    for (std::size_t index = 0;
         index < trace.sequence_lengths.size();
         ++index) {
        RequestId const request_id = next_request_id++;
        static_cast<void>(recordKvOperation(
            result,
            OperationKind::Create,
            0,
            0,
            KvCacheError::None,
            [&manager, request_id]() {
                return manager.createRequest(request_id);
            }
        ));

        switch (trace.sequence_lengths[index] % 3) {
        case 0:
            static_cast<void>(recordKvOperation(
                result,
                OperationKind::Append,
                72,
                0,
                KvCacheError::ResourceExhausted,
                [&manager, request_id]() {
                    return manager.append(request_id, 72);
                },
                "expected_oom"
            ));
            break;
        case 1: {
            std::uint32_t const length = std::min<std::uint32_t>(
                trace.sequence_lengths[index],
                kExtentPageTokenCapacity
            );
            static_cast<void>(recordKvOperation(
                result,
                OperationKind::Append,
                length,
                0,
                KvCacheError::None,
                [&manager, request_id, length]() {
                    return manager.append(request_id, length);
                },
                "cancel_path"
            ));
            break;
        }
        case 2:
            static_cast<void>(recordKvOperation(
                result,
                OperationKind::Append,
                kExtentPageTokenCapacity,
                0,
                KvCacheError::None,
                [&manager, request_id]() {
                    return manager.append(
                        request_id,
                        kExtentPageTokenCapacity
                    );
                }
            ));
            static_cast<void>(recordKvOperation(
                result,
                OperationKind::PromotionRollback,
                kExtentPageTokenCapacity,
                0,
                KvCacheError::None,
                [&manager, request_id]() {
                    PromotionPrepareResult const prepared =
                        manager.preparePromotion(request_id, 0);
                    if (!prepared.ok()) {
                        return prepared.error;
                    }
                    return manager.rollbackPromotion(prepared.promotion_id);
                },
                "simulated_copy_error"
            ));
            break;
        }

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

    finalizeWorkload(result, manager, workload_begin);
    return result;
}

[[nodiscard]] WorkloadResult runCpuWorkload(
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

[[nodiscard]] EnvironmentInfo cpuEnvironment(BenchmarkConfig const& config)
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

[[nodiscard]] std::uint64_t parseUnsigned(
    std::string const& text,
    char const* option)
{
    std::size_t consumed = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(text, &consumed, 0);
    } catch (std::exception const&) {
        throw std::invalid_argument(
            std::string("invalid value for ") + option + ": " + text
        );
    }
    if (consumed != text.size()) {
        throw std::invalid_argument(
            std::string("invalid value for ") + option + ": " + text
        );
    }
    return value;
}

[[nodiscard]] std::uint32_t parseUint32(
    std::string const& text,
    char const* option)
{
    std::uint64_t const value = parseUnsigned(text, option);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string("value for ") + option + " exceeds uint32 range"
        );
    }
    return static_cast<std::uint32_t>(value);
}

[[nodiscard]] std::string jsonEscape(std::string_view input)
{
    std::ostringstream output;
    for (char const character : input) {
        switch (character) {
        case '\\':
            output << "\\\\";
            break;
        case '"':
            output << "\\\"";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            output << character;
            break;
        }
    }
    return output.str();
}

[[nodiscard]] std::string csvCell(std::string_view input)
{
    bool const quote = input.find_first_of(",\"\n\r")
        != std::string_view::npos;
    if (!quote) {
        return std::string(input);
    }

    std::string output{"\""};
    for (char const character : input) {
        if (character == '"') {
            output += "\"\"";
        } else {
            output += character;
        }
    }
    output += '"';
    return output;
}

[[nodiscard]] std::string decimal(double value)
{
    std::ostringstream output;
    output << std::setprecision(10) << value;
    return output.str();
}

void writeCsvRow(
    std::ofstream& output,
    std::vector<std::string> const& cells)
{
    for (std::size_t index = 0; index < cells.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << csvCell(cells[index]);
    }
    output << '\n';
}

void ensureParentDirectory(std::string const& path)
{
    std::filesystem::path const parent =
        std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

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

WorkloadTrace generateWorkload(
    WorkloadKind workload,
    BenchmarkConfig const& config)
{
    std::string error;
    if (!validateConfig(config, error)) {
        throw std::invalid_argument(error);
    }

    WorkloadTrace trace{};
    trace.workload = workload;
    trace.seed = config.seed;
    trace.sequence_lengths.reserve(config.request_count);
    std::mt19937_64 generator(
        config.seed ^ (static_cast<std::uint64_t>(workload) << 48U)
    );

    constexpr std::array<std::uint32_t, 3> kPrefixes{64, 128, 256};
    constexpr std::array<std::uint32_t, 5> kForkPoints{1, 7, 8, 9, 63};
    constexpr std::array<std::uint32_t, 3> kFanouts{2, 4, 8};

    if (workload == WorkloadKind::SharedPrompt
        || workload == WorkloadKind::ForkCow) {
        std::size_t group_index = 0;
        while (trace.sequence_lengths.size() < config.request_count) {
            std::uint32_t const fanout =
                kFanouts[group_index % kFanouts.size()];
            std::uint32_t const base = workload == WorkloadKind::SharedPrompt
                ? std::min(
                    kPrefixes[group_index % kPrefixes.size()],
                    config.maximum_sequence_length - 1
                )
                : std::min(
                    kForkPoints[group_index % kForkPoints.size()],
                    config.maximum_sequence_length - 1
                );

            std::size_t const group_begin = trace.sequence_lengths.size();
            for (std::uint32_t child = 0;
                 child < fanout
                    && trace.sequence_lengths.size() < config.request_count;
                 ++child) {
                std::uint32_t const suffix =
                    workload == WorkloadKind::SharedPrompt
                    ? uniformLength(
                        generator,
                        1,
                        std::min<std::uint32_t>(
                            63,
                            config.maximum_sequence_length - base
                        )
                    )
                    : 1;
                trace.sequence_lengths.push_back(base + suffix);
            }
            std::size_t const group_size =
                trace.sequence_lengths.size() - group_begin;
            if (group_size > 1) {
                trace.reused_tokens += static_cast<std::uint64_t>(base)
                    * (group_size - 1);
            }
            ++group_index;
        }
        return trace;
    }

    for (std::uint32_t index = 0;
         index < config.request_count;
         ++index) {
        std::uint32_t length = 1;
        switch (workload) {
        case WorkloadKind::Short:
            length = uniformLength(
                generator,
                1,
                std::min<std::uint32_t>(63, config.maximum_sequence_length)
            );
            break;
        case WorkloadKind::Mixed: {
            std::uniform_int_distribution<std::uint32_t> bucket(0, 99);
            std::uint32_t const choice = bucket(generator);
            if (choice < 70 || config.maximum_sequence_length <= 128) {
                length = uniformLength(
                    generator,
                    1,
                    std::min<std::uint32_t>(
                        128,
                        config.maximum_sequence_length
                    )
                );
            } else if (choice < 90
                       || config.maximum_sequence_length <= 384) {
                length = uniformLength(
                    generator,
                    129,
                    std::min<std::uint32_t>(
                        384,
                        config.maximum_sequence_length
                    )
                );
            } else {
                length = uniformLength(
                    generator,
                    385,
                    config.maximum_sequence_length
                );
            }
            break;
        }
        case WorkloadKind::Adversarial: {
            std::uint32_t const count =
                (config.maximum_sequence_length - 1) / 64 + 1;
            length = 1 + 64 * (index % count);
            break;
        }
        case WorkloadKind::Long:
            length = uniformLength(
                generator,
                std::min<std::uint32_t>(
                    448,
                    config.maximum_sequence_length
                ),
                config.maximum_sequence_length
            );
            break;
        case WorkloadKind::SharedPrompt:
        case WorkloadKind::ForkCow:
            throw std::logic_error("group workload generation fell through");
        case WorkloadKind::Fault:
            length = uniformLength(
                generator,
                1,
                config.maximum_sequence_length
            );
            break;
        }
        trace.sequence_lengths.push_back(length);
    }

    return trace;
}

std::vector<CapacityResult> calculateCapacity(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config)
{
    std::size_t bytes_per_token_size = 0;
    if (!config.layout.bytesForTokens(1, bytes_per_token_size)) {
        throw std::overflow_error("bytes per token overflow");
    }
    std::uint64_t const bytes_per_token = bytes_per_token_size;
    std::uint64_t const budget_tokens =
        config.capacity_budget_bytes / bytes_per_token;

    std::vector<CapacityResult> output;
    output.reserve(kBaselines.size());

    for (BaselineKind const baseline : kBaselines) {
        CapacityResult result{};
        result.workload = trace.workload;
        result.baseline = baseline;

        for (std::uint32_t const length : trace.sequence_lengths) {
            std::uint64_t reserved = 0;
            std::uint64_t entries = 0;
            std::uint64_t micro_pages = 0;
            std::uint64_t extent_pages = 0;

            switch (baseline) {
            case BaselineKind::ContiguousMax:
                reserved = config.maximum_sequence_length;
                entries = 1;
                break;
            case BaselineKind::Fixed8:
                entries = ceilDivide(length, 8);
                reserved = entries * 8;
                break;
            case BaselineKind::Fixed16:
                entries = ceilDivide(length, 16);
                reserved = entries * 16;
                break;
            case BaselineKind::Fixed32:
                entries = ceilDivide(length, 32);
                reserved = entries * 32;
                break;
            case BaselineKind::Fixed64:
                entries = ceilDivide(length, 64);
                reserved = entries * 64;
                break;
            case BaselineKind::HeteroWithoutPromotion:
                micro_pages = ceilDivide(length, 8);
                entries = micro_pages;
                reserved = micro_pages * 8;
                break;
            case BaselineKind::HeteroWithPromotion:
                extent_pages = length / 64;
                micro_pages = ceilDivide(length % 64, 8);
                entries = extent_pages + micro_pages;
                reserved = extent_pages * 64 + micro_pages * 8;
                break;
            }

            result.used_tokens += length;
            result.reserved_tokens += reserved;
            result.block_table_entries += entries;
            result.micro_pages += micro_pages;
            result.extent_pages += extent_pages;
        }

        result.peak_reserved_tokens = result.reserved_tokens;
        if (baseline == BaselineKind::HeteroWithPromotion) {
            result.peak_reserved_tokens += std::min<std::uint64_t>(
                result.extent_pages,
                config.concurrency
            ) * kExtentPageTokenCapacity;
        }
        result.internal_fragmentation_tokens =
            result.reserved_tokens - result.used_tokens;
        result.reused_tokens = trace.reused_tokens;
        auto const toBytes = [bytes_per_token](std::uint64_t tokens) {
            if (tokens > std::numeric_limits<std::uint64_t>::max()
                / bytes_per_token) {
                throw std::overflow_error("capacity byte count overflow");
            }
            return tokens * bytes_per_token;
        };
        result.used_bytes = toBytes(result.used_tokens);
        result.reserved_bytes = toBytes(result.reserved_tokens);
        result.peak_reserved_bytes = toBytes(result.peak_reserved_tokens);
        result.internal_fragmentation_bytes = toBytes(
            result.internal_fragmentation_tokens
        );
        result.reused_bytes = toBytes(result.reused_tokens);
        result.utilization = result.reserved_tokens == 0
            ? 0.0
            : static_cast<double>(result.used_tokens)
                / static_cast<double>(result.reserved_tokens);
        if (result.reserved_tokens != 0) {
            long double const scaled_budget =
                static_cast<long double>(budget_tokens)
                * static_cast<long double>(trace.sequence_lengths.size());
            result.admitted_requests = std::min<std::uint64_t>(
                trace.sequence_lengths.size(),
                static_cast<std::uint64_t>(
                    scaled_budget
                    / static_cast<long double>(result.reserved_tokens)
                )
            );
        }
        output.push_back(result);
    }

    return output;
}

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

BenchmarkReport runCpuBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads)
{
    std::string error;
    if (!validateConfig(config, error)) {
        throw std::invalid_argument(error);
    }
    if (workloads.empty()) {
        throw std::invalid_argument("at least one workload is required");
    }

    BenchmarkReport report{};
    report.suite = "cpu_metadata";
    report.config = config;
    if (report.config.git_commit.empty()) {
        report.config.git_commit = defaultGitCommit();
    }
    report.environment = cpuEnvironment(report.config);

    for (WorkloadKind const workload : workloads) {
        WorkloadTrace const trace = generateWorkload(workload, report.config);
        std::vector<CapacityResult> const capacity =
            calculateCapacity(trace, report.config);
        report.capacity.insert(
            report.capacity.end(),
            capacity.begin(),
            capacity.end()
        );
        report.workloads.push_back(runCpuWorkload(trace, report.config));
    }
    return report;
}

bool parseCommandLine(
    int argc,
    char const* const* argv,
    CommandLineOptions& options,
    std::string& error)
{
    try {
        for (int index = 1; index < argc; ++index) {
            std::string const argument = argv[index];
            auto const nextValue = [&]() -> std::string {
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "missing value after " + argument
                    );
                }
                return argv[++index];
            };

            if (argument == "--help" || argument == "-h") {
                options.show_help = true;
            } else if (argument == "--seed") {
                options.config.seed = parseUnsigned(
                    nextValue(),
                    "--seed"
                );
            } else if (argument == "--requests") {
                options.config.request_count = parseUint32(
                    nextValue(),
                    "--requests"
                );
            } else if (argument == "--concurrency") {
                options.config.concurrency = parseUint32(
                    nextValue(),
                    "--concurrency"
                );
            } else if (argument == "--max-sequence-length") {
                options.config.maximum_sequence_length =
                    parseUint32(nextValue(), "--max-sequence-length");
            } else if (argument == "--warmup") {
                options.config.warmup_iterations =
                    parseUint32(nextValue(), "--warmup");
            } else if (argument == "--iterations") {
                options.config.measured_iterations =
                    parseUint32(nextValue(), "--iterations");
            } else if (argument == "--capacity-budget-mib") {
                std::uint64_t const mib = parseUnsigned(
                    nextValue(),
                    "--capacity-budget-mib"
                );
                if (mib > std::numeric_limits<std::uint64_t>::max()
                    / (1024ULL * 1024ULL)) {
                    throw std::invalid_argument(
                        "--capacity-budget-mib overflows bytes"
                    );
                }
                options.config.capacity_budget_bytes = mib * 1024ULL * 1024ULL;
            } else if (argument == "--layers") {
                options.config.layout.layer_count =
                    parseUint32(nextValue(), "--layers");
            } else if (argument == "--kv-heads") {
                options.config.layout.kv_head_count =
                    parseUint32(nextValue(), "--kv-heads");
            } else if (argument == "--head-dimension") {
                options.config.layout.head_dimension =
                    parseUint32(nextValue(), "--head-dimension");
            } else if (argument == "--git-commit") {
                options.config.git_commit = nextValue();
            } else if (argument == "--output-dir") {
                options.output_directory = nextValue();
            } else if (argument == "--workload") {
                std::string const value = nextValue();
                if (value == "all") {
                    options.workloads = allWorkloads();
                    continue;
                }
                std::optional<WorkloadKind> const workload =
                    parseWorkload(value);
                if (!workload.has_value()) {
                    throw std::invalid_argument(
                        "unknown workload: " + value
                    );
                }
                options.workloads.push_back(*workload);
            } else {
                throw std::invalid_argument("unknown option: " + argument);
            }
        }

        if (options.workloads.empty()) {
            options.workloads = allWorkloads();
        }
        return validateConfig(options.config, error);
    } catch (std::exception const& exception) {
        error = exception.what();
        return false;
    }
}

std::string benchmarkUsage(std::string_view program_name)
{
    std::ostringstream output;
    output
        << "Usage: " << program_name << " [options]\n"
        << "  --workload NAME          short|mixed|adversarial|long|"
           "shared_prompt|fork_cow|fault|all\n"
        << "  --seed N                 deterministic random seed\n"
        << "  --requests N             CPU trace request count (default 10000)\n"
        << "  --concurrency N          concurrent request/fork width\n"
        << "  --max-sequence-length N  maximum sequence length (default 544)\n"
        << "  --warmup N               CUDA warmup iterations\n"
        << "  --iterations N           CUDA measured iterations\n"
        << "  --capacity-budget-mib N  admission accounting budget\n"
        << "  --layers N               model layer count\n"
        << "  --kv-heads N             KV head count\n"
        << "  --head-dimension N       attention head dimension\n"
        << "  --git-commit HASH        override compiled Git revision\n"
        << "  --output-dir PATH        JSON/CSV destination directory\n"
        << "  --help                    show this message\n";
    return output.str();
}

std::string defaultGitCommit()
{
#ifdef HETEROPAGE_KV_GIT_COMMIT
    return HETEROPAGE_KV_GIT_COMMIT;
#else
    return "unknown";
#endif
}

void writeJsonReport(
    BenchmarkReport const& report,
    std::string const& path)
{
    ensureParentDirectory(path);
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open JSON report: " + path);
    }

    output << std::setprecision(10);
    output << "{\n";
    output << "  \"schema_version\": 1,\n";
    output << "  \"suite\": \"" << jsonEscape(report.suite) << "\",\n";
    output << "  \"successful\": "
           << (report.successful() ? "true" : "false") << ",\n";
    output << "  \"environment\": {"
           << "\"timestamp_utc\":\""
           << jsonEscape(report.environment.timestamp_utc) << "\","
           << "\"git_commit\":\""
           << jsonEscape(report.environment.git_commit) << "\","
           << "\"hostname\":\""
           << jsonEscape(report.environment.hostname) << "\","
           << "\"compiler\":\""
           << jsonEscape(report.environment.compiler) << "\","
           << "\"gpu_name\":\""
           << jsonEscape(report.environment.gpu_name) << "\","
           << "\"cuda_runtime\":\""
           << jsonEscape(report.environment.cuda_runtime) << "\"},\n";
    output << "  \"config\": {"
           << "\"seed\":" << report.config.seed << ','
           << "\"request_count\":" << report.config.request_count << ','
           << "\"concurrency\":" << report.config.concurrency << ','
           << "\"maximum_sequence_length\":"
           << report.config.maximum_sequence_length << ','
           << "\"warmup_iterations\":"
           << report.config.warmup_iterations << ','
           << "\"measured_iterations\":"
           << report.config.measured_iterations << ','
           << "\"capacity_budget_bytes\":"
           << report.config.capacity_budget_bytes << ','
           << "\"micro_page_tokens\":"
           << kMicroPageTokenCapacity << ','
           << "\"extent_page_tokens\":"
           << kExtentPageTokenCapacity << ','
           << "\"layout\":{"
           << "\"layers\":" << report.config.layout.layer_count << ','
           << "\"kv_heads\":" << report.config.layout.kv_head_count << ','
           << "\"head_dimension\":"
           << report.config.layout.head_dimension << "}},\n";

    output << "  \"capacity\": [\n";
    for (std::size_t index = 0; index < report.capacity.size(); ++index) {
        CapacityResult const& value = report.capacity[index];
        output << "    {"
               << "\"workload\":\"" << toString(value.workload) << "\","
               << "\"baseline\":\"" << toString(value.baseline) << "\","
               << "\"used_tokens\":" << value.used_tokens << ','
               << "\"reserved_tokens\":" << value.reserved_tokens << ','
               << "\"peak_reserved_tokens\":"
               << value.peak_reserved_tokens << ','
               << "\"internal_fragmentation_tokens\":"
               << value.internal_fragmentation_tokens << ','
               << "\"reused_tokens\":" << value.reused_tokens << ','
               << "\"used_bytes\":" << value.used_bytes << ','
               << "\"reserved_bytes\":" << value.reserved_bytes << ','
               << "\"peak_reserved_bytes\":"
               << value.peak_reserved_bytes << ','
               << "\"internal_fragmentation_bytes\":"
               << value.internal_fragmentation_bytes << ','
               << "\"reused_bytes\":" << value.reused_bytes << ','
               << "\"block_table_entries\":"
               << value.block_table_entries << ','
               << "\"micro_pages\":" << value.micro_pages << ','
               << "\"extent_pages\":" << value.extent_pages << ','
               << "\"admitted_requests\":"
               << value.admitted_requests << ','
               << "\"utilization\":" << value.utilization << '}';
        output << (index + 1 == report.capacity.size() ? "\n" : ",\n");
    }
    output << "  ],\n";

    output << "  \"workloads\": [\n";
    for (std::size_t workload_index = 0;
         workload_index < report.workloads.size();
         ++workload_index) {
        WorkloadResult const& workload = report.workloads[workload_index];
        output << "    {\"workload\":\"" << toString(workload.workload)
               << "\",\"trace_seed\":" << workload.trace_seed
               << ",\"trace_request_count\":"
               << workload.trace_request_count
               << ",\"minimum_sequence_length\":"
               << workload.minimum_sequence_length
               << ",\"maximum_sequence_length\":"
               << workload.maximum_sequence_length
               << ",\"mean_sequence_length\":"
               << workload.mean_sequence_length
               << ",\"elapsed_ns\":" << workload.elapsed_ns
               << ",\"completed_requests\":"
               << workload.completed_requests
               << ",\"requests_per_second\":"
               << workload.requests_per_second
               << ",\"failed_operations\":"
               << workload.failed_operations
               << ",\"invariants_ok\":"
               << (workload.invariants_ok ? "true" : "false")
               << ",\"resources_released\":"
               << (workload.resources_released ? "true" : "false")
               << ",\"latency\":[";

        for (std::size_t latency_index = 0;
             latency_index < workload.latency.size();
             ++latency_index) {
            LatencySummary const& value = workload.latency[latency_index];
            output << "{\"operation\":\"" << toString(value.operation)
                   << "\",\"sample_count\":" << value.sample_count
                   << ",\"failure_count\":" << value.failure_count
                   << ",\"minimum_ns\":" << value.minimum_ns
                   << ",\"maximum_ns\":" << value.maximum_ns
                   << ",\"mean_ns\":" << value.mean_ns
                   << ",\"p50_ns\":" << value.p50_ns
                   << ",\"p95_ns\":" << value.p95_ns
                   << ",\"p99_ns\":" << value.p99_ns
                   << ",\"total_bytes\":" << value.total_bytes
                   << ",\"effective_bandwidth_gbps\":"
                   << value.effective_bandwidth_gbps << '}';
            if (latency_index + 1 != workload.latency.size()) {
                output << ',';
            }
        }

        output << "],\"samples\":[";
        for (std::size_t sample_index = 0;
             sample_index < workload.samples.size();
             ++sample_index) {
            OperationSample const& value = workload.samples[sample_index];
            output << "{\"operation\":\"" << toString(value.operation)
                   << "\",\"sample_index\":" << value.sample_index
                   << ",\"duration_ns\":" << value.duration_ns
                   << ",\"bytes\":" << value.bytes
                   << ",\"token_count\":" << value.token_count
                   << ",\"success\":"
                   << (value.success ? "true" : "false")
                   << ",\"detail\":\"" << jsonEscape(value.detail)
                   << "\"}";
            if (sample_index + 1 != workload.samples.size()) {
                output << ',';
            }
        }
        output << "]}";
        output << (
            workload_index + 1 == report.workloads.size()
                ? "\n"
                : ",\n"
        );
    }
    output << "  ]\n}\n";
}

void writeCsvReport(
    BenchmarkReport const& report,
    std::string const& path)
{
    ensureParentDirectory(path);
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open CSV report: " + path);
    }

    constexpr std::size_t kColumnCount = 45;
    writeCsvRow(output, {
        "record_type",
        "suite",
        "workload",
        "baseline",
        "operation",
        "sample_index",
        "duration_ns",
        "bytes",
        "token_count",
        "success",
        "detail",
        "used_tokens",
        "reserved_tokens",
        "peak_reserved_tokens",
        "fragmentation_tokens",
        "entries",
        "micro_pages",
        "extent_pages",
        "admitted_requests",
        "utilization",
        "p50_ns",
        "p95_ns",
        "p99_ns",
        "git_commit",
        "seed",
        "request_count",
        "concurrency",
        "requests_per_second",
        "mean_ns",
        "total_bytes",
        "effective_bandwidth_gbps",
        "failure_count",
        "minimum_ns",
        "maximum_ns",
        "trace_seed",
        "trace_request_count",
        "minimum_sequence_length",
        "maximum_sequence_length",
        "mean_sequence_length",
        "reused_tokens",
        "used_bytes",
        "reserved_bytes",
        "peak_reserved_bytes",
        "fragmentation_bytes",
        "reused_bytes",
    });

    auto const baseRow = [&report]() {
        std::vector<std::string> row(kColumnCount);
        row[1] = report.suite;
        row[23] = report.environment.git_commit;
        row[24] = std::to_string(report.config.seed);
        row[25] = std::to_string(report.config.request_count);
        row[26] = std::to_string(report.config.concurrency);
        return row;
    };

    for (CapacityResult const& value : report.capacity) {
        std::vector<std::string> row = baseRow();
        row[0] = "capacity";
        row[2] = toString(value.workload);
        row[3] = toString(value.baseline);
        row[11] = std::to_string(value.used_tokens);
        row[12] = std::to_string(value.reserved_tokens);
        row[13] = std::to_string(value.peak_reserved_tokens);
        row[14] = std::to_string(value.internal_fragmentation_tokens);
        row[15] = std::to_string(value.block_table_entries);
        row[16] = std::to_string(value.micro_pages);
        row[17] = std::to_string(value.extent_pages);
        row[18] = std::to_string(value.admitted_requests);
        row[19] = decimal(value.utilization);
        row[39] = std::to_string(value.reused_tokens);
        row[40] = std::to_string(value.used_bytes);
        row[41] = std::to_string(value.reserved_bytes);
        row[42] = std::to_string(value.peak_reserved_bytes);
        row[43] = std::to_string(value.internal_fragmentation_bytes);
        row[44] = std::to_string(value.reused_bytes);
        writeCsvRow(output, row);
    }

    for (WorkloadResult const& workload : report.workloads) {
        auto const workloadRow = [&]() {
            std::vector<std::string> row = baseRow();
            row[2] = toString(workload.workload);
            row[27] = decimal(workload.requests_per_second);
            row[34] = std::to_string(workload.trace_seed);
            row[35] = std::to_string(workload.trace_request_count);
            row[36] = std::to_string(workload.minimum_sequence_length);
            row[37] = std::to_string(workload.maximum_sequence_length);
            row[38] = decimal(workload.mean_sequence_length);
            return row;
        };

        for (LatencySummary const& value : workload.latency) {
            std::vector<std::string> row = workloadRow();
            row[0] = "summary";
            row[4] = toString(value.operation);
            row[20] = std::to_string(value.p50_ns);
            row[21] = std::to_string(value.p95_ns);
            row[22] = std::to_string(value.p99_ns);
            row[28] = std::to_string(value.mean_ns);
            row[29] = std::to_string(value.total_bytes);
            row[30] = decimal(value.effective_bandwidth_gbps);
            row[31] = std::to_string(value.failure_count);
            row[32] = std::to_string(value.minimum_ns);
            row[33] = std::to_string(value.maximum_ns);
            writeCsvRow(output, row);
        }

        for (OperationSample const& value : workload.samples) {
            std::vector<std::string> row = workloadRow();
            row[0] = "sample";
            row[4] = toString(value.operation);
            row[5] = std::to_string(value.sample_index);
            row[6] = std::to_string(value.duration_ns);
            row[7] = std::to_string(value.bytes);
            row[8] = std::to_string(value.token_count);
            row[9] = value.success ? "true" : "false";
            row[10] = value.detail;
            writeCsvRow(output, row);
        }
    }
}

} // namespace kimkvcache::benchmark
