#include "cpu_benchmark_internal.h"

#include <algorithm>
#include <array>
#include <vector>

namespace kimkvcache::benchmark::cpu_detail {

WorkloadResult runSharedPrompt(
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

WorkloadResult runForkCow(
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

} // namespace kimkvcache::benchmark::cpu_detail
