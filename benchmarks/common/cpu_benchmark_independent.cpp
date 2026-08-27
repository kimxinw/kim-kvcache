#include "cpu_benchmark_internal.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace kimkvcache::benchmark::cpu_detail {

WorkloadResult runIndependentRequests(
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

} // namespace kimkvcache::benchmark::cpu_detail
