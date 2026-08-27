#include "cpu_benchmark_internal.h"

#include <algorithm>

namespace kimkvcache::benchmark::cpu_detail {

WorkloadResult runFaultWorkload(WorkloadTrace const& trace)
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

} // namespace kimkvcache::benchmark::cpu_detail
