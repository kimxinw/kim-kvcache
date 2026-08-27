#include "benchmark_workload.h"

#include <stdexcept>

namespace kimkvcache::benchmark::detail {

void runFaultIteration(
    CudaKvCache& cache,
    DeviceBuffer<KvScalar>& input,
    BenchmarkStream const& stream,
    GpuTimer& timer,
    WorkloadResult& result,
    BenchmarkConfig const& config,
    RequestId& next_request_id,
    std::uint32_t iteration,
    bool measured)
{
    RequestId const request_id = next_request_id++;
    if (cache.createRequest(request_id) != KvCacheError::None) {
        throw std::runtime_error("fault request creation failed");
    }

    if (((config.seed ^ iteration) & 1ULL) == 0) {
        cache.injectFailureOnce(CudaFailurePoint::Completion);
        static_cast<void>(recordCudaOperation(
            result,
            timer,
            stream,
            OperationKind::Append,
            8,
            bytesForTokens(config, 8),
            measured,
            false,
            [&cache, &input, &stream, request_id]() {
                return cache.append(
                    request_id,
                    8,
                    input.data(),
                    stream.opaque()
                );
            },
            "expected_completion_failure"
        ));
        if (cache.blockTable(request_id).has_value()) {
            recordMetadataFailure(
                result,
                OperationKind::Release,
                KvCacheError::InvalidState,
                "failed_append_request_not_cancelled"
            );
        }
    } else {
        static_cast<void>(recordCudaOperation(
            result,
            timer,
            stream,
            OperationKind::Append,
            kExtentPageTokenCapacity,
            bytesForTokens(config, kExtentPageTokenCapacity),
            measured,
            true,
            [&cache, &input, &stream, request_id]() {
                return cache.append(
                    request_id,
                    kExtentPageTokenCapacity,
                    input.data(),
                    stream.opaque()
                );
            }
        ));
        cache.injectFailureOnce(CudaFailurePoint::Completion);
        static_cast<void>(recordCudaOperation(
            result,
            timer,
            stream,
            OperationKind::PromotionRollback,
            kExtentPageTokenCapacity,
            bytesForTokens(config, kExtentPageTokenCapacity),
            measured,
            false,
            [&cache, &stream, request_id]() {
                return cache.promote(request_id, 0, stream.opaque());
            },
            "expected_copy_failure"
        ));
        KvCacheError const release_error = cache.releaseRequest(request_id);
        recordReleaseFailure(result, release_error, measured);
    }
    if (measured) {
        ++result.completed_requests;
    }
}

} // namespace kimkvcache::benchmark::detail
