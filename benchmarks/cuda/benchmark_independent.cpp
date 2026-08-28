#include "benchmark_workload.h"

namespace kimkvcache::benchmark::detail {

void runIndependentIteration(
    CudaBenchmarkCache& cache,
    DeviceBuffer<KvScalar>& input,
    DeviceBuffer<KvScalar>& output,
    DeviceBuffer<float>& query,
    DeviceBuffer<float>& attention,
    BenchmarkStream const& stream,
    GpuTimer& timer,
    WorkloadResult& result,
    BenchmarkConfig const& config,
    WorkloadKind workload,
    RequestId request_id,
    std::uint32_t iteration,
    bool measured)
{
    std::uint32_t const length = representativeLength(
        workload,
        iteration,
        config
    );
    KvCacheError const create_error = cache.createRequest(request_id);
    if (create_error != KvCacheError::None) {
        recordMetadataFailure(
            result,
            OperationKind::Create,
            create_error,
            "create_failed"
        );
        return;
    }

    CudaKvOperationResult const append_result = recordCudaOperation(
        result,
        timer,
        stream,
        OperationKind::Append,
        length,
        bytesForTokens(config, length),
        measured,
        true,
        [&cache, &input, &stream, request_id, length]() {
            return cache.append(
                request_id,
                length,
                input.data(),
                stream.opaque()
            );
        }
    );

    if (append_result.ok()) {
        if (cache.supportsPromotion()) {
            std::uint32_t const promotable =
                length / kExtentPageTokenCapacity
                * kExtentPageTokenCapacity;
            for (std::uint32_t logical_begin = 0;
                 logical_begin < promotable;
                 logical_begin += kExtentPageTokenCapacity) {
                static_cast<void>(recordCudaOperation(
                    result,
                    timer,
                    stream,
                    OperationKind::Promote,
                    kExtentPageTokenCapacity,
                    bytesForTokens(config, kExtentPageTokenCapacity),
                    measured,
                    true,
                    [&cache, &stream, request_id, logical_begin]() {
                        return cache.promote(
                            request_id,
                            logical_begin,
                            stream.opaque()
                        );
                    }
                ));
            }
        }

        static_cast<void>(recordCudaOperation(
            result,
            timer,
            stream,
            OperationKind::Gather,
            length,
            bytesForTokens(config, length),
            measured,
            true,
            [&cache, &output, &stream, request_id]() {
                return cache.gather(
                    request_id,
                    output.data(),
                    stream.opaque()
                );
            }
        ));

        std::uint64_t const attention_bytes =
            bytesForTokens(config, length)
            + 2ULL
                * config.layout.layer_count
                * config.layout.kv_head_count
                * config.layout.head_dimension
                * sizeof(float);
        static_cast<void>(recordCudaOperation(
            result,
            timer,
            stream,
            OperationKind::Attention,
            length,
            attention_bytes,
            measured,
            true,
            [&cache, &query, &attention, &stream, request_id]() {
                return cache.referenceAttention(
                    request_id,
                    query.data(),
                    attention.data(),
                    stream.opaque()
                );
            }
        ));
    }

    KvCacheError const release_error = cache.releaseRequest(request_id);
    recordReleaseFailure(result, release_error, measured);
    if (release_error == KvCacheError::None && measured) {
        ++result.completed_requests;
    }
}

} // namespace kimkvcache::benchmark::detail
