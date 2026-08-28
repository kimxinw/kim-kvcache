#include "benchmark_workload.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace kimkvcache::benchmark::detail {

void runSharedIteration(
    CudaBenchmarkCache& cache,
    DeviceBuffer<KvScalar>& input,
    DeviceBuffer<KvScalar>& output,
    BenchmarkStream const& stream,
    GpuTimer& timer,
    WorkloadResult& result,
    BenchmarkConfig const& config,
    RequestId& next_request_id,
    std::uint32_t iteration,
    bool measured)
{
    constexpr std::array<std::uint32_t, 3> kPrefixes{64, 128, 256};
    constexpr std::array<std::uint32_t, 3> kFanouts{2, 4, 8};
    std::uint32_t const prefix = std::min(
        kPrefixes[iteration % kPrefixes.size()],
        config.maximum_sequence_length - 1
    );
    std::uint32_t const fanout = kFanouts[iteration % kFanouts.size()];
    RequestId const parent = next_request_id++;
    if (cache.createRequest(parent) != KvCacheError::None) {
        throw std::runtime_error("shared prompt parent creation failed");
    }

    static_cast<void>(recordCudaOperation(
        result,
        timer,
        stream,
        OperationKind::Append,
        prefix,
        bytesForTokens(config, prefix),
        measured,
        true,
        [&cache, &input, &stream, parent, prefix]() {
            return cache.append(
                parent,
                prefix,
                input.data(),
                stream.opaque()
            );
        }
    ));

    std::vector<RequestId> children;
    children.reserve(fanout);
    for (std::uint32_t child_index = 0;
         child_index < fanout;
         ++child_index) {
        RequestId const child = next_request_id++;
        KvCacheError const fork_error = cache.forkRequest(parent, child);
        if (fork_error != KvCacheError::None) {
            recordMetadataFailure(
                result,
                OperationKind::Fork,
                fork_error,
                "shared_prompt_fork"
            );
            continue;
        }
        children.push_back(child);
        static_cast<void>(recordCudaOperation(
            result,
            timer,
            stream,
            OperationKind::Append,
            1,
            bytesForTokens(config, 1),
            measured,
            true,
            [&cache, &input, &stream, child]() {
                return cache.append(child, 1, input.data(), stream.opaque());
            }
        ));
        static_cast<void>(recordCudaOperation(
            result,
            timer,
            stream,
            OperationKind::Gather,
            prefix + 1,
            bytesForTokens(config, prefix + 1),
            measured,
            true,
            [&cache, &output, &stream, child]() {
                return cache.gather(child, output.data(), stream.opaque());
            }
        ));
    }

    for (RequestId const child : children) {
        KvCacheError const release_error = cache.releaseRequest(child);
        recordReleaseFailure(result, release_error, measured);
        if (release_error == KvCacheError::None && measured) {
            ++result.completed_requests;
        }
    }
    recordReleaseFailure(result, cache.releaseRequest(parent), measured);
}

void runForkCowIteration(
    CudaBenchmarkCache& cache,
    DeviceBuffer<KvScalar>& input,
    DeviceBuffer<KvScalar>& output,
    BenchmarkStream const& stream,
    GpuTimer& timer,
    WorkloadResult& result,
    BenchmarkConfig const& config,
    RequestId& next_request_id,
    std::uint32_t iteration,
    bool measured)
{
    constexpr std::array<std::uint32_t, 5> kForkPoints{1, 7, 8, 9, 63};
    std::uint32_t const point = std::min(
        kForkPoints[iteration % kForkPoints.size()],
        config.maximum_sequence_length - 1
    );
    RequestId const parent = next_request_id++;
    RequestId const child = next_request_id++;
    if (cache.createRequest(parent) != KvCacheError::None) {
        throw std::runtime_error("Fork/COW parent creation failed");
    }
    static_cast<void>(recordCudaOperation(
        result,
        timer,
        stream,
        OperationKind::Append,
        point,
        bytesForTokens(config, point),
        measured,
        true,
        [&cache, &input, &stream, parent, point]() {
            return cache.append(parent, point, input.data(), stream.opaque());
        }
    ));
    KvCacheError const seal_error = cache.sealTail(parent);
    KvCacheError const fork_error = seal_error == KvCacheError::None
        ? cache.forkRequest(parent, child)
        : seal_error;
    if (fork_error != KvCacheError::None) {
        recordMetadataFailure(
            result,
            OperationKind::Fork,
            fork_error,
            "fork_cow_setup"
        );
    } else {
        static_cast<void>(recordCudaOperation(
            result,
            timer,
            stream,
            OperationKind::CowAppend,
            1,
            bytesForTokens(config, 1),
            measured,
            true,
            [&cache, &input, &stream, child]() {
                return cache.append(child, 1, input.data(), stream.opaque());
            }
        ));
        static_cast<void>(recordCudaOperation(
            result,
            timer,
            stream,
            OperationKind::Gather,
            point + 1,
            bytesForTokens(config, point + 1),
            measured,
            true,
            [&cache, &output, &stream, child]() {
                return cache.gather(child, output.data(), stream.opaque());
            }
        ));
        KvCacheError const child_release = cache.releaseRequest(child);
        recordReleaseFailure(result, child_release, measured);
        if (child_release == KvCacheError::None && measured) {
            ++result.completed_requests;
        }
    }
    recordReleaseFailure(result, cache.releaseRequest(parent), measured);
}

} // namespace kimkvcache::benchmark::detail
