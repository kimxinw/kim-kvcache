#include "heteropage_kv/benchmark.h"

#include "heteropage_kv/cuda_kv_cache.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kimkvcache::benchmark {
namespace {

template <typename T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t element_count)
        : element_count_(element_count)
    {
        if (element_count_ != 0) {
            error_ = cudaMalloc(
                reinterpret_cast<void**>(&data_),
                element_count_ * sizeof(T)
            );
        }
    }

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    [[nodiscard]] bool ok() const noexcept
    {
        return error_ == cudaSuccess;
    }

    [[nodiscard]] T* data() noexcept
    {
        return data_;
    }

    [[nodiscard]] bool clear() noexcept
    {
        return element_count_ == 0
            || cudaMemset(
                data_,
                0,
                element_count_ * sizeof(T)
            ) == cudaSuccess;
    }

private:
    T* data_{nullptr};
    std::size_t element_count_{0};
    cudaError_t error_{cudaSuccess};
};

class BenchmarkStream final {
public:
    BenchmarkStream()
    {
        error_ = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    }

    ~BenchmarkStream()
    {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }
    }

    BenchmarkStream(BenchmarkStream const&) = delete;
    BenchmarkStream& operator=(BenchmarkStream const&) = delete;

    [[nodiscard]] bool ok() const noexcept
    {
        return error_ == cudaSuccess;
    }

    [[nodiscard]] cudaStream_t native() const noexcept
    {
        return stream_;
    }

    [[nodiscard]] CudaStream opaque() const noexcept
    {
        return reinterpret_cast<CudaStream>(stream_);
    }

private:
    cudaStream_t stream_{nullptr};
    cudaError_t error_{cudaSuccess};
};

class GpuTimer final {
public:
    GpuTimer()
    {
        cudaError_t const start_error = cudaEventCreate(&start_);
        cudaError_t const stop_error = start_error == cudaSuccess
            ? cudaEventCreate(&stop_)
            : start_error;
        error_ = start_error == cudaSuccess ? stop_error : start_error;
    }

    ~GpuTimer()
    {
        if (start_ != nullptr) {
            static_cast<void>(cudaEventDestroy(start_));
        }
        if (stop_ != nullptr) {
            static_cast<void>(cudaEventDestroy(stop_));
        }
    }

    GpuTimer(GpuTimer const&) = delete;
    GpuTimer& operator=(GpuTimer const&) = delete;

    [[nodiscard]] bool ok() const noexcept
    {
        return error_ == cudaSuccess;
    }

    template <typename Callable>
    [[nodiscard]] std::pair<CudaKvOperationResult, std::uint64_t> measure(
        cudaStream_t stream,
        Callable&& callable)
    {
        if (cudaEventRecord(start_, stream) != cudaSuccess) {
            throw std::runtime_error("cannot record CUDA benchmark start event");
        }
        CudaKvOperationResult result = callable();
        if (cudaEventRecord(stop_, stream) != cudaSuccess
            || cudaEventSynchronize(stop_) != cudaSuccess) {
            throw std::runtime_error("cannot record CUDA benchmark stop event");
        }

        float milliseconds = 0.0F;
        if (cudaEventElapsedTime(&milliseconds, start_, stop_)
            != cudaSuccess) {
            throw std::runtime_error("cannot read CUDA benchmark event time");
        }
        std::uint64_t const nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(milliseconds) * 1'000'000.0)
        );
        return {result, nanoseconds};
    }

private:
    cudaEvent_t start_{nullptr};
    cudaEvent_t stop_{nullptr};
    cudaError_t error_{cudaSuccess};
};

[[nodiscard]] std::uint64_t bytesForTokens(
    BenchmarkConfig const& config,
    std::uint32_t token_count)
{
    std::size_t bytes = 0;
    if (!config.layout.bytesForTokens(token_count, bytes)) {
        throw std::overflow_error("CUDA benchmark byte count overflow");
    }
    return bytes;
}

[[nodiscard]] std::string operationDetail(
    CudaKvOperationResult const& result,
    std::string prefix = {})
{
    if (!prefix.empty()) {
        prefix += ':';
    }
    prefix += "metadata=";
    prefix += toString(result.metadata_error);
    prefix += ";cuda=";
    prefix += toString(result.cuda_status.error);
    prefix += ";native=";
    prefix += std::to_string(result.cuda_status.native_error);
    return prefix;
}

template <typename Callable>
CudaKvOperationResult recordCudaOperation(
    WorkloadResult& workload,
    GpuTimer& timer,
    BenchmarkStream const& stream,
    OperationKind operation,
    std::uint32_t token_count,
    std::uint64_t bytes,
    bool measured,
    bool expect_success,
    Callable&& callable,
    std::string detail_prefix = {})
{
    auto measured_result = timer.measure(
        stream.native(),
        std::forward<Callable>(callable)
    );
    bool const success = measured_result.first.ok() == expect_success;

    if (measured) {
        workload.samples.push_back(OperationSample{
            operation,
            workload.samples.size(),
            measured_result.second,
            bytes,
            token_count,
            success,
            operationDetail(
                measured_result.first,
                std::move(detail_prefix)
            ),
        });
        if (!success) {
            ++workload.failed_operations;
        }
    } else if (!success) {
        throw std::runtime_error(
            "CUDA warmup operation returned an unexpected status"
        );
    }

    return measured_result.first;
}

void recordMetadataFailure(
    WorkloadResult& result,
    OperationKind operation,
    KvCacheError error,
    std::string detail)
{
    result.samples.push_back(OperationSample{
        operation,
        result.samples.size(),
        0,
        0,
        0,
        false,
        std::move(detail) + ':' + std::string(toString(error)),
    });
    ++result.failed_operations;
}

[[nodiscard]] std::uint32_t representativeLength(
    WorkloadKind workload,
    std::uint32_t iteration,
    BenchmarkConfig const& config)
{
    constexpr std::array<std::uint32_t, 3> kMixed{64, 256, 544};
    constexpr std::array<std::uint32_t, 3> kAdversarial{65, 257, 513};
    switch (workload) {
    case WorkloadKind::Short:
        return std::min<std::uint32_t>(32, config.maximum_sequence_length);
    case WorkloadKind::Mixed:
        return std::min(
            kMixed[iteration % kMixed.size()],
            config.maximum_sequence_length
        );
    case WorkloadKind::Adversarial:
        return std::min(
            kAdversarial[iteration % kAdversarial.size()],
            config.maximum_sequence_length
        );
    case WorkloadKind::Long:
        return std::min<std::uint32_t>(512, config.maximum_sequence_length);
    case WorkloadKind::SharedPrompt:
        return std::min<std::uint32_t>(128, config.maximum_sequence_length);
    case WorkloadKind::ForkCow:
        return std::min<std::uint32_t>(63, config.maximum_sequence_length);
    case WorkloadKind::Fault:
        return std::min<std::uint32_t>(64, config.maximum_sequence_length);
    }
    return 1;
}

void recordReleaseFailure(
    WorkloadResult& result,
    KvCacheError error,
    bool measured)
{
    if (error != KvCacheError::None && measured) {
        recordMetadataFailure(
            result,
            OperationKind::Release,
            error,
            "release_failed"
        );
    } else if (error != KvCacheError::None) {
        throw std::runtime_error("CUDA warmup request release failed");
    }
}

void runIndependentIteration(
    CudaKvCache& cache,
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

void runSharedIteration(
    CudaKvCache& cache,
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
    CudaKvCache& cache,
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

[[nodiscard]] WorkloadResult runCudaWorkload(
    WorkloadKind workload,
    BenchmarkConfig const& config)
{
    std::uint64_t const micro_capacity_value =
        (static_cast<std::uint64_t>(config.maximum_sequence_length) + 7) / 8
        + 32;
    std::uint64_t const extent_capacity_value =
        (static_cast<std::uint64_t>(config.maximum_sequence_length) + 63) / 64
        + 8;
    if (micro_capacity_value > std::numeric_limits<std::uint32_t>::max()
        || extent_capacity_value >
            std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("CUDA benchmark pool capacity overflow");
    }
    CudaKvCache cache(
        config.layout,
        static_cast<std::uint32_t>(micro_capacity_value),
        static_cast<std::uint32_t>(extent_capacity_value)
    );
    if (!cache.status().ok()) {
        throw std::runtime_error("cannot initialize CUDA KV cache");
    }

    std::size_t maximum_elements = 0;
    if (!config.layout.elementsForTokens(
            config.maximum_sequence_length,
            maximum_elements)) {
        throw std::overflow_error("maximum CUDA input size overflow");
    }
    std::size_t attention_elements =
        static_cast<std::size_t>(config.layout.layer_count)
        * config.layout.kv_head_count
        * config.layout.head_dimension;

    DeviceBuffer<KvScalar> input(maximum_elements);
    DeviceBuffer<KvScalar> output(maximum_elements);
    DeviceBuffer<float> query(attention_elements);
    DeviceBuffer<float> attention(attention_elements);
    BenchmarkStream stream;
    GpuTimer timer;
    if (!input.ok() || !output.ok() || !query.ok() || !attention.ok()
        || !stream.ok() || !timer.ok()
        || !input.clear() || !output.clear()
        || !query.clear() || !attention.clear()) {
        throw std::runtime_error("cannot allocate CUDA benchmark resources");
    }

    WorkloadResult result{};
    result.workload = workload;
    RequestId next_request_id = 1;
    std::uint32_t const total_iterations =
        config.warmup_iterations + config.measured_iterations;

    auto const wall_begin = std::chrono::steady_clock::now();
    for (std::uint32_t iteration = 0;
         iteration < total_iterations;
         ++iteration) {
        bool const measured = iteration >= config.warmup_iterations;
        std::uint32_t const measured_index = measured
            ? iteration - config.warmup_iterations
            : iteration;

        switch (workload) {
        case WorkloadKind::Short:
        case WorkloadKind::Mixed:
        case WorkloadKind::Adversarial:
        case WorkloadKind::Long:
            runIndependentIteration(
                cache,
                input,
                output,
                query,
                attention,
                stream,
                timer,
                result,
                config,
                workload,
                next_request_id++,
                measured_index,
                measured
            );
            break;
        case WorkloadKind::SharedPrompt:
            runSharedIteration(
                cache,
                input,
                output,
                stream,
                timer,
                result,
                config,
                next_request_id,
                measured_index,
                measured
            );
            break;
        case WorkloadKind::ForkCow:
            runForkCowIteration(
                cache,
                input,
                output,
                stream,
                timer,
                result,
                config,
                next_request_id,
                measured_index,
                measured
            );
            break;
        case WorkloadKind::Fault:
            runFaultIteration(
                cache,
                input,
                stream,
                timer,
                result,
                config,
                next_request_id,
                measured_index,
                measured
            );
            break;
        }
    }

    result.elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wall_begin
        ).count()
    );
    result.requests_per_second = result.elapsed_ns == 0
        ? 0.0
        : static_cast<double>(result.completed_requests) * 1.0e9
            / static_cast<double>(result.elapsed_ns);
    result.invariants_ok = cache.checkInvariants();
    KvCacheManagerSnapshot const snapshot = cache.metadataSnapshot();
    result.resources_released = snapshot.request_count == 0
        && snapshot.promotion_count == 0
        && snapshot.page_lease_count == 0
        && snapshot.micro_pool.allocated_slots == 0
        && snapshot.extent_pool.allocated_slots == 0;

    constexpr std::array<OperationKind, 6> kCudaOperations{
        OperationKind::Append,
        OperationKind::CowAppend,
        OperationKind::Promote,
        OperationKind::PromotionRollback,
        OperationKind::Gather,
        OperationKind::Attention,
    };
    for (OperationKind const operation : kCudaOperations) {
        LatencySummary const summary = summarizeLatency(
            operation,
            result.samples
        );
        if (summary.sample_count != 0) {
            result.latency.push_back(summary);
        }
    }
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

[[nodiscard]] EnvironmentInfo cudaEnvironment(BenchmarkConfig const& config)
{
    int device = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess
        || cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        throw std::runtime_error("cannot query CUDA device properties");
    }
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        throw std::runtime_error("cannot query CUDA runtime version");
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess) {
        throw std::runtime_error("cannot query CUDA driver version");
    }
    std::ostringstream cuda_version;
    cuda_version << "runtime=" << runtime_version / 1000 << '.'
                 << (runtime_version % 1000) / 10
                 << ";driver=" << driver_version / 1000 << '.'
                 << (driver_version % 1000) / 10;
    char const* hostname = std::getenv("HOSTNAME");

    return EnvironmentInfo{
        timestampUtc(),
        config.git_commit.empty() ? defaultGitCommit() : config.git_commit,
        hostname == nullptr ? "unknown" : hostname,
        "NVCC/C++17",
        properties.name,
        cuda_version.str(),
    };
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

} // namespace

BenchmarkReport runCudaBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads)
{
    std::string error;
    if (!validateConfig(config, error)) {
        throw std::invalid_argument(error);
    }
    if (workloads.empty()) {
        throw std::invalid_argument("at least one CUDA workload is required");
    }

    BenchmarkReport report{};
    report.suite = "cuda_data_path";
    report.config = config;
    if (report.config.git_commit.empty()) {
        report.config.git_commit = defaultGitCommit();
    }
    report.environment = cudaEnvironment(report.config);

    for (WorkloadKind const workload : workloads) {
        WorkloadTrace const trace = generateWorkload(workload, report.config);
        std::vector<CapacityResult> const capacity =
            calculateCapacity(trace, report.config);
        report.capacity.insert(
            report.capacity.end(),
            capacity.begin(),
            capacity.end()
        );
        WorkloadResult result = runCudaWorkload(workload, report.config);
        attachTraceStatistics(result, trace);
        report.workloads.push_back(std::move(result));
    }
    return report;
}

} // namespace kimkvcache::benchmark
