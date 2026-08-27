#pragma once

#include "heteropage_kv/benchmark/benchmark.h"
#include "heteropage_kv/cuda/cuda_kv_cache.h"

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace kimkvcache::benchmark::detail {

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
            || cudaMemset(data_, 0, element_count_ * sizeof(T))
                == cudaSuccess;
    }

private:
    T* data_{nullptr};
    std::size_t element_count_{0};
    cudaError_t error_{cudaSuccess};
};

class BenchmarkStream final {
public:
    BenchmarkStream();
    ~BenchmarkStream();

    BenchmarkStream(BenchmarkStream const&) = delete;
    BenchmarkStream& operator=(BenchmarkStream const&) = delete;

    [[nodiscard]] bool ok() const noexcept;
    [[nodiscard]] cudaStream_t native() const noexcept;
    [[nodiscard]] CudaStream opaque() const noexcept;

private:
    cudaStream_t stream_{nullptr};
    cudaError_t error_{cudaSuccess};
};

class GpuTimer final {
public:
    GpuTimer();
    ~GpuTimer();

    GpuTimer(GpuTimer const&) = delete;
    GpuTimer& operator=(GpuTimer const&) = delete;

    [[nodiscard]] bool ok() const noexcept;

    template <typename Callable>
    [[nodiscard]] std::pair<CudaKvOperationResult, std::uint64_t> measure(
        cudaStream_t stream,
        Callable&& callable)
    {
        if (cudaEventRecord(start_, stream) != cudaSuccess) {
            throw std::runtime_error(
                "cannot record CUDA benchmark start event"
            );
        }
        CudaKvOperationResult result = callable();
        if (cudaEventRecord(stop_, stream) != cudaSuccess
            || cudaEventSynchronize(stop_) != cudaSuccess) {
            throw std::runtime_error(
                "cannot record CUDA benchmark stop event"
            );
        }

        float milliseconds = 0.0F;
        if (cudaEventElapsedTime(&milliseconds, start_, stop_)
            != cudaSuccess) {
            throw std::runtime_error(
                "cannot read CUDA benchmark event time"
            );
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
    std::uint32_t token_count);

[[nodiscard]] std::string operationDetail(
    CudaKvOperationResult const& result,
    std::string prefix = {});

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
    std::string detail);

[[nodiscard]] std::uint32_t representativeLength(
    WorkloadKind workload,
    std::uint32_t iteration,
    BenchmarkConfig const& config);

void recordReleaseFailure(
    WorkloadResult& result,
    KvCacheError error,
    bool measured);

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
    bool measured);

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
    bool measured);

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
    bool measured);

void runFaultIteration(
    CudaKvCache& cache,
    DeviceBuffer<KvScalar>& input,
    BenchmarkStream const& stream,
    GpuTimer& timer,
    WorkloadResult& result,
    BenchmarkConfig const& config,
    RequestId& next_request_id,
    std::uint32_t iteration,
    bool measured);

[[nodiscard]] WorkloadResult runCudaWorkload(
    WorkloadKind workload,
    BenchmarkConfig const& config);

} // namespace kimkvcache::benchmark::detail
