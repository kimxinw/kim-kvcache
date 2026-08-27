#include "benchmark_workload.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace kimkvcache::benchmark::detail {

BenchmarkStream::BenchmarkStream()
{
    error_ = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
}

BenchmarkStream::~BenchmarkStream()
{
    if (stream_ != nullptr) {
        static_cast<void>(cudaStreamDestroy(stream_));
    }
}

bool BenchmarkStream::ok() const noexcept
{
    return error_ == cudaSuccess;
}

cudaStream_t BenchmarkStream::native() const noexcept
{
    return stream_;
}

CudaStream BenchmarkStream::opaque() const noexcept
{
    return reinterpret_cast<CudaStream>(stream_);
}

GpuTimer::GpuTimer()
{
    cudaError_t const start_error = cudaEventCreate(&start_);
    cudaError_t const stop_error = start_error == cudaSuccess
        ? cudaEventCreate(&stop_)
        : start_error;
    error_ = start_error == cudaSuccess ? stop_error : start_error;
}

GpuTimer::~GpuTimer()
{
    if (start_ != nullptr) {
        static_cast<void>(cudaEventDestroy(start_));
    }
    if (stop_ != nullptr) {
        static_cast<void>(cudaEventDestroy(stop_));
    }
}

bool GpuTimer::ok() const noexcept
{
    return error_ == cudaSuccess;
}

std::uint64_t bytesForTokens(
    BenchmarkConfig const& config,
    std::uint32_t token_count)
{
    std::size_t bytes = 0;
    if (!config.layout.bytesForTokens(token_count, bytes)) {
        throw std::overflow_error("CUDA benchmark byte count overflow");
    }
    return bytes;
}

std::string operationDetail(
    CudaKvOperationResult const& result,
    std::string prefix)
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

std::uint32_t representativeLength(
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

} // namespace kimkvcache::benchmark::detail
