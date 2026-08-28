#include "heteropage_kv/benchmark/benchmark.h"

#include "benchmark_workload.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace kimkvcache::benchmark {
namespace {

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

[[nodiscard]] BenchmarkReport runCudaBenchmarkImpl(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads,
    std::uint16_t fixed_page_tokens)
{
    std::string error;
    if (!validateConfig(config, error)) {
        throw std::invalid_argument(error);
    }
    if (workloads.empty()) {
        throw std::invalid_argument("at least one CUDA workload is required");
    }
    if (fixed_page_tokens != 0
        && fixed_page_tokens != 8
        && fixed_page_tokens != 16
        && fixed_page_tokens != 32
        && fixed_page_tokens != 64) {
        throw std::invalid_argument(
            "Fixed CUDA page tokens must be one of 8/16/32/64"
        );
    }
    if (fixed_page_tokens != 0
        && std::find(
            workloads.begin(),
            workloads.end(),
            WorkloadKind::Fault
        ) != workloads.end()) {
        throw std::invalid_argument(
            "Fixed CUDA benchmark does not support fault workload"
        );
    }

    BenchmarkReport report{};
    report.suite = fixed_page_tokens == 0
        ? "cuda_data_path"
        : "cuda_data_path_fixed_" + std::to_string(fixed_page_tokens);
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
        WorkloadResult result = detail::runCudaWorkload(
            workload,
            report.config,
            fixed_page_tokens
        );
        attachTraceStatistics(result, trace);
        report.workloads.push_back(std::move(result));
    }
    return report;
}

} // namespace

BenchmarkReport runCudaBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads)
{
    return runCudaBenchmarkImpl(config, workloads, 0);
}

BenchmarkReport runCudaFixedBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads,
    std::uint16_t tokens_per_page)
{
    return runCudaBenchmarkImpl(config, workloads, tokens_per_page);
}

} // namespace kimkvcache::benchmark
