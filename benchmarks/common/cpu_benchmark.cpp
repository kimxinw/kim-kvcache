#include "heteropage_kv/benchmark/benchmark.h"

#include "cpu_benchmark_internal.h"

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

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

[[nodiscard]] std::string compilerName()
{
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
    return std::string("MSVC ") + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

[[nodiscard]] EnvironmentInfo cpuEnvironment(BenchmarkConfig const& config)
{
    char const* hostname = std::getenv("HOSTNAME");
    return EnvironmentInfo{
        timestampUtc(),
        config.git_commit.empty() ? defaultGitCommit() : config.git_commit,
        hostname == nullptr ? "unknown" : hostname,
        compilerName(),
        "not_applicable",
        "not_applicable",
    };
}

} // namespace

BenchmarkReport runCpuBenchmark(
    BenchmarkConfig const& config,
    std::vector<WorkloadKind> const& workloads)
{
    std::string error;
    if (!validateConfig(config, error)) {
        throw std::invalid_argument(error);
    }
    if (workloads.empty()) {
        throw std::invalid_argument("at least one workload is required");
    }

    BenchmarkReport report{};
    report.suite = "cpu_metadata";
    report.config = config;
    if (report.config.git_commit.empty()) {
        report.config.git_commit = defaultGitCommit();
    }
    report.environment = cpuEnvironment(report.config);

    for (WorkloadKind const workload : workloads) {
        WorkloadTrace const trace = generateWorkload(workload, report.config);
        std::vector<CapacityResult> const capacity =
            calculateCapacity(trace, report.config);
        report.capacity.insert(
            report.capacity.end(),
            capacity.begin(),
            capacity.end()
        );
        report.workloads.push_back(
            cpu_detail::runCpuWorkload(trace, report.config)
        );
    }
    return report;
}

} // namespace kimkvcache::benchmark
