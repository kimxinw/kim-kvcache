#include "heteropage_kv/benchmark/benchmark.h"

#include "cpu_benchmark_internal.h"

#include <stdexcept>

namespace kimkvcache::benchmark {

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
    report.environment = cpu_detail::benchmarkEnvironment(report.config);

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
