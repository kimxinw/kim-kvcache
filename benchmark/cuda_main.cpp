#include "heteropage_kv/benchmark.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char const* const* argv)
{
    using namespace kimkvcache::benchmark;

    CommandLineOptions options{};
    std::string error;
    if (!parseCommandLine(argc, argv, options, error)) {
        std::cerr << "error: " << error << "\n\n"
                  << benchmarkUsage(argv[0]);
        return 2;
    }
    if (options.show_help) {
        std::cout << benchmarkUsage(argv[0]);
        return 0;
    }

    try {
        BenchmarkReport const report = runCudaBenchmark(
            options.config,
            options.workloads
        );
        std::filesystem::path const output_directory =
            options.output_directory;
        std::string const json_path =
            (output_directory / "cuda_data_path.json").string();
        std::string const csv_path =
            (output_directory / "cuda_data_path.csv").string();

        writeJsonReport(report, json_path);
        writeCsvReport(report, csv_path);

        std::cout << "CUDA benchmark "
                  << (report.successful() ? "completed" : "failed")
                  << "\nJSON: " << json_path
                  << "\nCSV:  " << csv_path << '\n';
        return report.successful() ? 0 : 1;
    } catch (std::exception const& exception) {
        std::cerr << "benchmark failed: " << exception.what() << '\n';
        return 1;
    }
}
