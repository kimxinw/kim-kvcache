#include "kim-kv/benchmark/benchmark.h"

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
        BenchmarkReport const report = options.fixed_page_tokens == 0
            ? runCudaBenchmark(options.config, options.workloads)
            : runCudaFixedBenchmark(
                options.config,
                options.workloads,
                options.fixed_page_tokens
            );
        std::filesystem::path const output_directory =
            options.output_directory;
        std::string const stem = options.fixed_page_tokens == 0
            ? "cuda_data_path"
            : "cuda_data_path_fixed_"
                + std::to_string(options.fixed_page_tokens);
        std::string const json_path =
            (output_directory / (stem + ".json")).string();
        std::string const csv_path =
            (output_directory / (stem + ".csv")).string();

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
