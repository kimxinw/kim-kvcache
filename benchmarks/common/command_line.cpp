#include "kim-kv/benchmark/benchmark.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace kimkvcache::benchmark {
namespace {

[[nodiscard]] std::uint64_t parseUnsigned(
    std::string const& text,
    char const* option)
{
    std::size_t consumed = 0;
    std::uint64_t value = 0;
    try {
        value = std::stoull(text, &consumed, 0);
    } catch (std::exception const&) {
        throw std::invalid_argument(
            std::string("invalid value for ") + option + ": " + text
        );
    }
    if (consumed != text.size()) {
        throw std::invalid_argument(
            std::string("invalid value for ") + option + ": " + text
        );
    }
    return value;
}

[[nodiscard]] std::uint32_t parseUint32(
    std::string const& text,
    char const* option)
{
    std::uint64_t const value = parseUnsigned(text, option);
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            std::string("value for ") + option + " exceeds uint32 range"
        );
    }
    return static_cast<std::uint32_t>(value);
}

} // namespace

bool parseCommandLine(
    int argc,
    char const* const* argv,
    CommandLineOptions& options,
    std::string& error)
{
    try {
        for (int index = 1; index < argc; ++index) {
            std::string const argument = argv[index];
            auto const nextValue = [&]() -> std::string {
                if (index + 1 >= argc) {
                    throw std::invalid_argument(
                        "missing value after " + argument
                    );
                }
                return argv[++index];
            };

            if (argument == "--help" || argument == "-h") {
                options.show_help = true;
            } else if (argument == "--seed") {
                options.config.seed = parseUnsigned(
                    nextValue(),
                    "--seed"
                );
            } else if (argument == "--requests") {
                options.config.request_count = parseUint32(
                    nextValue(),
                    "--requests"
                );
            } else if (argument == "--concurrency") {
                options.config.concurrency = parseUint32(
                    nextValue(),
                    "--concurrency"
                );
            } else if (argument == "--max-sequence-length") {
                options.config.maximum_sequence_length =
                    parseUint32(nextValue(), "--max-sequence-length");
            } else if (argument == "--warmup") {
                options.config.warmup_iterations =
                    parseUint32(nextValue(), "--warmup");
            } else if (argument == "--iterations") {
                options.config.measured_iterations =
                    parseUint32(nextValue(), "--iterations");
            } else if (argument == "--capacity-budget-mib") {
                std::uint64_t const mib = parseUnsigned(
                    nextValue(),
                    "--capacity-budget-mib"
                );
                if (mib > std::numeric_limits<std::uint64_t>::max()
                    / (1024ULL * 1024ULL)) {
                    throw std::invalid_argument(
                        "--capacity-budget-mib overflows bytes"
                    );
                }
                options.config.capacity_budget_bytes = mib * 1024ULL * 1024ULL;
            } else if (argument == "--layers") {
                options.config.layout.layer_count =
                    parseUint32(nextValue(), "--layers");
            } else if (argument == "--kv-heads") {
                options.config.layout.kv_head_count =
                    parseUint32(nextValue(), "--kv-heads");
            } else if (argument == "--head-dimension") {
                options.config.layout.head_dimension =
                    parseUint32(nextValue(), "--head-dimension");
            } else if (argument == "--git-commit") {
                options.config.git_commit = nextValue();
            } else if (argument == "--output-dir") {
                options.output_directory = nextValue();
            } else if (argument == "--fixed-page-tokens") {
                std::uint32_t const value = parseUint32(
                    nextValue(),
                    "--fixed-page-tokens"
                );
                if (value != 8 && value != 16
                    && value != 32 && value != 64) {
                    throw std::invalid_argument(
                        "--fixed-page-tokens must be one of 8/16/32/64"
                    );
                }
                options.fixed_page_tokens =
                    static_cast<std::uint16_t>(value);
            } else if (argument == "--workload") {
                std::string const value = nextValue();
                if (value == "all") {
                    options.workloads = allWorkloads();
                    continue;
                }
                std::optional<WorkloadKind> const workload =
                    parseWorkload(value);
                if (!workload.has_value()) {
                    throw std::invalid_argument(
                        "unknown workload: " + value
                    );
                }
                options.workloads.push_back(*workload);
            } else {
                throw std::invalid_argument("unknown option: " + argument);
            }
        }

        bool const uses_default_workloads = options.workloads.empty();
        if (uses_default_workloads) {
            options.workloads = allWorkloads();
        }

        if (options.fixed_page_tokens != 0) {
            auto const fault = std::find(
                options.workloads.begin(),
                options.workloads.end(),
                WorkloadKind::Fault
            );
            if (uses_default_workloads) {
                if (fault != options.workloads.end()) {
                    options.workloads.erase(fault);
                }
            } else if (fault != options.workloads.end()) {
                throw std::invalid_argument(
                    "fault workload requires promotion semantics and is "
                    "not available for fixed-page baselines"
                );
            }
        }
        return validateConfig(options.config, error);
    } catch (std::exception const& exception) {
        error = exception.what();
        return false;
    }
}

std::string benchmarkUsage(std::string_view program_name)
{
    std::ostringstream output;
    output
        << "Usage: " << program_name << " [options]\n"
        << "  --workload NAME          short|mixed|adversarial|long|"
           "shared_prompt|fork_cow|fault|all\n"
        << "  --seed N                 deterministic random seed\n"
        << "  --requests N             CPU trace request count (default 10000)\n"
        << "  --concurrency N          concurrent request/fork width\n"
        << "  --max-sequence-length N  maximum sequence length (default 544)\n"
        << "  --warmup N               CUDA warmup iterations\n"
        << "  --iterations N           CUDA measured iterations\n"
        << "  --capacity-budget-mib N  admission accounting budget\n"
        << "  --layers N               model layer count\n"
        << "  --kv-heads N             KV head count\n"
        << "  --head-dimension N       attention head dimension\n"
        << "  --git-commit HASH        override compiled Git revision\n"
        << "  --output-dir PATH        JSON/CSV destination directory\n"
        << "  --fixed-page-tokens N    executable fixed-page baseline; "
           "N=8|16|32|64\n"
        << "                            default workloads exclude fault\n"
        << "  --help                    show this message\n";
    return output.str();
}

} // namespace kimkvcache::benchmark
