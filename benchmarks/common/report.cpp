#include "heteropage_kv/benchmark/benchmark.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace kimkvcache::benchmark {
namespace {

[[nodiscard]] std::string jsonEscape(std::string_view input)
{
    std::ostringstream output;
    for (char const character : input) {
        switch (character) {
        case '\\':
            output << "\\\\";
            break;
        case '"':
            output << "\\\"";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            output << character;
            break;
        }
    }
    return output.str();
}

[[nodiscard]] std::string csvCell(std::string_view input)
{
    bool const quote = input.find_first_of(",\"\n\r")
        != std::string_view::npos;
    if (!quote) {
        return std::string(input);
    }

    std::string output{"\""};
    for (char const character : input) {
        if (character == '"') {
            output += "\"\"";
        } else {
            output += character;
        }
    }
    output += '"';
    return output;
}

[[nodiscard]] std::string decimal(double value)
{
    std::ostringstream output;
    output << std::setprecision(10) << value;
    return output.str();
}

void writeCsvRow(
    std::ofstream& output,
    std::vector<std::string> const& cells)
{
    for (std::size_t index = 0; index < cells.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << csvCell(cells[index]);
    }
    output << '\n';
}

void ensureParentDirectory(std::string const& path)
{
    std::filesystem::path const parent =
        std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

} // namespace

void writeJsonReport(
    BenchmarkReport const& report,
    std::string const& path)
{
    ensureParentDirectory(path);
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open JSON report: " + path);
    }

    output << std::setprecision(10);
    output << "{\n";
    output << "  \"schema_version\": 1,\n";
    output << "  \"suite\": \"" << jsonEscape(report.suite) << "\",\n";
    output << "  \"successful\": "
           << (report.successful() ? "true" : "false") << ",\n";
    output << "  \"environment\": {"
           << "\"timestamp_utc\":\""
           << jsonEscape(report.environment.timestamp_utc) << "\","
           << "\"git_commit\":\""
           << jsonEscape(report.environment.git_commit) << "\","
           << "\"hostname\":\""
           << jsonEscape(report.environment.hostname) << "\","
           << "\"compiler\":\""
           << jsonEscape(report.environment.compiler) << "\","
           << "\"gpu_name\":\""
           << jsonEscape(report.environment.gpu_name) << "\","
           << "\"cuda_runtime\":\""
           << jsonEscape(report.environment.cuda_runtime) << "\"},\n";
    output << "  \"config\": {"
           << "\"seed\":" << report.config.seed << ','
           << "\"request_count\":" << report.config.request_count << ','
           << "\"concurrency\":" << report.config.concurrency << ','
           << "\"maximum_sequence_length\":"
           << report.config.maximum_sequence_length << ','
           << "\"warmup_iterations\":"
           << report.config.warmup_iterations << ','
           << "\"measured_iterations\":"
           << report.config.measured_iterations << ','
           << "\"capacity_budget_bytes\":"
           << report.config.capacity_budget_bytes << ','
           << "\"micro_page_tokens\":"
           << kMicroPageTokenCapacity << ','
           << "\"extent_page_tokens\":"
           << kExtentPageTokenCapacity << ','
           << "\"layout\":{"
           << "\"layers\":" << report.config.layout.layer_count << ','
           << "\"kv_heads\":" << report.config.layout.kv_head_count << ','
           << "\"head_dimension\":"
           << report.config.layout.head_dimension << "}},\n";

    output << "  \"capacity\": [\n";
    for (std::size_t index = 0; index < report.capacity.size(); ++index) {
        CapacityResult const& value = report.capacity[index];
        output << "    {"
               << "\"workload\":\"" << toString(value.workload) << "\","
               << "\"baseline\":\"" << toString(value.baseline) << "\","
               << "\"used_tokens\":" << value.used_tokens << ','
               << "\"reserved_tokens\":" << value.reserved_tokens << ','
               << "\"peak_reserved_tokens\":"
               << value.peak_reserved_tokens << ','
               << "\"internal_fragmentation_tokens\":"
               << value.internal_fragmentation_tokens << ','
               << "\"reused_tokens\":" << value.reused_tokens << ','
               << "\"used_bytes\":" << value.used_bytes << ','
               << "\"reserved_bytes\":" << value.reserved_bytes << ','
               << "\"peak_reserved_bytes\":"
               << value.peak_reserved_bytes << ','
               << "\"internal_fragmentation_bytes\":"
               << value.internal_fragmentation_bytes << ','
               << "\"reused_bytes\":" << value.reused_bytes << ','
               << "\"block_table_entries\":"
               << value.block_table_entries << ','
               << "\"micro_pages\":" << value.micro_pages << ','
               << "\"extent_pages\":" << value.extent_pages << ','
               << "\"admitted_requests\":"
               << value.admitted_requests << ','
               << "\"utilization\":" << value.utilization << '}';
        output << (index + 1 == report.capacity.size() ? "\n" : ",\n");
    }
    output << "  ],\n";

    output << "  \"workloads\": [\n";
    for (std::size_t workload_index = 0;
         workload_index < report.workloads.size();
         ++workload_index) {
        WorkloadResult const& workload = report.workloads[workload_index];
        output << "    {\"workload\":\"" << toString(workload.workload)
               << "\",\"trace_seed\":" << workload.trace_seed
               << ",\"trace_request_count\":"
               << workload.trace_request_count
               << ",\"minimum_sequence_length\":"
               << workload.minimum_sequence_length
               << ",\"maximum_sequence_length\":"
               << workload.maximum_sequence_length
               << ",\"mean_sequence_length\":"
               << workload.mean_sequence_length
               << ",\"elapsed_ns\":" << workload.elapsed_ns
               << ",\"completed_requests\":"
               << workload.completed_requests
               << ",\"requests_per_second\":"
               << workload.requests_per_second
               << ",\"failed_operations\":"
               << workload.failed_operations
               << ",\"invariants_ok\":"
               << (workload.invariants_ok ? "true" : "false")
               << ",\"resources_released\":"
               << (workload.resources_released ? "true" : "false")
               << ",\"latency\":[";

        for (std::size_t latency_index = 0;
             latency_index < workload.latency.size();
             ++latency_index) {
            LatencySummary const& value = workload.latency[latency_index];
            output << "{\"operation\":\"" << toString(value.operation)
                   << "\",\"sample_count\":" << value.sample_count
                   << ",\"failure_count\":" << value.failure_count
                   << ",\"minimum_ns\":" << value.minimum_ns
                   << ",\"maximum_ns\":" << value.maximum_ns
                   << ",\"mean_ns\":" << value.mean_ns
                   << ",\"p50_ns\":" << value.p50_ns
                   << ",\"p95_ns\":" << value.p95_ns
                   << ",\"p99_ns\":" << value.p99_ns
                   << ",\"total_bytes\":" << value.total_bytes
                   << ",\"effective_bandwidth_gbps\":"
                   << value.effective_bandwidth_gbps << '}';
            if (latency_index + 1 != workload.latency.size()) {
                output << ',';
            }
        }

        output << "],\"samples\":[";
        for (std::size_t sample_index = 0;
             sample_index < workload.samples.size();
             ++sample_index) {
            OperationSample const& value = workload.samples[sample_index];
            output << "{\"operation\":\"" << toString(value.operation)
                   << "\",\"sample_index\":" << value.sample_index
                   << ",\"duration_ns\":" << value.duration_ns
                   << ",\"bytes\":" << value.bytes
                   << ",\"token_count\":" << value.token_count
                   << ",\"success\":"
                   << (value.success ? "true" : "false")
                   << ",\"detail\":\"" << jsonEscape(value.detail)
                   << "\"}";
            if (sample_index + 1 != workload.samples.size()) {
                output << ',';
            }
        }
        output << "]}";
        output << (
            workload_index + 1 == report.workloads.size()
                ? "\n"
                : ",\n"
        );
    }
    output << "  ]\n}\n";
}

void writeCsvReport(
    BenchmarkReport const& report,
    std::string const& path)
{
    ensureParentDirectory(path);
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open CSV report: " + path);
    }

    constexpr std::size_t kColumnCount = 45;
    writeCsvRow(output, {
        "record_type",
        "suite",
        "workload",
        "baseline",
        "operation",
        "sample_index",
        "duration_ns",
        "bytes",
        "token_count",
        "success",
        "detail",
        "used_tokens",
        "reserved_tokens",
        "peak_reserved_tokens",
        "fragmentation_tokens",
        "entries",
        "micro_pages",
        "extent_pages",
        "admitted_requests",
        "utilization",
        "p50_ns",
        "p95_ns",
        "p99_ns",
        "git_commit",
        "seed",
        "request_count",
        "concurrency",
        "requests_per_second",
        "mean_ns",
        "total_bytes",
        "effective_bandwidth_gbps",
        "failure_count",
        "minimum_ns",
        "maximum_ns",
        "trace_seed",
        "trace_request_count",
        "minimum_sequence_length",
        "maximum_sequence_length",
        "mean_sequence_length",
        "reused_tokens",
        "used_bytes",
        "reserved_bytes",
        "peak_reserved_bytes",
        "fragmentation_bytes",
        "reused_bytes",
    });

    auto const baseRow = [&report]() {
        std::vector<std::string> row(kColumnCount);
        row[1] = report.suite;
        row[23] = report.environment.git_commit;
        row[24] = std::to_string(report.config.seed);
        row[25] = std::to_string(report.config.request_count);
        row[26] = std::to_string(report.config.concurrency);
        return row;
    };

    for (CapacityResult const& value : report.capacity) {
        std::vector<std::string> row = baseRow();
        row[0] = "capacity";
        row[2] = toString(value.workload);
        row[3] = toString(value.baseline);
        row[11] = std::to_string(value.used_tokens);
        row[12] = std::to_string(value.reserved_tokens);
        row[13] = std::to_string(value.peak_reserved_tokens);
        row[14] = std::to_string(value.internal_fragmentation_tokens);
        row[15] = std::to_string(value.block_table_entries);
        row[16] = std::to_string(value.micro_pages);
        row[17] = std::to_string(value.extent_pages);
        row[18] = std::to_string(value.admitted_requests);
        row[19] = decimal(value.utilization);
        row[39] = std::to_string(value.reused_tokens);
        row[40] = std::to_string(value.used_bytes);
        row[41] = std::to_string(value.reserved_bytes);
        row[42] = std::to_string(value.peak_reserved_bytes);
        row[43] = std::to_string(value.internal_fragmentation_bytes);
        row[44] = std::to_string(value.reused_bytes);
        writeCsvRow(output, row);
    }

    for (WorkloadResult const& workload : report.workloads) {
        auto const workloadRow = [&]() {
            std::vector<std::string> row = baseRow();
            row[2] = toString(workload.workload);
            row[27] = decimal(workload.requests_per_second);
            row[34] = std::to_string(workload.trace_seed);
            row[35] = std::to_string(workload.trace_request_count);
            row[36] = std::to_string(workload.minimum_sequence_length);
            row[37] = std::to_string(workload.maximum_sequence_length);
            row[38] = decimal(workload.mean_sequence_length);
            return row;
        };

        for (LatencySummary const& value : workload.latency) {
            std::vector<std::string> row = workloadRow();
            row[0] = "summary";
            row[4] = toString(value.operation);
            row[20] = std::to_string(value.p50_ns);
            row[21] = std::to_string(value.p95_ns);
            row[22] = std::to_string(value.p99_ns);
            row[28] = std::to_string(value.mean_ns);
            row[29] = std::to_string(value.total_bytes);
            row[30] = decimal(value.effective_bandwidth_gbps);
            row[31] = std::to_string(value.failure_count);
            row[32] = std::to_string(value.minimum_ns);
            row[33] = std::to_string(value.maximum_ns);
            writeCsvRow(output, row);
        }

        for (OperationSample const& value : workload.samples) {
            std::vector<std::string> row = workloadRow();
            row[0] = "sample";
            row[4] = toString(value.operation);
            row[5] = std::to_string(value.sample_index);
            row[6] = std::to_string(value.duration_ns);
            row[7] = std::to_string(value.bytes);
            row[8] = std::to_string(value.token_count);
            row[9] = value.success ? "true" : "false";
            row[10] = value.detail;
            writeCsvRow(output, row);
        }
    }
}

} // namespace kimkvcache::benchmark
