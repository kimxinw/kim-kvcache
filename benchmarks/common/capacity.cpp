#include "kim-kv/benchmark/benchmark.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace kimkvcache::benchmark {
namespace {

constexpr std::array<BaselineKind, 7> kBaselines{
    BaselineKind::ContiguousMax,
    BaselineKind::Fixed8,
    BaselineKind::Fixed16,
    BaselineKind::Fixed32,
    BaselineKind::Fixed64,
    BaselineKind::HeteroWithoutPromotion,
    BaselineKind::HeteroWithPromotion,
};

[[nodiscard]] constexpr std::uint64_t ceilDivide(
    std::uint64_t value,
    std::uint64_t divisor) noexcept
{
    return value == 0 ? 0 : 1 + (value - 1) / divisor;
}

} // namespace

std::vector<CapacityResult> calculateCapacity(
    WorkloadTrace const& trace,
    BenchmarkConfig const& config)
{
    std::size_t bytes_per_token_size = 0;
    if (!config.layout.bytesForTokens(1, bytes_per_token_size)) {
        throw std::overflow_error("bytes per token overflow");
    }
    std::uint64_t const bytes_per_token = bytes_per_token_size;
    std::uint64_t const budget_tokens =
        config.capacity_budget_bytes / bytes_per_token;

    std::vector<CapacityResult> output;
    output.reserve(kBaselines.size());

    for (BaselineKind const baseline : kBaselines) {
        CapacityResult result{};
        result.workload = trace.workload;
        result.baseline = baseline;

        for (std::uint32_t const length : trace.sequence_lengths) {
            std::uint64_t reserved = 0;
            std::uint64_t entries = 0;
            std::uint64_t micro_pages = 0;
            std::uint64_t extent_pages = 0;

            switch (baseline) {
            case BaselineKind::ContiguousMax:
                reserved = config.maximum_sequence_length;
                entries = 1;
                break;
            case BaselineKind::Fixed8:
                entries = ceilDivide(length, 8);
                reserved = entries * 8;
                break;
            case BaselineKind::Fixed16:
                entries = ceilDivide(length, 16);
                reserved = entries * 16;
                break;
            case BaselineKind::Fixed32:
                entries = ceilDivide(length, 32);
                reserved = entries * 32;
                break;
            case BaselineKind::Fixed64:
                entries = ceilDivide(length, 64);
                reserved = entries * 64;
                break;
            case BaselineKind::HeteroWithoutPromotion:
                micro_pages = ceilDivide(length, 8);
                entries = micro_pages;
                reserved = micro_pages * 8;
                break;
            case BaselineKind::HeteroWithPromotion:
                extent_pages = length / 64;
                micro_pages = ceilDivide(length % 64, 8);
                entries = extent_pages + micro_pages;
                reserved = extent_pages * 64 + micro_pages * 8;
                break;
            }

            result.used_tokens += length;
            result.reserved_tokens += reserved;
            result.block_table_entries += entries;
            result.micro_pages += micro_pages;
            result.extent_pages += extent_pages;
        }

        result.peak_reserved_tokens = result.reserved_tokens;
        if (baseline == BaselineKind::HeteroWithPromotion) {
            result.peak_reserved_tokens += std::min<std::uint64_t>(
                result.extent_pages,
                config.concurrency
            ) * kExtentPageTokenCapacity;
        }
        result.internal_fragmentation_tokens =
            result.reserved_tokens - result.used_tokens;
        result.reused_tokens = trace.reused_tokens;
        auto const toBytes = [bytes_per_token](std::uint64_t tokens) {
            if (tokens > std::numeric_limits<std::uint64_t>::max()
                / bytes_per_token) {
                throw std::overflow_error("capacity byte count overflow");
            }
            return tokens * bytes_per_token;
        };
        result.used_bytes = toBytes(result.used_tokens);
        result.reserved_bytes = toBytes(result.reserved_tokens);
        result.peak_reserved_bytes = toBytes(result.peak_reserved_tokens);
        result.internal_fragmentation_bytes = toBytes(
            result.internal_fragmentation_tokens
        );
        result.reused_bytes = toBytes(result.reused_tokens);
        result.utilization = result.reserved_tokens == 0
            ? 0.0
            : static_cast<double>(result.used_tokens)
                / static_cast<double>(result.reserved_tokens);
        if (result.reserved_tokens != 0) {
            long double const scaled_budget =
                static_cast<long double>(budget_tokens)
                * static_cast<long double>(trace.sequence_lengths.size());
            result.admitted_requests = std::min<std::uint64_t>(
                trace.sequence_lengths.size(),
                static_cast<std::uint64_t>(
                    scaled_budget
                    / static_cast<long double>(result.reserved_tokens)
                )
            );
        }
        output.push_back(result);
    }

    return output;
}

} // namespace kimkvcache::benchmark
