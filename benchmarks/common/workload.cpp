#include "kim-kv/benchmark/benchmark.h"

#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>

namespace kimkvcache::benchmark {
namespace {

[[nodiscard]] std::uint32_t uniformLength(
    std::mt19937_64& generator,
    std::uint32_t minimum,
    std::uint32_t maximum)
{
    if (minimum >= maximum) {
        return maximum;
    }

    std::uniform_int_distribution<std::uint32_t> distribution(
        minimum,
        maximum
    );
    return distribution(generator);
}

} // namespace

WorkloadTrace generateWorkload(
    WorkloadKind workload,
    BenchmarkConfig const& config)
{
    std::string error;
    if (!validateConfig(config, error)) {
        throw std::invalid_argument(error);
    }

    WorkloadTrace trace{};
    trace.workload = workload;
    trace.seed = config.seed;
    trace.sequence_lengths.reserve(config.request_count);
    std::mt19937_64 generator(
        config.seed ^ (static_cast<std::uint64_t>(workload) << 48U)
    );

    constexpr std::array<std::uint32_t, 3> kPrefixes{64, 128, 256};
    constexpr std::array<std::uint32_t, 5> kForkPoints{1, 7, 8, 9, 63};
    constexpr std::array<std::uint32_t, 3> kFanouts{2, 4, 8};

    if (workload == WorkloadKind::SharedPrompt
        || workload == WorkloadKind::ForkCow) {
        std::size_t group_index = 0;
        while (trace.sequence_lengths.size() < config.request_count) {
            std::uint32_t const fanout =
                kFanouts[group_index % kFanouts.size()];
            std::uint32_t const base = workload == WorkloadKind::SharedPrompt
                ? std::min(
                    kPrefixes[group_index % kPrefixes.size()],
                    config.maximum_sequence_length - 1
                )
                : std::min(
                    kForkPoints[group_index % kForkPoints.size()],
                    config.maximum_sequence_length - 1
                );

            std::size_t const group_begin = trace.sequence_lengths.size();
            for (std::uint32_t child = 0;
                 child < fanout
                    && trace.sequence_lengths.size() < config.request_count;
                 ++child) {
                std::uint32_t const suffix =
                    workload == WorkloadKind::SharedPrompt
                    ? uniformLength(
                        generator,
                        1,
                        std::min<std::uint32_t>(
                            63,
                            config.maximum_sequence_length - base
                        )
                    )
                    : 1;
                trace.sequence_lengths.push_back(base + suffix);
            }
            std::size_t const group_size =
                trace.sequence_lengths.size() - group_begin;
            if (group_size > 1) {
                trace.reused_tokens += static_cast<std::uint64_t>(base)
                    * (group_size - 1);
            }
            ++group_index;
        }
        return trace;
    }

    for (std::uint32_t index = 0;
         index < config.request_count;
         ++index) {
        std::uint32_t length = 1;
        switch (workload) {
        case WorkloadKind::Short:
            length = uniformLength(
                generator,
                1,
                std::min<std::uint32_t>(63, config.maximum_sequence_length)
            );
            break;
        case WorkloadKind::Mixed: {
            std::uniform_int_distribution<std::uint32_t> bucket(0, 99);
            std::uint32_t const choice = bucket(generator);
            if (choice < 70 || config.maximum_sequence_length <= 128) {
                length = uniformLength(
                    generator,
                    1,
                    std::min<std::uint32_t>(
                        128,
                        config.maximum_sequence_length
                    )
                );
            } else if (choice < 90
                       || config.maximum_sequence_length <= 384) {
                length = uniformLength(
                    generator,
                    129,
                    std::min<std::uint32_t>(
                        384,
                        config.maximum_sequence_length
                    )
                );
            } else {
                length = uniformLength(
                    generator,
                    385,
                    config.maximum_sequence_length
                );
            }
            break;
        }
        case WorkloadKind::Adversarial: {
            std::uint32_t const count =
                (config.maximum_sequence_length - 1) / 64 + 1;
            length = 1 + 64 * (index % count);
            break;
        }
        case WorkloadKind::Long:
            length = uniformLength(
                generator,
                std::min<std::uint32_t>(
                    448,
                    config.maximum_sequence_length
                ),
                config.maximum_sequence_length
            );
            break;
        case WorkloadKind::SharedPrompt:
        case WorkloadKind::ForkCow:
            throw std::logic_error("group workload generation fell through");
        case WorkloadKind::Fault:
            length = uniformLength(
                generator,
                1,
                config.maximum_sequence_length
            );
            break;
        }
        trace.sequence_lengths.push_back(length);
    }

    return trace;
}

} // namespace kimkvcache::benchmark
