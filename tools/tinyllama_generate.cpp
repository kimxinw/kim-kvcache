#include "kim-kv/cuda/cuda_engine_kv_backend.h"
#include "kim-kv/cuda/cuda_model_runner.h"
#include "kim-kv/engine/generation.h"
#include "kim-kv/model/weight_manifest.h"

#include <cuda_runtime_api.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace kimkvcache;

struct Options final {
    std::string manifest{};
    std::string weights{};
    std::string output{};
    std::vector<std::uint32_t> tokens{};
    std::uint32_t max_new_tokens{0};
    std::uint32_t repetitions{1};
    std::uint32_t eos_token_id{std::numeric_limits<std::uint32_t>::max()};
};

[[nodiscard]] bool parseUnsigned(
    std::string_view encoded,
    std::uint32_t& value)
{
    auto const parsed = std::from_chars(
        encoded.data(), encoded.data() + encoded.size(), value
    );
    return !encoded.empty() && parsed.ec == std::errc{}
        && parsed.ptr == encoded.data() + encoded.size();
}

[[nodiscard]] bool parseTokens(
    std::string const& encoded,
    std::vector<std::uint32_t>& tokens)
{
    std::size_t begin = 0;
    while (begin < encoded.size()) {
        std::size_t const end = encoded.find(',', begin);
        std::string_view const part(
            encoded.data() + begin,
            (end == std::string::npos ? encoded.size() : end) - begin
        );
        std::uint32_t token = 0;
        if (!parseUnsigned(part, token)) {
            return false;
        }
        tokens.push_back(token);
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return !tokens.empty();
}

[[nodiscard]] bool parseOptions(int argc, char** argv, Options& options)
{
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            return false;
        }
        std::string const key = argv[index];
        std::string const value = argv[index + 1];
        if (key == "--manifest") {
            options.manifest = value;
        } else if (key == "--weights") {
            options.weights = value;
        } else if (key == "--output") {
            options.output = value;
        } else if (key == "--tokens") {
            if (!parseTokens(value, options.tokens)) {
                return false;
            }
        } else if (key == "--max-new-tokens") {
            if (!parseUnsigned(value, options.max_new_tokens)) {
                return false;
            }
        } else if (key == "--repetitions") {
            if (!parseUnsigned(value, options.repetitions)) {
                return false;
            }
        } else if (key == "--eos-token-id") {
            if (!parseUnsigned(value, options.eos_token_id)) {
                return false;
            }
        } else {
            return false;
        }
    }
    return !options.manifest.empty() && !options.weights.empty()
        && !options.output.empty() && !options.tokens.empty()
        && options.max_new_tokens != 0 && options.repetitions != 0;
}

template <typename Value>
void writeNumbers(std::ostream& output, std::vector<Value> const& values)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << values[index];
    }
    output << ']';
}

[[nodiscard]] int fail(std::string const& message)
{
    std::cerr << "[FAILED] " << message << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parseOptions(argc, argv, options)) {
        return fail("usage: --manifest PATH --weights PATH --tokens ID,... "
            "--max-new-tokens N --output PATH [--repetitions N] "
            "[--eos-token-id ID]");
    }
    WeightManifestLoadResult loaded = loadWeightManifest(options.manifest);
    if (!loaded.ok()) {
        return fail("manifest: " + loaded.status.detail);
    }
    TinyLlamaConfig const config = loaded.manifest.config;
    std::uint32_t const eos_token = options.eos_token_id
        == std::numeric_limits<std::uint32_t>::max()
        ? config.eos_token_id
        : options.eos_token_id;
    if (eos_token >= config.vocabulary_size) {
        return fail("EOS token is outside vocabulary");
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        return fail("cudaStreamCreate");
    }
    std::size_t const attention_workspace =
        static_cast<std::size_t>(config.attention_head_count)
        * config.max_position_embeddings * sizeof(float);
    EngineKvConfig const kv_config{
        KvLayout{
            config.layer_count,
            config.kv_head_count,
            config.head_dimension,
        },
        config.attention_head_count,
        attention_workspace,
    };
    std::unique_ptr<EngineKvBackend> backend =
        createHeterogeneousCudaEngineKvBackend(kv_config, 512, 64);
    if (backend == nullptr) {
        static_cast<void>(cudaStreamDestroy(stream));
        return fail("create generation KV backend");
    }
    CudaModelRunnerCreateResult created = createCudaTinyLlamaModelRunner(
        options.manifest,
        options.weights,
        *backend,
        reinterpret_cast<EngineStream>(stream)
    );
    if (!created.ok()) {
        static_cast<void>(cudaStreamDestroy(stream));
        return fail(std::string(toString(created.status.error)) + ": "
            + created.status.detail);
    }

    SingleRequestGenerationRuntime runtime(*backend, *created.runner);
    std::vector<std::uint32_t> reference_output;
    GenerationTerminal first_terminal;
    bool outputs_consistent = true;
    bool resources_reclaimed = true;
    std::size_t baseline_free_bytes = 0;
    std::size_t total_bytes = 0;
    for (std::uint32_t repetition = 0;
         repetition < options.repetitions;
         ++repetition) {
        GenerationTerminal terminal = runtime.generate(GenerationRequest{
            static_cast<RequestId>(repetition) + 1,
            options.tokens,
            SamplingConfig{options.max_new_tokens, eos_token},
            {},
        });
        if (!terminal.ok()) {
            created.runner.reset();
            backend.reset();
            static_cast<void>(cudaStreamDestroy(stream));
            return fail(std::string(toString(terminal.error)) + ": "
                + terminal.detail);
        }
        if (repetition == 0) {
            reference_output = terminal.output_token_ids;
            first_terminal = terminal;
        } else if (terminal.output_token_ids != reference_output) {
            outputs_consistent = false;
        }
        EngineKvBackendSnapshot const snapshot = backend->snapshot();
        resources_reclaimed = resources_reclaimed
            && snapshot.request_count == 0
            && snapshot.active_transaction_count == 0
            && snapshot.committed_token_count == 0
            && backend->checkInvariants();
        if (repetition == 0) {
            if (cudaMemGetInfo(&baseline_free_bytes, &total_bytes)
                != cudaSuccess) {
                return fail("cudaMemGetInfo after warmup");
            }
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return fail("generation stream synchronize");
    }
    std::size_t final_free_bytes = 0;
    if (cudaMemGetInfo(&final_free_bytes, &total_bytes) != cudaSuccess) {
        return fail("cudaMemGetInfo after repetitions");
    }
    std::int64_t const free_memory_delta =
        static_cast<std::int64_t>(final_free_bytes)
        - static_cast<std::int64_t>(baseline_free_bytes);

    std::ofstream output(options.output);
    if (!output) {
        return fail("open output JSON");
    }
    output << "{\n  \"checkpoint\": \"" << loaded.manifest.checkpoint
        << "\",\n  \"checkpoint_revision\": \""
        << loaded.manifest.checkpoint_revision << "\",\n"
        << "  \"input_tokens\": ";
    writeNumbers(output, options.tokens);
    output << ",\n  \"max_new_tokens\": " << options.max_new_tokens
        << ",\n  \"eos_token_id\": " << eos_token
        << ",\n  \"output_tokens\": ";
    writeNumbers(output, reference_output);
    output << ",\n  \"terminal_reason\": \""
        << toString(first_terminal.reason)
        << "\",\n  \"usage\": {\"prompt_tokens\": "
        << first_terminal.usage.prompt_tokens
        << ", \"completion_tokens\": "
        << first_terminal.usage.completion_tokens
        << ", \"total_tokens\": " << first_terminal.usage.total_tokens
        << "},\n  \"metrics_ns\": {\"ttft\": "
        << first_terminal.metrics.ttft_ns << ", \"tpot\": "
        << first_terminal.metrics.tpot_ns << ", \"e2e\": "
        << first_terminal.metrics.e2e_ns
        << "},\n  \"repetitions\": " << options.repetitions
        << ",\n  \"outputs_consistent\": "
        << (outputs_consistent ? "true" : "false")
        << ",\n  \"resources_reclaimed\": "
        << (resources_reclaimed ? "true" : "false")
        << ",\n  \"gpu_free_bytes_after_warmup\": "
        << baseline_free_bytes
        << ",\n  \"gpu_free_bytes_after_repetitions\": "
        << final_free_bytes
        << ",\n  \"gpu_free_bytes_delta\": " << free_memory_delta
        << "\n}\n";
    output.close();

    created.runner.reset();
    backend.reset();
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        return fail("destroy generation stream");
    }
    if (!outputs_consistent || !resources_reclaimed
        || free_memory_delta != 0) {
        return fail("generation repetition/resource stability check");
    }
    std::cout << "generation output: " << options.output << '\n';
    return 0;
}
