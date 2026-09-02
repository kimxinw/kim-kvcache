#include "kim-kv/cuda/cuda_engine_kv_backend.h"
#include "kim-kv/cuda/cuda_model_runner.h"
#include "kim-kv/model/weight_manifest.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
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
};

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
        auto const parsed = std::from_chars(
            part.data(), part.data() + part.size(), token
        );
        if (part.empty() || parsed.ec != std::errc{}
            || parsed.ptr != part.data() + part.size()) {
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
        } else {
            return false;
        }
    }
    return !options.manifest.empty() && !options.weights.empty()
        && !options.output.empty() && !options.tokens.empty();
}

float fp32(KvScalar value)
{
    return __half2float(__ushort_as_half(value));
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

void writeHalfNumbers(std::ostream& output, std::vector<KvScalar> const& values)
{
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            output << ',';
        }
        output << fp32(values[index]);
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
            "--output PATH");
    }
    WeightManifestLoadResult loaded = loadWeightManifest(options.manifest);
    if (!loaded.ok()) {
        return fail("manifest: " + loaded.status.detail);
    }
    TinyLlamaConfig const config = loaded.manifest.config;
    for (std::uint32_t token : options.tokens) {
        if (token >= config.vocabulary_size) {
            return fail("input token is outside vocabulary");
        }
    }
    if (options.tokens.size() > config.max_position_embeddings) {
        return fail("input exceeds max_position_embeddings");
    }

    cudaStream_t stream = nullptr;
    cudaError_t const stream_status = cudaStreamCreate(&stream);
    if (stream_status != cudaSuccess) {
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
    if (backend == nullptr || !backend->createRequest(1).ok()) {
        static_cast<void>(cudaStreamDestroy(stream));
        return fail("create validation KV request");
    }

    CudaModelRunnerCreateResult created = createCudaTinyLlamaModelRunner(
        options.manifest,
        options.weights,
        *backend,
        reinterpret_cast<EngineStream>(stream)
    );
    if (!created.ok()) {
        static_cast<void>(backend->releaseRequest(1));
        static_cast<void>(cudaStreamDestroy(stream));
        return fail(std::string(toString(created.status.error)) + ": "
            + created.status.detail);
    }

    std::vector<std::uint32_t> greedy_tokens;
    ModelRunnerDebugCapture final_debug;
    ModelTokenResult final_result;
    for (std::size_t position = 0; position < options.tokens.size(); ++position) {
        ModelRunnerDebugCapture* debug = nullptr;
        if (position + 1 == options.tokens.size()) {
            final_debug.layer_indices = {
                0,
                config.layer_count / 2,
                config.layer_count - 1,
            };
            debug = &final_debug;
        }
        ModelTokenResult token_result = created.runner->forwardToken(
            1,
            options.tokens[position],
            static_cast<std::uint32_t>(position),
            debug
        );
        if (!token_result.ok()) {
            created.runner.reset();
            static_cast<void>(backend->releaseRequest(1));
            static_cast<void>(cudaStreamDestroy(stream));
            return fail(std::string(toString(token_result.status.error)) + ": "
                + token_result.status.detail);
        }
        greedy_tokens.push_back(token_result.greedy_token_id);
        if (debug != nullptr) {
            final_result = std::move(token_result);
        }
    }

    EngineKvBackendSnapshot const committed = backend->snapshot();
    if (committed.committed_token_count != options.tokens.size()
        || committed.active_transaction_count != 0
        || !backend->checkInvariants()) {
        return fail("post-forward KV invariant");
    }
    if (!backend->releaseRequest(1).ok()) {
        return fail("release validation request");
    }
    EngineKvBackendSnapshot const released = backend->snapshot();

    std::ofstream output(options.output);
    if (!output) {
        return fail("open output JSON");
    }
    output << std::setprecision(9);
    output << "{\n  \"checkpoint\": \"" << loaded.manifest.checkpoint
        << "\",\n  \"checkpoint_revision\": \""
        << loaded.manifest.checkpoint_revision << "\",\n"
        << "  \"config_sha256\": \"" << loaded.manifest.config_sha256
        << "\",\n  \"input_tokens\": ";
    writeNumbers(output, options.tokens);
    output << ",\n  \"greedy_tokens\": ";
    writeNumbers(output, greedy_tokens);
    output << ",\n  \"selected_layers\": ";
    writeNumbers(output, final_debug.layer_indices);
    output << ",\n  \"embedding\": ";
    writeHalfNumbers(output, final_debug.embedding);
    output << ",\n  \"layer_hidden_states\": [";
    for (std::size_t index = 0;
         index < final_debug.layer_hidden_states.size();
         ++index) {
        if (index != 0) {
            output << ',';
        }
        writeHalfNumbers(output, final_debug.layer_hidden_states[index]);
    }
    output << "],\n  \"final_hidden_state\": ";
    writeHalfNumbers(output, final_debug.final_hidden_state);
    output << ",\n  \"logits\": ";
    writeNumbers(output, final_result.logits);
    output << ",\n  \"device_weight_bytes\": "
        << created.runner->deviceWeightBytes()
        << ",\n  \"device_workspace_bytes\": "
        << created.runner->deviceWorkspaceBytes()
        << ",\n  \"released_request_count\": " << released.request_count
        << ",\n  \"released_committed_token_count\": "
        << released.committed_token_count << "\n}\n";
    output.close();

    created.runner.reset();
    backend.reset();
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        return fail("destroy validation stream");
    }
    std::cout << "validation output: " << options.output << '\n';
    return 0;
}
