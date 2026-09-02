#include "kim-kv/engine/generation.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using namespace kimkvcache;

int failures = 0;

void expect(bool condition, std::string const& message)
{
    if (!condition) {
        std::cerr << "[FAILED] " << message << '\n';
        ++failures;
    }
}

class FakeKvBackend final : public EngineKvBackend {
public:
    [[nodiscard]] EngineKvBackendKind kind() const noexcept override
    {
        return EngineKvBackendKind::Heterogeneous;
    }

    [[nodiscard]] EngineKvConfig config() const noexcept override
    {
        return {};
    }

    [[nodiscard]] EngineKvStatus createRequest(
        RequestId request_id) override
    {
        if (request_id == kInvalidRequestId) {
            return {EngineKvError::InvalidArgument};
        }
        if (!requests_.insert(request_id).second) {
            return {EngineKvError::RequestAlreadyExists};
        }
        return {};
    }

    [[nodiscard]] EngineKvStatus forkRequest(
        RequestId,
        RequestId) override
    {
        return {EngineKvError::InvalidState};
    }

    [[nodiscard]] EngineKvStatus releaseRequest(
        RequestId request_id) override
    {
        return requests_.erase(request_id) == 1
            ? EngineKvStatus{}
            : EngineKvStatus{EngineKvError::RequestNotFound};
    }

    [[nodiscard]] TokenReserveResult reserveToken(
        ReserveTokenRequest const&) override
    {
        TokenReserveResult result;
        result.status = {EngineKvError::InvalidState};
        return result;
    }

    [[nodiscard]] EngineKvBackendSnapshot snapshot() const override
    {
        return EngineKvBackendSnapshot{requests_.size(), 0, 0};
    }

    [[nodiscard]] bool checkInvariants() const override
    {
        return true;
    }

private:
    std::unordered_set<RequestId> requests_{};
};

class FakeModelRunner final : public GenerationModelRunner {
public:
    TinyLlamaConfig model_config{
        8, 16, 2, 4, 2, 2, 256, 256, 1, 255,
        1.0e-5F, 10000.0F, false,
    };
    std::shared_ptr<GenerationCancellationToken> cancel_on_call{};
    SingleRequestGenerationRuntime* stop_runtime_on_call{nullptr};
    std::uint64_t action_call{0};
    std::uint64_t fail_call{0};
    std::uint64_t calls{0};
    std::uint32_t forced_token{0};
    std::uint64_t force_call{0};

    [[nodiscard]] TinyLlamaConfig generationConfig() const noexcept override
    {
        return model_config;
    }

    [[nodiscard]] GenerationStepResult generationForwardToken(
        RequestId,
        std::uint32_t token_id,
        std::uint32_t expected_position) override
    {
        ++calls;
        if (action_call != 0 && calls == action_call) {
            if (cancel_on_call != nullptr) {
                cancel_on_call->cancel();
            }
            if (stop_runtime_on_call != nullptr) {
                stop_runtime_on_call->stop();
            }
        }
        if (fail_call != 0 && calls == fail_call) {
            return {false, 0, "injected model failure"};
        }
        if (force_call != 0 && calls == force_call) {
            return {true, forced_token, {}};
        }
        return {
            true,
            static_cast<std::uint32_t>(
                (token_id * 7 + expected_position * 3 + 1) % 251
            ),
            {},
        };
    }
};

std::vector<std::uint32_t> prompt(std::uint32_t length)
{
    std::vector<std::uint32_t> tokens(length);
    for (std::uint32_t index = 0; index < length; ++index) {
        tokens[index] = (index * 11 + 3) % 251;
    }
    return tokens;
}

std::vector<std::uint32_t> referenceGreedy(
    std::vector<std::uint32_t> const& input,
    std::uint32_t output_length)
{
    std::uint32_t next = 0;
    for (std::uint32_t position = 0; position < input.size(); ++position) {
        next = (input[position] * 7 + position * 3 + 1) % 251;
    }
    std::vector<std::uint32_t> output;
    output.reserve(output_length);
    for (std::uint32_t generated = 0; generated < output_length; ++generated) {
        output.push_back(next);
        if (generated + 1 != output_length) {
            std::uint32_t const position = static_cast<std::uint32_t>(
                input.size()
            ) + generated;
            next = (next * 7 + position * 3 + 1) % 251;
        }
    }
    return output;
}

GenerationRequest request(
    RequestId request_id,
    std::uint32_t input_length,
    std::uint32_t output_length)
{
    return GenerationRequest{
        request_id,
        prompt(input_length),
        SamplingConfig{output_length, 255},
        {},
    };
}

void expectReleased(FakeKvBackend const& backend, std::string const& context)
{
    EngineKvBackendSnapshot const snapshot = backend.snapshot();
    expect(snapshot.request_count == 0, context + " releases request");
    expect(snapshot.active_transaction_count == 0,
        context + " leaves no transaction");
    expect(snapshot.committed_token_count == 0,
        context + " leaves no committed token");
    expect(backend.checkInvariants(), context + " preserves invariants");
}

void testLengthAndReference()
{
    FakeKvBackend backend;
    FakeModelRunner runner;
    SingleRequestGenerationRuntime runtime(backend, runner);

    GenerationRequest osl1 = request(1, 32, 1);
    GenerationTerminal first = runtime.generate(osl1);
    expect(first.ok(), "OSL1 completes successfully");
    expect(first.reason == GenerationTerminalReason::MaxNewTokens,
        "OSL1 has length terminal");
    expect(first.output_token_ids == referenceGreedy(
        osl1.prompt_token_ids, 1), "ISL32 OSL1 matches reference");
    expect(first.usage.prompt_tokens == 32
        && first.usage.completion_tokens == 1
        && first.usage.total_tokens == 33,
        "OSL1 usage is complete");
    expect(first.metrics.tpot_ns == 0, "OSL1 TPOT is zero by definition");
    expectReleased(backend, "OSL1");

    GenerationRequest osl32 = request(2, 128, 32);
    GenerationTerminal second = runtime.generate(osl32);
    expect(second.ok(), "OSL32 completes successfully");
    expect(second.output_token_ids == referenceGreedy(
        osl32.prompt_token_ids, 32), "ISL128 OSL32 matches reference");
    expect(second.usage.prompt_tokens == 128
        && second.usage.completion_tokens == 32
        && second.usage.total_tokens == 160,
        "OSL32 usage is complete");
    expect(second.metrics.e2e_ns >= second.metrics.ttft_ns,
        "E2E includes TTFT");
    expectReleased(backend, "OSL32");
}

void testEosTerminal()
{
    FakeKvBackend backend;
    FakeModelRunner runner;
    runner.force_call = 4;
    runner.forced_token = runner.model_config.eos_token_id;
    SingleRequestGenerationRuntime runtime(backend, runner);

    GenerationRequest value = request(3, 4, 32);
    value.sampling.eos_token_id.reset();
    GenerationTerminal terminal = runtime.generate(value);
    expect(terminal.ok(), "EOS completes successfully");
    expect(terminal.reason == GenerationTerminalReason::EosToken,
        "EOS has EOS terminal");
    expect(terminal.output_token_ids
            == std::vector<std::uint32_t>{runner.model_config.eos_token_id},
        "EOS is returned exactly once");
    expect(terminal.usage.completion_tokens == 1,
        "EOS counts as a completion token");
    expectReleased(backend, "EOS");
}

void testCancellationAndStop()
{
    {
        FakeKvBackend backend;
        FakeModelRunner runner;
        auto cancellation = std::make_shared<GenerationCancellationToken>();
        runner.cancel_on_call = cancellation;
        runner.action_call = 5;
        SingleRequestGenerationRuntime runtime(backend, runner);
        GenerationRequest value = request(4, 16, 8);
        value.cancellation = cancellation;
        GenerationTerminal terminal = runtime.generate(value);
        expect(!terminal.ok()
            && terminal.reason == GenerationTerminalReason::Cancelled,
            "cancellation reaches a unique cancelled terminal");
        expectReleased(backend, "cancellation");
    }
    {
        FakeKvBackend backend;
        FakeModelRunner runner;
        SingleRequestGenerationRuntime runtime(backend, runner);
        runner.stop_runtime_on_call = &runtime;
        runner.action_call = 5;
        GenerationTerminal terminal = runtime.generate(request(5, 16, 8));
        expect(!terminal.ok()
            && terminal.reason == GenerationTerminalReason::RuntimeStopped
            && terminal.error == GenerationError::RuntimeStopped,
            "runtime stop reaches a unique stopped terminal");
        expect(runtime.stopped(), "runtime remains stopped");
        GenerationTerminal rejected = runtime.generate(request(6, 2, 1));
        expect(rejected.reason == GenerationTerminalReason::RuntimeStopped,
            "stopped runtime rejects subsequent work");
        expectReleased(backend, "runtime stop");
    }
}

void testFailureAndValidation()
{
    FakeKvBackend backend;
    FakeModelRunner runner;
    runner.fail_call = 3;
    SingleRequestGenerationRuntime runtime(backend, runner);
    GenerationTerminal failed = runtime.generate(request(7, 8, 2));
    expect(failed.reason == GenerationTerminalReason::Failed
        && failed.error == GenerationError::ModelFailed,
        "model failure reaches caller");
    expect(failed.detail == "injected model failure",
        "model failure detail is retained");
    expectReleased(backend, "model failure");

    runner.fail_call = 0;
    GenerationRequest empty = request(8, 1, 1);
    empty.prompt_token_ids.clear();
    expect(runtime.generate(empty).error == GenerationError::InvalidArgument,
        "empty prompt is rejected");
    GenerationRequest too_long = request(9, 256, 2);
    expect(runtime.generate(too_long).error
        == GenerationError::InvalidArgument,
        "position overflow is rejected");
    GenerationRequest invalid_eos = request(10, 1, 1);
    invalid_eos.sampling.eos_token_id = 256;
    expect(runtime.generate(invalid_eos).error
        == GenerationError::InvalidArgument,
        "out-of-vocabulary EOS is rejected");
    runner.force_call = runner.calls + 1;
    runner.forced_token = runner.model_config.vocabulary_size;
    GenerationTerminal invalid_model_token = runtime.generate(
        request(11, 1, 1)
    );
    expect(invalid_model_token.error == GenerationError::ModelFailed,
        "out-of-vocabulary model output is rejected");
    expectReleased(backend, "validation failures");
}

void testRepeatedResourceStability()
{
    FakeKvBackend backend;
    FakeModelRunner runner;
    SingleRequestGenerationRuntime runtime(backend, runner);
    EngineKvBackendSnapshot const baseline = backend.snapshot();
    for (RequestId id = 100; id < 200; ++id) {
        GenerationTerminal terminal = runtime.generate(request(id, 32, 32));
        expect(terminal.ok(), "100-run generation succeeds");
        expect(backend.snapshot().request_count == baseline.request_count,
            "100-run request count returns to baseline");
        expect(backend.snapshot().active_transaction_count
                == baseline.active_transaction_count,
            "100-run transaction count returns to baseline");
        expect(backend.snapshot().committed_token_count
                == baseline.committed_token_count,
            "100-run committed token count returns to baseline");
    }
    expectReleased(backend, "100-run stability");
}

} // namespace

int main()
{
    testLengthAndReference();
    testEosTerminal();
    testCancellationAndStop();
    testFailureAndValidation();
    testRepeatedResourceStability();
    if (failures == 0) {
        std::cout << "Generation runtime contract passed\n";
    }
    return failures == 0 ? 0 : 1;
}
