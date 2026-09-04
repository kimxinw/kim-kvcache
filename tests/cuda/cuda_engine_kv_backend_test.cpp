#include "kim-kv/cuda/cuda_engine_kv_backend.h"
#include "storage.h"
#include "kim-kv/runtime/kv_cache_manager.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
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

KvScalar fp16(float value)
{
    __half const half = __float2half(value);
    KvScalar bits = 0;
    static_assert(sizeof(bits) == sizeof(half));
    std::memcpy(&bits, &half, sizeof(bits));
    return bits;
}

float fp32(KvScalar bits)
{
    return __half2float(__ushort_as_half(bits));
}

struct DeviceBuffers final {
    KvScalar* key{nullptr};
    KvScalar* value{nullptr};
    KvScalar* query{nullptr};
    KvScalar* output{nullptr};
    void* workspace{nullptr};
    cudaStream_t stream{nullptr};
    bool owns_stream{true};

    explicit DeviceBuffers(cudaStream_t shared_stream = nullptr)
        : stream(shared_stream)
        , owns_stream(shared_stream == nullptr)
    {
        if (owns_stream) {
            expect(cudaStreamCreate(&stream) == cudaSuccess, "create stream");
        }
        expect(cudaMalloc(reinterpret_cast<void**>(&key), 4) == cudaSuccess,
            "allocate key");
        expect(cudaMalloc(reinterpret_cast<void**>(&value), 4) == cudaSuccess,
            "allocate value");
        expect(cudaMalloc(reinterpret_cast<void**>(&query), 8) == cudaSuccess,
            "allocate query");
        expect(cudaMalloc(reinterpret_cast<void**>(&output), 8) == cudaSuccess,
            "allocate output");
        expect(cudaMalloc(&workspace, 4096) == cudaSuccess,
            "allocate workspace");
    }

    ~DeviceBuffers()
    {
        if (owns_stream) {
            static_cast<void>(cudaStreamSynchronize(stream));
        }
        static_cast<void>(cudaFree(workspace));
        static_cast<void>(cudaFree(output));
        static_cast<void>(cudaFree(query));
        static_cast<void>(cudaFree(value));
        static_cast<void>(cudaFree(key));
        if (owns_stream) {
            static_cast<void>(cudaStreamDestroy(stream));
        }
    }
};

enum class FaultMode {
    None,
    Submission,
    Completion,
};

std::vector<float> referenceOutput(
    std::vector<int> const& history,
    int new_token)
{
    std::vector<int> tokens = history;
    tokens.push_back(new_token);
    std::vector<float> result(4, 0.0F);
    float const scale = 1.0F / std::sqrt(2.0F);
    float const queries[2][2]{{0.5F, 0.25F}, {-0.2F, 0.4F}};
    for (std::size_t query_head = 0; query_head < 2; ++query_head) {
        std::vector<float> scores;
        for (int token : tokens) {
            float const key[2]{
                0.1F * static_cast<float>(token + 1) + 0.02F,
                -0.05F * static_cast<float>(token) + 0.01F,
            };
            scores.push_back((queries[query_head][0] * key[0]
                + queries[query_head][1] * key[1]) * scale);
        }
        float const maximum = *std::max_element(scores.begin(), scores.end());
        float denominator = 0.0F;
        for (float score : scores) {
            denominator += std::exp(score - maximum);
        }
        for (std::size_t index = 0; index < tokens.size(); ++index) {
            float const probability = std::exp(scores[index] - maximum)
                / denominator;
            float const token = static_cast<float>(tokens[index] + 1);
            result[query_head * 2] += probability * (token + 0.2F);
            result[query_head * 2 + 1] += probability * (token * 0.5F - 0.1F);
        }
    }
    return result;
}

bool runToken(
    EngineKvBackend& backend,
    RequestId request_id,
    std::vector<int>& history,
    int token,
    DeviceBuffers& device,
    FaultMode fault = FaultMode::None)
{
    TokenReserveResult reserved = backend.reserveToken(ReserveTokenRequest{
        request_id,
        static_cast<std::uint32_t>(history.size()),
        reinterpret_cast<EngineStream>(device.stream),
    });
    expect(reserved.ok(), "reserve engine token");
    if (!reserved.ok()) {
        return false;
    }
    expect(
        backend.snapshot().active_transaction_count == 1,
        "reserve increments active transaction count"
    );

    std::vector<KvScalar> const host_query{
        fp16(0.5F), fp16(0.25F), fp16(-0.2F), fp16(0.4F),
    };
    expect(cudaMemcpyAsync(
        device.query,
        host_query.data(),
        host_query.size() * sizeof(KvScalar),
        cudaMemcpyHostToDevice,
        device.stream
    ) == cudaSuccess, "upload query");

    if (fault == FaultMode::Submission) {
        expect(injectCudaEngineFailureOnce(
            backend, CudaFailurePoint::Submission), "inject submission");
    }

    for (std::uint32_t layer = 0; layer < 2; ++layer) {
        float const layer_offset = layer == 1 ? 0.02F : 0.0F;
        std::vector<KvScalar> const host_key{
            fp16(0.1F * static_cast<float>(token + 1) + layer_offset),
            fp16(-0.05F * static_cast<float>(token) + layer_offset * 0.5F),
        };
        std::vector<KvScalar> const host_value{
            fp16(static_cast<float>(token + 1) + layer_offset * 10.0F),
            fp16(static_cast<float>(token + 1) * 0.5F
                - layer_offset * 5.0F),
        };
        expect(cudaMemcpyAsync(
            device.key,
            host_key.data(),
            host_key.size() * sizeof(KvScalar),
            cudaMemcpyHostToDevice,
            device.stream
        ) == cudaSuccess, "upload layer key");
        expect(cudaMemcpyAsync(
            device.value,
            host_value.data(),
            host_value.size() * sizeof(KvScalar),
            cudaMemcpyHostToDevice,
            device.stream
        ) == cudaSuccess, "upload layer value");

        EngineKvStatus const write = reserved.transaction.writeLayer(
            LayerKvWrite{layer, device.key, device.value}
        );
        if (fault == FaultMode::Submission && layer == 0) {
            expect(
                write.error == EngineKvError::SubmissionFailed,
                "submission failure reaches transaction"
            );
            expect(reserved.transaction.rollback().ok(), "fault rollback");
            expect(backend.checkInvariants(), "submission rollback invariants");
            return false;
        }
        expect(write.ok(), "enqueue layer write");
        EngineKvStatus const attention = reserved.transaction.attendLayer(
            PagedDecodeRequest{
                layer,
                device.query,
                device.output,
                device.workspace,
                4096,
                1.0F / std::sqrt(2.0F),
            }
        );
        expect(attention.ok(), "enqueue direct paged attention");
    }

    if (fault == FaultMode::Completion) {
        expect(injectCudaEngineFailureOnce(
            backend, CudaFailurePoint::Completion), "inject completion");
        expect(
            reserved.transaction.commit().error
                == EngineKvError::ExecutionFailed,
            "completion failure rolls back at token boundary"
        );
        expect(backend.checkInvariants(), "completion rollback invariants");
        return false;
    }

    expect(reserved.transaction.commit().ok(), "commit engine token");
    std::vector<KvScalar> host_output(4);
    expect(cudaMemcpy(
        host_output.data(),
        device.output,
        host_output.size() * sizeof(KvScalar),
        cudaMemcpyDeviceToHost
    ) == cudaSuccess, "download attention output");
    std::vector<float> const expected = referenceOutput(history, token);
    for (std::size_t index = 0; index < expected.size(); ++index) {
        float const actual = fp32(host_output[index]);
        expect(
            std::abs(actual - expected[index]) < 0.035F,
            "paged GQA output matches contiguous reference"
        );
    }
    history.push_back(token);
    expect(backend.checkInvariants(), "post-commit engine invariants");
    expect(
        backend.snapshot().active_transaction_count == 0,
        "commit clears active transaction"
    );
    return true;
}

void testBackend(
    EngineKvBackendKind kind,
    std::uint16_t fixed_page_tokens = 8)
{
    EngineKvConfig const config{KvLayout{2, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        kind == EngineKvBackendKind::Heterogeneous
        ? createHeterogeneousCudaEngineKvBackend(config, 32, 4)
        : createFixedCudaEngineKvBackend(config, fixed_page_tokens, 32);
    expect(backend != nullptr, "create CUDA engine backend");
    expect(backend->kind() == kind, "backend kind");
    EngineKvBackendSnapshot const initial = backend->snapshot();
    expect(initial.primary_page_tokens
            == (kind == EngineKvBackendKind::Heterogeneous
                ? kMicroPageTokenCapacity
                : fixed_page_tokens),
        "snapshot exposes primary page size");
    expect(initial.primary_page_capacity == 32,
        "snapshot exposes primary page capacity");
    expect(initial.secondary_page_capacity
            == (kind == EngineKvBackendKind::Heterogeneous ? 4U : 0U),
        "snapshot distinguishes heterogeneous secondary pool");
    expect(initial.storage_reserved_bytes != 0,
        "snapshot exposes reserved CUDA storage bytes");
    expect(backend->createRequest(1).ok(), "create engine request");

    DeviceBuffers device;
    std::vector<int> parent_history;
    for (int token = 0; token < 10; ++token) {
        expect(runToken(
            *backend, 1, parent_history, token, device),
            "multi-page token"
        );
    }
    expect(
        backend->snapshot().committed_token_count == 10,
        "ten committed tokens"
    );
    expect(backend->snapshot().allocated_primary_pages != 0
        && backend->snapshot().successful_primary_allocations != 0,
        "snapshot exposes live and cumulative page allocations");

    expect(backend->forkRequest(1, 2).ok(), "fork engine request");
    std::vector<int> child_history = parent_history;
    expect(runToken(
        *backend, 2, child_history, 21, device),
        "forked partial-tail COW token"
    );
    expect(
        backend->snapshot().committed_token_count == 21,
        "snapshot sums parent and child lengths"
    );

    std::uint64_t const before_fault =
        backend->snapshot().committed_token_count;
    expect(!runToken(
        *backend,
        1,
        parent_history,
        30,
        device,
        FaultMode::Submission), "submission failure is not committed");
    expect(
        backend->snapshot().committed_token_count == before_fault,
        "submission failure preserves length"
    );
    expect(!runToken(
        *backend,
        1,
        parent_history,
        31,
        device,
        FaultMode::Completion), "completion failure is not committed");
    expect(
        backend->snapshot().committed_token_count == before_fault,
        "completion failure preserves length"
    );

    expect(backend->releaseRequest(2).ok(), "release child");
    expect(backend->releaseRequest(1).ok(), "release parent");
    EngineKvBackendSnapshot const empty = backend->snapshot();
    expect(empty.request_count == 0, "all requests released");
    expect(empty.committed_token_count == 0, "released token count cleared");
    expect(empty.allocated_primary_pages == 0
        && empty.allocated_secondary_pages == 0,
        "released backend reports no allocated pages");
    expect(backend->checkInvariants(), "final engine invariants");
}

void testExtentPagedAttention()
{
    KvLayout const layout{1, 1, 2};
    KvCacheManager manager(9, 1);
    CudaKvStorage storage(layout, 9, 1);
    expect(storage.status().ok(), "extent storage initialized");
    expect(manager.createRequest(50) == KvCacheError::None, "extent request");

    std::vector<KvScalar> host_input(256, fp16(0.0F));
    for (std::uint32_t token = 0; token < 64; ++token) {
        host_input[layout.offset(
            0, KvComponent::Key, token, 0, 0, 64)] = fp16(1.0F);
        host_input[layout.offset(
            0, KvComponent::Key, token, 0, 1, 64)] = fp16(1.0F);
        host_input[layout.offset(
            0, KvComponent::Value, token, 0, 0, 64)] =
            fp16(static_cast<float>(token) / 64.0F);
        host_input[layout.offset(
            0, KvComponent::Value, token, 0, 1, 64)] =
            fp16(-static_cast<float>(token) / 128.0F);
    }
    KvScalar* device_input = nullptr;
    expect(cudaMalloc(
        reinterpret_cast<void**>(&device_input),
        host_input.size() * sizeof(KvScalar)) == cudaSuccess,
        "extent input allocation"
    );
    expect(cudaMemcpy(
        device_input,
        host_input.data(),
        host_input.size() * sizeof(KvScalar),
        cudaMemcpyHostToDevice) == cudaSuccess,
        "extent input upload"
    );

    PageLeaseAcquireResult before = manager.acquireRequestReadLease(50);
    expect(before.ok(), "extent before lease");
    expect(manager.append(50, 64) == KvCacheError::None, "append 64 micro tokens");
    PageLeaseAcquireResult after = manager.acquireRequestReadLease(50);
    expect(after.ok(), "extent after lease");
    CudaSubmission append = storage.appendAsync(
        before.table, after.table, 0, 64, device_input
    );
    expect(append.submissionStatus().ok() && append.wait().ok(),
        "upload 64-token KV");
    expect(manager.releasePageLease(after.lease_id) == KvCacheError::None,
        "release after lease");
    expect(manager.releasePageLease(before.lease_id) == KvCacheError::None,
        "release before lease");

    PromotionPrepareResult promotion = manager.preparePromotion(50, 0);
    expect(promotion.ok(), "prepare extent promotion");
    PageLeaseAcquireResult promotion_lease =
        manager.acquirePromotionIoLease(promotion.promotion_id);
    expect(promotion_lease.ok(), "promotion IO lease");
    CudaSubmission promoted = storage.promoteAsync(promotion);
    expect(promoted.submissionStatus().ok() && promoted.wait().ok(),
        "copy micro pages to extent");
    expect(manager.releasePageLease(promotion_lease.lease_id)
            == KvCacheError::None,
        "release promotion lease");
    expect(manager.commitPromotion(promotion.promotion_id)
            == KvCacheError::None,
        "commit extent promotion");
    expect(
        manager.blockTable(50)->entries().size() == 1
            && manager.blockTable(50)->entries().front().kind
                == PageKind::Extent,
        "history is represented by one extent"
    );

    TokenReservationResult reservation = manager.reserveToken(50, 64);
    expect(reservation.ok(), "reserve token after extent");
    PageLeaseAcquireResult reservation_lease =
        manager.acquireTokenReservationLease(reservation.reservation_id);
    expect(reservation_lease.ok(), "extent transaction lease");
    CudaEngineTransactionBeginResult transaction =
        storage.beginEngineTransaction(
            reservation.before, reservation.reserved, 1, nullptr
        );
    expect(transaction.ok(), "begin extent engine transaction");

    KvScalar* key = nullptr;
    KvScalar* value = nullptr;
    KvScalar* query = nullptr;
    KvScalar* output = nullptr;
    void* workspace = nullptr;
    expect(cudaMalloc(reinterpret_cast<void**>(&key), 4) == cudaSuccess,
        "extent key allocation");
    expect(cudaMalloc(reinterpret_cast<void**>(&value), 4) == cudaSuccess,
        "extent value allocation");
    expect(cudaMalloc(reinterpret_cast<void**>(&query), 4) == cudaSuccess,
        "extent query allocation");
    expect(cudaMalloc(reinterpret_cast<void**>(&output), 4) == cudaSuccess,
        "extent output allocation");
    expect(cudaMalloc(&workspace, 65 * sizeof(float)) == cudaSuccess,
        "extent workspace allocation");
    std::vector<KvScalar> const current_key{fp16(1.0F), fp16(1.0F)};
    std::vector<KvScalar> const current_value{fp16(1.0F), fp16(-1.0F)};
    std::vector<KvScalar> const current_query{fp16(1.0F), fp16(1.0F)};
    expect(cudaMemcpy(key, current_key.data(), 4, cudaMemcpyHostToDevice)
            == cudaSuccess,
        "extent key upload");
    expect(cudaMemcpy(value, current_value.data(), 4, cudaMemcpyHostToDevice)
            == cudaSuccess,
        "extent value upload");
    expect(cudaMemcpy(query, current_query.data(), 4, cudaMemcpyHostToDevice)
            == cudaSuccess,
        "extent query upload");
    expect(transaction.transaction->writeLayer(
            LayerKvWrite{0, key, value}).ok(),
        "write token after extent");
    expect(transaction.transaction->attendLayer(PagedDecodeRequest{
            0,
            query,
            output,
            workspace,
            65 * sizeof(float),
            1.0F / std::sqrt(2.0F),
        }).ok(),
        "attend directly across extent and micro page");
    expect(transaction.transaction->finish().ok(), "finish extent attention");
    std::vector<KvScalar> host_output(2);
    expect(cudaMemcpy(
        host_output.data(), output, 4, cudaMemcpyDeviceToHost)
            == cudaSuccess,
        "extent output download");
    expect(std::abs(fp32(host_output[0]) - 0.5F) < 0.01F,
        "extent value dimension zero");
    expect(std::abs(fp32(host_output[1]) + 0.2577F) < 0.01F,
        "extent value dimension one");
    expect(manager.releasePageLease(reservation_lease.lease_id)
            == KvCacheError::None,
        "release extent transaction lease");
    expect(manager.commitTokenReservation(reservation.reservation_id)
            == KvCacheError::None,
        "commit token after extent");
    expect(manager.checkInvariants(), "extent transaction invariants");

    static_cast<void>(cudaFree(workspace));
    static_cast<void>(cudaFree(output));
    static_cast<void>(cudaFree(query));
    static_cast<void>(cudaFree(value));
    static_cast<void>(cudaFree(key));
    static_cast<void>(cudaFree(device_input));
}

void testBatchedAttentionFailureIsolation(EngineKvBackendKind kind)
{
    EngineKvConfig const config{KvLayout{1, 1, 2}, 2, 4096};
    std::unique_ptr<EngineKvBackend> backend =
        kind == EngineKvBackendKind::Heterogeneous
        ? createHeterogeneousCudaEngineKvBackend(config, 16, 2)
        : createFixedCudaEngineKvBackend(config, 8, 16);
    expect(backend != nullptr, "create batch isolation backend");
    expect(backend->createRequest(70).ok(), "create failing batch request");
    expect(backend->createRequest(71).ok(), "create healthy batch request");

    cudaStream_t stream = nullptr;
    expect(cudaStreamCreate(&stream) == cudaSuccess,
        "create shared batch stream");
    DeviceBuffers first(stream);
    DeviceBuffers second(stream);
    TokenReserveResult first_reserved = backend->reserveToken({
        70, 0, reinterpret_cast<EngineStream>(stream),
    });
    TokenReserveResult second_reserved = backend->reserveToken({
        71, 0, reinterpret_cast<EngineStream>(stream),
    });
    expect(first_reserved.ok() && second_reserved.ok(),
        "reserve two independent batch lanes");

    std::vector<KvScalar> const query{
        fp16(0.5F), fp16(0.25F), fp16(-0.2F), fp16(0.4F),
    };
    std::vector<KvScalar> const first_key{fp16(0.1F), fp16(0.2F)};
    std::vector<KvScalar> const first_value{fp16(1.0F), fp16(2.0F)};
    std::vector<KvScalar> const second_key{fp16(0.3F), fp16(0.4F)};
    std::vector<KvScalar> const second_value{fp16(3.0F), fp16(4.0F)};
    auto upload = [&](DeviceBuffers& device,
                      std::vector<KvScalar> const& key,
                      std::vector<KvScalar> const& value) {
        expect(cudaMemcpyAsync(
            device.query, query.data(), 8, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batch query");
        expect(cudaMemcpyAsync(
            device.key, key.data(), 4, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batch key");
        expect(cudaMemcpyAsync(
            device.value, value.data(), 4, cudaMemcpyHostToDevice, stream
        ) == cudaSuccess, "upload batch value");
    };
    upload(first, first_key, first_value);
    upload(second, second_key, second_value);
    expect(first_reserved.transaction.writeLayer(
            {0, first.key, first.value}).ok(),
        "write failing batch lane before attention");
    expect(second_reserved.transaction.writeLayer(
            {0, second.key, second.value}).ok(),
        "write healthy batch lane before attention");

    expect(injectCudaEngineFailureOnce(
        *backend, CudaFailurePoint::Submission),
        "inject one batch attention submission failure");
    std::array<PagedDecodeBatchItem, 2> items{{
        {&first_reserved.transaction,
            {0, first.query, first.output, first.workspace, 4096,
                1.0F / std::sqrt(2.0F)}},
        {&second_reserved.transaction,
            {0, second.query, second.output, second.workspace, 4096,
                1.0F / std::sqrt(2.0F)}},
    }};
    std::array<DevicePagedDecodeBatchItem, 2> host_items{};
    DevicePagedDecodeBatchItem* device_items = nullptr;
    expect(cudaMalloc(
        reinterpret_cast<void**>(&device_items),
        sizeof(DevicePagedDecodeBatchItem) * items.size()
    ) == cudaSuccess, "allocate batch lane metadata");
    PagedDecodeBatch batch{
        items.data(),
        items.size(),
        host_items.data(),
        device_items,
        host_items.size(),
    };
    attendLayerBatch(batch);
    expect(items[0].status.error == EngineKvError::SubmissionFailed,
        "injected batch lane fails closed");
    expect(items[1].status.ok(),
        "healthy batch lane survives another lane failure");
    expect(first_reserved.transaction.rollback().ok(),
        "failed batch lane rolls back");
    expect(second_reserved.transaction.commit().ok(),
        "healthy batch lane commits");

    std::array<KvScalar, 4> output{};
    expect(cudaMemcpy(
        output.data(), second.output, 8, cudaMemcpyDeviceToHost
    ) == cudaSuccess, "download healthy batch output");
    expect(std::abs(fp32(output[0]) - 3.0F) < 0.01F
            && std::abs(fp32(output[1]) - 4.0F) < 0.01F
            && std::abs(fp32(output[2]) - 3.0F) < 0.01F
            && std::abs(fp32(output[3]) - 4.0F) < 0.01F,
        "healthy batch lane computes its independent GQA output");
    EngineKvBackendSnapshot const snapshot = backend->snapshot();
    expect(snapshot.committed_token_count == 1
            && snapshot.active_transaction_count == 0,
        "batch failure publishes only the healthy lane");
    expect(snapshot.batched_attention_submissions == 1
            && snapshot.batched_attention_lanes == 1,
        "batch telemetry records the surviving submitted lane");
    expect(backend->releaseRequest(71).ok(), "release healthy batch request");
    expect(backend->releaseRequest(70).ok(), "release failed batch request");
    expect(backend->checkInvariants(), "batch failure isolation invariants");

    static_cast<void>(cudaFree(device_items));
    static_cast<void>(cudaStreamDestroy(stream));
}

} // namespace

int main()
{
    testBackend(EngineKvBackendKind::Heterogeneous);
    for (std::uint16_t const page_tokens : {8, 16, 32, 64}) {
        testBackend(EngineKvBackendKind::Fixed, page_tokens);
    }
    testExtentPagedAttention();
    testBatchedAttentionFailureIsolation(
        EngineKvBackendKind::Heterogeneous
    );
    testBatchedAttentionFailureIsolation(EngineKvBackendKind::Fixed);
    if (failures != 0) {
        std::cerr << failures << " CUDA engine KV checks failed\n";
        return 1;
    }
    std::cout << "CUDA engine KV checks passed\n";
    return 0;
}
