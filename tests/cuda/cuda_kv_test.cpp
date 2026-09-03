#include "kim-kv/cuda/cuda_kv_cache.h"
#include "kim-kv/cuda/fixed_cuda_kv_cache.h"
#include "storage.h"
#include "kim-kv/reference/kv_reference.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace kimkvcache;

int failures = 0;

void expect(bool condition, std::string const& message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

template <typename T>
class DeviceBuffer final {
public:
    explicit DeviceBuffer(std::size_t element_count)
        : element_count_(element_count)
    {
        if (element_count_ != 0) {
            error_ = cudaMalloc(
                reinterpret_cast<void**>(&data_),
                element_count_ * sizeof(T)
            );
        }
    }

    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            static_cast<void>(cudaFree(data_));
        }
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    [[nodiscard]] bool ok() const noexcept
    {
        return error_ == cudaSuccess;
    }

    [[nodiscard]] T* data() noexcept
    {
        return data_;
    }

    [[nodiscard]] T const* data() const noexcept
    {
        return data_;
    }

    [[nodiscard]] bool upload(std::vector<T> const& source)
    {
        return source.size() == element_count_
            && cudaMemcpy(
                data_,
                source.data(),
                source.size() * sizeof(T),
                cudaMemcpyHostToDevice
            ) == cudaSuccess;
    }

    [[nodiscard]] bool download(std::vector<T>& target) const
    {
        target.resize(element_count_);
        return cudaMemcpy(
            target.data(),
            data_,
            target.size() * sizeof(T),
            cudaMemcpyDeviceToHost
        ) == cudaSuccess;
    }

private:
    T* data_{nullptr};
    std::size_t element_count_{0};
    cudaError_t error_{cudaSuccess};
};

class TestStream final {
public:
    TestStream()
    {
        error_ = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    }

    ~TestStream()
    {
        if (stream_ != nullptr) {
            static_cast<void>(cudaStreamDestroy(stream_));
        }
    }

    [[nodiscard]] bool ok() const noexcept
    {
        return error_ == cudaSuccess;
    }

    [[nodiscard]] CudaStream get() const noexcept
    {
        return reinterpret_cast<CudaStream>(stream_);
    }

private:
    cudaStream_t stream_{nullptr};
    cudaError_t error_{cudaSuccess};
};

[[nodiscard]] std::vector<KvScalar> makeKv(
    KvLayout const& layout,
    std::uint32_t logical_token_begin,
    std::uint32_t token_count)
{
    std::size_t element_count = 0;
    static_cast<void>(layout.elementsForTokens(token_count, element_count));
    std::vector<KvScalar> result(element_count);

    for (std::uint32_t layer = 0; layer < layout.layer_count; ++layer) {
        for (std::uint32_t component = 0; component < 2; ++component) {
            for (std::uint32_t token = 0; token < token_count; ++token) {
                for (std::uint32_t head = 0;
                     head < layout.kv_head_count;
                     ++head) {
                    for (std::uint32_t dimension = 0;
                         dimension < layout.head_dimension;
                         ++dimension) {
                        std::uint32_t const logical_token =
                            logical_token_begin + token;
                        float const value =
                            static_cast<float>(layer) * 0.125F
                            + static_cast<float>(component) * 0.25F
                            + static_cast<float>(logical_token) * 0.0078125F
                            + static_cast<float>(head) * 0.03125F
                            + static_cast<float>(dimension) * 0.00390625F
                            - 0.5F;
                        std::size_t const offset = layout.offset(
                            layer,
                            component == 0
                                ? KvComponent::Key
                                : KvComponent::Value,
                            token,
                            head,
                            dimension,
                            token_count
                        );
                        result[offset] = floatToKvScalar(value);
                    }
                }
            }
        }
    }

    return result;
}

template <typename Cache>
[[nodiscard]] bool appendChunk(
    Cache& cache,
    KvLayout const& layout,
    RequestId request_id,
    std::uint32_t logical_begin,
    std::uint32_t token_count,
    CudaStream stream)
{
    std::vector<KvScalar> const host =
        makeKv(layout, logical_begin, token_count);
    DeviceBuffer<KvScalar> input(host.size());

    if (!input.ok() || !input.upload(host)) {
        expect(false, "append input allocation/upload must succeed");
        return false;
    }

    CudaKvOperationResult const result = cache.append(
        request_id,
        token_count,
        input.data(),
        stream
    );
    expect(result.ok(), "CUDA append must succeed");
    return result.ok();
}

template <typename Cache>
void expectGather(
    Cache& cache,
    KvLayout const& layout,
    RequestId request_id,
    std::uint32_t token_count,
    std::vector<KvScalar> const& expected,
    CudaStream stream,
    std::string const& message)
{
    std::size_t elements = 0;
    static_cast<void>(layout.elementsForTokens(token_count, elements));
    DeviceBuffer<KvScalar> output(elements);
    expect(output.ok(), message + ": output allocation");

    CudaKvOperationResult const result = cache.gather(
        request_id,
        output.data(),
        stream
    );
    expect(result.ok(), message + ": gather succeeds");

    std::vector<KvScalar> actual;
    expect(output.download(actual), message + ": output download");
    expect(actual == expected, message + ": byte-exact FP16 data");
}

template <typename Cache>
void expectAttention(
    Cache& cache,
    KvLayout const& layout,
    RequestId request_id,
    std::uint32_t token_count,
    std::vector<KvScalar> const& contiguous_kv,
    CudaStream stream,
    std::string const& message)
{
    std::size_t const query_elements =
        static_cast<std::size_t>(layout.layer_count)
        * layout.kv_head_count
        * layout.head_dimension;
    std::vector<float> query(query_elements);
    for (std::size_t index = 0; index < query.size(); ++index) {
        query[index] = static_cast<float>((index % 11) + 1) * 0.03125F;
    }

    std::vector<float> reference;
    expect(
        referenceAttention(
            layout,
            contiguous_kv,
            token_count,
            query,
            reference
        ),
        message + ": CPU reference"
    );

    DeviceBuffer<float> device_query(query.size());
    DeviceBuffer<float> device_output(query.size());
    expect(
        device_query.ok() && device_output.ok(),
        message + ": device buffers"
    );
    expect(device_query.upload(query), message + ": query upload");

    CudaKvOperationResult const attention = cache.referenceAttention(
        request_id,
        device_query.data(),
        device_output.data(),
        stream
    );
    expect(attention.ok(), message + ": CUDA attention");

    std::vector<float> actual;
    expect(device_output.download(actual), message + ": output download");
    expect(actual.size() == reference.size(), message + ": output size");
    if (actual.size() == reference.size()) {
        for (std::size_t index = 0; index < actual.size(); ++index) {
            float const error = std::abs(actual[index] - reference[index]);
            expect(error <= 5.0e-4F, message + ": reference tolerance");
        }
    }
}

void testAppendGatherPromotionAndAttention(
    KvLayout const& layout,
    CudaStream stream)
{
    CudaKvCache cache(layout, 16, 2);
    expect(cache.status().ok(), "CUDA cache initialization");

    CudaStorageSnapshot const storage = cache.storageSnapshot();
    std::size_t token_bytes = 0;
    static_cast<void>(layout.bytesForTokens(1, token_bytes));
    expect(
        storage.micro_reserved_bytes == 16 * 8 * token_bytes,
        "Micro allocation accounting"
    );
    expect(
        storage.extent_reserved_bytes == 2 * 64 * token_bytes,
        "Extent allocation accounting"
    );

    expect(cache.createRequest(1) == KvCacheError::None, "create request 1");
    if (!appendChunk(cache, layout, 1, 0, 5, stream)
        || !appendChunk(cache, layout, 1, 5, 11, stream)
        || !appendChunk(cache, layout, 1, 16, 48, stream)) {
        return;
    }

    std::vector<KvScalar> const expected = makeKv(layout, 0, 64);
    expectGather(
        cache,
        layout,
        1,
        64,
        expected,
        stream,
        "pre-promotion gather"
    );

    CudaKvOperationResult const promotion = cache.promote(1, 0, stream);
    expect(promotion.ok(), "transactional CUDA promotion must succeed");
    auto const table = cache.blockTable(1);
    expect(table.has_value(), "promoted table exists");
    if (table.has_value()) {
        expect(table->entries().size() == 1, "promotion creates one entry");
        expect(
            table->entries()[0].kind == PageKind::Extent,
            "promotion publishes Extent"
        );
    }

    expectGather(
        cache,
        layout,
        1,
        64,
        expected,
        stream,
        "post-promotion gather"
    );

    expectAttention(cache, layout, 1, 64, expected, stream, "hetero");

    expect(cache.checkInvariants(), "K4 main path invariants");
    expect(cache.releaseRequest(1) == KvCacheError::None, "release request 1");
    expect(cache.checkInvariants(), "K4 main cleanup invariants");
}

void testPartialTailCow(KvLayout const& layout, CudaStream stream)
{
    CudaKvCache cache(layout, 4, 1);
    expect(cache.status().ok(), "COW cache initialization");
    expect(cache.createRequest(10) == KvCacheError::None, "create COW parent");
    if (!appendChunk(cache, layout, 10, 0, 5, stream)) {
        return;
    }

    expect(
        cache.forkRequest(10, 11) == KvCacheError::None,
        "fork partial tail"
    );
    if (!appendChunk(cache, layout, 11, 5, 3, stream)) {
        return;
    }

    expectGather(
        cache,
        layout,
        10,
        5,
        makeKv(layout, 0, 5),
        stream,
        "parent remains unchanged"
    );
    expectGather(
        cache,
        layout,
        11,
        8,
        makeKv(layout, 0, 8),
        stream,
        "child COW prefix plus append"
    );
    expect(cache.checkInvariants(), "COW invariants");
    expect(cache.releaseRequest(11) == KvCacheError::None, "release child");
    expect(cache.releaseRequest(10) == KvCacheError::None, "release parent");
}

void testFaultRollback(KvLayout const& layout, CudaStream stream)
{
    CudaKvCache cache(layout, 16, 1);
    expect(cache.status().ok(), "fault cache initialization");
    expect(cache.createRequest(20) == KvCacheError::None, "create fault req");
    if (!appendChunk(cache, layout, 20, 0, 64, stream)) {
        return;
    }

    cache.injectFailureOnce(CudaFailurePoint::Submission);
    CudaKvOperationResult const submit_failure = cache.promote(20, 0, stream);
    expect(
        submit_failure.cuda_status.error == CudaError::SubmissionFailed,
        "injected submission failure must surface"
    );
    auto table = cache.blockTable(20);
    expect(table.has_value(), "request survives promotion submit failure");
    if (table.has_value()) {
        expect(table->entries().size() == 8, "submit failure preserves Micro");
    }
    expect(cache.metadataSnapshot().promotion_count == 0, "submit rollback");
    expect(cache.metadataSnapshot().extent_pool.allocated_slots == 0,
        "submit failure frees target");

    cache.injectFailureOnce(CudaFailurePoint::Completion);
    CudaKvOperationResult const completion_failure =
        cache.promote(20, 0, stream);
    expect(
        completion_failure.cuda_status.error == CudaError::ExecutionFailed,
        "injected Event failure must surface"
    );
    table = cache.blockTable(20);
    expect(table.has_value(), "request survives promotion Event failure");
    if (table.has_value()) {
        expect(table->entries().size() == 8, "Event failure preserves Micro");
    }
    expect(cache.metadataSnapshot().extent_pool.allocated_slots == 0,
        "Event failure frees target");
    expect(cache.checkInvariants(), "promotion fault invariants");

    expect(cache.createRequest(21) == KvCacheError::None, "create append fault");
    std::vector<KvScalar> const one_token = makeKv(layout, 0, 1);
    DeviceBuffer<KvScalar> input(one_token.size());
    expect(input.ok() && input.upload(one_token), "fault input upload");
    cache.injectFailureOnce(CudaFailurePoint::Submission);
    CudaKvOperationResult const append_failure = cache.append(
        21,
        1,
        input.data(),
        stream
    );
    expect(
        append_failure.cuda_status.error == CudaError::SubmissionFailed,
        "append submission failure surfaces"
    );
    expect(
        !cache.blockTable(21).has_value(),
        "failed device Append cancels unsafe request"
    );
    expect(cache.checkInvariants(), "append failure cleanup invariants");
    expect(cache.releaseRequest(20) == KvCacheError::None, "release fault req");
}

void testCancellationDuringPromotionIo(
    KvLayout const& layout,
    CudaStream stream)
{
    KvCacheManager manager(8, 1);
    CudaKvStorage storage(layout, 8, 1);
    expect(storage.status().ok(), "low-level storage initialization");
    expect(manager.createRequest(30) == KvCacheError::None, "create cancel req");
    BlockTable const empty_table = *manager.blockTable(30);
    expect(manager.append(30, 64) == KvCacheError::None, "append cancel req");
    BlockTable const populated_table = *manager.blockTable(30);

    std::vector<KvScalar> const host_input = makeKv(layout, 0, 64);
    DeviceBuffer<KvScalar> device_input(host_input.size());
    expect(
        device_input.ok() && device_input.upload(host_input),
        "cancellation input upload"
    );
    CudaSubmission append = storage.appendAsync(
        empty_table,
        populated_table,
        0,
        64,
        device_input.data(),
        stream
    );
    expect(append.wait().ok(), "initialize cancellation source pages");

    PromotionPrepareResult const promotion = manager.preparePromotion(30, 0);
    PageLeaseAcquireResult const lease =
        manager.acquirePromotionIoLease(promotion.promotion_id);
    expect(promotion.ok() && lease.ok(), "prepare cancellation operation");

    CudaSubmission submission = storage.promoteAsync(promotion, stream);
    expect(submission.submitted(), "promotion copy submits");
    expect(
        manager.releaseRequest(30) == KvCacheError::None,
        "request cancels while CUDA Event is pending"
    );
    expect(
        manager.snapshot().micro_pool.allocated_slots == 8,
        "leased sources cannot be reused before Event"
    );
    expect(submission.wait().ok(), "already submitted copy reaches Event");
    expect(
        manager.releasePageLease(lease.lease_id) == KvCacheError::None,
        "Event completion releases retiring pages"
    );
    expect(manager.snapshot().micro_pool.allocated_slots == 0, "sources free");
    expect(manager.snapshot().extent_pool.allocated_slots == 0, "target free");
    expect(manager.checkInvariants(), "cancellation race invariants");
}

void testFixedCudaPages(KvLayout const& layout, CudaStream stream)
{
    constexpr std::uint16_t kPageSizes[]{8, 16, 32, 64};
    std::size_t token_bytes = 0;
    static_cast<void>(layout.bytesForTokens(1, token_bytes));

    for (std::uint16_t const page_tokens : kPageSizes) {
        FixedCudaKvCache cache(layout, page_tokens, 24);
        std::string const label = "Fixed-" + std::to_string(page_tokens);
        expect(cache.status().ok(), label + ": initialization");
        expect(cache.tokensPerPage() == page_tokens, label + ": page tokens");

        CudaStorageSnapshot const storage = cache.storageSnapshot();
        expect(
            storage.micro_page_tokens == page_tokens,
            label + ": storage page tokens"
        );
        expect(storage.extent_capacity == 0, label + ": no extent pool");
        expect(
            storage.micro_reserved_bytes
                == 24ULL * page_tokens * token_bytes,
            label + ": allocation accounting"
        );

        RequestId const request_id = 100 + page_tokens;
        std::uint32_t const first = page_tokens - 1;
        std::uint32_t const second = page_tokens + 4;
        std::uint32_t const total = first + second;
        expect(
            cache.createRequest(request_id) == KvCacheError::None,
            label + ": create"
        );
        if (!appendChunk(
                cache,
                layout,
                request_id,
                0,
                first,
                stream
            )
            || !appendChunk(
                cache,
                layout,
                request_id,
                first,
                second,
                stream
            )) {
            continue;
        }

        std::vector<KvScalar> const expected = makeKv(layout, 0, total);
        expectGather(
            cache,
            layout,
            request_id,
            total,
            expected,
            stream,
            label + ": multi-page gather"
        );
        expectAttention(
            cache,
            layout,
            request_id,
            total,
            expected,
            stream,
            label + ": attention"
        );

        RequestId const child_id = request_id + 1'000;
        expect(
            cache.forkRequest(request_id, child_id) == KvCacheError::None,
            label + ": fork"
        );
        if (appendChunk(
                cache,
                layout,
                child_id,
                total,
                1,
                stream
            )) {
            expectGather(
                cache,
                layout,
                request_id,
                total,
                expected,
                stream,
                label + ": parent after COW"
            );
            expectGather(
                cache,
                layout,
                child_id,
                total + 1,
                makeKv(layout, 0, total + 1),
                stream,
                label + ": child after COW"
            );
        }

        expect(cache.checkInvariants(), label + ": active invariants");
        expect(
            cache.releaseRequest(child_id) == KvCacheError::None,
            label + ": release child"
        );
        expect(
            cache.releaseRequest(request_id) == KvCacheError::None,
            label + ": release parent"
        );
        FixedPageManagerSnapshot const metadata = cache.metadataSnapshot();
        expect(metadata.request_count == 0, label + ": no requests");
        expect(
            metadata.pool.allocated_slots == 0,
            label + ": no allocated pages"
        );
        expect(cache.checkInvariants(), label + ": cleanup invariants");
    }

    FixedCudaKvCache failure_cache(layout, 8, 4);
    expect(
        failure_cache.createRequest(9'000) == KvCacheError::None,
        "Fixed failure: create"
    );
    std::vector<KvScalar> const one_token = makeKv(layout, 0, 1);
    DeviceBuffer<KvScalar> input(one_token.size());
    expect(input.ok() && input.upload(one_token), "Fixed failure: upload");
    failure_cache.injectFailureOnce(CudaFailurePoint::Completion);
    CudaKvOperationResult const failure = failure_cache.append(
        9'000,
        1,
        input.data(),
        stream
    );
    expect(
        failure.cuda_status.error == CudaError::ExecutionFailed,
        "Fixed failure: completion error"
    );
    expect(
        !failure_cache.blockTable(9'000).has_value(),
        "Fixed failure: unsafe request cancelled"
    );
    expect(failure_cache.checkInvariants(), "Fixed failure: invariants");
}

} // namespace

int main()
{
    int device_count = 0;
    cudaError_t const device_error = cudaGetDeviceCount(&device_count);
    if (device_error != cudaSuccess || device_count == 0) {
        std::cerr << "FAILED: cuda_kv_contract requires a visible CUDA GPU\n";
        return 1;
    }

    KvLayout const layout{2, 2, 8};
    TestStream stream;
    expect(stream.ok(), "non-blocking CUDA stream creation");
    if (!stream.ok()) {
        return 1;
    }

    testAppendGatherPromotionAndAttention(layout, stream.get());
    testPartialTailCow(layout, stream.get());
    testFaultRollback(layout, stream.get());
    testCancellationDuringPromotionIo(layout, stream.get());
    testFixedCudaPages(layout, stream.get());

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All Hetero/K4 and Fixed/K6 CUDA tests passed\n";
    return 0;
}
