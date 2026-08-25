#include "heteropage_kv/cuda_kv_storage.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace kimkvcache {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;

struct DeviceLayout final {
    std::uint32_t layers;
    std::uint32_t heads;
    std::uint32_t dimensions;
};

struct DeviceBlockDescriptor final {
    KvScalar const* data;
    std::uint32_t logical_token_begin;
    std::uint16_t valid_tokens;
    std::uint16_t token_capacity;
};

struct PromotionSlots final {
    std::uint32_t slots[kPromotionSourcePageCount];
};

[[nodiscard]] cudaStream_t nativeStream(CudaStream stream) noexcept
{
    return reinterpret_cast<cudaStream_t>(stream);
}

[[nodiscard]] CudaStatus mapInitializationError(cudaError_t error) noexcept
{
    if (error == cudaSuccess) {
        return CudaStatus{};
    }

    CudaError mapped = CudaError::RuntimeUnavailable;
    if (error == cudaErrorMemoryAllocation) {
        mapped = CudaError::AllocationFailed;
    }

    return CudaStatus{mapped, static_cast<int>(error)};
}

[[nodiscard]] CudaStatus mapSubmissionError(cudaError_t error) noexcept
{
    if (error == cudaErrorMemoryAllocation) {
        return CudaStatus{
            CudaError::AllocationFailed,
            static_cast<int>(error),
        };
    }

    return error == cudaSuccess
        ? CudaStatus{}
        : CudaStatus{
            CudaError::SubmissionFailed,
            static_cast<int>(error),
        };
}

[[nodiscard]] std::size_t pageElements(
    KvLayout const& layout,
    PageKind kind) noexcept
{
    std::size_t result = 0;
    return layout.pageElements(kind, result) ? result : 0;
}

[[nodiscard]] std::size_t operationElements(
    KvLayout const& layout,
    std::uint32_t token_count) noexcept
{
    std::size_t result = 0;
    return layout.elementsForTokens(token_count, result) ? result : 0;
}

[[nodiscard]] dim3 gridFor(std::size_t element_count) noexcept
{
    std::size_t const blocks =
        (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    return dim3(static_cast<unsigned int>(blocks));
}

__device__ void decodeElement(
    std::size_t index,
    DeviceLayout layout,
    std::uint32_t token_count,
    std::uint32_t& layer,
    std::uint32_t& component,
    std::uint32_t& token,
    std::uint32_t& head,
    std::uint32_t& dimension)
{
    dimension = static_cast<std::uint32_t>(index % layout.dimensions);
    index /= layout.dimensions;
    head = static_cast<std::uint32_t>(index % layout.heads);
    index /= layout.heads;
    token = static_cast<std::uint32_t>(index % token_count);
    index /= token_count;
    component = static_cast<std::uint32_t>(index % 2);
    index /= 2;
    layer = static_cast<std::uint32_t>(index);
}

__device__ std::size_t tensorOffset(
    DeviceLayout layout,
    std::uint32_t layer,
    std::uint32_t component,
    std::uint32_t token,
    std::uint32_t head,
    std::uint32_t dimension,
    std::uint32_t token_capacity)
{
    return (((static_cast<std::size_t>(layer) * 2 + component)
                * token_capacity + token)
                * layout.heads + head)
                * layout.dimensions
        + dimension;
}

__global__ void copyPageTokensKernel(
    KvScalar const* source,
    std::uint32_t source_capacity,
    KvScalar* target,
    std::uint32_t target_capacity,
    std::uint32_t token_count,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= element_count) {
        return;
    }

    std::uint32_t layer = 0;
    std::uint32_t component = 0;
    std::uint32_t token = 0;
    std::uint32_t head = 0;
    std::uint32_t dimension = 0;
    decodeElement(
        index,
        layout,
        token_count,
        layer,
        component,
        token,
        head,
        dimension
    );

    target[tensorOffset(
        layout,
        layer,
        component,
        token,
        head,
        dimension,
        target_capacity
    )] = source[tensorOffset(
        layout,
        layer,
        component,
        token,
        head,
        dimension,
        source_capacity
    )];
}

__global__ void appendTokensKernel(
    KvScalar const* source,
    std::uint32_t source_token_capacity,
    std::uint32_t source_token_begin,
    KvScalar* target,
    std::uint32_t target_token_capacity,
    std::uint32_t target_token_begin,
    std::uint32_t segment_token_count,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= element_count) {
        return;
    }

    std::uint32_t layer = 0;
    std::uint32_t component = 0;
    std::uint32_t token = 0;
    std::uint32_t head = 0;
    std::uint32_t dimension = 0;
    decodeElement(
        index,
        layout,
        segment_token_count,
        layer,
        component,
        token,
        head,
        dimension
    );

    target[tensorOffset(
        layout,
        layer,
        component,
        target_token_begin + token,
        head,
        dimension,
        target_token_capacity
    )] = source[tensorOffset(
        layout,
        layer,
        component,
        source_token_begin + token,
        head,
        dimension,
        source_token_capacity
    )];
}

__global__ void promotionKernel(
    KvScalar const* micro_pool,
    std::size_t micro_page_elements,
    PromotionSlots source_slots,
    KvScalar* target,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= element_count) {
        return;
    }

    std::uint32_t layer = 0;
    std::uint32_t component = 0;
    std::uint32_t token = 0;
    std::uint32_t head = 0;
    std::uint32_t dimension = 0;
    decodeElement(
        index,
        layout,
        kExtentPageTokenCapacity,
        layer,
        component,
        token,
        head,
        dimension
    );

    std::uint32_t const source_page = token / kMicroPageTokenCapacity;
    std::uint32_t const source_token = token % kMicroPageTokenCapacity;
    KvScalar const* source = micro_pool
        + static_cast<std::size_t>(source_slots.slots[source_page])
            * micro_page_elements;

    target[tensorOffset(
        layout,
        layer,
        component,
        token,
        head,
        dimension,
        kExtentPageTokenCapacity
    )] = source[tensorOffset(
        layout,
        layer,
        component,
        source_token,
        head,
        dimension,
        kMicroPageTokenCapacity
    )];
}

__global__ void gatherKernel(
    DeviceBlockDescriptor const* descriptors,
    std::uint32_t descriptor_count,
    KvScalar* output,
    std::uint32_t total_tokens,
    DeviceLayout layout,
    std::size_t element_count)
{
    std::size_t const index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= element_count) {
        return;
    }

    std::uint32_t layer = 0;
    std::uint32_t component = 0;
    std::uint32_t token = 0;
    std::uint32_t head = 0;
    std::uint32_t dimension = 0;
    decodeElement(
        index,
        layout,
        total_tokens,
        layer,
        component,
        token,
        head,
        dimension
    );

    DeviceBlockDescriptor descriptor{};
    bool found = false;

    for (std::uint32_t descriptor_index = 0;
         descriptor_index < descriptor_count;
         ++descriptor_index) {
        DeviceBlockDescriptor const candidate =
            descriptors[descriptor_index];
        std::uint32_t const end = candidate.logical_token_begin
            + candidate.valid_tokens;

        if (token >= candidate.logical_token_begin && token < end) {
            descriptor = candidate;
            found = true;
            break;
        }
    }

    if (!found) {
        return;
    }

    std::uint32_t const page_token = token - descriptor.logical_token_begin;
    output[index] = descriptor.data[tensorOffset(
        layout,
        layer,
        component,
        page_token,
        head,
        dimension,
        descriptor.token_capacity
    )];
}

__global__ void attentionScoresKernel(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* query,
    float* scores,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    std::uint32_t const layer_head = blockIdx.x;
    std::uint32_t const layer = layer_head / layout.heads;
    std::uint32_t const head = layer_head % layout.heads;
    std::size_t const query_base =
        static_cast<std::size_t>(layer_head) * layout.dimensions;
    float const scale = rsqrtf(static_cast<float>(layout.dimensions));

    for (std::uint32_t token = 0; token < token_count; ++token) {
        float partial = 0.0F;

        for (std::uint32_t dimension = threadIdx.x;
             dimension < layout.dimensions;
             dimension += blockDim.x) {
            std::size_t const key_offset = tensorOffset(
                layout,
                layer,
                0,
                token,
                head,
                dimension,
                token_count
            );
            __half const key =
                reinterpret_cast<__half const*>(contiguous_kv)[key_offset];
            partial += query[query_base + dimension] * __half2float(key);
        }

        reduction[threadIdx.x] = partial;
        __syncthreads();

        for (unsigned int stride = blockDim.x / 2;
             stride != 0;
             stride /= 2) {
            if (threadIdx.x < stride) {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            scores[static_cast<std::size_t>(layer_head) * token_count
                + token] = reduction[0] * scale;
        }
        __syncthreads();
    }
}

__global__ void attentionOutputKernel(
    KvScalar const* contiguous_kv,
    std::uint32_t token_count,
    float const* scores,
    float* output,
    DeviceLayout layout)
{
    extern __shared__ float reduction[];
    std::uint32_t const layer_head = blockIdx.x;
    std::uint32_t const layer = layer_head / layout.heads;
    std::uint32_t const head = layer_head % layout.heads;
    float const* block_scores = scores
        + static_cast<std::size_t>(layer_head) * token_count;

    float local_maximum = -FLT_MAX;
    for (std::uint32_t token = threadIdx.x;
         token < token_count;
         token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, block_scores[token]);
    }

    reduction[threadIdx.x] = local_maximum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x],
                reduction[threadIdx.x + stride]
            );
        }
        __syncthreads();
    }

    float const maximum = reduction[0];
    // Every thread must cache the maximum before the shared reduction buffer
    // is reused for the softmax sum. Without this barrier, faster threads can
    // overwrite reduction[0] while slower threads are still reading it.
    __syncthreads();
    float local_sum = 0.0F;
    for (std::uint32_t token = threadIdx.x;
         token < token_count;
         token += blockDim.x) {
        local_sum += expf(block_scores[token] - maximum);
    }

    reduction[threadIdx.x] = local_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2;
         stride != 0;
         stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float const denominator = reduction[0];
    std::size_t const output_base =
        static_cast<std::size_t>(layer_head) * layout.dimensions;

    for (std::uint32_t dimension = threadIdx.x;
         dimension < layout.dimensions;
         dimension += blockDim.x) {
        float weighted_value = 0.0F;

        for (std::uint32_t token = 0; token < token_count; ++token) {
            std::size_t const value_offset = tensorOffset(
                layout,
                layer,
                1,
                token,
                head,
                dimension,
                token_count
            );
            __half const value =
                reinterpret_cast<__half const*>(contiguous_kv)[value_offset];
            weighted_value += expf(block_scores[token] - maximum)
                * __half2float(value);
        }

        output[output_base + dimension] = weighted_value / denominator;
    }
}

} // namespace

struct CudaKvStorage::Impl final {
    KvLayout layout{};
    std::uint32_t micro_capacity{0};
    std::uint32_t extent_capacity{0};
    std::size_t micro_page_elements{0};
    std::size_t extent_page_elements{0};
    std::size_t micro_reserved_bytes{0};
    std::size_t extent_reserved_bytes{0};
    KvScalar* micro_data{nullptr};
    KvScalar* extent_data{nullptr};
    CudaStatus initialization_status{};
    bool fail_submission_once{false};
    bool fail_completion_once{false};
    mutable std::mutex fault_mutex;

    ~Impl()
    {
        // 所有 Allocation 属于同一共享状态；仍存活的 Submission 也持有
        // shared_ptr，因此只有最后一个使用者退出后才会到这里。
        static_cast<void>(cudaDeviceSynchronize());
        if (micro_data != nullptr) {
            static_cast<void>(cudaFree(micro_data));
        }
        if (extent_data != nullptr) {
            static_cast<void>(cudaFree(extent_data));
        }
    }

    [[nodiscard]] bool consumeFailure(CudaFailurePoint point) noexcept
    {
        std::lock_guard<std::mutex> lock(fault_mutex);
        bool* flag = point == CudaFailurePoint::Submission
            ? &fail_submission_once
            : &fail_completion_once;
        bool const result = *flag;
        *flag = false;
        return result;
    }

    [[nodiscard]] KvScalar* pagePointer(PageHandle handle) const noexcept
    {
        if (!handle.isStructurallyValid()) {
            return nullptr;
        }

        switch (handle.kind) {
        case PageKind::Micro:
            if (handle.slot >= micro_capacity || micro_data == nullptr) {
                return nullptr;
            }
            return micro_data
                + static_cast<std::size_t>(handle.slot)
                    * micro_page_elements;
        case PageKind::Extent:
            if (handle.slot >= extent_capacity || extent_data == nullptr) {
                return nullptr;
            }
            return extent_data
                + static_cast<std::size_t>(handle.slot)
                    * extent_page_elements;
        }

        return nullptr;
    }

    [[nodiscard]] bool validTable(BlockTable const& table) const noexcept
    {
        if (!table.checkInvariants()) {
            return false;
        }

        for (MappingEntry const& entry : table.entries()) {
            if (pagePointer(entry.handle) == nullptr) {
                return false;
            }
        }

        return true;
    }
};

struct CudaSubmission::Impl final {
    std::shared_ptr<void> storage_owner{};
    cudaEvent_t event{nullptr};
    cudaStream_t stream{nullptr};
    std::vector<void*> temporaries{};
    CudaStatus submission_status{};
    CudaStatus final_status{};
    bool inject_completion_failure{false};
    bool finished{false};

    void cleanup() noexcept
    {
        for (void* allocation : temporaries) {
            if (allocation != nullptr) {
                static_cast<void>(cudaFree(allocation));
            }
        }
        temporaries.clear();

        if (event != nullptr) {
            static_cast<void>(cudaEventDestroy(event));
            event = nullptr;
        }
    }

    [[nodiscard]] CudaStatus complete(bool wait) noexcept
    {
        if (!submission_status.ok()) {
            return submission_status;
        }
        if (finished) {
            return final_status;
        }
        if (event == nullptr) {
            final_status = CudaStatus{CudaError::InternalError, 0};
            finished = true;
            cleanup();
            return final_status;
        }

        cudaError_t const event_status = wait
            ? cudaEventSynchronize(event)
            : cudaEventQuery(event);

        if (!wait && event_status == cudaErrorNotReady) {
            return CudaStatus{
                CudaError::NotReady,
                static_cast<int>(event_status),
            };
        }

        if (event_status != cudaSuccess) {
            final_status = CudaStatus{
                CudaError::ExecutionFailed,
                static_cast<int>(event_status),
            };
        } else if (inject_completion_failure) {
            final_status = CudaStatus{
                CudaError::ExecutionFailed,
                static_cast<int>(cudaErrorUnknown),
            };
        } else {
            final_status = CudaStatus{};
        }

        finished = true;
        cleanup();
        return final_status;
    }

    ~Impl()
    {
        if (!finished && submission_status.ok() && event != nullptr) {
            static_cast<void>(cudaEventSynchronize(event));
        }
        cleanup();
    }
};

namespace {

[[nodiscard]] std::unique_ptr<CudaSubmission::Impl> failedOperation(
    CudaStatus status)
{
    auto operation = std::make_unique<CudaSubmission::Impl>();
    operation->submission_status = status;
    operation->final_status = status;
    operation->finished = true;
    return operation;
}

[[nodiscard]] std::unique_ptr<CudaSubmission::Impl> beginOperation(
    std::shared_ptr<CudaKvStorage::Impl> const& storage,
    CudaStream stream)
{
    if (!storage || !storage->initialization_status.ok()) {
        return failedOperation(
            storage
                ? storage->initialization_status
                : CudaStatus{CudaError::AllocationFailed, 0}
        );
    }

    if (storage->consumeFailure(CudaFailurePoint::Submission)) {
        return failedOperation(
            CudaStatus{CudaError::SubmissionFailed, 0}
        );
    }

    auto operation = std::make_unique<CudaSubmission::Impl>();
    operation->storage_owner = storage;
    operation->stream = nativeStream(stream);
    operation->inject_completion_failure =
        storage->consumeFailure(CudaFailurePoint::Completion);

    cudaError_t const event_error = cudaEventCreateWithFlags(
        &operation->event,
        cudaEventDisableTiming
    );

    if (event_error != cudaSuccess) {
        operation->submission_status = mapSubmissionError(event_error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
    }

    return operation;
}

[[nodiscard]] CudaStatus recordOperation(
    CudaSubmission::Impl& operation) noexcept
{
    cudaError_t error = cudaGetLastError();

    if (error == cudaSuccess) {
        error = cudaEventRecord(operation.event, operation.stream);
    }

    if (error != cudaSuccess) {
        // 先前已成功入队的工作可能仍引用临时 Descriptor；错误路径允许
        // 同步 Stream，以保证 cleanup 不提前释放它们。
        static_cast<void>(cudaStreamSynchronize(operation.stream));
        operation.submission_status = mapSubmissionError(error);
        operation.final_status = operation.submission_status;
        operation.finished = true;
        operation.cleanup();
        return operation.submission_status;
    }

    operation.submission_status = CudaStatus{};
    return operation.submission_status;
}

[[nodiscard]] bool addTemporary(
    CudaSubmission::Impl& operation,
    void* allocation)
{
    try {
        operation.temporaries.push_back(allocation);
        return true;
    } catch (...) {
        static_cast<void>(cudaFree(allocation));
        operation.submission_status = CudaStatus{
            CudaError::AllocationFailed,
            0,
        };
        operation.final_status = operation.submission_status;
        operation.finished = true;
        operation.cleanup();
        return false;
    }
}

[[nodiscard]] DeviceLayout deviceLayout(KvLayout const& layout) noexcept
{
    return DeviceLayout{
        layout.layer_count,
        layout.kv_head_count,
        layout.head_dimension,
    };
}

} // namespace

CudaSubmission::CudaSubmission() noexcept = default;

CudaSubmission::CudaSubmission(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl))
{
}

CudaSubmission::~CudaSubmission() = default;
CudaSubmission::CudaSubmission(CudaSubmission&&) noexcept = default;
CudaSubmission& CudaSubmission::operator=(CudaSubmission&&) noexcept = default;

bool CudaSubmission::submitted() const noexcept
{
    return impl_ != nullptr && impl_->submission_status.ok();
}

CudaStatus CudaSubmission::submissionStatus() const noexcept
{
    return impl_ != nullptr
        ? impl_->submission_status
        : CudaStatus{CudaError::InvalidArgument, 0};
}

CudaStatus CudaSubmission::query() noexcept
{
    return impl_ != nullptr
        ? impl_->complete(false)
        : CudaStatus{CudaError::InvalidArgument, 0};
}

CudaStatus CudaSubmission::wait() noexcept
{
    return impl_ != nullptr
        ? impl_->complete(true)
        : CudaStatus{CudaError::InvalidArgument, 0};
}

CudaKvStorage::CudaKvStorage(
    KvLayout layout,
    std::uint32_t micro_capacity,
    std::uint32_t extent_capacity) noexcept
{
    try {
        impl_ = std::make_shared<Impl>();
    } catch (...) {
        return;
    }

    impl_->layout = layout;
    impl_->micro_capacity = micro_capacity;
    impl_->extent_capacity = extent_capacity;
    impl_->micro_page_elements = pageElements(layout, PageKind::Micro);
    impl_->extent_page_elements = pageElements(layout, PageKind::Extent);

    if (!layout.valid()
        || impl_->micro_page_elements == 0
        || impl_->extent_page_elements == 0) {
        impl_->initialization_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        return;
    }

    if (micro_capacity != 0
        && impl_->micro_page_elements >
            std::numeric_limits<std::size_t>::max()
                / micro_capacity / sizeof(KvScalar)) {
        impl_->initialization_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        return;
    }
    if (extent_capacity != 0
        && impl_->extent_page_elements >
            std::numeric_limits<std::size_t>::max()
                / extent_capacity / sizeof(KvScalar)) {
        impl_->initialization_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        return;
    }

    impl_->micro_reserved_bytes = impl_->micro_page_elements
        * micro_capacity * sizeof(KvScalar);
    impl_->extent_reserved_bytes = impl_->extent_page_elements
        * extent_capacity * sizeof(KvScalar);

    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess || device_count == 0) {
        impl_->initialization_status = error == cudaSuccess
            ? CudaStatus{CudaError::RuntimeUnavailable, 0}
            : mapInitializationError(error);
        return;
    }

    if (impl_->micro_reserved_bytes != 0) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&impl_->micro_data),
            impl_->micro_reserved_bytes
        );
        if (error == cudaSuccess) {
            error = cudaMemset(
                impl_->micro_data,
                0,
                impl_->micro_reserved_bytes
            );
        }
    }

    if (error == cudaSuccess && impl_->extent_reserved_bytes != 0) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&impl_->extent_data),
            impl_->extent_reserved_bytes
        );
        if (error == cudaSuccess) {
            error = cudaMemset(
                impl_->extent_data,
                0,
                impl_->extent_reserved_bytes
            );
        }
    }

    impl_->initialization_status = mapInitializationError(error);
}

CudaKvStorage::~CudaKvStorage() = default;

CudaStatus CudaKvStorage::status() const noexcept
{
    return impl_ != nullptr
        ? impl_->initialization_status
        : CudaStatus{CudaError::AllocationFailed, 0};
}

KvLayout CudaKvStorage::layout() const noexcept
{
    return impl_ != nullptr ? impl_->layout : KvLayout{};
}

CudaStorageSnapshot CudaKvStorage::snapshot() const noexcept
{
    if (impl_ == nullptr) {
        return CudaStorageSnapshot{};
    }

    return CudaStorageSnapshot{
        impl_->micro_capacity,
        impl_->extent_capacity,
        impl_->micro_reserved_bytes,
        impl_->extent_reserved_bytes,
    };
}

CudaSubmission CudaKvStorage::appendAsync(
    BlockTable const& before,
    BlockTable const& after,
    std::uint32_t append_token_begin,
    std::uint32_t token_count,
    KvScalar const* device_input,
    CudaStream stream)
{
    auto operation = beginOperation(impl_, stream);
    if (!operation->submission_status.ok()) {
        return CudaSubmission(std::move(operation));
    }

    std::uint64_t const expected_end =
        static_cast<std::uint64_t>(append_token_begin) + token_count;
    if (device_input == nullptr
        || token_count == 0
        || expected_end > std::numeric_limits<std::uint32_t>::max()
        || !impl_->validTable(before)
        || !impl_->validTable(after)
        || before.tokenCount() != append_token_begin
        || after.tokenCount() != expected_end) {
        operation->submission_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    DeviceLayout const layout = deviceLayout(impl_->layout);

    // 唯一允许改变旧逻辑区间 Handle 的情况是 Partial Tail COW。
    for (MappingEntry const& old_entry : before.entries()) {
        auto const replacement = std::find_if(
            after.entries().begin(),
            after.entries().end(),
            [&old_entry](MappingEntry const& entry) {
                return entry.logical_token_begin
                    == old_entry.logical_token_begin;
            }
        );

        if (replacement == after.entries().end()) {
            operation->submission_status = CudaStatus{
                CudaError::InvalidArgument,
                0,
            };
            operation->final_status = operation->submission_status;
            operation->finished = true;
            operation->cleanup();
            return CudaSubmission(std::move(operation));
        }

        if (replacement->handle == old_entry.handle) {
            continue;
        }

        bool const is_tail = old_entry.logicalTokenEnd()
            == append_token_begin;
        if (!is_tail
            || old_entry.kind != PageKind::Micro
            || replacement->kind != PageKind::Micro
            || old_entry.valid_tokens >= kMicroPageTokenCapacity
            || replacement->valid_tokens < old_entry.valid_tokens) {
            operation->submission_status = CudaStatus{
                CudaError::InvalidArgument,
                0,
            };
            operation->final_status = operation->submission_status;
            operation->finished = true;
            operation->cleanup();
            return CudaSubmission(std::move(operation));
        }

        std::size_t const elements = operationElements(
            impl_->layout,
            old_entry.valid_tokens
        );
        copyPageTokensKernel<<<
            gridFor(elements),
            kThreadsPerBlock,
            0,
            operation->stream
        >>>(
            impl_->pagePointer(old_entry.handle),
            kMicroPageTokenCapacity,
            impl_->pagePointer(replacement->handle),
            kMicroPageTokenCapacity,
            old_entry.valid_tokens,
            layout,
            elements
        );
    }

    std::uint32_t const append_end =
        append_token_begin + token_count;
    for (MappingEntry const& entry : after.entries()) {
        std::uint32_t const segment_begin = std::max(
            append_token_begin,
            entry.logical_token_begin
        );
        std::uint32_t const segment_end = std::min(
            append_end,
            entry.logicalTokenEnd()
        );

        if (segment_begin >= segment_end) {
            continue;
        }

        std::uint32_t const segment_tokens = segment_end - segment_begin;
        std::size_t const elements = operationElements(
            impl_->layout,
            segment_tokens
        );
        appendTokensKernel<<<
            gridFor(elements),
            kThreadsPerBlock,
            0,
            operation->stream
        >>>(
            device_input,
            token_count,
            segment_begin - append_token_begin,
            impl_->pagePointer(entry.handle),
            pageTokenCapacity(entry.kind),
            segment_begin - entry.logical_token_begin,
            segment_tokens,
            layout,
            elements
        );
    }

    static_cast<void>(recordOperation(*operation));
    return CudaSubmission(std::move(operation));
}

CudaSubmission CudaKvStorage::gatherAsync(
    BlockTable const& table,
    KvScalar* device_output,
    CudaStream stream)
{
    auto operation = beginOperation(impl_, stream);
    if (!operation->submission_status.ok()) {
        return CudaSubmission(std::move(operation));
    }

    std::uint32_t const token_count = table.tokenCount();
    if (device_output == nullptr
        || token_count == 0
        || !impl_->validTable(table)) {
        operation->submission_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    std::vector<DeviceBlockDescriptor> descriptors;
    try {
        descriptors.reserve(table.entries().size());
        for (MappingEntry const& entry : table.entries()) {
            descriptors.push_back(DeviceBlockDescriptor{
                impl_->pagePointer(entry.handle),
                entry.logical_token_begin,
                entry.valid_tokens,
                pageTokenCapacity(entry.kind),
            });
        }
    } catch (...) {
        operation->submission_status = CudaStatus{
            CudaError::AllocationFailed,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    DeviceBlockDescriptor* device_descriptors = nullptr;
    std::size_t const descriptor_bytes =
        descriptors.size() * sizeof(DeviceBlockDescriptor);
    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_descriptors),
        descriptor_bytes
    );

    if (error != cudaSuccess
        || !addTemporary(*operation, device_descriptors)) {
        if (error != cudaSuccess) {
            operation->submission_status = mapSubmissionError(error);
            operation->final_status = operation->submission_status;
            operation->finished = true;
            operation->cleanup();
        }
        return CudaSubmission(std::move(operation));
    }

    error = cudaMemcpyAsync(
        device_descriptors,
        descriptors.data(),
        descriptor_bytes,
        cudaMemcpyHostToDevice,
        operation->stream
    );

    if (error == cudaSuccess) {
        std::size_t const elements = operationElements(
            impl_->layout,
            token_count
        );
        gatherKernel<<<
            gridFor(elements),
            kThreadsPerBlock,
            0,
            operation->stream
        >>>(
            device_descriptors,
            static_cast<std::uint32_t>(descriptors.size()),
            device_output,
            token_count,
            deviceLayout(impl_->layout),
            elements
        );
    } else {
        operation->submission_status = mapSubmissionError(error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    static_cast<void>(recordOperation(*operation));
    return CudaSubmission(std::move(operation));
}

CudaSubmission CudaKvStorage::promoteAsync(
    PromotionPrepareResult const& promotion,
    CudaStream stream)
{
    auto operation = beginOperation(impl_, stream);
    if (!operation->submission_status.ok()) {
        return CudaSubmission(std::move(operation));
    }

    if (!promotion.ok()
        || promotion.target_handle.kind != PageKind::Extent
        || impl_->pagePointer(promotion.target_handle) == nullptr) {
        operation->submission_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    PromotionSlots slots{};
    for (std::size_t index = 0;
         index < promotion.source_handles.size();
         ++index) {
        PageHandle const source = promotion.source_handles[index];
        if (source.kind != PageKind::Micro
            || impl_->pagePointer(source) == nullptr) {
            operation->submission_status = CudaStatus{
                CudaError::InvalidArgument,
                0,
            };
            operation->final_status = operation->submission_status;
            operation->finished = true;
            operation->cleanup();
            return CudaSubmission(std::move(operation));
        }
        slots.slots[index] = source.slot;
    }

    std::size_t const elements = impl_->extent_page_elements;
    promotionKernel<<<
        gridFor(elements),
        kThreadsPerBlock,
        0,
        operation->stream
    >>>(
        impl_->micro_data,
        impl_->micro_page_elements,
        slots,
        impl_->pagePointer(promotion.target_handle),
        deviceLayout(impl_->layout),
        elements
    );

    static_cast<void>(recordOperation(*operation));
    return CudaSubmission(std::move(operation));
}

CudaSubmission CudaKvStorage::referenceAttentionAsync(
    BlockTable const& table,
    float const* device_query,
    float* device_output,
    CudaStream stream)
{
    auto operation = beginOperation(impl_, stream);
    if (!operation->submission_status.ok()) {
        return CudaSubmission(std::move(operation));
    }

    std::uint32_t const token_count = table.tokenCount();
    if (device_query == nullptr
        || device_output == nullptr
        || token_count == 0
        || !impl_->validTable(table)) {
        operation->submission_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    std::size_t kv_bytes = 0;
    std::size_t const score_count =
        static_cast<std::size_t>(impl_->layout.layer_count)
        * impl_->layout.kv_head_count
        * token_count;
    if (!impl_->layout.bytesForTokens(token_count, kv_bytes)
        || score_count >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        operation->submission_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    KvScalar* contiguous_kv = nullptr;
    float* scores = nullptr;
    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&contiguous_kv),
        kv_bytes
    );
    if (error == cudaSuccess
        && !addTemporary(*operation, contiguous_kv)) {
        return CudaSubmission(std::move(operation));
    }
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&scores),
            score_count * sizeof(float)
        );
    }
    if (error == cudaSuccess && !addTemporary(*operation, scores)) {
        return CudaSubmission(std::move(operation));
    }
    if (error != cudaSuccess) {
        operation->submission_status = mapSubmissionError(error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    // 与 gatherAsync 相同的 Descriptor 路径，但共享同一个 Event 和临时区。
    std::vector<DeviceBlockDescriptor> descriptors;
    try {
        descriptors.reserve(table.entries().size());
        for (MappingEntry const& entry : table.entries()) {
            descriptors.push_back(DeviceBlockDescriptor{
                impl_->pagePointer(entry.handle),
                entry.logical_token_begin,
                entry.valid_tokens,
                pageTokenCapacity(entry.kind),
            });
        }
    } catch (...) {
        operation->submission_status = CudaStatus{
            CudaError::AllocationFailed,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    DeviceBlockDescriptor* device_descriptors = nullptr;
    std::size_t const descriptor_bytes =
        descriptors.size() * sizeof(DeviceBlockDescriptor);
    error = cudaMalloc(
        reinterpret_cast<void**>(&device_descriptors),
        descriptor_bytes
    );
    if (error == cudaSuccess
        && !addTemporary(*operation, device_descriptors)) {
        return CudaSubmission(std::move(operation));
    }
    if (error != cudaSuccess) {
        operation->submission_status = mapSubmissionError(error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    error = cudaMemcpyAsync(
        device_descriptors,
        descriptors.data(),
        descriptor_bytes,
        cudaMemcpyHostToDevice,
        operation->stream
    );
    if (error != cudaSuccess) {
        operation->submission_status = mapSubmissionError(error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    std::size_t const elements = operationElements(
        impl_->layout,
        token_count
    );
    gatherKernel<<<
        gridFor(elements),
        kThreadsPerBlock,
        0,
        operation->stream
    >>>(
        device_descriptors,
        static_cast<std::uint32_t>(descriptors.size()),
        contiguous_kv,
        token_count,
        deviceLayout(impl_->layout),
        elements
    );

    std::uint32_t const attention_blocks =
        impl_->layout.layer_count * impl_->layout.kv_head_count;
    std::size_t const shared_bytes =
        kThreadsPerBlock * sizeof(float);
    attentionScoresKernel<<<
        attention_blocks,
        kThreadsPerBlock,
        shared_bytes,
        operation->stream
    >>>(
        contiguous_kv,
        token_count,
        device_query,
        scores,
        deviceLayout(impl_->layout)
    );
    attentionOutputKernel<<<
        attention_blocks,
        kThreadsPerBlock,
        shared_bytes,
        operation->stream
    >>>(
        contiguous_kv,
        token_count,
        scores,
        device_output,
        deviceLayout(impl_->layout)
    );

    static_cast<void>(recordOperation(*operation));
    return CudaSubmission(std::move(operation));
}

void CudaKvStorage::injectFailureOnce(CudaFailurePoint point) noexcept
{
    if (impl_ == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(impl_->fault_mutex);
    if (point == CudaFailurePoint::Submission) {
        impl_->fail_submission_once = true;
    } else {
        impl_->fail_completion_once = true;
    }
}

} // namespace kimkvcache
