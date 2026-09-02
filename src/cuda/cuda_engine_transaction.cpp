#include "cuda_kv_storage_internal.cuh"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace kimkvcache {
namespace {

[[nodiscard]] CudaStatus invalidArgument() noexcept
{
    return CudaStatus{CudaError::InvalidArgument, 0};
}

[[nodiscard]] CudaStatus injectedSubmissionFailure() noexcept
{
    return CudaStatus{CudaError::SubmissionFailed, 0};
}

} // namespace

CudaEngineTransaction::Impl::~Impl()
{
    static_cast<void>(complete());
    if (device_descriptors != nullptr) {
        static_cast<void>(cudaFree(device_descriptors));
    }
    if (completion_event != nullptr) {
        static_cast<void>(cudaEventDestroy(completion_event));
    }
}

CudaStatus CudaEngineTransaction::Impl::failSubmission(
    cudaError_t error) noexcept
{
    submission_status = cuda_storage_detail::mapSubmissionError(error);
    final_status = submission_status;
    return submission_status;
}

CudaStatus CudaEngineTransaction::Impl::complete() noexcept
{
    if (finished) {
        return final_status;
    }
    if (!submission_status.ok()) {
        static_cast<void>(cudaStreamSynchronize(stream));
        final_status = submission_status;
        finished = true;
        return final_status;
    }
    if (completion_event == nullptr) {
        final_status = CudaStatus{CudaError::InternalError, 0};
        finished = true;
        return final_status;
    }

    cudaError_t error = cudaEventRecord(completion_event, stream);
    if (error == cudaSuccess) {
        error = cudaEventSynchronize(completion_event);
    }
    if (error != cudaSuccess) {
        final_status = CudaStatus{
            CudaError::ExecutionFailed,
            static_cast<int>(error),
        };
    } else if (storage->consumeFailure(CudaFailurePoint::Completion)) {
        final_status = CudaStatus{
            CudaError::ExecutionFailed,
            static_cast<int>(cudaErrorUnknown),
        };
    } else {
        final_status = {};
    }
    finished = true;
    return final_status;
}

CudaEngineTransaction::CudaEngineTransaction(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl))
{
}

CudaEngineTransaction::~CudaEngineTransaction() = default;
CudaEngineTransaction::CudaEngineTransaction(
    CudaEngineTransaction&&) noexcept = default;
CudaEngineTransaction& CudaEngineTransaction::operator=(
    CudaEngineTransaction&&) noexcept = default;

CudaStatus CudaEngineTransaction::status() const noexcept
{
    return impl_ != nullptr ? impl_->submission_status : invalidArgument();
}

CudaStatus CudaEngineTransaction::writeLayer(
    LayerKvWrite const& write) noexcept
{
    if (impl_ == nullptr
        || impl_->finished
        || !impl_->submission_status.ok()
        || write.layer >= impl_->storage->layout.layer_count
        || write.device_key == nullptr
        || write.device_value == nullptr) {
        return impl_ != nullptr && !impl_->submission_status.ok()
            ? impl_->submission_status
            : invalidArgument();
    }
    if (impl_->storage->consumeFailure(CudaFailurePoint::Submission)) {
        impl_->submission_status = injectedSubmissionFailure();
        impl_->final_status = impl_->submission_status;
        return impl_->submission_status;
    }

    MappingEntry const& target_entry = impl_->reserved.entries().back();
    KvScalar* target = impl_->storage->pagePointer(target_entry.handle);
    std::uint16_t const target_capacity =
        impl_->storage->pageTokenCapacity(target_entry.kind);
    if (target == nullptr
        || impl_->reserved.tokenCount() == 0
        || impl_->reserved.tokenCount() - 1
            < target_entry.logical_token_begin) {
        return impl_->failSubmission(cudaErrorInvalidValue);
    }

    if (!impl_->before.entries().empty()) {
        MappingEntry const& old_tail = impl_->before.entries().back();
        if (old_tail.logical_token_begin
                == target_entry.logical_token_begin
            && old_tail.handle != target_entry.handle) {
            KvScalar const* source = impl_->storage->pagePointer(
                old_tail.handle
            );
            if (source == nullptr) {
                return impl_->failSubmission(cudaErrorInvalidValue);
            }
            cuda_detail::launchCopyLayerTokens(
                source,
                impl_->storage->pageTokenCapacity(old_tail.kind),
                target,
                target_capacity,
                old_tail.valid_tokens,
                write.layer,
                cuda_storage_detail::deviceLayout(impl_->storage->layout),
                impl_->stream
            );
        }
    }

    std::uint32_t const page_token = impl_->reserved.tokenCount() - 1
        - target_entry.logical_token_begin;
    cuda_detail::launchWriteLayerToken(
        write.device_key,
        write.device_value,
        target,
        target_capacity,
        page_token,
        write.layer,
        cuda_storage_detail::deviceLayout(impl_->storage->layout),
        impl_->stream
    );
    cudaError_t const error = cudaGetLastError();
    return error == cudaSuccess
        ? CudaStatus{}
        : impl_->failSubmission(error);
}

CudaStatus CudaEngineTransaction::attendLayer(
    PagedDecodeRequest const& request) noexcept
{
    if (impl_ == nullptr
        || impl_->finished
        || !impl_->submission_status.ok()
        || request.layer >= impl_->storage->layout.layer_count
        || request.device_query == nullptr
        || request.device_output == nullptr
        || request.device_workspace == nullptr
        || !(request.attention_scale > 0.0F)) {
        return impl_ != nullptr && !impl_->submission_status.ok()
            ? impl_->submission_status
            : invalidArgument();
    }
    std::uint32_t const token_count = impl_->reserved.tokenCount();
    std::size_t const score_count =
        static_cast<std::size_t>(impl_->query_head_count) * token_count;
    if (token_count == 0
        || score_count > std::numeric_limits<std::size_t>::max()
            / sizeof(float)
        || request.workspace_bytes < score_count * sizeof(float)) {
        return invalidArgument();
    }
    if (impl_->storage->consumeFailure(CudaFailurePoint::Submission)) {
        impl_->submission_status = injectedSubmissionFailure();
        impl_->final_status = impl_->submission_status;
        return impl_->submission_status;
    }

    cuda_detail::launchPagedDecodeAttention(
        impl_->device_descriptors,
        impl_->descriptor_count,
        impl_->storage->micro_data,
        impl_->storage->micro_page_elements,
        impl_->storage->extent_data,
        impl_->storage->extent_page_elements,
        token_count,
        request.layer,
        impl_->query_head_count,
        request.device_query,
        static_cast<float*>(request.device_workspace),
        request.device_output,
        request.attention_scale,
        cuda_storage_detail::deviceLayout(impl_->storage->layout),
        impl_->stream
    );
    cudaError_t const error = cudaGetLastError();
    return error == cudaSuccess
        ? CudaStatus{}
        : impl_->failSubmission(error);
}

CudaStatus CudaEngineTransaction::finish() noexcept
{
    return impl_ != nullptr ? impl_->complete() : invalidArgument();
}

CudaEngineTransactionBeginResult CudaKvStorage::beginEngineTransaction(
    BlockTable const& before,
    BlockTable const& reserved,
    std::uint32_t query_head_count,
    CudaStream stream)
{
    CudaEngineTransactionBeginResult result;
    if (impl_ == nullptr || !impl_->initialization_status.ok()) {
        result.status = impl_ != nullptr
            ? impl_->initialization_status
            : CudaStatus{CudaError::AllocationFailed, 0};
        return result;
    }
    if (!impl_->validTable(before)
        || !impl_->validTable(reserved)
        || reserved.tokenCount() != before.tokenCount() + 1
        || query_head_count < impl_->layout.kv_head_count
        || query_head_count % impl_->layout.kv_head_count != 0
        || reserved.entries().empty()) {
        result.status = invalidArgument();
        return result;
    }
    if (impl_->consumeFailure(CudaFailurePoint::Submission)) {
        result.status = injectedSubmissionFailure();
        return result;
    }

    std::unique_ptr<CudaEngineTransaction::Impl> transaction;
    std::vector<::kimkvcache::DeviceBlockDescriptor> descriptors;
    try {
        transaction = std::make_unique<CudaEngineTransaction::Impl>();
        transaction->storage = impl_;
        transaction->before = before;
        transaction->reserved = reserved;
        transaction->descriptor_count = static_cast<std::uint32_t>(
            reserved.entries().size()
        );
        transaction->query_head_count = query_head_count;
        transaction->stream = cuda_storage_detail::nativeStream(stream);
        descriptors.reserve(reserved.entries().size());
        for (MappingEntry const& entry : reserved.entries()) {
            descriptors.push_back(::kimkvcache::DeviceBlockDescriptor{
                entry.handle.slot,
                entry.handle.generation,
                entry.logical_token_begin,
                entry.valid_tokens,
                impl_->pageTokenCapacity(entry.kind),
                entry.kind,
            });
        }
    } catch (...) {
        result.status = CudaStatus{CudaError::AllocationFailed, 0};
        return result;
    }

    cudaError_t error = cudaEventCreateWithFlags(
        &transaction->completion_event,
        cudaEventDisableTiming
    );
    std::size_t const descriptor_bytes = descriptors.size()
        * sizeof(::kimkvcache::DeviceBlockDescriptor);
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&transaction->device_descriptors),
            descriptor_bytes
        );
    }
    if (error == cudaSuccess) {
        error = cudaMemcpyAsync(
            transaction->device_descriptors,
            descriptors.data(),
            descriptor_bytes,
            cudaMemcpyHostToDevice,
            transaction->stream
        );
    }
    if (error != cudaSuccess) {
        result.status = cuda_storage_detail::mapSubmissionError(error);
        return result;
    }

    transaction->submission_status = {};
    result.transaction = std::unique_ptr<CudaEngineTransaction>(
        new CudaEngineTransaction(std::move(transaction))
    );
    result.status = {};
    return result;
}

} // namespace kimkvcache
