#include "kim-kv/cuda/fixed_cuda_kv_cache.h"

#include <utility>

namespace kimkvcache {
namespace {

[[nodiscard]] CudaStatus finalStatus(CudaSubmission& submission) noexcept
{
    CudaStatus const submitted = submission.submissionStatus();
    return submitted.ok() ? submission.wait() : submitted;
}

} // namespace

FixedCudaKvCache::FixedCudaKvCache(
    KvLayout layout,
    std::uint16_t tokens_per_page,
    std::uint32_t page_capacity)
    : tokens_per_page_(tokens_per_page)
    , manager_(tokens_per_page, page_capacity)
    , storage_(
        layout,
        page_capacity,
        0,
        tokens_per_page,
        kExtentPageTokenCapacity
    )
{
}

CudaStatus FixedCudaKvCache::status() const noexcept
{
    return storage_.status();
}

std::uint16_t FixedCudaKvCache::tokensPerPage() const noexcept
{
    return tokens_per_page_;
}

KvCacheError FixedCudaKvCache::createRequest(RequestId request_id)
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    if (!status().ok()) {
        return KvCacheError::InvalidState;
    }
    return manager_.createRequest(request_id);
}

KvCacheError FixedCudaKvCache::sealTail(RequestId request_id)
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    return manager_.sealTail(request_id);
}

KvCacheError FixedCudaKvCache::forkRequest(
    RequestId source_request_id,
    RequestId child_request_id)
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    return manager_.forkRequest(source_request_id, child_request_id);
}

KvCacheError FixedCudaKvCache::releaseRequest(RequestId request_id)
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    return manager_.releaseRequest(request_id);
}

CudaKvOperationResult FixedCudaKvCache::metadataFailure(
    KvCacheError error) noexcept
{
    return CudaKvOperationResult{error, CudaStatus{}};
}

CudaKvOperationResult FixedCudaKvCache::append(
    RequestId request_id,
    std::uint32_t token_count,
    KvScalar const* device_input,
    CudaStream stream)
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    if (request_id == kInvalidRequestId
        || token_count == 0
        || device_input == nullptr) {
        return metadataFailure(KvCacheError::InvalidArgument);
    }
    if (!status().ok()) {
        return CudaKvOperationResult{KvCacheError::None, status()};
    }

    std::optional<BlockTable> const before = manager_.blockTable(request_id);
    if (!before.has_value()) {
        return metadataFailure(KvCacheError::RequestNotFound);
    }

    std::uint32_t const append_begin = before->tokenCount();
    KvCacheError const append_error = manager_.append(
        request_id,
        token_count
    );
    if (append_error != KvCacheError::None) {
        return metadataFailure(append_error);
    }

    std::optional<BlockTable> const after = manager_.blockTable(request_id);
    if (!after.has_value()) {
        return metadataFailure(KvCacheError::RequestNotFound);
    }

    CudaSubmission submission = storage_.appendAsync(
        *before,
        *after,
        append_begin,
        token_count,
        device_input,
        stream
    );
    CudaStatus const cuda_status = finalStatus(submission);
    if (!cuda_status.ok()) {
        KvCacheError const cancel_error = manager_.releaseRequest(request_id);
        if (cancel_error != KvCacheError::None
            && cancel_error != KvCacheError::RequestNotFound) {
            return CudaKvOperationResult{cancel_error, cuda_status};
        }
    }
    return CudaKvOperationResult{KvCacheError::None, cuda_status};
}

CudaKvOperationResult FixedCudaKvCache::gather(
    RequestId request_id,
    KvScalar* device_output,
    CudaStream stream)
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    if (request_id == kInvalidRequestId || device_output == nullptr) {
        return metadataFailure(KvCacheError::InvalidArgument);
    }
    std::optional<BlockTable> const table = manager_.blockTable(request_id);
    if (!table.has_value()) {
        return metadataFailure(KvCacheError::RequestNotFound);
    }
    CudaSubmission submission = storage_.gatherAsync(
        *table,
        device_output,
        stream
    );
    return CudaKvOperationResult{KvCacheError::None, finalStatus(submission)};
}

CudaKvOperationResult FixedCudaKvCache::referenceAttention(
    RequestId request_id,
    float const* device_query,
    float* device_output,
    CudaStream stream)
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    if (request_id == kInvalidRequestId
        || device_query == nullptr
        || device_output == nullptr) {
        return metadataFailure(KvCacheError::InvalidArgument);
    }
    std::optional<BlockTable> const table = manager_.blockTable(request_id);
    if (!table.has_value()) {
        return metadataFailure(KvCacheError::RequestNotFound);
    }
    CudaSubmission submission = storage_.referenceAttentionAsync(
        *table,
        device_query,
        device_output,
        stream
    );
    return CudaKvOperationResult{KvCacheError::None, finalStatus(submission)};
}

std::optional<BlockTable> FixedCudaKvCache::blockTable(
    RequestId request_id) const
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    return manager_.blockTable(request_id);
}

FixedPageManagerSnapshot FixedCudaKvCache::metadataSnapshot() const
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    return manager_.snapshot();
}

CudaStorageSnapshot FixedCudaKvCache::storageSnapshot() const noexcept
{
    return storage_.snapshot();
}

bool FixedCudaKvCache::checkInvariants() const
{
    std::lock_guard<std::mutex> lock(orchestration_mutex_);
    return manager_.checkInvariants();
}

void FixedCudaKvCache::injectFailureOnce(
    CudaFailurePoint point) noexcept
{
    storage_.injectFailureOnce(point);
}

} // namespace kimkvcache
