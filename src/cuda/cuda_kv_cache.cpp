#include "kim-kv/cuda/cuda_kv_cache.h"

#include <utility>

namespace kimkvcache {
namespace {

[[nodiscard]] CudaStatus finalStatus(CudaSubmission& submission) noexcept
{
    CudaStatus const submitted = submission.submissionStatus();
    return submitted.ok() ? submission.wait() : submitted;
}

} // namespace

CudaKvCache::CudaKvCache(
    KvLayout layout,
    std::uint32_t micro_capacity,
    std::uint32_t extent_capacity)
    : manager_(micro_capacity, extent_capacity)
    , storage_(layout, micro_capacity, extent_capacity)
{
}

CudaStatus CudaKvCache::status() const noexcept
{
    return storage_.status();
}

KvCacheError CudaKvCache::createRequest(RequestId request_id)
{
    if (!status().ok()) {
        return KvCacheError::InvalidState;
    }
    return manager_.createRequest(request_id);
}

KvCacheError CudaKvCache::sealTail(RequestId request_id)
{
    return manager_.sealTail(request_id);
}

KvCacheError CudaKvCache::forkRequest(
    RequestId source_request_id,
    RequestId child_request_id)
{
    return manager_.forkRequest(source_request_id, child_request_id);
}

KvCacheError CudaKvCache::releaseRequest(RequestId request_id)
{
    return manager_.releaseRequest(request_id);
}

CudaKvOperationResult CudaKvCache::metadataFailure(
    KvCacheError error) noexcept
{
    return CudaKvOperationResult{error, CudaStatus{}};
}

CudaKvOperationResult CudaKvCache::append(
    RequestId request_id,
    std::uint32_t token_count,
    KvScalar const* device_input,
    CudaStream stream)
{
    if (request_id == kInvalidRequestId
        || token_count == 0
        || device_input == nullptr) {
        return metadataFailure(KvCacheError::InvalidArgument);
    }

    CudaStatus const storage_status = status();
    if (!storage_status.ok()) {
        return CudaKvOperationResult{KvCacheError::None, storage_status};
    }

    PageLeaseAcquireResult before =
        manager_.acquireRequestReadLease(request_id);

    if (!before.ok()) {
        return metadataFailure(before.error);
    }

    std::uint32_t const append_begin = before.table.tokenCount();
    KvCacheError const append_error =
        manager_.append(request_id, token_count);

    if (append_error != KvCacheError::None) {
        KvCacheError const lease_error =
            manager_.releasePageLease(before.lease_id);
        return metadataFailure(
            lease_error == KvCacheError::None
                ? append_error
                : lease_error
        );
    }

    PageLeaseAcquireResult after =
        manager_.acquireRequestReadLease(request_id);

    if (!after.ok()) {
        KvCacheError const before_release =
            manager_.releasePageLease(before.lease_id);

        if (after.error != KvCacheError::RequestNotFound) {
            static_cast<void>(manager_.releaseRequest(request_id));
        }

        return metadataFailure(
            before_release == KvCacheError::None
                ? after.error
                : before_release
        );
    }

    CudaSubmission submission = storage_.appendAsync(
        before.table,
        after.table,
        append_begin,
        token_count,
        device_input,
        stream
    );
    CudaStatus const cuda_status = finalStatus(submission);

    KvCacheError const after_release =
        manager_.releasePageLease(after.lease_id);
    KvCacheError const before_release =
        manager_.releasePageLease(before.lease_id);

    if (after_release != KvCacheError::None) {
        return metadataFailure(after_release);
    }
    if (before_release != KvCacheError::None) {
        return metadataFailure(before_release);
    }

    if (!cuda_status.ok()) {
        // Metadata Append 已提交但设备数据不可用，不能继续暴露该请求。
        // 取消请求比保留一张含未初始化 KV 的 BlockTable 更安全。
        KvCacheError const cancel_error = manager_.releaseRequest(request_id);
        if (cancel_error != KvCacheError::None
            && cancel_error != KvCacheError::RequestNotFound) {
            return CudaKvOperationResult{cancel_error, cuda_status};
        }
        return CudaKvOperationResult{KvCacheError::None, cuda_status};
    }

    if (!manager_.blockTable(request_id).has_value()) {
        return metadataFailure(KvCacheError::RequestNotFound);
    }

    return CudaKvOperationResult{};
}

CudaKvOperationResult CudaKvCache::promote(
    RequestId request_id,
    std::uint32_t logical_token_begin,
    CudaStream stream)
{
    CudaStatus const storage_status = status();
    if (!storage_status.ok()) {
        return CudaKvOperationResult{KvCacheError::None, storage_status};
    }

    PromotionPrepareResult const prepared =
        manager_.preparePromotion(request_id, logical_token_begin);

    if (!prepared.ok()) {
        return metadataFailure(prepared.error);
    }

    PageLeaseAcquireResult lease =
        manager_.acquirePromotionIoLease(prepared.promotion_id);

    if (!lease.ok()) {
        KvCacheError const rollback_error =
            manager_.rollbackPromotion(prepared.promotion_id);
        return metadataFailure(
            rollback_error == KvCacheError::None
                ? lease.error
                : rollback_error
        );
    }

    CudaSubmission submission = storage_.promoteAsync(prepared, stream);
    CudaStatus const cuda_status = finalStatus(submission);
    KvCacheError const lease_error =
        manager_.releasePageLease(lease.lease_id);

    if (lease_error != KvCacheError::None) {
        static_cast<void>(manager_.rollbackPromotion(
            prepared.promotion_id
        ));
        return metadataFailure(lease_error);
    }

    if (!cuda_status.ok()) {
        KvCacheError const rollback_error =
            manager_.rollbackPromotion(prepared.promotion_id);

        if (rollback_error != KvCacheError::None
            && rollback_error != KvCacheError::PromotionNotFound) {
            return CudaKvOperationResult{rollback_error, cuda_status};
        }
        return CudaKvOperationResult{KvCacheError::None, cuda_status};
    }

    KvCacheError const commit_error =
        manager_.commitPromotion(prepared.promotion_id);

    return CudaKvOperationResult{commit_error, cuda_status};
}

CudaKvOperationResult CudaKvCache::finishReadOperation(
    PageLeaseId lease_id,
    CudaSubmission submission)
{
    CudaStatus const cuda_status = finalStatus(submission);
    KvCacheError const lease_error = manager_.releasePageLease(lease_id);
    return CudaKvOperationResult{lease_error, cuda_status};
}

CudaKvOperationResult CudaKvCache::gather(
    RequestId request_id,
    KvScalar* device_output,
    CudaStream stream)
{
    if (request_id == kInvalidRequestId || device_output == nullptr) {
        return metadataFailure(KvCacheError::InvalidArgument);
    }

    CudaStatus const storage_status = status();
    if (!storage_status.ok()) {
        return CudaKvOperationResult{KvCacheError::None, storage_status};
    }

    PageLeaseAcquireResult lease =
        manager_.acquireRequestReadLease(request_id);

    if (!lease.ok()) {
        return metadataFailure(lease.error);
    }

    return finishReadOperation(
        lease.lease_id,
        storage_.gatherAsync(lease.table, device_output, stream)
    );
}

CudaKvOperationResult CudaKvCache::referenceAttention(
    RequestId request_id,
    float const* device_query,
    float* device_output,
    CudaStream stream)
{
    if (request_id == kInvalidRequestId
        || device_query == nullptr
        || device_output == nullptr) {
        return metadataFailure(KvCacheError::InvalidArgument);
    }

    CudaStatus const storage_status = status();
    if (!storage_status.ok()) {
        return CudaKvOperationResult{KvCacheError::None, storage_status};
    }

    PageLeaseAcquireResult lease =
        manager_.acquireRequestReadLease(request_id);

    if (!lease.ok()) {
        return metadataFailure(lease.error);
    }

    return finishReadOperation(
        lease.lease_id,
        storage_.referenceAttentionAsync(
            lease.table,
            device_query,
            device_output,
            stream
        )
    );
}

std::optional<BlockTable> CudaKvCache::blockTable(
    RequestId request_id) const
{
    return manager_.blockTable(request_id);
}

KvCacheManagerSnapshot CudaKvCache::metadataSnapshot() const
{
    return manager_.snapshot();
}

CudaStorageSnapshot CudaKvCache::storageSnapshot() const noexcept
{
    return storage_.snapshot();
}

bool CudaKvCache::checkInvariants() const
{
    return manager_.checkInvariants();
}

void CudaKvCache::injectFailureOnce(CudaFailurePoint point) noexcept
{
    storage_.injectFailureOnce(point);
}

} // namespace kimkvcache
