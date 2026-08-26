#include "heteropage_kv/runtime/kv_cache_manager.h"

#include <limits>
#include <utility>

namespace kimkvcache {

PageLeaseId KvCacheManager::allocatePageLeaseIdLocked()
{
    std::size_t const maximum_attempts = page_leases_.size() + 1;

    for (std::size_t attempt = 0;
         attempt < maximum_attempts;
         ++attempt) {
        PageLeaseId const candidate = next_page_lease_id_;

        ++next_page_lease_id_;
        if (next_page_lease_id_ == kInvalidPageLeaseId) {
            next_page_lease_id_ = 1;
        }

        if (candidate != kInvalidPageLeaseId
            && page_leases_.find(candidate) == page_leases_.end()) {
            return candidate;
        }
    }

    return kInvalidPageLeaseId;
}

PageLeaseAcquireResult KvCacheManager::acquirePageLeaseLocked(
    std::vector<PageHandle> handles,
    BlockTable table)
{
    PageLeaseAcquireResult result;

    for (PageHandle const handle : handles) {
        RuntimeSlot const* runtime = runtimeSlotLocked(handle);

        if (runtime == nullptr
            || runtime->state == PageState::Free
            || runtime->state == PageState::Retiring
            || runtime->inflight_readers ==
                std::numeric_limits<std::uint32_t>::max()) {
            result.error = KvCacheError::InvalidState;
            return result;
        }
    }

    PageLeaseId const lease_id = allocatePageLeaseIdLocked();

    if (lease_id == kInvalidPageLeaseId) {
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }

    // 所有可能分配内存的 Result 构造都发生在修改 Runtime 计数之前。
    result.table = std::move(table);
    result.handles = handles;

    auto const insertion = page_leases_.emplace(
        lease_id,
        PageLease{lease_id, std::move(handles)}
    );

    if (!insertion.second) {
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }

    for (PageHandle const handle : insertion.first->second.handles) {
        RuntimeSlot* runtime = runtimeSlotLocked(handle);
        ++runtime->inflight_readers;
    }

    result.error = KvCacheError::None;
    result.lease_id = lease_id;
    return result;
}

PageLeaseAcquireResult KvCacheManager::acquireRequestReadLease(
    RequestId request_id)
{
    PageLeaseAcquireResult result;

    if (request_id == kInvalidRequestId) {
        result.error = KvCacheError::InvalidArgument;
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto const request_iterator = requests_.find(request_id);

    if (request_iterator == requests_.end()) {
        result.error = KvCacheError::RequestNotFound;
        return result;
    }

    BlockTable const& table = request_iterator->second.table;
    std::vector<PageHandle> handles;
    handles.reserve(table.entries().size());

    for (MappingEntry const& entry : table.entries()) {
        handles.push_back(entry.handle);
    }

    return acquirePageLeaseLocked(std::move(handles), table);
}

PageLeaseAcquireResult KvCacheManager::acquirePromotionIoLease(
    PromotionId promotion_id)
{
    PageLeaseAcquireResult result;

    if (promotion_id == kInvalidPromotionId) {
        result.error = KvCacheError::InvalidArgument;
        return result;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto const transaction_iterator = promotions_.find(promotion_id);

    if (transaction_iterator == promotions_.end()) {
        result.error = KvCacheError::PromotionNotFound;
        return result;
    }

    PromotionTransaction const& transaction =
        transaction_iterator->second;

    std::vector<PageHandle> handles;
    handles.reserve(kPromotionSourcePageCount + 1);

    for (PageHandle const source : transaction.source_handles) {
        handles.push_back(source);
    }
    handles.push_back(transaction.target_handle);

    return acquirePageLeaseLocked(std::move(handles), BlockTable{});
}

KvCacheError KvCacheManager::releasePageLease(PageLeaseId lease_id)
{
    if (lease_id == kInvalidPageLeaseId) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto const lease_iterator = page_leases_.find(lease_id);

    if (lease_iterator == page_leases_.end()) {
        return KvCacheError::InvalidState;
    }

    for (PageHandle const handle : lease_iterator->second.handles) {
        RuntimeSlot const* runtime = runtimeSlotLocked(handle);

        if (runtime == nullptr
            || runtime->state == PageState::Free
            || runtime->inflight_readers == 0) {
            return KvCacheError::InternalInvariantViolation;
        }
    }

    std::vector<PageHandle> const handles =
        lease_iterator->second.handles;
    page_leases_.erase(lease_iterator);

    for (PageHandle const handle : handles) {
        RuntimeSlot* runtime = runtimeSlotLocked(handle);
        --runtime->inflight_readers;

        if (runtime->state == PageState::Retiring
            && runtime->ref_count == 0
            && runtime->promotion_pins == 0
            && runtime->inflight_readers == 0) {
            KvCacheError const release_error = freePageLocked(handle);

            if (release_error != KvCacheError::None) {
                return release_error;
            }
        }
    }

    return KvCacheError::None;
}

} // namespace kimkvcache
