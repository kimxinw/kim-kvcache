#include "kim-kv/runtime/kv_cache_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace kimkvcache {

KvCacheError KvCacheManager::sealTail(RequestId request_id)
{
    if (request_id == kInvalidRequestId) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto request_iterator = requests_.find(request_id);

    if (request_iterator == requests_.end()) {
        return KvCacheError::RequestNotFound;
    }

    if (hasTokenReservationLocked(request_id)) {
        return KvCacheError::RequestConflict;
    }

    BlockTable& table = request_iterator->second.table;

    if (table.entries_.empty()) {
        return KvCacheError::InvalidState;
    }

    MappingEntry const& tail = table.entries_.back();
    RuntimeSlot* runtime = runtimeSlotLocked(tail.handle);

    if (runtime == nullptr
        || runtime->valid_tokens != tail.valid_tokens) {
        return KvCacheError::InternalInvariantViolation;
    }

    if (runtime->state == PageState::Sealed) {
        return KvCacheError::None;
    }

    if (runtime->state != PageState::Mutable
        || runtime->ref_count != 1
        || runtime->mutable_owner != request_id) {
        return KvCacheError::InvalidState;
    }

    runtime->state = PageState::Sealed;
    runtime->mutable_owner = kInvalidRequestId;

    // Mapping 没有变化，因此 BlockTable Version 不增加。
    return KvCacheError::None;
}

KvCacheError KvCacheManager::forkRequest(
    RequestId source_request_id,
    RequestId child_request_id)
{
    if (source_request_id == kInvalidRequestId
        || child_request_id == kInvalidRequestId
        || source_request_id == child_request_id) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto source_iterator = requests_.find(source_request_id);

    if (source_iterator == requests_.end()) {
        return KvCacheError::RequestNotFound;
    }

    if (hasTokenReservationLocked(source_request_id)) {
        return KvCacheError::RequestConflict;
    }

    if (requests_.find(child_request_id) != requests_.end()) {
        return KvCacheError::RequestAlreadyExists;
    }

    BlockTable const& source_table = source_iterator->second.table;

    for (std::size_t index = 0;
         index < source_table.entries_.size();
         ++index) {
        MappingEntry const& entry = source_table.entries_[index];
        RuntimeSlot* runtime = runtimeSlotLocked(entry.handle);

        if (runtime == nullptr
            || runtime->valid_tokens != entry.valid_tokens
            || runtime->ref_count == 0
            || runtime->ref_count ==
                std::numeric_limits<std::uint32_t>::max()) {
            return KvCacheError::InternalInvariantViolation;
        }

        if (runtime->state == PageState::Mutable) {
            bool const is_tail =
                index + 1 == source_table.entries_.size();

            if (!is_tail
                || runtime->ref_count != 1
                || runtime->mutable_owner != source_request_id) {
                return KvCacheError::InternalInvariantViolation;
            }
        } else if (runtime->state != PageState::Sealed) {
            return KvCacheError::InvalidState;
        }
    }

    BlockTable child_table = source_table;

    // Child Version 是独立的，不继承 Source 的历史版本号。
    child_table.version_ = child_table.entries_.empty() ? 0 : 1;

    auto const insertion = requests_.emplace(
        child_request_id,
        RequestState{std::move(child_table)}
    );

    if (!insertion.second) {
        return KvCacheError::RequestAlreadyExists;
    }

    for (MappingEntry const& entry : source_table.entries_) {
        RuntimeSlot* runtime = runtimeSlotLocked(entry.handle);

        if (runtime->state == PageState::Mutable) {
            runtime->state = PageState::Sealed;
            runtime->mutable_owner = kInvalidRequestId;
        }

        ++runtime->ref_count;
    }

    return KvCacheError::None;
}

KvCacheError KvCacheManager::releaseRequest(RequestId request_id)
{
    if (request_id == kInvalidRequestId) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto request_iterator = requests_.find(request_id);

    if (request_iterator == requests_.end()) {
        return KvCacheError::RequestNotFound;
    }

    if (hasTokenReservationLocked(request_id)) {
        return KvCacheError::RequestConflict;
    }

    KvCacheError const rollback_error =
        rollbackPromotionsForRequestLocked(request_id);

    if (rollback_error != KvCacheError::None) {
        return rollback_error;
    }

    BlockTable const& table = request_iterator->second.table;

    for (MappingEntry const& entry : table.entries_) {
        RuntimeSlot const* runtime = runtimeSlotLocked(entry.handle);

        if (runtime == nullptr
            || runtime->state == PageState::Free
            || runtime->state == PageState::CopyTarget
            || runtime->ref_count == 0) {
            return KvCacheError::InternalInvariantViolation;
        }
    }

    for (MappingEntry const& entry : table.entries_) {
        KvCacheError const decrement_error =
            decrementReferenceLocked(entry.handle);

        if (decrement_error != KvCacheError::None) {
            return decrement_error;
        }
    }

    requests_.erase(request_iterator);

    return KvCacheError::None;
}

} // namespace kimkvcache
