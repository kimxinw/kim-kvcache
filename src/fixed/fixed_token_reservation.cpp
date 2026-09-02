#include "kim-kv/fixed/fixed_page_manager.h"

#include <limits>
#include <utility>

namespace kimkvcache {

bool FixedPageManager::hasTokenReservationLocked(
    RequestId request_id) const noexcept
{
    return request_token_reservations_.find(request_id)
        != request_token_reservations_.end();
}

KvTokenReservationId FixedPageManager::allocateTokenReservationIdLocked()
{
    std::size_t const maximum_attempts = token_reservations_.size() + 1;
    for (std::size_t attempt = 0; attempt < maximum_attempts; ++attempt) {
        KvTokenReservationId const candidate = next_token_reservation_id_++;
        if (next_token_reservation_id_ == kInvalidKvTokenReservationId) {
            next_token_reservation_id_ = 1;
        }
        if (candidate != kInvalidKvTokenReservationId
            && token_reservations_.find(candidate)
                == token_reservations_.end()) {
            return candidate;
        }
    }
    return kInvalidKvTokenReservationId;
}

TokenReservationResult FixedPageManager::reserveToken(
    RequestId request_id,
    std::uint32_t expected_committed_tokens)
{
    TokenReservationResult result;
    result.request_id = request_id;
    result.logical_token_position = expected_committed_tokens;
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
    if (hasTokenReservationLocked(request_id)) {
        result.error = KvCacheError::RequestConflict;
        return result;
    }

    BlockTable const& before = request_iterator->second.table;
    if (before.tokenCount() != expected_committed_tokens
        || before.version_ == std::numeric_limits<std::uint64_t>::max()
        || expected_committed_tokens
            == std::numeric_limits<std::uint32_t>::max()) {
        result.error = KvCacheError::InvalidState;
        return result;
    }

    BlockTable candidate = before;
    candidate.entries_.reserve(candidate.entries_.size() + 1);
    PageHandle staged_target = PageHandle::invalid();
    PageHandle existing_mutable = PageHandle::invalid();
    PageHandle replaced_sealed_tail = PageHandle::invalid();
    if (candidate.entries_.empty()
        || candidate.entries_.back().valid_tokens == tokens_per_page_) {
        KvCacheError const allocation_error =
            allocateStagedLocked(0, staged_target);
        if (allocation_error != KvCacheError::None) {
            result.error = allocation_error;
            return result;
        }
        candidate.entries_.push_back(MappingEntry{
            expected_committed_tokens,
            1,
            PageKind::Micro,
            staged_target,
        });
    } else {
        MappingEntry& tail = candidate.entries_.back();
        RuntimeSlot* runtime = runtimeSlotLocked(tail.handle);
        if (runtime == nullptr
            || runtime->valid_tokens != tail.valid_tokens) {
            result.error = KvCacheError::InternalInvariantViolation;
            return result;
        }
        if (runtime->state == PageState::Mutable) {
            if (runtime->ref_count != 1
                || runtime->mutable_owner != request_id) {
                result.error = KvCacheError::InternalInvariantViolation;
                return result;
            }
            existing_mutable = tail.handle;
        } else if (runtime->state == PageState::Sealed) {
            KvCacheError const allocation_error = allocateStagedLocked(
                tail.valid_tokens,
                staged_target
            );
            if (allocation_error != KvCacheError::None) {
                result.error = allocation_error;
                return result;
            }
            replaced_sealed_tail = tail.handle;
            tail.handle = staged_target;
        } else {
            result.error = KvCacheError::InvalidState;
            return result;
        }
        ++tail.valid_tokens;
    }

    candidate.version_ = before.version_ + 1;
    if (!candidate.checkInvariants(tokens_per_page_)) {
        if (staged_target.isStructurallyValid()) {
            static_cast<void>(freePageLocked(staged_target));
        }
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }

    KvTokenReservationId const reservation_id =
        allocateTokenReservationIdLocked();
    if (reservation_id == kInvalidKvTokenReservationId) {
        if (staged_target.isStructurallyValid()) {
            static_cast<void>(freePageLocked(staged_target));
        }
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }
    auto const insertion = token_reservations_.emplace(
        reservation_id,
        TokenReservation{
            reservation_id,
            request_id,
            before.version_,
            candidate,
            staged_target,
            existing_mutable,
            replaced_sealed_tail,
        }
    );
    if (!insertion.second
        || !request_token_reservations_.emplace(
            request_id, reservation_id).second) {
        token_reservations_.erase(reservation_id);
        if (staged_target.isStructurallyValid()) {
            static_cast<void>(freePageLocked(staged_target));
        }
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }

    result.error = KvCacheError::None;
    result.reservation_id = reservation_id;
    result.before = before;
    result.reserved = std::move(candidate);
    return result;
}

KvCacheError FixedPageManager::commitTokenReservation(
    KvTokenReservationId reservation_id)
{
    if (reservation_id == kInvalidKvTokenReservationId) {
        return KvCacheError::InvalidArgument;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto transaction_iterator = token_reservations_.find(reservation_id);
    if (transaction_iterator == token_reservations_.end()) {
        return KvCacheError::TokenReservationNotFound;
    }
    TokenReservation& transaction = transaction_iterator->second;
    auto request_iterator = requests_.find(transaction.request_id);
    if (request_iterator == requests_.end()
        || request_iterator->second.table.version_
            != transaction.prepared_table_version) {
        return KvCacheError::RequestConflict;
    }

    MappingEntry const& committed_tail =
        transaction.candidate.entries_.back();
    RuntimeSlot* staged = transaction.staged_target.isStructurallyValid()
        ? runtimeSlotLocked(transaction.staged_target)
        : nullptr;
    RuntimeSlot* existing = transaction.existing_mutable.isStructurallyValid()
        ? runtimeSlotLocked(transaction.existing_mutable)
        : nullptr;
    RuntimeSlot* replaced =
        transaction.replaced_sealed_tail.isStructurallyValid()
        ? runtimeSlotLocked(transaction.replaced_sealed_tail)
        : nullptr;
    if ((staged != nullptr
            && (staged->state != PageState::CopyTarget
                || staged->ref_count != 0))
        || (existing != nullptr
            && (existing->state != PageState::Mutable
                || existing->ref_count != 1
                || existing->mutable_owner != transaction.request_id))
        || (replaced != nullptr
            && (replaced->state != PageState::Sealed
                || replaced->ref_count == 0))) {
        return KvCacheError::InternalInvariantViolation;
    }

    if (transaction.replaced_sealed_tail.isStructurallyValid()) {
        KvCacheError const decrement_error = decrementReferenceLocked(
            transaction.replaced_sealed_tail
        );
        if (decrement_error != KvCacheError::None) {
            return decrement_error;
        }
    }
    if (existing != nullptr) {
        existing->valid_tokens = committed_tail.valid_tokens;
        if (committed_tail.valid_tokens == tokens_per_page_) {
            existing->state = PageState::Sealed;
            existing->mutable_owner = kInvalidRequestId;
        }
    }
    if (staged != nullptr) {
        staged->valid_tokens = committed_tail.valid_tokens;
        staged->ref_count = 1;
        if (committed_tail.valid_tokens == tokens_per_page_) {
            staged->state = PageState::Sealed;
            staged->mutable_owner = kInvalidRequestId;
        } else {
            staged->state = PageState::Mutable;
            staged->mutable_owner = transaction.request_id;
        }
    }

    request_iterator->second.table = std::move(transaction.candidate);
    request_token_reservations_.erase(transaction.request_id);
    token_reservations_.erase(transaction_iterator);
    return KvCacheError::None;
}

KvCacheError FixedPageManager::rollbackTokenReservationLocked(
    KvTokenReservationId reservation_id)
{
    auto const iterator = token_reservations_.find(reservation_id);
    if (iterator == token_reservations_.end()) {
        return KvCacheError::TokenReservationNotFound;
    }
    TokenReservation const& transaction = iterator->second;
    if (transaction.staged_target.isStructurallyValid()) {
        KvCacheError const free_error = freePageLocked(
            transaction.staged_target
        );
        if (free_error != KvCacheError::None) {
            return free_error;
        }
    }
    request_token_reservations_.erase(transaction.request_id);
    token_reservations_.erase(iterator);
    return KvCacheError::None;
}

KvCacheError FixedPageManager::rollbackTokenReservation(
    KvTokenReservationId reservation_id)
{
    if (reservation_id == kInvalidKvTokenReservationId) {
        return KvCacheError::InvalidArgument;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return rollbackTokenReservationLocked(reservation_id);
}

} // namespace kimkvcache
