#include "kim-kv/runtime/kv_cache_manager.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>

namespace kimkvcache {

KvCacheError KvCacheManager::allocatePromotionTargetLocked(
    PageHandle& handle)
{
    PageAllocationResult const allocation = extent_pool_.allocate();

    if (!allocation.ok()) {
        if (allocation.error == PagePoolError::Exhausted) {
            return KvCacheError::ResourceExhausted;
        }

        return KvCacheError::InternalInvariantViolation;
    }

    handle = allocation.handle;

    if (handle.slot >= extent_runtime_.size()) {
        static_cast<void>(extent_pool_.release(handle));
        handle = PageHandle::invalid();
        return KvCacheError::InternalInvariantViolation;
    }

    RuntimeSlot& runtime = extent_runtime_[handle.slot];

    if (runtime.state != PageState::Free
        || runtime.ref_count != 0
        || runtime.promotion_pins != 0
        || runtime.inflight_readers != 0) {
        static_cast<void>(extent_pool_.release(handle));
        handle = PageHandle::invalid();
        return KvCacheError::InternalInvariantViolation;
    }

    runtime.state = PageState::CopyTarget;
    runtime.generation = handle.generation;
    runtime.valid_tokens = kExtentPageTokenCapacity;
    runtime.ref_count = 0;
    runtime.promotion_pins = 0;
    runtime.inflight_readers = 0;
    runtime.mutable_owner = kInvalidRequestId;

    return KvCacheError::None;
}

PromotionId KvCacheManager::allocatePromotionIdLocked()
{
    // 在最多 N+1 个候选中一定能找到一个未被 N 个活跃事务占用的 ID。
    std::size_t const maximum_attempts = promotions_.size() + 1;

    for (std::size_t attempt = 0;
         attempt < maximum_attempts;
         ++attempt) {
        PromotionId const candidate = next_promotion_id_;

        ++next_promotion_id_;
        if (next_promotion_id_ == kInvalidPromotionId) {
            next_promotion_id_ = 1;
        }

        if (candidate != kInvalidPromotionId
            && promotions_.find(candidate) == promotions_.end()) {
            return candidate;
        }
    }

    return kInvalidPromotionId;
}

PromotionPrepareResult KvCacheManager::preparePromotion(
    RequestId request_id,
    std::uint32_t logical_token_begin)
{
    PromotionPrepareResult result;
    result.request_id = request_id;
    result.logical_token_begin = logical_token_begin;

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

    if (table.version_ == std::numeric_limits<std::uint64_t>::max()) {
        result.error = KvCacheError::PromotionConflict;
        return result;
    }

    auto const first_entry = std::find_if(
        table.entries_.begin(),
        table.entries_.end(),
        [logical_token_begin](MappingEntry const& entry) {
            return entry.logical_token_begin == logical_token_begin;
        }
    );

    if (first_entry == table.entries_.end()) {
        result.error = KvCacheError::PromotionNotEligible;
        return result;
    }

    std::size_t const source_entry_index =
        static_cast<std::size_t>(
            std::distance(table.entries_.begin(), first_entry)
        );

    if (table.entries_.size() - source_entry_index
        < kPromotionSourcePageCount) {
        result.error = KvCacheError::PromotionNotEligible;
        return result;
    }

    std::array<PageHandle, kPromotionSourcePageCount> source_handles{};

    for (std::size_t offset = 0;
         offset < kPromotionSourcePageCount;
         ++offset) {
        MappingEntry const& entry =
            table.entries_[source_entry_index + offset];

        std::uint64_t const expected_begin =
            static_cast<std::uint64_t>(logical_token_begin)
            + offset * kMicroPageTokenCapacity;

        RuntimeSlot const* runtime = runtimeSlotLocked(entry.handle);

        if (expected_begin > std::numeric_limits<std::uint32_t>::max()
            || entry.logical_token_begin != expected_begin
            || entry.kind != PageKind::Micro
            || entry.handle.kind != PageKind::Micro
            || entry.valid_tokens != kMicroPageTokenCapacity
            || runtime == nullptr
            || runtime->state != PageState::Sealed
            || runtime->valid_tokens != kMicroPageTokenCapacity
            || runtime->ref_count != 1
            || runtime->promotion_pins != 0
            || runtime->mutable_owner != kInvalidRequestId
            || micro_pool_.validate(entry.handle)
                != PagePoolError::None) {
            result.error = KvCacheError::PromotionNotEligible;
            return result;
        }

        source_handles[offset] = entry.handle;
    }

    PageHandle target_handle = PageHandle::invalid();
    KvCacheError const allocation_error =
        allocatePromotionTargetLocked(target_handle);

    if (allocation_error != KvCacheError::None) {
        result.error = allocation_error;
        return result;
    }

    PromotionId const promotion_id = allocatePromotionIdLocked();

    if (promotion_id == kInvalidPromotionId) {
        static_cast<void>(freePageLocked(target_handle));
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }

    PromotionTransaction transaction{
        promotion_id,
        request_id,
        table.version_,
        logical_token_begin,
        source_entry_index,
        source_handles,
        target_handle,
    };

    auto const insertion = promotions_.emplace(
        promotion_id,
        std::move(transaction)
    );

    if (!insertion.second) {
        static_cast<void>(freePageLocked(target_handle));
        result.error = KvCacheError::InternalInvariantViolation;
        return result;
    }

    for (PageHandle const source_handle : source_handles) {
        RuntimeSlot* runtime = runtimeSlotLocked(source_handle);
        // 上面的资格检查和全局锁保证 runtime 有效且 Pin 仍为 0。
        ++runtime->promotion_pins;
    }

    result.error = KvCacheError::None;
    result.promotion_id = promotion_id;
    result.source_handles = source_handles;
    result.target_handle = target_handle;
    return result;
}

KvCacheError KvCacheManager::rollbackPromotionLocked(
    PromotionId promotion_id)
{
    auto const transaction_iterator = promotions_.find(promotion_id);

    if (transaction_iterator == promotions_.end()) {
        return KvCacheError::PromotionNotFound;
    }

    PromotionTransaction const& transaction = transaction_iterator->second;
    RuntimeSlot* target = runtimeSlotLocked(transaction.target_handle);

    if (target == nullptr
        || target->state != PageState::CopyTarget
        || target->valid_tokens != kExtentPageTokenCapacity
        || target->ref_count != 0
        || target->promotion_pins != 0
        || target->mutable_owner != kInvalidRequestId
        || extent_pool_.validate(transaction.target_handle)
            != PagePoolError::None) {
        return KvCacheError::InternalInvariantViolation;
    }

    for (PageHandle const source_handle : transaction.source_handles) {
        RuntimeSlot const* source = runtimeSlotLocked(source_handle);

        if (source == nullptr
            || (
                source->state != PageState::Sealed
                && source->state != PageState::Retiring
            )
            || source->valid_tokens != kMicroPageTokenCapacity
            || source->promotion_pins != 1
            || source->mutable_owner != kInvalidRequestId
            || micro_pool_.validate(source_handle)
                != PagePoolError::None) {
            return KvCacheError::InternalInvariantViolation;
        }
    }

    KvCacheError const target_release_error =
        freePageLocked(transaction.target_handle);

    if (target_release_error != KvCacheError::None) {
        return target_release_error;
    }

    for (PageHandle const source_handle : transaction.source_handles) {
        RuntimeSlot* source = runtimeSlotLocked(source_handle);
        --source->promotion_pins;

        if (source->ref_count == 0) {
            KvCacheError const release_error =
                freePageLocked(source_handle);

            if (release_error != KvCacheError::None) {
                return release_error;
            }
        }
    }

    promotions_.erase(transaction_iterator);
    return KvCacheError::None;
}

KvCacheError KvCacheManager::rollbackPromotion(
    PromotionId promotion_id)
{
    if (promotion_id == kInvalidPromotionId) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    return rollbackPromotionLocked(promotion_id);
}

KvCacheError KvCacheManager::rollbackPromotionsForRequestLocked(
    RequestId request_id)
{
    while (true) {
        auto const transaction_iterator = std::find_if(
            promotions_.begin(),
            promotions_.end(),
            [request_id](auto const& item) {
                return item.second.request_id == request_id;
            }
        );

        if (transaction_iterator == promotions_.end()) {
            return KvCacheError::None;
        }

        PromotionId const promotion_id = transaction_iterator->first;
        KvCacheError const rollback_error =
            rollbackPromotionLocked(promotion_id);

        if (rollback_error != KvCacheError::None) {
            return rollback_error;
        }
    }
}

KvCacheError KvCacheManager::commitPromotion(
    PromotionId promotion_id)
{
    if (promotion_id == kInvalidPromotionId) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto const transaction_iterator = promotions_.find(promotion_id);

    if (transaction_iterator == promotions_.end()) {
        return KvCacheError::PromotionNotFound;
    }

    PromotionTransaction const& transaction = transaction_iterator->second;

    auto rollback_with = [this, promotion_id](KvCacheError error) {
        KvCacheError const rollback_error =
            rollbackPromotionLocked(promotion_id);

        return rollback_error == KvCacheError::None
            ? error
            : rollback_error;
    };

    auto request_iterator = requests_.find(transaction.request_id);

    if (request_iterator == requests_.end()) {
        return rollback_with(KvCacheError::PromotionConflict);
    }

    BlockTable& table = request_iterator->second.table;

    if (table.version_ != transaction.prepared_table_version
        || transaction.source_entry_index > table.entries_.size()
        || table.entries_.size() - transaction.source_entry_index
            < kPromotionSourcePageCount
        || table.version_ == std::numeric_limits<std::uint64_t>::max()) {
        return rollback_with(KvCacheError::PromotionConflict);
    }

    RuntimeSlot* target = runtimeSlotLocked(transaction.target_handle);

    if (target == nullptr
        || target->state != PageState::CopyTarget
        || target->valid_tokens != kExtentPageTokenCapacity
        || target->ref_count != 0
        || target->promotion_pins != 0
        || target->inflight_readers != 0
        || target->mutable_owner != kInvalidRequestId
        || extent_pool_.validate(transaction.target_handle)
            != PagePoolError::None) {
        return rollback_with(KvCacheError::PromotionConflict);
    }

    for (std::size_t offset = 0;
         offset < kPromotionSourcePageCount;
         ++offset) {
        MappingEntry const& entry =
            table.entries_[transaction.source_entry_index + offset];
        PageHandle const source_handle = transaction.source_handles[offset];
        RuntimeSlot const* source = runtimeSlotLocked(source_handle);
        std::uint64_t const expected_begin =
            static_cast<std::uint64_t>(transaction.logical_token_begin)
            + offset * kMicroPageTokenCapacity;

        if (expected_begin > std::numeric_limits<std::uint32_t>::max()
            || entry.logical_token_begin != expected_begin
            || entry.kind != PageKind::Micro
            || entry.valid_tokens != kMicroPageTokenCapacity
            || entry.handle != source_handle
            || source == nullptr
            || source->state != PageState::Sealed
            || source->valid_tokens != kMicroPageTokenCapacity
            || source->ref_count != 1
            || source->promotion_pins != 1
            || source->mutable_owner != kInvalidRequestId
            || micro_pool_.validate(source_handle)
                != PagePoolError::None) {
            return rollback_with(KvCacheError::PromotionConflict);
        }
    }

    BlockTable candidate = table;
    MappingEntry const promoted_entry{
        transaction.logical_token_begin,
        kExtentPageTokenCapacity,
        PageKind::Extent,
        transaction.target_handle,
    };

    candidate.entries_[transaction.source_entry_index] = promoted_entry;
    candidate.entries_.erase(
        candidate.entries_.begin()
            + static_cast<std::ptrdiff_t>(
                transaction.source_entry_index + 1
            ),
        candidate.entries_.begin()
            + static_cast<std::ptrdiff_t>(
                transaction.source_entry_index
                + kPromotionSourcePageCount
            )
    );
    candidate.version_ = table.version_ + 1;

    if (!candidate.checkInvariants()) {
        return rollback_with(KvCacheError::InternalInvariantViolation);
    }

    // 从这里开始进入无异常 Commit 区：先切换唯一可见映射，再完成
    // Target/Source 生命周期转换。全程仍由同一把全局锁保护。
    table = std::move(candidate);

    target->state = PageState::Sealed;
    target->ref_count = 1;

    for (PageHandle const source_handle : transaction.source_handles) {
        RuntimeSlot* source = runtimeSlotLocked(source_handle);

        --source->ref_count;
        --source->promotion_pins;

        KvCacheError const release_error = freePageLocked(source_handle);

        if (release_error != KvCacheError::None) {
            return release_error;
        }
    }

    promotions_.erase(transaction_iterator);
    return KvCacheError::None;
}

std::optional<PromotionTransactionSnapshot>
KvCacheManager::promotionTransaction(PromotionId promotion_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto const iterator = promotions_.find(promotion_id);

    if (iterator == promotions_.end()) {
        return std::nullopt;
    }

    PromotionTransaction const& transaction = iterator->second;

    return PromotionTransactionSnapshot{
        transaction.promotion_id,
        transaction.request_id,
        transaction.prepared_table_version,
        transaction.logical_token_begin,
        transaction.source_handles,
        transaction.target_handle,
    };
}

} // namespace kimkvcache
