#include "heteropage_kv/kv_cache_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace kimkvcache {

KvCacheManager::KvCacheManager(
    std::uint32_t micro_capacity,
    std::uint32_t extent_capacity)
    : micro_pool_(PageKind::Micro, micro_capacity)
    , extent_pool_(PageKind::Extent, extent_capacity)
    , micro_runtime_(micro_capacity)
    , extent_runtime_(extent_capacity)
{
}

KvCacheError KvCacheManager::createRequest(RequestId request_id)
{
    if (request_id == kInvalidRequestId) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto const result = requests_.emplace(
        request_id,
        RequestState{}
    );

    if (!result.second) {
        return KvCacheError::RequestAlreadyExists;
    }

    return KvCacheError::None;
}

KvCacheManager::RuntimeSlot* KvCacheManager::runtimeSlotLocked(
    PageHandle handle) noexcept
{
    if (!handle.isStructurallyValid()) {
        return nullptr;
    }

    std::vector<RuntimeSlot>* runtime = nullptr;

    switch (handle.kind) {
    case PageKind::Micro:
        runtime = &micro_runtime_;
        break;
    case PageKind::Extent:
        runtime = &extent_runtime_;
        break;
    }

    if (runtime == nullptr || handle.slot >= runtime->size()) {
        return nullptr;
    }

    RuntimeSlot& slot = (*runtime)[handle.slot];

    if (slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

KvCacheManager::RuntimeSlot const* KvCacheManager::runtimeSlotLocked(
    PageHandle handle) const noexcept
{
    if (!handle.isStructurallyValid()) {
        return nullptr;
    }

    std::vector<RuntimeSlot> const* runtime = nullptr;

    switch (handle.kind) {
    case PageKind::Micro:
        runtime = &micro_runtime_;
        break;
    case PageKind::Extent:
        runtime = &extent_runtime_;
        break;
    }

    if (runtime == nullptr || handle.slot >= runtime->size()) {
        return nullptr;
    }

    RuntimeSlot const& slot = (*runtime)[handle.slot];

    if (slot.generation != handle.generation) {
        return nullptr;
    }

    return &slot;
}

KvCacheError KvCacheManager::allocateStagedMicroLocked(
    std::uint16_t initial_valid_tokens,
    PageHandle& handle)
{
    if (initial_valid_tokens >= kMicroPageTokenCapacity) {
        return KvCacheError::InvalidArgument;
    }

    PageAllocationResult const allocation = micro_pool_.allocate();

    if (!allocation.ok()) {
        if (allocation.error == PagePoolError::Exhausted) {
            return KvCacheError::ResourceExhausted;
        }

        return KvCacheError::InternalInvariantViolation;
    }

    handle = allocation.handle;

    if (handle.slot >= micro_runtime_.size()) {
        static_cast<void>(micro_pool_.release(handle));
        handle = PageHandle::invalid();
        return KvCacheError::InternalInvariantViolation;
    }

    RuntimeSlot& runtime = micro_runtime_[handle.slot];

    if (runtime.state != PageState::Free
        || runtime.ref_count != 0
        || runtime.promotion_pins != 0
        || runtime.inflight_readers != 0) {
        static_cast<void>(micro_pool_.release(handle));
        handle = PageHandle::invalid();
        return KvCacheError::InternalInvariantViolation;
    }

    runtime.state = PageState::CopyTarget;
    runtime.generation = handle.generation;
    runtime.valid_tokens = initial_valid_tokens;
    runtime.ref_count = 0;
    runtime.promotion_pins = 0;
    runtime.inflight_readers = 0;
    runtime.mutable_owner = kInvalidRequestId;

    return KvCacheError::None;
}

void KvCacheManager::resetFreedRuntimeLocked(
    RuntimeSlot& runtime) noexcept
{
    // generation 保留，用于识别当前 Free Slot 的最后一代 Handle。
    runtime.state = PageState::Free;
    runtime.valid_tokens = 0;
    runtime.ref_count = 0;
    runtime.promotion_pins = 0;
    runtime.inflight_readers = 0;
    runtime.mutable_owner = kInvalidRequestId;
}

void KvCacheManager::rollbackStagedPagesLocked(
    std::vector<StagedPage> const& staged_pages) noexcept
{
    for (auto iterator = staged_pages.rbegin();
         iterator != staged_pages.rend();
         ++iterator) {
        RuntimeSlot* runtime = runtimeSlotLocked(iterator->handle);

        if (runtime == nullptr
            || runtime->state != PageState::CopyTarget
            || runtime->ref_count != 0) {
            continue;
        }

        PagePoolError const release_error =
            micro_pool_.release(iterator->handle);

        if (release_error == PagePoolError::None) {
            resetFreedRuntimeLocked(*runtime);
        }
    }
}

KvCacheError KvCacheManager::freePageLocked(PageHandle handle)
{
    RuntimeSlot* runtime = runtimeSlotLocked(handle);

    if (runtime == nullptr || runtime->state == PageState::Free) {
        return KvCacheError::InternalInvariantViolation;
    }

    if (runtime->ref_count != 0) {
        return KvCacheError::InvalidState;
    }

    runtime->mutable_owner = kInvalidRequestId;

    if (runtime->promotion_pins != 0
        || runtime->inflight_readers != 0) {
        runtime->state = PageState::Retiring;
        return KvCacheError::None;
    }

    PagePool* pool = nullptr;

    switch (handle.kind) {
    case PageKind::Micro:
        pool = &micro_pool_;
        break;
    case PageKind::Extent:
        pool = &extent_pool_;
        break;
    }

    if (pool == nullptr
        || pool->validate(handle) != PagePoolError::None
        || pool->release(handle) != PagePoolError::None) {
        return KvCacheError::InternalInvariantViolation;
    }

    resetFreedRuntimeLocked(*runtime);

    return KvCacheError::None;
}

KvCacheError KvCacheManager::decrementReferenceLocked(
    PageHandle handle)
{
    RuntimeSlot* runtime = runtimeSlotLocked(handle);

    if (runtime == nullptr
        || runtime->state == PageState::Free
        || runtime->state == PageState::CopyTarget
        || runtime->ref_count == 0) {
        return KvCacheError::InternalInvariantViolation;
    }

    --runtime->ref_count;

    if (runtime->ref_count != 0) {
        return KvCacheError::None;
    }

    return freePageLocked(handle);
}

KvCacheError KvCacheManager::append(
    RequestId request_id,
    std::uint32_t token_count)
{
    if (request_id == kInvalidRequestId || token_count == 0) {
        return KvCacheError::InvalidArgument;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto request_iterator = requests_.find(request_id);

    if (request_iterator == requests_.end()) {
        return KvCacheError::RequestNotFound;
    }

    RequestState& request = request_iterator->second;
    std::uint32_t const old_token_count = request.table.tokenCount();

    if (token_count >
        std::numeric_limits<std::uint32_t>::max() - old_token_count) {
        return KvCacheError::InvalidArgument;
    }

    BlockTable candidate = request.table;

    std::size_t const maximum_new_pages =
        (static_cast<std::size_t>(token_count)
            + kMicroPageTokenCapacity - 1)
        / kMicroPageTokenCapacity
        + 1;

    candidate.entries_.reserve(
        candidate.entries_.size() + maximum_new_pages
    );

    std::vector<StagedPage> staged_pages;
    staged_pages.reserve(maximum_new_pages);

    constexpr std::size_t kNoActiveEntry =
        std::numeric_limits<std::size_t>::max();

    std::size_t active_entry_index = kNoActiveEntry;
    PageHandle existing_mutable = PageHandle::invalid();
    PageHandle replaced_sealed_tail = PageHandle::invalid();

    if (!candidate.entries_.empty()) {
        std::size_t const tail_index = candidate.entries_.size() - 1;
        MappingEntry& tail = candidate.entries_[tail_index];
        RuntimeSlot* tail_runtime = runtimeSlotLocked(tail.handle);

        if (tail_runtime == nullptr
            || tail_runtime->valid_tokens != tail.valid_tokens) {
            return KvCacheError::InternalInvariantViolation;
        }

        if (tail.kind == PageKind::Micro
            && tail.valid_tokens < kMicroPageTokenCapacity) {
            if (tail_runtime->state == PageState::Mutable) {
                if (tail_runtime->ref_count != 1
                    || tail_runtime->mutable_owner != request_id) {
                    return KvCacheError::InternalInvariantViolation;
                }

                existing_mutable = tail.handle;
                active_entry_index = tail_index;
            } else if (tail_runtime->state == PageState::Sealed) {
                PageHandle copy_target = PageHandle::invalid();

                KvCacheError const allocation_error =
                    allocateStagedMicroLocked(
                        tail.valid_tokens,
                        copy_target
                    );

                if (allocation_error != KvCacheError::None) {
                    return allocation_error;
                }

                staged_pages.push_back(
                    StagedPage{copy_target, tail_index}
                );

                replaced_sealed_tail = tail.handle;
                tail.kind = PageKind::Micro;
                tail.handle = copy_target;
                active_entry_index = tail_index;
            } else {
                return KvCacheError::InvalidState;
            }
        } else if (tail_runtime->state != PageState::Sealed) {
            return KvCacheError::InternalInvariantViolation;
        }
    }

    std::uint32_t remaining_tokens = token_count;
    std::uint32_t logical_end = old_token_count;

    while (remaining_tokens != 0) {
        if (active_entry_index == kNoActiveEntry) {
            PageHandle new_page = PageHandle::invalid();

            KvCacheError const allocation_error =
                allocateStagedMicroLocked(0, new_page);

            if (allocation_error != KvCacheError::None) {
                rollbackStagedPagesLocked(staged_pages);
                return allocation_error;
            }

            std::size_t const new_index = candidate.entries_.size();

            candidate.entries_.push_back(
                MappingEntry{
                    logical_end,
                    0,
                    PageKind::Micro,
                    new_page,
                }
            );

            staged_pages.push_back(
                StagedPage{new_page, new_index}
            );

            active_entry_index = new_index;
        }

        MappingEntry& active = candidate.entries_[active_entry_index];

        std::uint32_t const available =
            kMicroPageTokenCapacity - active.valid_tokens;

        std::uint32_t const appended =
            std::min(remaining_tokens, available);

        active.valid_tokens = static_cast<std::uint16_t>(
            active.valid_tokens + appended
        );

        logical_end += appended;
        remaining_tokens -= appended;

        if (active.valid_tokens == kMicroPageTokenCapacity) {
            active_entry_index = kNoActiveEntry;
        }
    }

    candidate.version_ = request.table.version_ + 1;

    if (!candidate.checkInvariants()) {
        rollbackStagedPagesLocked(staged_pages);
        return KvCacheError::InternalInvariantViolation;
    }

    if (existing_mutable.isStructurallyValid()) {
        RuntimeSlot* runtime = runtimeSlotLocked(existing_mutable);

        if (runtime == nullptr
            || runtime->state != PageState::Mutable
            || runtime->ref_count != 1
            || runtime->mutable_owner != request_id) {
            rollbackStagedPagesLocked(staged_pages);
            return KvCacheError::InternalInvariantViolation;
        }
    }

    if (replaced_sealed_tail.isStructurallyValid()) {
        RuntimeSlot* runtime = runtimeSlotLocked(replaced_sealed_tail);

        if (runtime == nullptr
            || runtime->state != PageState::Sealed
            || runtime->ref_count == 0) {
            rollbackStagedPagesLocked(staged_pages);
            return KvCacheError::InternalInvariantViolation;
        }
    }

    for (StagedPage const& staged : staged_pages) {
        RuntimeSlot* runtime = runtimeSlotLocked(staged.handle);

        if (runtime == nullptr
            || runtime->state != PageState::CopyTarget
            || runtime->ref_count != 0
            || staged.entry_index >= candidate.entries_.size()
            || candidate.entries_[staged.entry_index].handle
                != staged.handle) {
            rollbackStagedPagesLocked(staged_pages);
            return KvCacheError::InternalInvariantViolation;
        }
    }

    // 从这里开始进入无异常 Commit 区。
    if (replaced_sealed_tail.isStructurallyValid()) {
        KvCacheError const decrement_error =
            decrementReferenceLocked(replaced_sealed_tail);

        if (decrement_error != KvCacheError::None) {
            rollbackStagedPagesLocked(staged_pages);
            return decrement_error;
        }
    }

    if (existing_mutable.isStructurallyValid()) {
        RuntimeSlot* runtime = runtimeSlotLocked(existing_mutable);

        for (MappingEntry const& entry : candidate.entries_) {
            if (entry.handle != existing_mutable) {
                continue;
            }

            runtime->valid_tokens = entry.valid_tokens;

            if (entry.valid_tokens == kMicroPageTokenCapacity) {
                runtime->state = PageState::Sealed;
                runtime->mutable_owner = kInvalidRequestId;
            }

            break;
        }
    }

    for (StagedPage const& staged : staged_pages) {
        RuntimeSlot* runtime = runtimeSlotLocked(staged.handle);
        MappingEntry const& entry =
            candidate.entries_[staged.entry_index];

        runtime->valid_tokens = entry.valid_tokens;
        runtime->ref_count = 1;

        if (entry.valid_tokens == kMicroPageTokenCapacity) {
            runtime->state = PageState::Sealed;
            runtime->mutable_owner = kInvalidRequestId;
        } else {
            runtime->state = PageState::Mutable;
            runtime->mutable_owner = request_id;
        }
    }

    request.table = std::move(candidate);

    return KvCacheError::None;
}

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

std::optional<BlockTable> KvCacheManager::blockTable(
    RequestId request_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto const iterator = requests_.find(request_id);

    if (iterator == requests_.end()) {
        return std::nullopt;
    }

    return iterator->second.table;
}

std::optional<PageMetadataSnapshot> KvCacheManager::pageMetadata(
    PageHandle handle) const
{
    std::lock_guard<std::mutex> lock(mutex_);

    RuntimeSlot const* runtime = runtimeSlotLocked(handle);

    if (runtime == nullptr) {
        return std::nullopt;
    }

    return PageMetadataSnapshot{
        handle,
        runtime->state,
        runtime->valid_tokens,
        runtime->ref_count,
        runtime->promotion_pins,
        runtime->inflight_readers,
        runtime->mutable_owner,
    };
}

KvCacheManagerSnapshot KvCacheManager::snapshot() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    return KvCacheManagerSnapshot{
        static_cast<std::uint64_t>(requests_.size()),
        static_cast<std::uint64_t>(promotions_.size()),
        micro_pool_.snapshot(),
        extent_pool_.snapshot(),
    };
}

bool KvCacheManager::checkInvariants() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return checkInvariantsLocked();
}

bool KvCacheManager::checkInvariantsLocked() const
{
    if (!micro_pool_.checkInvariants()
        || !extent_pool_.checkInvariants()) {
        return false;
    }

    std::vector<std::uint32_t> expected_micro_refs(
        micro_runtime_.size(),
        0
    );

    std::vector<std::uint32_t> expected_extent_refs(
        extent_runtime_.size(),
        0
    );

    std::vector<std::uint32_t> expected_micro_pins(
        micro_runtime_.size(),
        0
    );

    std::vector<std::uint32_t> expected_extent_pins(
        extent_runtime_.size(),
        0
    );

    std::vector<std::uint8_t> expected_micro_targets(
        micro_runtime_.size(),
        0
    );

    std::vector<std::uint8_t> expected_extent_targets(
        extent_runtime_.size(),
        0
    );

    for (auto const& request_pair : requests_) {
        RequestId const request_id = request_pair.first;
        BlockTable const& table = request_pair.second.table;

        if (request_id == kInvalidRequestId
            || !table.checkInvariants()) {
            return false;
        }

        for (std::size_t index = 0;
             index < table.entries_.size();
             ++index) {
            MappingEntry const& entry = table.entries_[index];
            RuntimeSlot const* runtime =
                runtimeSlotLocked(entry.handle);

            if (runtime == nullptr
                || runtime->state == PageState::Free
                || runtime->state == PageState::CopyTarget
                || runtime->state == PageState::Retiring
                || runtime->valid_tokens != entry.valid_tokens
                || runtime->ref_count == 0) {
                return false;
            }

            if (runtime->state == PageState::Mutable) {
                bool const is_tail =
                    index + 1 == table.entries_.size();

                if (!is_tail
                    || entry.kind != PageKind::Micro
                    || runtime->ref_count != 1
                    || runtime->mutable_owner != request_id
                    || runtime->valid_tokens == 0
                    || runtime->valid_tokens >=
                        kMicroPageTokenCapacity) {
                    return false;
                }
            } else if (runtime->state == PageState::Sealed) {
                if (runtime->mutable_owner != kInvalidRequestId) {
                    return false;
                }
            } else {
                return false;
            }

            std::vector<std::uint32_t>* expected_refs = nullptr;

            switch (entry.kind) {
            case PageKind::Micro:
                expected_refs = &expected_micro_refs;
                break;
            case PageKind::Extent:
                expected_refs = &expected_extent_refs;
                break;
            }

            if (expected_refs == nullptr
                || entry.handle.slot >= expected_refs->size()
                || (*expected_refs)[entry.handle.slot] ==
                    std::numeric_limits<std::uint32_t>::max()) {
                return false;
            }

            ++(*expected_refs)[entry.handle.slot];
        }
    }

    for (auto const& promotion_pair : promotions_) {
        PromotionId const promotion_id = promotion_pair.first;
        PromotionTransaction const& transaction = promotion_pair.second;

        if (promotion_id == kInvalidPromotionId
            || transaction.promotion_id != promotion_id
            || transaction.request_id == kInvalidRequestId
            || transaction.target_handle.kind != PageKind::Extent) {
            return false;
        }

        auto const request_iterator = requests_.find(transaction.request_id);

        if (request_iterator == requests_.end()) {
            return false;
        }

        BlockTable const& table = request_iterator->second.table;

        if (transaction.prepared_table_version > table.version_) {
            return false;
        }

        RuntimeSlot const* target =
            runtimeSlotLocked(transaction.target_handle);

        if (target == nullptr
            || target->state != PageState::CopyTarget
            || target->valid_tokens != kExtentPageTokenCapacity
            || target->ref_count != 0
            || target->promotion_pins != 0
            || target->inflight_readers != 0
            || target->mutable_owner != kInvalidRequestId
            || transaction.target_handle.slot >=
                expected_extent_targets.size()
            || expected_extent_targets[
                transaction.target_handle.slot] != 0) {
            return false;
        }

        expected_extent_targets[transaction.target_handle.slot] = 1;

        auto const first_source = std::find_if(
            table.entries_.begin(),
            table.entries_.end(),
            [&transaction](MappingEntry const& entry) {
                return entry.handle == transaction.source_handles[0];
            }
        );

        if (first_source == table.entries_.end()) {
            return false;
        }

        std::size_t const current_source_index =
            static_cast<std::size_t>(
                std::distance(table.entries_.begin(), first_source)
            );

        if (table.entries_.size() - current_source_index
            < kPromotionSourcePageCount) {
            return false;
        }

        for (std::size_t offset = 0;
             offset < kPromotionSourcePageCount;
             ++offset) {
            PageHandle const source_handle =
                transaction.source_handles[offset];
            MappingEntry const& entry =
                table.entries_[current_source_index + offset];
            RuntimeSlot const* source = runtimeSlotLocked(source_handle);
            std::uint64_t const expected_begin =
                static_cast<std::uint64_t>(
                    transaction.logical_token_begin
                ) + offset * kMicroPageTokenCapacity;

            if (expected_begin >
                    std::numeric_limits<std::uint32_t>::max()
                || source_handle.kind != PageKind::Micro
                || entry.handle != source_handle
                || entry.kind != PageKind::Micro
                || entry.logical_token_begin != expected_begin
                || entry.valid_tokens != kMicroPageTokenCapacity
                || source == nullptr
                || source->state != PageState::Sealed
                || source->valid_tokens != kMicroPageTokenCapacity
                || source->ref_count == 0
                || source->promotion_pins == 0
                || source->mutable_owner != kInvalidRequestId
                || source_handle.slot >= expected_micro_pins.size()
                || expected_micro_pins[source_handle.slot] != 0) {
                return false;
            }

            expected_micro_pins[source_handle.slot] = 1;
        }
    }

    auto const validate_pool = [](
        PageKind kind,
        PagePool const& pool,
        std::vector<RuntimeSlot> const& runtime_slots,
        std::vector<std::uint32_t> const& expected_refs,
        std::vector<std::uint32_t> const& expected_pins,
        std::vector<std::uint8_t> const& expected_targets
    ) {
        if (runtime_slots.size() != expected_refs.size()
            || runtime_slots.size() != expected_pins.size()
            || runtime_slots.size() != expected_targets.size()) {
            return false;
        }

        std::uint32_t observed_allocated = 0;

        for (std::size_t index = 0;
             index < runtime_slots.size();
             ++index) {
            RuntimeSlot const& runtime = runtime_slots[index];

            if (runtime.state == PageState::Free) {
                if (runtime.valid_tokens != 0
                    || runtime.ref_count != 0
                    || runtime.promotion_pins != 0
                    || runtime.inflight_readers != 0
                    || runtime.mutable_owner != kInvalidRequestId
                    || expected_refs[index] != 0
                    || expected_pins[index] != 0
                    || expected_targets[index] != 0) {
                    return false;
                }

                continue;
            }

            ++observed_allocated;

            if (runtime.generation ==
                PageHandle::kInvalidGeneration) {
                return false;
            }

            PageHandle const handle{
                kind,
                static_cast<std::uint32_t>(index),
                runtime.generation,
            };

            if (pool.validate(handle) != PagePoolError::None
                || runtime.ref_count != expected_refs[index]
                || runtime.promotion_pins != expected_pins[index]) {
                return false;
            }

            std::uint16_t const capacity = pageTokenCapacity(kind);
            bool const is_expected_target =
                expected_targets[index] != 0;

            switch (runtime.state) {
            case PageState::Free:
                return false;

            case PageState::Mutable:
                if (kind != PageKind::Micro
                    || runtime.ref_count != 1
                    || runtime.valid_tokens == 0
                    || runtime.valid_tokens >= capacity
                    || runtime.mutable_owner == kInvalidRequestId
                    || runtime.promotion_pins != 0
                    || is_expected_target) {
                    return false;
                }
                break;

            case PageState::Sealed:
                if (runtime.ref_count == 0
                    || runtime.valid_tokens == 0
                    || runtime.valid_tokens > capacity
                    || runtime.mutable_owner != kInvalidRequestId
                    || is_expected_target) {
                    return false;
                }
                break;

            case PageState::CopyTarget:
                if (kind != PageKind::Extent
                    || !is_expected_target
                    || runtime.ref_count != 0
                    || runtime.valid_tokens != capacity
                    || runtime.promotion_pins != 0
                    || runtime.inflight_readers != 0
                    || runtime.mutable_owner != kInvalidRequestId) {
                    return false;
                }
                break;

            case PageState::Retiring:
                if (runtime.ref_count != 0
                    || runtime.mutable_owner != kInvalidRequestId
                    || is_expected_target
                    || (
                        runtime.promotion_pins == 0
                        && runtime.inflight_readers == 0
                    )) {
                    return false;
                }
                break;
            }
        }

        PagePoolSnapshot const pool_snapshot = pool.snapshot();

        return pool_snapshot.allocated_slots == observed_allocated
            && pool_snapshot.capacityBalanced();
    };

    return validate_pool(
               PageKind::Micro,
               micro_pool_,
               micro_runtime_,
               expected_micro_refs,
               expected_micro_pins,
               expected_micro_targets
           )
        && validate_pool(
               PageKind::Extent,
               extent_pool_,
               extent_runtime_,
               expected_extent_refs,
               expected_extent_pins,
               expected_extent_targets
           );
}

} // namespace kimkvcache
