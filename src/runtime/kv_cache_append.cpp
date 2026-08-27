#include "heteropage_kv/runtime/kv_cache_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace kimkvcache {

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

} // namespace kimkvcache
