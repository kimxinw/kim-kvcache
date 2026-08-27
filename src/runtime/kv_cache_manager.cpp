#include "heteropage_kv/runtime/kv_cache_manager.h"

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

} // namespace kimkvcache
