#include "heteropage_kv/runtime/kv_cache_manager.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace kimkvcache {

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
        static_cast<std::uint64_t>(page_leases_.size()),
        micro_pool_.snapshot(),
        extent_pool_.snapshot(),
    };
}

} // namespace kimkvcache
