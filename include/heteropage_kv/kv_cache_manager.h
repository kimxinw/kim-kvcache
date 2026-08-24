#pragma once

#include "heteropage_kv/block_table.h"
#include "heteropage_kv/page_pool.h"
#include "heteropage_kv/page_state.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kimkvcache {

using RequestId = std::uint64_t;

constexpr RequestId kInvalidRequestId = 0;

struct PageMetadataSnapshot final {
    PageHandle handle{PageHandle::invalid()};
    PageState state{PageState::Free};
    std::uint16_t valid_tokens{0};
    std::uint32_t ref_count{0};
    std::uint32_t promotion_pins{0};
    std::uint32_t inflight_readers{0};
    RequestId mutable_owner{kInvalidRequestId};

    [[nodiscard]] constexpr bool shared() const noexcept
    {
        return state == PageState::Sealed && ref_count > 1;
    }
};

struct KvCacheManagerSnapshot final {
    std::uint64_t request_count{0};
    PagePoolSnapshot micro_pool{};
    PagePoolSnapshot extent_pool{};
};

class KvCacheManager final {
public:
    KvCacheManager(
        std::uint32_t micro_capacity,
        std::uint32_t extent_capacity
    );

    ~KvCacheManager() = default;

    KvCacheManager(KvCacheManager const&) = delete;
    KvCacheManager& operator=(KvCacheManager const&) = delete;

    KvCacheManager(KvCacheManager&&) = delete;
    KvCacheManager& operator=(KvCacheManager&&) = delete;

    [[nodiscard]] KvCacheError createRequest(RequestId request_id);

    [[nodiscard]] KvCacheError append(
        RequestId request_id,
        std::uint32_t token_count
    );

    [[nodiscard]] KvCacheError sealTail(RequestId request_id);

    [[nodiscard]] KvCacheError forkRequest(
        RequestId source_request_id,
        RequestId child_request_id
    );

    [[nodiscard]] KvCacheError releaseRequest(RequestId request_id);

    [[nodiscard]] std::optional<BlockTable> blockTable(
        RequestId request_id) const;

    [[nodiscard]] std::optional<PageMetadataSnapshot> pageMetadata(
        PageHandle handle) const;

    [[nodiscard]] KvCacheManagerSnapshot snapshot() const;

    [[nodiscard]] bool checkInvariants() const;

private:
    struct RuntimeSlot final {
        PageState state{PageState::Free};
        std::uint32_t generation{0};
        std::uint16_t valid_tokens{0};
        std::uint32_t ref_count{0};
        std::uint32_t promotion_pins{0};
        std::uint32_t inflight_readers{0};
        RequestId mutable_owner{kInvalidRequestId};
    };

    struct RequestState final {
        BlockTable table;
    };

    struct StagedPage final {
        PageHandle handle{PageHandle::invalid()};
        std::size_t entry_index{0};
    };

    [[nodiscard]] RuntimeSlot* runtimeSlotLocked(
        PageHandle handle) noexcept;

    [[nodiscard]] RuntimeSlot const* runtimeSlotLocked(
        PageHandle handle) const noexcept;

    [[nodiscard]] KvCacheError allocateStagedMicroLocked(
        std::uint16_t initial_valid_tokens,
        PageHandle& handle
    );

    void rollbackStagedPagesLocked(
        std::vector<StagedPage> const& staged_pages) noexcept;

    [[nodiscard]] KvCacheError decrementReferenceLocked(
        PageHandle handle
    );

    [[nodiscard]] KvCacheError freePageLocked(PageHandle handle);

    void resetFreedRuntimeLocked(RuntimeSlot& runtime) noexcept;

    [[nodiscard]] bool checkInvariantsLocked() const;

    PagePool micro_pool_;
    PagePool extent_pool_;

    std::vector<RuntimeSlot> micro_runtime_;
    std::vector<RuntimeSlot> extent_runtime_;

    std::unordered_map<RequestId, RequestState> requests_;

    mutable std::mutex mutex_;
};

} // namespace kimkvcache
