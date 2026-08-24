#pragma once

#include "heteropage_kv/block_table.h"
#include "heteropage_kv/page_pool.h"
#include "heteropage_kv/page_state.h"
#include "heteropage_kv/promotion.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kimkvcache {

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
    std::uint64_t promotion_count{0};
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

    // Prepare 只固定元数据和物理页生命周期，不执行数据复制。
    // 调用方应在异步 Copy 成功后调用 commitPromotion；任何失败路径
    // 都必须调用 rollbackPromotion。K4 会在这两个调用之间接入 CUDA。
    [[nodiscard]] PromotionPrepareResult preparePromotion(
        RequestId request_id,
        std::uint32_t logical_token_begin
    );

    [[nodiscard]] KvCacheError commitPromotion(
        PromotionId promotion_id
    );

    [[nodiscard]] KvCacheError rollbackPromotion(
        PromotionId promotion_id
    );

    [[nodiscard]] std::optional<BlockTable> blockTable(
        RequestId request_id) const;

    [[nodiscard]] std::optional<PageMetadataSnapshot> pageMetadata(
        PageHandle handle) const;

    [[nodiscard]] std::optional<PromotionTransactionSnapshot>
    promotionTransaction(PromotionId promotion_id) const;

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

    struct PromotionTransaction final {
        PromotionId promotion_id{kInvalidPromotionId};
        RequestId request_id{kInvalidRequestId};
        std::uint64_t prepared_table_version{0};
        std::uint32_t logical_token_begin{0};
        std::size_t source_entry_index{0};
        std::array<PageHandle, kPromotionSourcePageCount> source_handles{};
        PageHandle target_handle{PageHandle::invalid()};
    };

    [[nodiscard]] RuntimeSlot* runtimeSlotLocked(
        PageHandle handle) noexcept;

    [[nodiscard]] RuntimeSlot const* runtimeSlotLocked(
        PageHandle handle) const noexcept;

    [[nodiscard]] KvCacheError allocateStagedMicroLocked(
        std::uint16_t initial_valid_tokens,
        PageHandle& handle
    );

    [[nodiscard]] KvCacheError allocatePromotionTargetLocked(
        PageHandle& handle
    );

    void rollbackStagedPagesLocked(
        std::vector<StagedPage> const& staged_pages) noexcept;

    [[nodiscard]] KvCacheError decrementReferenceLocked(
        PageHandle handle
    );

    [[nodiscard]] KvCacheError freePageLocked(PageHandle handle);

    [[nodiscard]] PromotionId allocatePromotionIdLocked();

    [[nodiscard]] KvCacheError rollbackPromotionLocked(
        PromotionId promotion_id
    );

    [[nodiscard]] KvCacheError rollbackPromotionsForRequestLocked(
        RequestId request_id
    );

    void resetFreedRuntimeLocked(RuntimeSlot& runtime) noexcept;

    [[nodiscard]] bool checkInvariantsLocked() const;

    PagePool micro_pool_;
    PagePool extent_pool_;

    std::vector<RuntimeSlot> micro_runtime_;
    std::vector<RuntimeSlot> extent_runtime_;

    std::unordered_map<RequestId, RequestState> requests_;
    std::unordered_map<PromotionId, PromotionTransaction> promotions_;

    PromotionId next_promotion_id_{1};

    mutable std::mutex mutex_;
};

} // namespace kimkvcache
