#pragma once

#include "heteropage_kv/core/block_table.h"
#include "heteropage_kv/core/page_pool.h"
#include "heteropage_kv/core/page_state.h"
// 复用全项目统一的 RequestId / kInvalidRequestId 定义。
#include "heteropage_kv/runtime/promotion.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace kimkvcache {

struct FixedPageManagerSnapshot final {
    std::uint64_t request_count{0};
    std::uint64_t successful_allocations{0};
    std::uint64_t peak_allocated_pages{0};
    PagePoolSnapshot pool{};
};

// K6 固定页运行时对照：单一页种类、可配置 token 容量的页式 KV 映射，
// 作为 HeteroPageKV 双池 + Promotion 运行时的等价对照 Baseline。
//
// 语义与 KvCacheManager 的 Create/Append/Seal/Fork/COW/Release 子集对齐；
// 没有 Promotion，因此不提供 Lease/Promotion API。物理 token 容量完全
// 由构造参数 tokens_per_page 决定；BlockTable 中的所有页仍标记为
// PageKind::Micro。该 kind 在本模块内只作为 Pool/Handle 的命名空间；
// BlockTable 和 RuntimeSlot 的容量由 tokens_per_page 显式校验。
class FixedPageManager final {
public:
    // tokens_per_page 必须大于 0，否则抛出 std::invalid_argument。
    FixedPageManager(
        std::uint16_t tokens_per_page,
        std::uint32_t page_capacity
    );

    ~FixedPageManager() = default;

    FixedPageManager(FixedPageManager const&) = delete;
    FixedPageManager& operator=(FixedPageManager const&) = delete;

    FixedPageManager(FixedPageManager&&) = delete;
    FixedPageManager& operator=(FixedPageManager&&) = delete;

    [[nodiscard]] std::uint16_t tokensPerPage() const noexcept;

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

    [[nodiscard]] FixedPageManagerSnapshot snapshot() const;

    [[nodiscard]] bool checkInvariants() const;

private:
    struct RuntimeSlot final {
        PageState state{PageState::Free};
        std::uint32_t generation{0};
        std::uint16_t valid_tokens{0};
        std::uint32_t ref_count{0};
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

    [[nodiscard]] KvCacheError allocateStagedLocked(
        std::uint16_t initial_valid_tokens,
        PageHandle& handle
    );

    void rollbackStagedPagesLocked(
        std::vector<StagedPage> const& staged_pages) noexcept;

    void resetFreedRuntimeLocked(RuntimeSlot& runtime) noexcept;

    [[nodiscard]] KvCacheError freePageLocked(PageHandle handle);

    [[nodiscard]] KvCacheError decrementReferenceLocked(
        PageHandle handle
    );

    [[nodiscard]] bool checkInvariantsLocked() const;

    std::uint16_t tokens_per_page_;
    PagePool pool_;
    std::vector<RuntimeSlot> runtime_;

    std::unordered_map<RequestId, RequestState> requests_;

    std::uint64_t peak_allocated_pages_{0};

    mutable std::mutex mutex_;
};

} // namespace kimkvcache
