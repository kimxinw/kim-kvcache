#pragma once

#include "heteropage_kv/core/block_table.h"
#include "heteropage_kv/core/page_state.h"

#include <cstdint>
#include <vector>

namespace kimkvcache {

using PageLeaseId = std::uint64_t;

constexpr PageLeaseId kInvalidPageLeaseId = 0;

// PageLease 是 Host 元数据与异步设备访问之间的生命周期屏障。
// 租约存在期间，所引用物理页即使失去全部 BlockTable 引用也只会进入
// RETIRING，直到 releasePageLease() 后才可能返回 PagePool。
struct PageLeaseAcquireResult final {
    KvCacheError error{KvCacheError::None};
    PageLeaseId lease_id{kInvalidPageLeaseId};
    BlockTable table{};
    std::vector<PageHandle> handles{};

    [[nodiscard]] bool ok() const noexcept
    {
        return error == KvCacheError::None;
    }

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return ok();
    }
};

} // namespace kimkvcache
