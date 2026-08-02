#pragma once

#include "heteropage_kv/page_handle.h"
#include "heteropage_kv/page_types.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace kimkvcache {

struct PageAllocationResult final {
    PagePoolError error{PagePoolError::None};
    PageHandle handle{PageHandle::invalid()};

    [[nodiscard]] constexpr bool ok() const noexcept {
        return error == PagePoolError::None;
    }


    [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return ok();
    }
};
struct PagePoolSnapshot final {
    PageKind kind{PageKind::Micro};

    std::uint32_t total_slots{0};
    std::uint32_t free_slots{0};
    std::uint32_t allocated_slots{0};

    std::uint64_t successful_allocations{0};
    std::uint64_t successful_releases{0};
    std::uint64_t failed_allocations{0};
    std::uint64_t rejected_releases{0};

    [[nodiscard]] constexpr bool capacityBalanced() const noexcept {
        return total_slots == free_slots + allocated_slots;
    }
};

class PagePool final {
public:
    PagePool(PageKind kind,std::uint32_t capacity);

    ~PagePool() = default;

    PagePool(PagePool const &) = delete;
    PagePool& operator=(PagePool const & ) = delete;

    PagePool(PagePool&&) = delete;
    PagePool& operator(PagePool&&) = delete;

    [[nodiscard]] PageKind kind() const noexcept;

    [[nodiscard]] std::uint32_t capacity() const noexcept;

    // Pool 耗尽属于预期失败，通过 Result 返回，不抛异常。
    [[nodiscard]] PageAllocationResult allocate();

    // 成功返回 None，非法释放返回具体错误码。
    [[nodiscard]] PagePoolError release(PageHandle handle);

    // 只验证 Handle，不改变 Pool 状态和统计计数。
    [[nodiscard]] PagePoolError validate(PageHandle handle) const;

    [[nodiscard]] PagePoolSnapshot snapshot() const;

    // 用于测试、Shutdown 和 Debug，不应放在热路径。
    [[nodiscard]] bool checkInvariants() const;
private:
    struct SlotMetadata final {
        std::uint32_t generation {0};
        bool allocated{false};
    };

    [[nodiscard]] static std::uint32_t nextGeneration(
        std::uint32_t current
    )noexcept;

    // 调用方必须已经持有 mutex_。
    [[nodiscard]] PagePoolError validateLocked(
        PageHandle handle) const noexcept;

    PageKind kind_;
    std::uint32_t capacity_;

    std::vector<SlotMetadata> slots_;
    std::vector<std::uint32_t> free_slots_;

    std::uint32_t allocated_slots_{0};

    std::uint64_t successful_allocations_{0};
    std::uint64_t successful_releases_{0};
    std::uint64_t failed_allocations_{0};
    std::uint64_t rejected_releases_{0};

    mutable std::mutex mutex_;
};

}//namespace kimkvcache