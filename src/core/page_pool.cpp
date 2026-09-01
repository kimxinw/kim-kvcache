#include "kim-kv/core/page_pool.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kimkvcache {
PagePool::PagePool(PageKind kind, std::uint32_t capacity)
      : kind_(kind)
      , capacity_(capacity)
  {
    if (!isKnownPageKind(kind_)) {
        throw std::invalid_argument(
            "PagePool kind must be Micro or Extent"
        );
    }

    if (capacity_ == 0) {
        throw std::invalid_argument(
            "PagePool capacity must be greater than zero"
        );
    }

    slots_.resize(capacity_);
    free_slots_.reserve(capacity_);

    // 逆序写入，之后通过 pop_back() 按 0、1、2……顺序分配。
    for (std::uint32_t slot = capacity_; slot > 0; --slot) {
        free_slots_.push_back(slot - 1);
    }
}

PageKind PagePool::kind() const noexcept
{
    return kind_;
}

std::uint32_t PagePool::capacity() const noexcept
{
    return capacity_;
}

std::uint32_t PagePool::nextGeneration(
      std::uint32_t current) noexcept
{
    ++current;

    // unsigned 溢出是定义良好的。Generation 0 保留为无效值。
    if (current == PageHandle::kInvalidGeneration) {
        current = 1;
    }

    return current;
}

PageAllocationResult PagePool::allocate(){
    std::lock_guard<std::mutex> lock(mutex_);

    if (free_slots_.empty()) {
        ++failed_allocations_;

        return {
            PagePoolError::Exhausted,
            PageHandle::invalid(),
        };
    }

    std::uint32_t const slot = free_slots_.back();

    // 以下情况只能由 PagePool 内部状态损坏导致。
    if (slot >= slots_.size()) {
        throw std::logic_error(
            "PagePool invariant violation: free slot is out of range"
        );
    }

    SlotMetadata& metadata = slots_[slot];

    if (metadata.allocated) {
        throw std::logic_error(
            "PagePool invariant violation: free list contains allocated slot"
        );
    }

    if (allocated_slots_ >= capacity_) {
        throw std::logic_error(
            "PagePool invariant violation: allocated count exceeds capacity"
        );
    }

    free_slots_.pop_back();

    metadata.generation = nextGeneration(metadata.generation);
    metadata.allocated = true;

    ++allocated_slots_;
    ++successful_allocations_;

    return {
        PagePoolError::None,
        PageHandle{
            kind_,
            slot,
            metadata.generation,
        },
    };
}

PagePoolError PagePool::release(PageHandle handle){
    std::lock_guard<std::mutex> lock(mutex_);

    PagePoolError const validation = validateLocked(handle);

    if(validation != PagePoolError::None){
        ++rejected_releases_;
        return validation;
    }

    if (allocated_slots_ == 0) {
        throw std::logic_error(
            "PagePool invariant violation: allocated count is already zero"
        );
    }

    if (free_slots_.size() >= slots_.size()) {
        throw std::logic_error(
            "PagePool invariant violation: free list exceeds capacity"
        );
    }

    SlotMetadata& metadata = slots_[handle.slot];

    // reserve(capacity_) 已经在构造阶段执行，因此这里不会扩容。
    // 先写入 Free List，可保证 push_back 异常时元数据仍保持 allocated。
    free_slots_.push_back(handle.slot);

    metadata.allocated = false;

    --allocated_slots_;
    ++successful_releases_;

    return PagePoolError::None;
}

PagePoolError PagePool::validate(PageHandle handle) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return validateLocked(handle);
}

PagePoolError PagePool::validateLocked(PageHandle handle) const noexcept
{
    if (!handle.isStructurallyValid()) {
        return PagePoolError::InvalidHandle;
    }

    if (handle.kind != kind_) {
        return PagePoolError::WrongKind;
    }

    if (handle.slot >= slots_.size()) {
        return PagePoolError::OutOfRange;
    }

    SlotMetadata const& metadata = slots_[handle.slot];

    // 先检查 Generation：
    // Slot 被重新分配后，旧 Handle 应报告 StaleGeneration，
    // 而不是 AlreadyFree。
    if (handle.generation != metadata.generation) {
        return PagePoolError::StaleGeneration;
    }

    if (!metadata.allocated) {
        return PagePoolError::AlreadyFree;
    }

    return PagePoolError::None;
}

PagePoolSnapshot PagePool::snapshot() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    return PagePoolSnapshot{
        kind_,
        capacity_,
        static_cast<std::uint32_t>(free_slots_.size()),
        allocated_slots_,
        successful_allocations_,
        successful_releases_,
        failed_allocations_,
        rejected_releases_,
    };
}

bool PagePool::checkInvariants() const
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (slots_.size() != capacity_) {
        return false;
    }

    if (free_slots_.size() + allocated_slots_ != capacity_) {
        return false;
    }

    std::vector<std::uint8_t> free_seen(slots_.size(), 0);

    for (std::uint32_t const slot : free_slots_) {
        if (slot >= slots_.size()) {
            return false;
        }

        if (free_seen[slot] != 0) {
            return false;
        }

        if (slots_[slot].allocated) {
            return false;
        }

        free_seen[slot] = 1;
    }

    std::uint32_t observed_allocated = 0;

    for (std::uint32_t slot = 0; slot < capacity_; ++slot) {
        SlotMetadata const& metadata = slots_[slot];

        if (metadata.allocated) {
            if (metadata.generation ==
                PageHandle::kInvalidGeneration) {
                return false;
            }

            if (free_seen[slot] != 0) {
                return false;
            }

            ++observed_allocated;
        } else {
            // 每一个未分配的 Slot 都必须且只能在 Free List 中出现一次。
            if (free_seen[slot] == 0) {
                return false;
            }
        }
    }

    if (observed_allocated != allocated_slots_) {
        return false;
    }

    if (successful_releases_ > successful_allocations_) {
        return false;
    }

    if (successful_allocations_ - successful_releases_ !=
        allocated_slots_) {
        return false;
    }

    return true;
}

}//namespace kimkvcache
