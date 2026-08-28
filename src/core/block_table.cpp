#include "heteropage_kv/core/block_table.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>

namespace kimkvcache {
namespace {

[[nodiscard]] bool checkBlockTableInvariants(
    std::vector<MappingEntry> const& entries,
    std::uint16_t uniform_page_token_capacity)
{
    std::uint64_t expected_begin = 0;

    for (std::size_t index = 0; index < entries.size(); ++index) {
        MappingEntry const& entry = entries[index];

        if (!entry.handle.isStructurallyValid()) {
            return false;
        }

        if (entry.kind != entry.handle.kind) {
            return false;
        }

        std::uint16_t const capacity = uniform_page_token_capacity == 0
            ? pageTokenCapacity(entry.kind)
            : uniform_page_token_capacity;

        if (capacity == 0
            || entry.valid_tokens == 0
            || entry.valid_tokens > capacity) {
            return false;
        }

        if (expected_begin >
            std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }

        if (entry.logical_token_begin != expected_begin) {
            return false;
        }

        expected_begin += entry.valid_tokens;

        if (expected_begin >
            std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }

        // 同一个请求不能用同一个物理页映射两个逻辑区间。
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (entries[previous].handle == entry.handle) {
                return false;
            }
        }
    }

    return true;
}

} // namespace

std::uint64_t BlockTable::version() const noexcept
{
    return version_;
}

std::uint32_t BlockTable::tokenCount() const noexcept
{
    if (entries_.empty()) {
        return 0;
    }

    return entries_.back().logicalTokenEnd();
}

bool BlockTable::empty() const noexcept
{
    return entries_.empty();
}

std::vector<MappingEntry> const& BlockTable::entries() const noexcept
{
    return entries_;
}

MappingEntry const* BlockTable::find(
    std::uint32_t logical_token) const noexcept
{
    auto const upper = std::upper_bound(
        entries_.begin(),
        entries_.end(),
        logical_token,
        [](std::uint32_t token, MappingEntry const& entry) {
            return token < entry.logical_token_begin;
        }
    );

    if (upper == entries_.begin()) {
        return nullptr;
    }

    auto const candidate = std::prev(upper);

    if (logical_token >= candidate->logicalTokenEnd()) {
        return nullptr;
    }

    return &(*candidate);
}

bool BlockTable::checkInvariants() const
{
    return checkBlockTableInvariants(entries_, 0);
}

bool BlockTable::checkInvariants(
    std::uint16_t uniform_page_token_capacity) const
{
    return uniform_page_token_capacity != 0
        && checkBlockTableInvariants(
            entries_,
            uniform_page_token_capacity
        );
}

} // namespace kimkvcache
