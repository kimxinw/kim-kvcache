#pragma once

#include "heteropage_kv/core/page_handle.h"
#include "heteropage_kv/core/page_state.h"

#include <cstdint>
#include <vector>

namespace kimkvcache {
    class KvCacheManager;

    struct MappingEntry final {
        std::uint32_t logical_token_begin{0};
        std::uint16_t valid_tokens{0};
        PageKind kind{PageKind::Micro};
        PageHandle handle{PageHandle::invalid()};

        [[nodiscard]] constexpr std::uint32_t logicalTokenEnd() const noexcept
        {
            return logical_token_begin + valid_tokens;
        }

        friend constexpr bool operator==(
            MappingEntry const& lhs,
            MappingEntry const& rhs) noexcept
        {
            return lhs.logical_token_begin == rhs.logical_token_begin
                && lhs.valid_tokens == rhs.valid_tokens
                && lhs.kind == rhs.kind
                && lhs.handle == rhs.handle;
        }

        friend constexpr bool operator!=(
            MappingEntry const& lhs,
            MappingEntry const& rhs) noexcept
        {
            return !(lhs == rhs);
        }
    };

    class BlockTable final {
    public:
        BlockTable() = default;

        [[nodiscard]] std::uint64_t version() const noexcept;

        [[nodiscard]] std::uint32_t tokenCount() const noexcept;

        [[nodiscard]] bool empty() const noexcept;

        [[nodiscard]] std::vector<MappingEntry> const& entries() const noexcept;

        [[nodiscard]] MappingEntry const* find(
            std::uint32_t logical_token) const noexcept;

        [[nodiscard]] bool checkInvariants() const;

    private:
        friend class KvCacheManager;

        std::vector<MappingEntry> entries_;
        std::uint64_t version_{0};
    };

}//namespace kimkvcache
