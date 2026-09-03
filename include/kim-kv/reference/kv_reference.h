#pragma once

#include "kim-kv/core/kv_layout.h"

#include <cstdint>
#include <vector>

namespace kimkvcache {

[[nodiscard]] KvScalar floatToKvScalar(float value) noexcept;

[[nodiscard]] float kvScalarToFloat(KvScalar value) noexcept;

// 连续布局 Reference Attention：
// KV   [layer][K/V][token][head][dim]
// Q/O  [layer][head][dim]
[[nodiscard]] bool referenceAttention(
    KvLayout const& layout,
    std::vector<KvScalar> const& contiguous_kv,
    std::uint32_t token_count,
    std::vector<float> const& query,
    std::vector<float>& output
);

} // namespace kimkvcache
