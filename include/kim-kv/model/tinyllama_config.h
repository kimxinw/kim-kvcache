#pragma once

#include <cstdint>

namespace kimkvcache {

// Minimal Llama-family model contract used by the E2 single-request runner.
// The first production baseline is TinyLlama 1.1B Chat, while small shapes are
// intentionally accepted so the complete CUDA path can be contract-tested.
struct TinyLlamaConfig final {
    std::uint32_t hidden_size{0};
    std::uint32_t intermediate_size{0};
    std::uint32_t layer_count{0};
    std::uint32_t attention_head_count{0};
    std::uint32_t kv_head_count{0};
    std::uint32_t head_dimension{0};
    std::uint32_t vocabulary_size{0};
    std::uint32_t max_position_embeddings{0};
    std::uint32_t bos_token_id{0};
    std::uint32_t eos_token_id{0};
    float rms_norm_epsilon{0.0F};
    float rope_theta{0.0F};
    bool tied_word_embeddings{false};

    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return hidden_size != 0
            && intermediate_size != 0
            && layer_count != 0
            && attention_head_count != 0
            && kv_head_count != 0
            && head_dimension != 0
            && hidden_size == attention_head_count * head_dimension
            && attention_head_count >= kv_head_count
            && attention_head_count % kv_head_count == 0
            && head_dimension % 2 == 0
            && vocabulary_size != 0
            && max_position_embeddings != 0
            && bos_token_id < vocabulary_size
            && eos_token_id < vocabulary_size
            && rms_norm_epsilon > 0.0F
            && rope_theta > 0.0F;
    }
};

[[nodiscard]] constexpr TinyLlamaConfig tinyLlama11bChatConfig() noexcept
{
    return TinyLlamaConfig{
        2048,
        5632,
        22,
        32,
        4,
        64,
        32000,
        2048,
        1,
        2,
        1.0e-5F,
        10000.0F,
        false,
    };
}

} // namespace kimkvcache
