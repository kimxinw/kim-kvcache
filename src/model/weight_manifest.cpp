#include "kim-kv/model/weight_manifest.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace kimkvcache {
namespace {

constexpr std::uint32_t kManifestVersion = 1;
constexpr std::size_t kSha256HexLength = 64;

[[nodiscard]] WeightManifestStatus failure(
    WeightManifestError error,
    std::string detail)
{
    return WeightManifestStatus{error, std::move(detail)};
}

[[nodiscard]] bool parseUnsigned(
    std::string_view text,
    std::uint64_t& value) noexcept
{
    if (text.empty()) {
        return false;
    }
    auto const parsed = std::from_chars(
        text.data(), text.data() + text.size(), value
    );
    return parsed.ec == std::errc{}
        && parsed.ptr == text.data() + text.size();
}

[[nodiscard]] bool parseFloat(std::string const& text, float& value)
{
    try {
        std::size_t parsed = 0;
        value = std::stof(text, &parsed);
        return parsed == text.size();
    } catch (...) {
        return false;
    }
}

[[nodiscard]] bool validSha256(std::string const& value) noexcept
{
    return value.size() == kSha256HexLength
        && std::all_of(value.begin(), value.end(), [](char character) {
            return (character >= '0' && character <= '9')
                || (character >= 'a' && character <= 'f');
        });
}

[[nodiscard]] std::vector<std::string> split(
    std::string const& value,
    char delimiter)
{
    std::vector<std::string> parts;
    std::size_t begin = 0;
    while (begin <= value.size()) {
        std::size_t const end = value.find(delimiter, begin);
        parts.emplace_back(value.substr(
            begin,
            end == std::string::npos ? std::string::npos : end - begin
        ));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return parts;
}

[[nodiscard]] bool checkedElements(
    std::vector<std::uint32_t> const& shape,
    std::uint64_t& elements) noexcept
{
    if (shape.empty() || shape.size() > 2) {
        return false;
    }
    elements = 1;
    for (std::uint32_t dimension : shape) {
        if (dimension == 0
            || elements > std::numeric_limits<std::uint64_t>::max()
                / dimension) {
            return false;
        }
        elements *= dimension;
    }
    return true;
}

using ExpectedShapes =
    std::unordered_map<std::string, std::vector<std::uint32_t>>;

[[nodiscard]] ExpectedShapes expectedShapes(TinyLlamaConfig const& config)
{
    ExpectedShapes expected;
    expected.emplace(
        "model.embed_tokens.weight",
        std::vector<std::uint32_t>{
            config.vocabulary_size, config.hidden_size,
        }
    );
    expected.emplace(
        "model.norm.weight",
        std::vector<std::uint32_t>{config.hidden_size}
    );
    expected.emplace(
        "lm_head.weight",
        std::vector<std::uint32_t>{
            config.vocabulary_size, config.hidden_size,
        }
    );

    std::uint32_t const kv_size =
        config.kv_head_count * config.head_dimension;
    for (std::uint32_t layer = 0; layer < config.layer_count; ++layer) {
        std::string const prefix = "model.layers."
            + std::to_string(layer) + ".";
        expected.emplace(
            prefix + "input_layernorm.weight",
            std::vector<std::uint32_t>{config.hidden_size}
        );
        expected.emplace(
            prefix + "post_attention_layernorm.weight",
            std::vector<std::uint32_t>{config.hidden_size}
        );
        expected.emplace(
            prefix + "self_attn.q_proj.weight",
            std::vector<std::uint32_t>{
                config.hidden_size, config.hidden_size,
            }
        );
        expected.emplace(
            prefix + "self_attn.k_proj.weight",
            std::vector<std::uint32_t>{kv_size, config.hidden_size}
        );
        expected.emplace(
            prefix + "self_attn.v_proj.weight",
            std::vector<std::uint32_t>{kv_size, config.hidden_size}
        );
        expected.emplace(
            prefix + "self_attn.o_proj.weight",
            std::vector<std::uint32_t>{
                config.hidden_size, config.hidden_size,
            }
        );
        expected.emplace(
            prefix + "mlp.gate_proj.weight",
            std::vector<std::uint32_t>{
                config.intermediate_size, config.hidden_size,
            }
        );
        expected.emplace(
            prefix + "mlp.up_proj.weight",
            std::vector<std::uint32_t>{
                config.intermediate_size, config.hidden_size,
            }
        );
        expected.emplace(
            prefix + "mlp.down_proj.weight",
            std::vector<std::uint32_t>{
                config.hidden_size, config.intermediate_size,
            }
        );
    }
    return expected;
}

// Compact SHA-256 implementation used only during model initialization.
class Sha256 final {
public:
    void update(std::uint8_t const* data, std::size_t size) noexcept
    {
        total_bytes_ += size;
        while (size != 0) {
            std::size_t const copied = std::min(
                size, block_.size() - block_size_
            );
            std::memcpy(block_.data() + block_size_, data, copied);
            block_size_ += copied;
            data += copied;
            size -= copied;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    [[nodiscard]] std::string finish() noexcept
    {
        std::uint64_t const bit_count = total_bytes_ * 8;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56) {
            std::fill(block_.begin() + block_size_, block_.end(), 0);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + block_size_, block_.begin() + 56, 0);
        for (std::size_t index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<std::uint8_t>(
                bit_count >> (index * 8)
            );
        }
        transform(block_.data());

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (std::uint32_t word : state_) {
            output << std::setw(8) << word;
        }
        return output.str();
    }

private:
    [[nodiscard]] static constexpr std::uint32_t rotateRight(
        std::uint32_t value,
        std::uint32_t count) noexcept
    {
        return (value >> count) | (value << (32 - count));
    }

    void transform(std::uint8_t const* data) noexcept
    {
        static constexpr std::array<std::uint32_t, 64> constants{
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        };
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            words[index] = (static_cast<std::uint32_t>(data[index * 4]) << 24)
                | (static_cast<std::uint32_t>(data[index * 4 + 1]) << 16)
                | (static_cast<std::uint32_t>(data[index * 4 + 2]) << 8)
                | static_cast<std::uint32_t>(data[index * 4 + 3]);
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            std::uint32_t const s0 = rotateRight(words[index - 15], 7)
                ^ rotateRight(words[index - 15], 18)
                ^ (words[index - 15] >> 3);
            std::uint32_t const s1 = rotateRight(words[index - 2], 17)
                ^ rotateRight(words[index - 2], 19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0
                + words[index - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            std::uint32_t const sigma1 = rotateRight(e, 6)
                ^ rotateRight(e, 11) ^ rotateRight(e, 25);
            std::uint32_t const choice = (e & f) ^ ((~e) & g);
            std::uint32_t const temporary1 = h + sigma1 + choice
                + constants[index] + words[index];
            std::uint32_t const sigma0 = rotateRight(a, 2)
                ^ rotateRight(a, 13) ^ rotateRight(a, 22);
            std::uint32_t const majority = (a & b) ^ (a & c) ^ (b & c);
            std::uint32_t const temporary2 = sigma0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };
    std::array<std::uint8_t, 64> block_{};
    std::size_t block_size_{0};
    std::uint64_t total_bytes_{0};
};

[[nodiscard]] bool setRequiredString(
    std::unordered_map<std::string, std::string> const& values,
    char const* key,
    std::string& target)
{
    auto const found = values.find(key);
    if (found == values.end() || found->second.empty()) {
        return false;
    }
    target = found->second;
    return true;
}

[[nodiscard]] bool setRequiredUnsigned(
    std::unordered_map<std::string, std::string> const& values,
    char const* key,
    std::uint32_t& target)
{
    auto const found = values.find(key);
    std::uint64_t value = 0;
    if (found == values.end() || !parseUnsigned(found->second, value)
        || value > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    target = static_cast<std::uint32_t>(value);
    return true;
}

} // namespace

WeightTensorDescriptor const* WeightManifest::find(
    std::string_view name) const noexcept
{
    auto const found = std::find_if(
        tensors.begin(), tensors.end(), [name](auto const& tensor) {
            return tensor.name == name;
        }
    );
    return found == tensors.end() ? nullptr : &*found;
}

WeightManifestStatus validateWeightManifest(
    WeightManifest const& manifest)
{
    if (manifest.version != kManifestVersion) {
        return failure(
            WeightManifestError::UnsupportedVersion,
            "manifest version must be 1"
        );
    }
    if (manifest.data_type != "fp16") {
        return failure(
            WeightManifestError::UnsupportedDataType,
            "only fp16 archives are supported"
        );
    }
    if (!manifest.config.valid()) {
        return failure(
            WeightManifestError::InvalidConfig,
            "model dimensions are inconsistent"
        );
    }
    if (manifest.checkpoint.empty()
        || manifest.checkpoint_revision.empty()
        || manifest.tokenizer_revision.empty()
        || !validSha256(manifest.config_sha256)
        || manifest.data_file.empty()
        || manifest.data_bytes == 0) {
        return failure(
            WeightManifestError::InvalidConfig,
            "model identity or data archive metadata is incomplete"
        );
    }

    ExpectedShapes expected = expectedShapes(manifest.config);
    std::unordered_set<std::string> names;
    struct Range final {
        std::uint64_t begin;
        std::uint64_t end;
        std::string name;
    };
    std::vector<Range> ranges;
    ranges.reserve(manifest.tensors.size());
    for (WeightTensorDescriptor const& tensor : manifest.tensors) {
        if (!names.emplace(tensor.name).second) {
            return failure(
                WeightManifestError::DuplicateTensor,
                "duplicate tensor: " + tensor.name
            );
        }
        auto const wanted = expected.find(tensor.name);
        if (wanted == expected.end()) {
            return failure(
                WeightManifestError::UnexpectedTensor,
                "unexpected tensor: " + tensor.name
            );
        }
        if (tensor.shape != wanted->second || !validSha256(tensor.sha256)) {
            return failure(
                WeightManifestError::InvalidTensor,
                "shape or checksum metadata mismatch: " + tensor.name
            );
        }
        std::uint64_t elements = 0;
        if (!checkedElements(tensor.shape, elements)
            || tensor.offset % alignof(std::uint16_t) != 0
            || elements > std::numeric_limits<std::uint64_t>::max() / 2
            || tensor.byte_size != elements * 2
            || tensor.offset > manifest.data_bytes
            || tensor.byte_size > manifest.data_bytes - tensor.offset) {
            return failure(
                WeightManifestError::InvalidTensor,
                "invalid byte range: " + tensor.name
            );
        }
        ranges.push_back(Range{
            tensor.offset, tensor.offset + tensor.byte_size, tensor.name,
        });
        expected.erase(wanted);
    }
    if (!expected.empty()) {
        return failure(
            WeightManifestError::MissingTensor,
            "missing tensor: " + expected.begin()->first
        );
    }
    std::sort(ranges.begin(), ranges.end(), [](auto const& left, auto const& right) {
        return left.begin < right.begin;
    });
    for (std::size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index].begin < ranges[index - 1].end) {
            return failure(
                WeightManifestError::InvalidTensor,
                "overlapping tensor ranges: " + ranges[index].name
            );
        }
    }
    return {};
}

WeightManifestLoadResult loadWeightManifest(
    std::string const& manifest_path)
{
    WeightManifestLoadResult result;
    std::ifstream input(manifest_path);
    if (!input) {
        result.status = failure(
            WeightManifestError::FileOpenFailed, manifest_path
        );
        return result;
    }

    std::unordered_map<std::string, std::string> values;
    std::vector<std::string> tensor_lines;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.empty() || line.front() == '#') {
            continue;
        }
        std::size_t const equal = line.find('=');
        if (equal == std::string::npos || equal == 0) {
            result.status = failure(
                WeightManifestError::ParseFailed,
                "invalid line " + std::to_string(line_number)
            );
            return result;
        }
        std::string const key = line.substr(0, equal);
        std::string const value = line.substr(equal + 1);
        if (key == "tensor") {
            tensor_lines.push_back(value);
        } else if (!values.emplace(key, value).second) {
            result.status = failure(
                WeightManifestError::ParseFailed,
                "duplicate key: " + key
            );
            return result;
        }
    }

    WeightManifest& manifest = result.manifest;
    std::uint32_t tensor_count = 0;
    bool fields_ok = setRequiredUnsigned(values, "version", manifest.version)
        && setRequiredString(values, "checkpoint", manifest.checkpoint)
        && setRequiredString(
            values, "checkpoint_revision", manifest.checkpoint_revision)
        && setRequiredString(
            values, "tokenizer_revision", manifest.tokenizer_revision)
        && setRequiredString(
            values, "config_sha256", manifest.config_sha256)
        && setRequiredString(values, "dtype", manifest.data_type)
        && setRequiredString(values, "data_file", manifest.data_file)
        && setRequiredUnsigned(values, "hidden_size", manifest.config.hidden_size)
        && setRequiredUnsigned(
            values, "intermediate_size", manifest.config.intermediate_size)
        && setRequiredUnsigned(values, "layer_count", manifest.config.layer_count)
        && setRequiredUnsigned(
            values, "attention_head_count",
            manifest.config.attention_head_count)
        && setRequiredUnsigned(
            values, "kv_head_count", manifest.config.kv_head_count)
        && setRequiredUnsigned(
            values, "head_dimension", manifest.config.head_dimension)
        && setRequiredUnsigned(
            values, "vocabulary_size", manifest.config.vocabulary_size)
        && setRequiredUnsigned(
            values, "max_position_embeddings",
            manifest.config.max_position_embeddings)
        && setRequiredUnsigned(
            values, "bos_token_id", manifest.config.bos_token_id)
        && setRequiredUnsigned(
            values, "eos_token_id", manifest.config.eos_token_id)
        && setRequiredUnsigned(values, "tensor_count", tensor_count);
    std::uint64_t data_bytes = 0;
    auto const data_bytes_value = values.find("data_bytes");
    fields_ok = fields_ok
        && data_bytes_value != values.end()
        && parseUnsigned(data_bytes_value->second, data_bytes);
    manifest.data_bytes = data_bytes;
    auto const epsilon = values.find("rms_norm_epsilon");
    auto const theta = values.find("rope_theta");
    fields_ok = fields_ok
        && epsilon != values.end()
        && theta != values.end()
        && parseFloat(epsilon->second, manifest.config.rms_norm_epsilon)
        && parseFloat(theta->second, manifest.config.rope_theta);
    auto const tied = values.find("tied_word_embeddings");
    fields_ok = fields_ok && tied != values.end()
        && (tied->second == "0" || tied->second == "1");
    if (tied != values.end()) {
        manifest.config.tied_word_embeddings = tied->second == "1";
    }
    if (!fields_ok || tensor_count != tensor_lines.size()) {
        result.status = failure(
            WeightManifestError::ParseFailed,
            "missing field or tensor_count mismatch"
        );
        return result;
    }

    manifest.tensors.reserve(tensor_lines.size());
    for (std::string const& encoded : tensor_lines) {
        std::vector<std::string> const parts = split(encoded, '|');
        if (parts.size() != 5) {
            result.status = failure(
                WeightManifestError::ParseFailed,
                "tensor record must contain five fields"
            );
            return result;
        }
        WeightTensorDescriptor tensor;
        tensor.name = parts[0];
        tensor.sha256 = parts[4];
        if (!parseUnsigned(parts[1], tensor.offset)
            || !parseUnsigned(parts[2], tensor.byte_size)) {
            result.status = failure(
                WeightManifestError::ParseFailed,
                "invalid tensor range: " + tensor.name
            );
            return result;
        }
        for (std::string const& dimension : split(parts[3], ',')) {
            std::uint64_t parsed = 0;
            if (!parseUnsigned(dimension, parsed)
                || parsed > std::numeric_limits<std::uint32_t>::max()) {
                result.status = failure(
                    WeightManifestError::ParseFailed,
                    "invalid tensor shape: " + tensor.name
                );
                return result;
            }
            tensor.shape.push_back(static_cast<std::uint32_t>(parsed));
        }
        manifest.tensors.push_back(std::move(tensor));
    }
    result.status = validateWeightManifest(manifest);
    return result;
}

WeightManifestStatus validateWeightDataFile(
    WeightManifest const& manifest,
    std::string const& data_path)
{
    WeightManifestStatus const metadata = validateWeightManifest(manifest);
    if (!metadata.ok()) {
        return metadata;
    }
    std::ifstream input(data_path, std::ios::binary | std::ios::ate);
    if (!input) {
        return failure(WeightManifestError::FileOpenFailed, data_path);
    }
    std::streamoff const file_size = input.tellg();
    if (file_size < 0
        || static_cast<std::uint64_t>(file_size) != manifest.data_bytes) {
        return failure(
            WeightManifestError::DataFileMismatch,
            "archive size differs from manifest"
        );
    }

    for (WeightTensorDescriptor const& tensor : manifest.tensors) {
        std::string actual;
        WeightManifestStatus const computed = computeWeightDataSha256(
            data_path, tensor.offset, tensor.byte_size, actual
        );
        if (!computed.ok()) {
            return computed;
        }
        if (actual != tensor.sha256) {
            return failure(
                WeightManifestError::ChecksumMismatch, tensor.name
            );
        }
    }
    return {};
}

WeightManifestStatus computeWeightDataSha256(
    std::string const& data_path,
    std::uint64_t offset,
    std::uint64_t byte_size,
    std::string& sha256)
{
    sha256.clear();
    if (data_path.empty() || byte_size == 0
        || offset > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max())) {
        return failure(
            WeightManifestError::InvalidArgument,
            "invalid checksum range"
        );
    }
    std::ifstream input(data_path, std::ios::binary);
    if (!input) {
        return failure(WeightManifestError::FileOpenFailed, data_path);
    }
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<std::uint8_t> buffer(4 * 1024 * 1024);
    Sha256 digest;
    std::uint64_t remaining = byte_size;
    while (remaining != 0) {
        std::size_t const wanted = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, buffer.size())
        );
        input.read(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(wanted)
        );
        if (input.gcount() != static_cast<std::streamsize>(wanted)) {
            return failure(
                WeightManifestError::DataFileMismatch,
                "checksum range exceeds data file"
            );
        }
        digest.update(buffer.data(), wanted);
        remaining -= wanted;
    }
    sha256 = digest.finish();
    return {};
}

} // namespace kimkvcache
