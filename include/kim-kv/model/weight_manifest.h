#pragma once

#include "kim-kv/model/tinyllama_config.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kimkvcache {

enum class WeightManifestError : std::uint8_t {
    None,
    InvalidArgument,
    FileOpenFailed,
    ParseFailed,
    UnsupportedVersion,
    UnsupportedDataType,
    InvalidConfig,
    InvalidTensor,
    DuplicateTensor,
    MissingTensor,
    UnexpectedTensor,
    DataFileMismatch,
    ChecksumMismatch,
};

[[nodiscard]] constexpr std::string_view toString(
    WeightManifestError error) noexcept
{
    switch (error) {
    case WeightManifestError::None:
        return "none";
    case WeightManifestError::InvalidArgument:
        return "invalid_argument";
    case WeightManifestError::FileOpenFailed:
        return "file_open_failed";
    case WeightManifestError::ParseFailed:
        return "parse_failed";
    case WeightManifestError::UnsupportedVersion:
        return "unsupported_version";
    case WeightManifestError::UnsupportedDataType:
        return "unsupported_data_type";
    case WeightManifestError::InvalidConfig:
        return "invalid_config";
    case WeightManifestError::InvalidTensor:
        return "invalid_tensor";
    case WeightManifestError::DuplicateTensor:
        return "duplicate_tensor";
    case WeightManifestError::MissingTensor:
        return "missing_tensor";
    case WeightManifestError::UnexpectedTensor:
        return "unexpected_tensor";
    case WeightManifestError::DataFileMismatch:
        return "data_file_mismatch";
    case WeightManifestError::ChecksumMismatch:
        return "checksum_mismatch";
    }
    return "unknown";
}

struct WeightManifestStatus final {
    WeightManifestError error{WeightManifestError::None};
    std::string detail{};

    [[nodiscard]] bool ok() const noexcept
    {
        return error == WeightManifestError::None;
    }

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return ok();
    }
};

struct WeightTensorDescriptor final {
    std::string name{};
    std::uint64_t offset{0};
    std::uint64_t byte_size{0};
    std::vector<std::uint32_t> shape{};
    std::string sha256{};
};

struct WeightManifest final {
    std::uint32_t version{0};
    std::string checkpoint{};
    std::string checkpoint_revision{};
    std::string tokenizer_revision{};
    std::string config_sha256{};
    std::string data_type{};
    std::string data_file{};
    std::uint64_t data_bytes{0};
    TinyLlamaConfig config{};
    std::vector<WeightTensorDescriptor> tensors{};

    [[nodiscard]] WeightTensorDescriptor const* find(
        std::string_view name) const noexcept;
};

struct WeightManifestLoadResult final {
    WeightManifestStatus status{};
    WeightManifest manifest{};

    [[nodiscard]] bool ok() const noexcept
    {
        return status.ok();
    }
};

[[nodiscard]] WeightManifestLoadResult loadWeightManifest(
    std::string const& manifest_path
);

[[nodiscard]] WeightManifestStatus validateWeightManifest(
    WeightManifest const& manifest
);

// Verifies the exact binary length and every per-tensor SHA-256. This is kept
// separate from parsing so callers can inspect a manifest without reading a
// multi-gigabyte archive.
[[nodiscard]] WeightManifestStatus validateWeightDataFile(
    WeightManifest const& manifest,
    std::string const& data_path
);

// Computes one binary range digest without loading the range into memory.
// Tooling uses this to create deterministic manifests for converted archives.
[[nodiscard]] WeightManifestStatus computeWeightDataSha256(
    std::string const& data_path,
    std::uint64_t offset,
    std::uint64_t byte_size,
    std::string& sha256
);

} // namespace kimkvcache
