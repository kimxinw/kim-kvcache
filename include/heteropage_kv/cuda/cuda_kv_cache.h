#pragma once

#include "heteropage_kv/cuda/cuda_kv_storage.h"
#include "heteropage_kv/runtime/kv_cache_manager.h"

#include <cstdint>
#include <optional>

namespace kimkvcache {

struct CudaKvOperationResult final {
    KvCacheError metadata_error{KvCacheError::None};
    CudaStatus cuda_status{};

    [[nodiscard]] constexpr bool ok() const noexcept
    {
        return metadata_error == KvCacheError::None
            && cuda_status.ok();
    }

    [[nodiscard]] explicit constexpr operator bool() const noexcept
    {
        return ok();
    }
};

// K4 的安全编排层。所有 CUDA 工作都异步提交到给定 Stream，但公共操作
// 在返回前等待 Event；等待期间不持 KvCacheManager 的全局 Mutex。
class CudaKvCache final {
public:
    CudaKvCache(
        KvLayout layout,
        std::uint32_t micro_capacity,
        std::uint32_t extent_capacity
    );

    ~CudaKvCache() = default;

    CudaKvCache(CudaKvCache const&) = delete;
    CudaKvCache& operator=(CudaKvCache const&) = delete;

    CudaKvCache(CudaKvCache&&) = delete;
    CudaKvCache& operator=(CudaKvCache&&) = delete;

    [[nodiscard]] CudaStatus status() const noexcept;

    [[nodiscard]] KvCacheError createRequest(RequestId request_id);
    [[nodiscard]] KvCacheError sealTail(RequestId request_id);
    [[nodiscard]] KvCacheError forkRequest(
        RequestId source_request_id,
        RequestId child_request_id
    );
    [[nodiscard]] KvCacheError releaseRequest(RequestId request_id);

    [[nodiscard]] CudaKvOperationResult append(
        RequestId request_id,
        std::uint32_t token_count,
        KvScalar const* device_input,
        CudaStream stream = nullptr
    );

    [[nodiscard]] CudaKvOperationResult promote(
        RequestId request_id,
        std::uint32_t logical_token_begin,
        CudaStream stream = nullptr
    );

    [[nodiscard]] CudaKvOperationResult gather(
        RequestId request_id,
        KvScalar* device_output,
        CudaStream stream = nullptr
    );

    [[nodiscard]] CudaKvOperationResult referenceAttention(
        RequestId request_id,
        float const* device_query,
        float* device_output,
        CudaStream stream = nullptr
    );

    [[nodiscard]] std::optional<BlockTable> blockTable(
        RequestId request_id) const;
    [[nodiscard]] KvCacheManagerSnapshot metadataSnapshot() const;
    [[nodiscard]] CudaStorageSnapshot storageSnapshot() const noexcept;
    [[nodiscard]] bool checkInvariants() const;

    void injectFailureOnce(CudaFailurePoint point) noexcept;

private:
    [[nodiscard]] static CudaKvOperationResult metadataFailure(
        KvCacheError error
    ) noexcept;

    [[nodiscard]] CudaKvOperationResult finishReadOperation(
        PageLeaseId lease_id,
        CudaSubmission submission
    );

    KvCacheManager manager_;
    CudaKvStorage storage_;
};

} // namespace kimkvcache
