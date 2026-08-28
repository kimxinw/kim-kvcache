#pragma once

#include "heteropage_kv/cuda/cuda_kv_cache.h"
#include "heteropage_kv/fixed/fixed_page_manager.h"

#include <cstdint>
#include <mutex>
#include <optional>

namespace kimkvcache {

// K6 固定页 CUDA 对照。所有公共操作通过 orchestration_mutex_ 串行化，
// 且 CUDA Submission 在锁内等待完成，因此 FixedPageManager 无需暴露
// Hetero 运行时的异步 PageLease API，也不会在设备访问期间复用 Slot。
class FixedCudaKvCache final {
public:
    FixedCudaKvCache(
        KvLayout layout,
        std::uint16_t tokens_per_page,
        std::uint32_t page_capacity
    );

    ~FixedCudaKvCache() = default;

    FixedCudaKvCache(FixedCudaKvCache const&) = delete;
    FixedCudaKvCache& operator=(FixedCudaKvCache const&) = delete;

    FixedCudaKvCache(FixedCudaKvCache&&) = delete;
    FixedCudaKvCache& operator=(FixedCudaKvCache&&) = delete;

    [[nodiscard]] CudaStatus status() const noexcept;
    [[nodiscard]] std::uint16_t tokensPerPage() const noexcept;

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
    [[nodiscard]] FixedPageManagerSnapshot metadataSnapshot() const;
    [[nodiscard]] CudaStorageSnapshot storageSnapshot() const noexcept;
    [[nodiscard]] bool checkInvariants() const;

    void injectFailureOnce(CudaFailurePoint point) noexcept;

private:
    [[nodiscard]] static CudaKvOperationResult metadataFailure(
        KvCacheError error
    ) noexcept;

    std::uint16_t tokens_per_page_;
    FixedPageManager manager_;
    CudaKvStorage storage_;
    mutable std::mutex orchestration_mutex_;
};

} // namespace kimkvcache
