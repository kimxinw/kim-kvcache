#include "cuda_kv_storage_internal.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace kimkvcache {
namespace {

[[nodiscard]] std::size_t pageElements(
    KvLayout const& layout,
    std::uint16_t token_capacity) noexcept
{
    std::size_t result = 0;
    return layout.elementsForTokens(token_capacity, result) ? result : 0;
}

} // namespace

CudaKvStorage::Impl::~Impl()
{
    // Submission 共享持有 Impl；最后一个使用者退出后才释放设备池。
    static_cast<void>(cudaDeviceSynchronize());
    if (micro_data != nullptr) {
        static_cast<void>(cudaFree(micro_data));
    }
    if (extent_data != nullptr) {
        static_cast<void>(cudaFree(extent_data));
    }
}

bool CudaKvStorage::Impl::consumeFailure(
    CudaFailurePoint point) noexcept
{
    std::lock_guard<std::mutex> lock(fault_mutex);
    bool* flag = point == CudaFailurePoint::Submission
        ? &fail_submission_once
        : &fail_completion_once;
    bool const result = *flag;
    *flag = false;
    return result;
}

KvScalar* CudaKvStorage::Impl::pagePointer(
    PageHandle handle) const noexcept
{
    if (!handle.isStructurallyValid()) {
        return nullptr;
    }

    switch (handle.kind) {
    case PageKind::Micro:
        if (handle.slot >= micro_capacity || micro_data == nullptr) {
            return nullptr;
        }
        return micro_data
            + static_cast<std::size_t>(handle.slot) * micro_page_elements;
    case PageKind::Extent:
        if (handle.slot >= extent_capacity || extent_data == nullptr) {
            return nullptr;
        }
        return extent_data
            + static_cast<std::size_t>(handle.slot) * extent_page_elements;
    }
    return nullptr;
}

std::uint16_t CudaKvStorage::Impl::pageTokenCapacity(
    PageKind kind) const noexcept
{
    return kind == PageKind::Micro
        ? micro_page_tokens
        : extent_page_tokens;
}

bool CudaKvStorage::Impl::validTable(BlockTable const& table) const noexcept
{
    bool const standard_layout =
        micro_page_tokens == kMicroPageTokenCapacity
        && extent_page_tokens == kExtentPageTokenCapacity;
    bool const valid_structure = standard_layout
        ? table.checkInvariants()
        : extent_capacity == 0
            && table.checkInvariants(micro_page_tokens);
    if (!valid_structure) {
        return false;
    }
    return std::all_of(
        table.entries().begin(),
        table.entries().end(),
        [this](MappingEntry const& entry) {
            return pagePointer(entry.handle) != nullptr;
        }
    );
}

CudaKvStorage::CudaKvStorage(
    KvLayout layout,
    std::uint32_t micro_capacity,
    std::uint32_t extent_capacity,
    std::uint16_t micro_page_tokens,
    std::uint16_t extent_page_tokens) noexcept
{
    try {
        impl_ = std::make_shared<Impl>();
    } catch (...) {
        return;
    }

    impl_->layout = layout;
    impl_->micro_capacity = micro_capacity;
    impl_->extent_capacity = extent_capacity;
    impl_->micro_page_tokens = micro_page_tokens;
    impl_->extent_page_tokens = extent_page_tokens;
    impl_->micro_page_elements = pageElements(layout, micro_page_tokens);
    impl_->extent_page_elements = pageElements(layout, extent_page_tokens);

    if (!layout.valid()
        || micro_page_tokens == 0
        || extent_page_tokens == 0
        || impl_->micro_page_elements == 0
        || impl_->extent_page_elements == 0) {
        impl_->initialization_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        return;
    }

    if ((micro_capacity != 0
         && impl_->micro_page_elements
            > std::numeric_limits<std::size_t>::max()
                / micro_capacity / sizeof(KvScalar))
        || (extent_capacity != 0
            && impl_->extent_page_elements
                > std::numeric_limits<std::size_t>::max()
                    / extent_capacity / sizeof(KvScalar))) {
        impl_->initialization_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        return;
    }

    impl_->micro_reserved_bytes = impl_->micro_page_elements
        * micro_capacity * sizeof(KvScalar);
    impl_->extent_reserved_bytes = impl_->extent_page_elements
        * extent_capacity * sizeof(KvScalar);

    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
        impl_->initialization_status = error == cudaSuccess
            ? CudaStatus{CudaError::RuntimeUnavailable, 0}
            : cuda_storage_detail::mapInitializationError(error);
        return;
    }

    if (impl_->micro_reserved_bytes != 0) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&impl_->micro_data),
            impl_->micro_reserved_bytes
        );
        if (error == cudaSuccess) {
            error = cudaMemset(
                impl_->micro_data,
                0,
                impl_->micro_reserved_bytes
            );
        }
    }

    if (error == cudaSuccess && impl_->extent_reserved_bytes != 0) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&impl_->extent_data),
            impl_->extent_reserved_bytes
        );
        if (error == cudaSuccess) {
            error = cudaMemset(
                impl_->extent_data,
                0,
                impl_->extent_reserved_bytes
            );
        }
    }
    impl_->initialization_status =
        cuda_storage_detail::mapInitializationError(error);
}

CudaKvStorage::~CudaKvStorage() = default;

CudaStatus CudaKvStorage::status() const noexcept
{
    return impl_ != nullptr
        ? impl_->initialization_status
        : CudaStatus{CudaError::AllocationFailed, 0};
}

KvLayout CudaKvStorage::layout() const noexcept
{
    return impl_ != nullptr ? impl_->layout : KvLayout{};
}

CudaStorageSnapshot CudaKvStorage::snapshot() const noexcept
{
    if (impl_ == nullptr) {
        return CudaStorageSnapshot{};
    }
    return CudaStorageSnapshot{
        impl_->micro_capacity,
        impl_->extent_capacity,
        impl_->micro_page_tokens,
        impl_->extent_page_tokens,
        impl_->micro_reserved_bytes,
        impl_->extent_reserved_bytes,
    };
}

void CudaKvStorage::injectFailureOnce(CudaFailurePoint point) noexcept
{
    if (impl_ == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(impl_->fault_mutex);
    if (point == CudaFailurePoint::Submission) {
        impl_->fail_submission_once = true;
    } else {
        impl_->fail_completion_once = true;
    }
}

} // namespace kimkvcache
