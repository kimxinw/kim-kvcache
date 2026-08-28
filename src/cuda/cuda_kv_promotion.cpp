#include "cuda_kv_storage_internal.cuh"

#include <cstddef>
#include <utility>

namespace kimkvcache {

CudaSubmission CudaKvStorage::promoteAsync(
    PromotionPrepareResult const& promotion,
    CudaStream stream)
{
    using namespace cuda_storage_detail;
    auto operation = beginOperation(impl_, stream);
    if (!operation->submission_status.ok()) {
        return CudaSubmission(std::move(operation));
    }

    if (impl_->micro_page_tokens != kMicroPageTokenCapacity
        || impl_->extent_page_tokens != kExtentPageTokenCapacity
        || !promotion.ok()
        || promotion.target_handle.kind != PageKind::Extent
        || impl_->pagePointer(promotion.target_handle) == nullptr) {
        operation->submission_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    cuda_detail::PromotionSlots slots{};
    for (std::size_t index = 0;
         index < promotion.source_handles.size();
         ++index) {
        PageHandle const source = promotion.source_handles[index];
        if (source.kind != PageKind::Micro
            || impl_->pagePointer(source) == nullptr) {
            operation->submission_status = CudaStatus{
                CudaError::InvalidArgument,
                0,
            };
            operation->final_status = operation->submission_status;
            operation->finished = true;
            operation->cleanup();
            return CudaSubmission(std::move(operation));
        }
        slots.slots[index] = source.slot;
    }

    cuda_detail::launchPromotion(
        impl_->micro_data,
        impl_->micro_page_elements,
        slots,
        impl_->pagePointer(promotion.target_handle),
        deviceLayout(impl_->layout),
        impl_->extent_page_elements,
        operation->stream
    );

    static_cast<void>(recordOperation(*operation));
    return CudaSubmission(std::move(operation));
}

} // namespace kimkvcache
