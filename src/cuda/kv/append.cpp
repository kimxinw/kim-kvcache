#include "storage_internal.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace kimkvcache {

CudaSubmission CudaKvStorage::appendAsync(
    BlockTable const& before,
    BlockTable const& after,
    std::uint32_t append_token_begin,
    std::uint32_t token_count,
    KvScalar const* device_input,
    CudaStream stream)
{
    using namespace cuda_storage_detail;
    auto operation = beginOperation(impl_, stream);
    if (!operation->submission_status.ok()) {
        return CudaSubmission(std::move(operation));
    }

    std::uint64_t const expected_end =
        static_cast<std::uint64_t>(append_token_begin) + token_count;
    if (device_input == nullptr
        || token_count == 0
        || expected_end > std::numeric_limits<std::uint32_t>::max()
        || !impl_->validTable(before)
        || !impl_->validTable(after)
        || before.tokenCount() != append_token_begin
        || after.tokenCount() != expected_end) {
        operation->submission_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    cuda_detail::DeviceLayout const layout = deviceLayout(impl_->layout);

    // 唯一允许改变旧逻辑区间 Handle 的情况是 Partial Tail COW。
    for (MappingEntry const& old_entry : before.entries()) {
        auto const replacement = std::find_if(
            after.entries().begin(),
            after.entries().end(),
            [&old_entry](MappingEntry const& entry) {
                return entry.logical_token_begin
                    == old_entry.logical_token_begin;
            }
        );

        if (replacement == after.entries().end()) {
            operation->submission_status = CudaStatus{
                CudaError::InvalidArgument,
                0,
            };
            operation->final_status = operation->submission_status;
            operation->finished = true;
            operation->cleanup();
            return CudaSubmission(std::move(operation));
        }

        if (replacement->handle == old_entry.handle) {
            continue;
        }

        bool const is_tail = old_entry.logicalTokenEnd()
            == append_token_begin;
        if (!is_tail
            || old_entry.kind != PageKind::Micro
            || replacement->kind != PageKind::Micro
            || old_entry.valid_tokens >= impl_->micro_page_tokens
            || replacement->valid_tokens < old_entry.valid_tokens) {
            operation->submission_status = CudaStatus{
                CudaError::InvalidArgument,
                0,
            };
            operation->final_status = operation->submission_status;
            operation->finished = true;
            operation->cleanup();
            return CudaSubmission(std::move(operation));
        }

        std::size_t const elements = operationElements(
            impl_->layout,
            old_entry.valid_tokens
        );
        cuda_detail::launchCopyPageTokens(
            impl_->pagePointer(old_entry.handle),
            impl_->micro_page_tokens,
            impl_->pagePointer(replacement->handle),
            impl_->micro_page_tokens,
            old_entry.valid_tokens,
            layout,
            elements,
            operation->stream
        );
    }

    std::uint32_t const append_end = append_token_begin + token_count;
    for (MappingEntry const& entry : after.entries()) {
        std::uint32_t const segment_begin = std::max(
            append_token_begin,
            entry.logical_token_begin
        );
        std::uint32_t const segment_end = std::min(
            append_end,
            entry.logicalTokenEnd()
        );

        if (segment_begin >= segment_end) {
            continue;
        }

        std::uint32_t const segment_tokens = segment_end - segment_begin;
        std::size_t const elements = operationElements(
            impl_->layout,
            segment_tokens
        );
        cuda_detail::launchAppendTokens(
            device_input,
            token_count,
            segment_begin - append_token_begin,
            impl_->pagePointer(entry.handle),
            impl_->pageTokenCapacity(entry.kind),
            segment_begin - entry.logical_token_begin,
            segment_tokens,
            layout,
            elements,
            operation->stream
        );
    }

    static_cast<void>(recordOperation(*operation));
    return CudaSubmission(std::move(operation));
}

} // namespace kimkvcache
