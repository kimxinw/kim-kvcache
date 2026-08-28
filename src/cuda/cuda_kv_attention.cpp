#include "cuda_kv_storage_internal.cuh"

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace kimkvcache {

CudaSubmission CudaKvStorage::referenceAttentionAsync(
    BlockTable const& table,
    float const* device_query,
    float* device_output,
    CudaStream stream)
{
    using namespace cuda_storage_detail;
    auto operation = beginOperation(impl_, stream);
    if (!operation->submission_status.ok()) {
        return CudaSubmission(std::move(operation));
    }

    std::uint32_t const token_count = table.tokenCount();
    if (device_query == nullptr
        || device_output == nullptr
        || token_count == 0
        || !impl_->validTable(table)) {
        operation->submission_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    std::size_t kv_bytes = 0;
    std::size_t const score_count =
        static_cast<std::size_t>(impl_->layout.layer_count)
        * impl_->layout.kv_head_count
        * token_count;
    if (!impl_->layout.bytesForTokens(token_count, kv_bytes)
        || score_count
            > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        operation->submission_status = CudaStatus{
            CudaError::InvalidArgument,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    KvScalar* contiguous_kv = nullptr;
    float* scores = nullptr;
    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&contiguous_kv),
        kv_bytes
    );
    if (error == cudaSuccess
        && !addTemporary(*operation, contiguous_kv)) {
        return CudaSubmission(std::move(operation));
    }
    if (error == cudaSuccess) {
        error = cudaMalloc(
            reinterpret_cast<void**>(&scores),
            score_count * sizeof(float)
        );
    }
    if (error == cudaSuccess && !addTemporary(*operation, scores)) {
        return CudaSubmission(std::move(operation));
    }
    if (error != cudaSuccess) {
        operation->submission_status = mapSubmissionError(error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    std::vector<cuda_detail::DeviceBlockDescriptor> descriptors;
    try {
        descriptors.reserve(table.entries().size());
        for (MappingEntry const& entry : table.entries()) {
            descriptors.push_back(cuda_detail::DeviceBlockDescriptor{
                impl_->pagePointer(entry.handle),
                entry.logical_token_begin,
                entry.valid_tokens,
                impl_->pageTokenCapacity(entry.kind),
            });
        }
    } catch (...) {
        operation->submission_status = CudaStatus{
            CudaError::AllocationFailed,
            0,
        };
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    cuda_detail::DeviceBlockDescriptor* device_descriptors = nullptr;
    std::size_t const descriptor_bytes =
        descriptors.size() * sizeof(cuda_detail::DeviceBlockDescriptor);
    error = cudaMalloc(
        reinterpret_cast<void**>(&device_descriptors),
        descriptor_bytes
    );
    if (error == cudaSuccess
        && !addTemporary(*operation, device_descriptors)) {
        return CudaSubmission(std::move(operation));
    }
    if (error != cudaSuccess) {
        operation->submission_status = mapSubmissionError(error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    error = cudaMemcpyAsync(
        device_descriptors,
        descriptors.data(),
        descriptor_bytes,
        cudaMemcpyHostToDevice,
        operation->stream
    );
    if (error != cudaSuccess) {
        operation->submission_status = mapSubmissionError(error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    std::size_t const elements = operationElements(
        impl_->layout,
        token_count
    );
    cuda_detail::launchGather(
        device_descriptors,
        static_cast<std::uint32_t>(descriptors.size()),
        contiguous_kv,
        token_count,
        deviceLayout(impl_->layout),
        elements,
        operation->stream
    );
    cuda_detail::launchAttentionScores(
        contiguous_kv,
        token_count,
        device_query,
        scores,
        deviceLayout(impl_->layout),
        operation->stream
    );
    cuda_detail::launchAttentionOutput(
        contiguous_kv,
        token_count,
        scores,
        device_output,
        deviceLayout(impl_->layout),
        operation->stream
    );

    static_cast<void>(recordOperation(*operation));
    return CudaSubmission(std::move(operation));
}

} // namespace kimkvcache
