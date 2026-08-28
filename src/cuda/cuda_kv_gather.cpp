#include "cuda_kv_storage_internal.cuh"

#include <cstddef>
#include <utility>
#include <vector>

namespace kimkvcache {

CudaSubmission CudaKvStorage::gatherAsync(
    BlockTable const& table,
    KvScalar* device_output,
    CudaStream stream)
{
    using namespace cuda_storage_detail;
    auto operation = beginOperation(impl_, stream);
    if (!operation->submission_status.ok()) {
        return CudaSubmission(std::move(operation));
    }

    std::uint32_t const token_count = table.tokenCount();
    if (device_output == nullptr
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
    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&device_descriptors),
        descriptor_bytes
    );

    if (error != cudaSuccess
        || !addTemporary(*operation, device_descriptors)) {
        if (error != cudaSuccess) {
            operation->submission_status = mapSubmissionError(error);
            operation->final_status = operation->submission_status;
            operation->finished = true;
            operation->cleanup();
        }
        return CudaSubmission(std::move(operation));
    }

    error = cudaMemcpyAsync(
        device_descriptors,
        descriptors.data(),
        descriptor_bytes,
        cudaMemcpyHostToDevice,
        operation->stream
    );

    if (error == cudaSuccess) {
        std::size_t const elements = operationElements(
            impl_->layout,
            token_count
        );
        cuda_detail::launchGather(
            device_descriptors,
            static_cast<std::uint32_t>(descriptors.size()),
            device_output,
            token_count,
            deviceLayout(impl_->layout),
            elements,
            operation->stream
        );
    } else {
        operation->submission_status = mapSubmissionError(error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
        operation->cleanup();
        return CudaSubmission(std::move(operation));
    }

    static_cast<void>(recordOperation(*operation));
    return CudaSubmission(std::move(operation));
}

} // namespace kimkvcache
