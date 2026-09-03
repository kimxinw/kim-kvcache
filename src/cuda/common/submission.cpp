#include "../kv/storage_internal.cuh"

#include <memory>
#include <new>
#include <utility>

namespace kimkvcache {

void CudaSubmission::Impl::cleanup() noexcept
{
    for (void* allocation : temporaries) {
        if (allocation != nullptr) {
            static_cast<void>(cudaFree(allocation));
        }
    }
    temporaries.clear();

    if (event != nullptr) {
        static_cast<void>(cudaEventDestroy(event));
        event = nullptr;
    }
}

CudaStatus CudaSubmission::Impl::complete(bool wait) noexcept
{
    if (!submission_status.ok()) {
        return submission_status;
    }
    if (finished) {
        return final_status;
    }
    if (event == nullptr) {
        final_status = CudaStatus{CudaError::InternalError, 0};
        finished = true;
        cleanup();
        return final_status;
    }

    cudaError_t const event_status = wait
        ? cudaEventSynchronize(event)
        : cudaEventQuery(event);

    if (!wait && event_status == cudaErrorNotReady) {
        return CudaStatus{
            CudaError::NotReady,
            static_cast<int>(event_status),
        };
    }

    if (event_status != cudaSuccess) {
        final_status = CudaStatus{
            CudaError::ExecutionFailed,
            static_cast<int>(event_status),
        };
    } else if (inject_completion_failure) {
        final_status = CudaStatus{
            CudaError::ExecutionFailed,
            static_cast<int>(cudaErrorUnknown),
        };
    } else {
        final_status = CudaStatus{};
    }

    finished = true;
    cleanup();
    return final_status;
}

CudaSubmission::Impl::~Impl()
{
    if (!finished && submission_status.ok() && event != nullptr) {
        static_cast<void>(cudaEventSynchronize(event));
    }
    cleanup();
}

namespace cuda_storage_detail {
namespace {

[[nodiscard]] std::unique_ptr<CudaSubmission::Impl> failedOperation(
    CudaStatus status)
{
    auto operation = std::make_unique<CudaSubmission::Impl>();
    operation->submission_status = status;
    operation->final_status = status;
    operation->finished = true;
    return operation;
}

} // namespace

std::unique_ptr<CudaSubmission::Impl> beginOperation(
    std::shared_ptr<CudaKvStorage::Impl> const& storage,
    CudaStream stream)
{
    if (!storage || !storage->initialization_status.ok()) {
        return failedOperation(
            storage
                ? storage->initialization_status
                : CudaStatus{CudaError::AllocationFailed, 0}
        );
    }

    if (storage->consumeFailure(CudaFailurePoint::Submission)) {
        return failedOperation(CudaStatus{CudaError::SubmissionFailed, 0});
    }

    auto operation = std::make_unique<CudaSubmission::Impl>();
    operation->storage_owner = storage;
    operation->stream = nativeStream(stream);
    operation->inject_completion_failure =
        storage->consumeFailure(CudaFailurePoint::Completion);

    cudaError_t const event_error = cudaEventCreateWithFlags(
        &operation->event,
        cudaEventDisableTiming
    );
    if (event_error != cudaSuccess) {
        operation->submission_status = mapSubmissionError(event_error);
        operation->final_status = operation->submission_status;
        operation->finished = true;
    }
    return operation;
}

CudaStatus recordOperation(CudaSubmission::Impl& operation) noexcept
{
    cudaError_t error = cudaGetLastError();
    if (error == cudaSuccess) {
        error = cudaEventRecord(operation.event, operation.stream);
    }

    if (error != cudaSuccess) {
        // 已入队的工作可能仍引用临时 Descriptor；同步后才能释放。
        static_cast<void>(cudaStreamSynchronize(operation.stream));
        operation.submission_status = mapSubmissionError(error);
        operation.final_status = operation.submission_status;
        operation.finished = true;
        operation.cleanup();
        return operation.submission_status;
    }

    operation.submission_status = CudaStatus{};
    return operation.submission_status;
}

bool addTemporary(CudaSubmission::Impl& operation, void* allocation)
{
    try {
        operation.temporaries.push_back(allocation);
        return true;
    } catch (...) {
        static_cast<void>(cudaFree(allocation));
        operation.submission_status = CudaStatus{
            CudaError::AllocationFailed,
            0,
        };
        operation.final_status = operation.submission_status;
        operation.finished = true;
        operation.cleanup();
        return false;
    }
}

} // namespace cuda_storage_detail

CudaSubmission::CudaSubmission() noexcept = default;

CudaSubmission::CudaSubmission(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl))
{
}

CudaSubmission::~CudaSubmission() = default;
CudaSubmission::CudaSubmission(CudaSubmission&&) noexcept = default;
CudaSubmission& CudaSubmission::operator=(CudaSubmission&&) noexcept = default;

bool CudaSubmission::submitted() const noexcept
{
    return impl_ != nullptr && impl_->submission_status.ok();
}

CudaStatus CudaSubmission::submissionStatus() const noexcept
{
    return impl_ != nullptr
        ? impl_->submission_status
        : CudaStatus{CudaError::InvalidArgument, 0};
}

CudaStatus CudaSubmission::query() noexcept
{
    return impl_ != nullptr
        ? impl_->complete(false)
        : CudaStatus{CudaError::InvalidArgument, 0};
}

CudaStatus CudaSubmission::wait() noexcept
{
    return impl_ != nullptr
        ? impl_->complete(true)
        : CudaStatus{CudaError::InvalidArgument, 0};
}

} // namespace kimkvcache
