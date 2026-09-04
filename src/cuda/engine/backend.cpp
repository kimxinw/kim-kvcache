#include "kim-kv/cuda/cuda_engine_kv_backend.h"

#include "../kv/storage.h"
#include "kim-kv/fixed/fixed_page_manager.h"
#include "kim-kv/runtime/kv_cache_manager.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <utility>

namespace kimkvcache {
namespace {

[[nodiscard]] EngineKvStatus mapMetadata(KvCacheError error) noexcept
{
    switch (error) {
    case KvCacheError::None:
        return {};
    case KvCacheError::InvalidArgument:
        return {EngineKvError::InvalidArgument};
    case KvCacheError::RequestNotFound:
        return {EngineKvError::RequestNotFound};
    case KvCacheError::RequestAlreadyExists:
        return {EngineKvError::RequestAlreadyExists};
    case KvCacheError::RequestConflict:
    case KvCacheError::PromotionConflict:
        return {EngineKvError::RequestConflict};
    case KvCacheError::ResourceExhausted:
        return {EngineKvError::ResourceExhausted};
    case KvCacheError::InvalidState:
        return {EngineKvError::InvalidState};
    case KvCacheError::TokenReservationNotFound:
    case KvCacheError::PromotionNotEligible:
    case KvCacheError::PromotionNotFound:
    case KvCacheError::InternalInvariantViolation:
        return {EngineKvError::InternalError};
    }
    return {EngineKvError::InternalError};
}

[[nodiscard]] EngineKvStatus mapCuda(CudaStatus status) noexcept
{
    switch (status.error) {
    case CudaError::None:
        return {};
    case CudaError::InvalidArgument:
        return {EngineKvError::InvalidArgument};
    case CudaError::AllocationFailed:
        return {EngineKvError::ResourceExhausted};
    case CudaError::SubmissionFailed:
        return {EngineKvError::SubmissionFailed};
    case CudaError::ExecutionFailed:
        return {EngineKvError::ExecutionFailed};
    case CudaError::NotReady:
    case CudaError::RuntimeUnavailable:
    case CudaError::InternalError:
        return {EngineKvError::InternalError};
    }
    return {EngineKvError::InternalError};
}

struct EngineBackendState final {
    EngineKvBackendKind kind{EngineKvBackendKind::Heterogeneous};
    EngineKvConfig config{};
    std::unique_ptr<KvCacheManager> heterogeneous{};
    std::unique_ptr<FixedPageManager> fixed{};
    CudaKvStorage storage;
    std::unordered_map<RequestId, std::uint32_t> committed_lengths{};
    std::uint64_t active_transactions{0};
    std::atomic<std::uint64_t> batched_attention_submissions{0};
    std::atomic<std::uint64_t> batched_attention_lanes{0};
    mutable std::mutex mutex{};

    EngineBackendState(
        EngineKvConfig value,
        std::uint32_t micro_capacity,
        std::uint32_t extent_capacity)
        : kind(EngineKvBackendKind::Heterogeneous)
        , config(value)
        , heterogeneous(std::make_unique<KvCacheManager>(
            micro_capacity, extent_capacity))
        , storage(value.kv_layout, micro_capacity, extent_capacity)
    {
    }

    EngineBackendState(
        EngineKvConfig value,
        std::uint16_t tokens_per_page,
        std::uint32_t page_capacity)
        : kind(EngineKvBackendKind::Fixed)
        , config(value)
        , fixed(std::make_unique<FixedPageManager>(
            tokens_per_page, page_capacity))
        , storage(
            value.kv_layout,
            page_capacity,
            0,
            tokens_per_page,
            kExtentPageTokenCapacity)
    {
    }

    [[nodiscard]] KvCacheError create(RequestId request_id)
    {
        return heterogeneous != nullptr
            ? heterogeneous->createRequest(request_id)
            : fixed->createRequest(request_id);
    }

    [[nodiscard]] KvCacheError fork(RequestId source, RequestId child)
    {
        return heterogeneous != nullptr
            ? heterogeneous->forkRequest(source, child)
            : fixed->forkRequest(source, child);
    }

    [[nodiscard]] KvCacheError release(RequestId request_id)
    {
        return heterogeneous != nullptr
            ? heterogeneous->releaseRequest(request_id)
            : fixed->releaseRequest(request_id);
    }

    [[nodiscard]] TokenReservationResult reserve(
        RequestId request_id,
        std::uint32_t expected)
    {
        return heterogeneous != nullptr
            ? heterogeneous->reserveToken(request_id, expected)
            : fixed->reserveToken(request_id, expected);
    }

    [[nodiscard]] KvCacheError commit(KvTokenReservationId id)
    {
        return heterogeneous != nullptr
            ? heterogeneous->commitTokenReservation(id)
            : fixed->commitTokenReservation(id);
    }

    [[nodiscard]] KvCacheError rollback(KvTokenReservationId id)
    {
        return heterogeneous != nullptr
            ? heterogeneous->rollbackTokenReservation(id)
            : fixed->rollbackTokenReservation(id);
    }

    [[nodiscard]] bool metadataInvariants() const
    {
        return heterogeneous != nullptr
            ? heterogeneous->checkInvariants()
            : fixed->checkInvariants();
    }

    [[nodiscard]] std::uint64_t reservationCount() const
    {
        return heterogeneous != nullptr
            ? heterogeneous->snapshot().token_reservation_count
            : fixed->snapshot().token_reservation_count;
    }
};

class CudaTokenTransactionBackend final
    : public TokenTransactionBackend {
public:
    CudaTokenTransactionBackend(
        std::shared_ptr<EngineBackendState> owner,
        KvTokenReservationId reservation_id,
        RequestId request_id,
        PageLeaseId lease_id,
        std::unique_ptr<CudaEngineTransaction> io)
        : owner_(std::move(owner))
        , reservation_id_(reservation_id)
        , request_id_(request_id)
        , lease_id_(lease_id)
        , io_(std::move(io))
    {
    }

    [[nodiscard]] EngineKvStatus writeLayer(
        LayerKvWrite const& write,
        EngineStream) override
    {
        return mapCuda(io_->writeLayer(write));
    }

    [[nodiscard]] EngineKvStatus attendLayer(
        PagedDecodeRequest const& request,
        EngineStream) override
    {
        return mapCuda(io_->attendLayer(request));
    }

    void attendLayerBatch(PagedDecodeBatch& batch) override
    {
        constexpr std::size_t kStackBatchCapacity = 64;
        std::array<CudaEngineTransaction::AttentionBatchItem,
            kStackBatchCapacity> cuda_items{};
        std::size_t cuda_count = 0;
        EngineStream common_stream = nullptr;

        for (std::size_t index = 0; index < batch.item_count; ++index) {
            PagedDecodeBatchItem const& item = batch.items[index];
            if (!item.status.ok()) {
                continue;
            }
            auto* const backend = dynamic_cast<CudaTokenTransactionBackend*>(
                batchBackend(item)
            );
            EngineStream const stream = batchStream(item);
            if (backend == nullptr || backend->owner_.get() != owner_.get()
                || (cuda_count != 0 && stream != common_stream)
                || cuda_count == kStackBatchCapacity) {
                TokenTransactionBackend::attendLayerBatch(batch);
                return;
            }
            if (cuda_count == 0) {
                common_stream = stream;
            }
            cuda_items[cuda_count++] = {
                backend->io_.get(), item.request, {},
            };
        }
        if (cuda_count < 2) {
            TokenTransactionBackend::attendLayerBatch(batch);
            return;
        }

        CudaEngineTransaction::attendLayerBatch(
            cuda_items.data(),
            cuda_count,
            batch.host_items,
            batch.device_items,
            batch.item_capacity
        );

        std::size_t cuda_index = 0;
        std::uint64_t successful_lanes = 0;
        for (std::size_t index = 0; index < batch.item_count; ++index) {
            PagedDecodeBatchItem& item = batch.items[index];
            if (!item.status.ok()) {
                continue;
            }
            item.status = mapCuda(cuda_items[cuda_index++].status);
            successful_lanes += item.status.ok() ? 1U : 0U;
        }
        if (successful_lanes != 0) {
            owner_->batched_attention_submissions.fetch_add(
                1, std::memory_order_relaxed
            );
            owner_->batched_attention_lanes.fetch_add(
                successful_lanes, std::memory_order_relaxed
            );
        }
    }

    [[nodiscard]] EngineKvStatus commit(EngineStream) override
    {
        CudaStatus const completion = io_->finish();
        if (!completion.ok()) {
            return mapCuda(completion);
        }
        std::lock_guard<std::mutex> lock(owner_->mutex);
        EngineKvStatus const lease_status = releaseLeaseLocked();
        if (!lease_status.ok()) {
            return lease_status;
        }
        KvCacheError const committed = owner_->commit(reservation_id_);
        if (committed != KvCacheError::None) {
            return mapMetadata(committed);
        }
        auto length = owner_->committed_lengths.find(request_id_);
        if (length == owner_->committed_lengths.end()) {
            return {EngineKvError::InternalError};
        }
        ++length->second;
        --owner_->active_transactions;
        resolved_ = true;
        return {};
    }

    void rollback(EngineStream) noexcept override
    {
        if (resolved_) {
            return;
        }
        static_cast<void>(io_->finish());
        std::lock_guard<std::mutex> lock(owner_->mutex);
        static_cast<void>(releaseLeaseLocked());
        static_cast<void>(owner_->rollback(reservation_id_));
        if (owner_->active_transactions != 0) {
            --owner_->active_transactions;
        }
        resolved_ = true;
    }

private:
    [[nodiscard]] EngineKvStatus releaseLeaseLocked() noexcept
    {
        if (lease_id_ == kInvalidPageLeaseId) {
            return {};
        }
        KvCacheError const released =
            owner_->heterogeneous->releasePageLease(lease_id_);
        if (released == KvCacheError::None) {
            lease_id_ = kInvalidPageLeaseId;
        }
        return mapMetadata(released);
    }

    std::shared_ptr<EngineBackendState> owner_;
    KvTokenReservationId reservation_id_{kInvalidKvTokenReservationId};
    RequestId request_id_{kInvalidRequestId};
    PageLeaseId lease_id_{kInvalidPageLeaseId};
    std::unique_ptr<CudaEngineTransaction> io_;
    bool resolved_{false};
};

class CudaEngineKvBackendImpl final : public EngineKvBackend {
public:
    explicit CudaEngineKvBackendImpl(
        std::shared_ptr<EngineBackendState> state)
        : state_(std::move(state))
    {
    }

    [[nodiscard]] EngineKvBackendKind kind() const noexcept override
    {
        return state_->kind;
    }

    [[nodiscard]] EngineKvConfig config() const noexcept override
    {
        return state_->config;
    }

    [[nodiscard]] EngineKvStatus createRequest(
        RequestId request_id) override
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->config.valid() || !state_->storage.status().ok()) {
            return {EngineKvError::InvalidState};
        }
        KvCacheError const created = state_->create(request_id);
        if (created == KvCacheError::None) {
            state_->committed_lengths.emplace(request_id, 0);
        }
        return mapMetadata(created);
    }

    [[nodiscard]] EngineKvStatus forkRequest(
        RequestId source_request_id,
        RequestId child_request_id) override
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        auto const source = state_->committed_lengths.find(source_request_id);
        if (source == state_->committed_lengths.end()) {
            return {EngineKvError::RequestNotFound};
        }
        KvCacheError const forked = state_->fork(
            source_request_id, child_request_id
        );
        if (forked == KvCacheError::None) {
            state_->committed_lengths.emplace(
                child_request_id, source->second
            );
        }
        return mapMetadata(forked);
    }

    [[nodiscard]] EngineKvStatus releaseRequest(
        RequestId request_id) override
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        KvCacheError const released = state_->release(request_id);
        if (released == KvCacheError::None) {
            state_->committed_lengths.erase(request_id);
        }
        return mapMetadata(released);
    }

    [[nodiscard]] TokenReserveResult reserveToken(
        ReserveTokenRequest const& request) override
    {
        TokenReserveResult result;
        std::lock_guard<std::mutex> lock(state_->mutex);
        auto const length = state_->committed_lengths.find(
            request.request_id
        );
        if (request.request_id == kInvalidRequestId) {
            result.status = {EngineKvError::InvalidArgument};
            return result;
        }
        if (length == state_->committed_lengths.end()) {
            result.status = {EngineKvError::RequestNotFound};
            return result;
        }

        TokenReservationResult reservation = state_->reserve(
            request.request_id, request.expected_committed_tokens
        );
        if (!reservation.ok()) {
            result.status = mapMetadata(reservation.error);
            return result;
        }

        PageLeaseId lease_id = kInvalidPageLeaseId;
        if (state_->heterogeneous != nullptr) {
            PageLeaseAcquireResult lease =
                state_->heterogeneous->acquireTokenReservationLease(
                    reservation.reservation_id
                );
            if (!lease.ok()) {
                static_cast<void>(state_->rollback(
                    reservation.reservation_id
                ));
                result.status = mapMetadata(lease.error);
                return result;
            }
            lease_id = lease.lease_id;
        }

        CudaEngineTransactionBeginResult io =
            state_->storage.beginEngineTransaction(
                reservation.before,
                reservation.reserved,
                state_->config.query_head_count,
                request.stream
            );
        if (!io.ok()) {
            if (lease_id != kInvalidPageLeaseId) {
                static_cast<void>(state_->heterogeneous->releasePageLease(
                    lease_id
                ));
            }
            static_cast<void>(state_->rollback(reservation.reservation_id));
            result.status = mapCuda(io.status);
            return result;
        }

        std::unique_ptr<CudaTokenTransactionBackend> transaction_backend;
        try {
            transaction_backend =
                std::make_unique<CudaTokenTransactionBackend>(
                    state_,
                    reservation.reservation_id,
                    reservation.request_id,
                    lease_id,
                    std::move(io.transaction)
                );
        } catch (...) {
            if (io.transaction != nullptr) {
                static_cast<void>(io.transaction->finish());
            }
            if (lease_id != kInvalidPageLeaseId) {
                static_cast<void>(state_->heterogeneous->releasePageLease(
                    lease_id
                ));
            }
            static_cast<void>(state_->rollback(reservation.reservation_id));
            result.status = {EngineKvError::ResourceExhausted};
            return result;
        }
        result.transaction = TokenTransaction(
            std::move(transaction_backend),
            reservation.reservation_id,
            reservation.request_id,
            reservation.logical_token_position,
            state_->config.kv_layout.layer_count,
            request.stream
        );
        if (!result.transaction.valid()) {
            result.status = {EngineKvError::InternalError};
            return result;
        }
        ++state_->active_transactions;
        result.status = {};
        return result;
    }

    [[nodiscard]] EngineKvBackendSnapshot snapshot() const override
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        std::uint64_t committed = 0;
        for (auto const& entry : state_->committed_lengths) {
            committed += entry.second;
        }
        EngineKvBackendSnapshot result;
        result.request_count = state_->committed_lengths.size();
        result.active_transaction_count = state_->active_transactions;
        result.committed_token_count = committed;
        result.batched_attention_submissions =
            state_->batched_attention_submissions.load(
                std::memory_order_relaxed
            );
        result.batched_attention_lanes =
            state_->batched_attention_lanes.load(std::memory_order_relaxed);
        CudaStorageSnapshot const storage = state_->storage.snapshot();
        result.storage_reserved_bytes = storage.totalReservedBytes();
        if (state_->heterogeneous != nullptr) {
            KvCacheManagerSnapshot const metadata =
                state_->heterogeneous->snapshot();
            result.primary_page_tokens = kMicroPageTokenCapacity;
            result.secondary_page_tokens = kExtentPageTokenCapacity;
            result.primary_page_capacity = metadata.micro_pool.total_slots;
            result.secondary_page_capacity = metadata.extent_pool.total_slots;
            result.allocated_primary_pages =
                metadata.micro_pool.allocated_slots;
            result.allocated_secondary_pages =
                metadata.extent_pool.allocated_slots;
            result.successful_primary_allocations =
                metadata.micro_pool.successful_allocations;
            result.successful_secondary_allocations =
                metadata.extent_pool.successful_allocations;
            result.failed_primary_allocations =
                metadata.micro_pool.failed_allocations;
            result.failed_secondary_allocations =
                metadata.extent_pool.failed_allocations;
        } else {
            FixedPageManagerSnapshot const metadata =
                state_->fixed->snapshot();
            result.primary_page_tokens = state_->fixed->tokensPerPage();
            result.primary_page_capacity = metadata.pool.total_slots;
            result.allocated_primary_pages = metadata.pool.allocated_slots;
            result.successful_primary_allocations =
                metadata.pool.successful_allocations;
            result.failed_primary_allocations =
                metadata.pool.failed_allocations;
        }
        return result;
    }

    [[nodiscard]] bool checkInvariants() const override
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        return state_->config.valid()
            && state_->storage.status().ok()
            && state_->metadataInvariants()
            && state_->active_transactions == state_->reservationCount();
    }

    void injectFailureOnce(CudaFailurePoint point) noexcept
    {
        state_->storage.injectFailureOnce(point);
    }

private:
    std::shared_ptr<EngineBackendState> state_;
};

} // namespace

std::unique_ptr<EngineKvBackend>
createHeterogeneousCudaEngineKvBackend(
    EngineKvConfig config,
    std::uint32_t micro_page_capacity,
    std::uint32_t extent_page_capacity)
{
    if (!config.valid()
        || micro_page_capacity == 0
        || extent_page_capacity == 0) {
        return nullptr;
    }
    try {
        return std::make_unique<CudaEngineKvBackendImpl>(
            std::make_shared<EngineBackendState>(
                config, micro_page_capacity, extent_page_capacity
            )
        );
    } catch (...) {
        return nullptr;
    }
}

std::unique_ptr<EngineKvBackend> createFixedCudaEngineKvBackend(
    EngineKvConfig config,
    std::uint16_t tokens_per_page,
    std::uint32_t page_capacity)
{
    if (!config.valid()
        || tokens_per_page == 0
        || page_capacity == 0) {
        return nullptr;
    }
    try {
        return std::make_unique<CudaEngineKvBackendImpl>(
            std::make_shared<EngineBackendState>(
                config, tokens_per_page, page_capacity
            )
        );
    } catch (...) {
        return nullptr;
    }
}

bool injectCudaEngineFailureOnce(
    EngineKvBackend& backend,
    CudaFailurePoint point) noexcept
{
    auto* concrete = dynamic_cast<CudaEngineKvBackendImpl*>(&backend);
    if (concrete == nullptr) {
        return false;
    }
    concrete->injectFailureOnce(point);
    return true;
}

} // namespace kimkvcache
