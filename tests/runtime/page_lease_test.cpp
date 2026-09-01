#include "kim-kv/runtime/kv_cache_manager.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

using namespace kimkvcache;

int failures = 0;

void expect(bool condition, std::string const& message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

void testRequestLeaseDefersPhysicalRelease()
{
    KvCacheManager manager(1, 1);

    expect(manager.createRequest(1) == KvCacheError::None, "create request");
    expect(manager.append(1, 8) == KvCacheError::None, "append full page");

    PageLeaseAcquireResult const lease =
        manager.acquireRequestReadLease(1);
    expect(lease.ok(), "request lease must be acquired");
    expect(lease.handles.size() == 1, "lease must pin one page");

    PageHandle const handle = lease.handles[0];
    auto metadata = manager.pageMetadata(handle);
    expect(metadata.has_value(), "leased page metadata must exist");
    if (metadata.has_value()) {
        expect(metadata->inflight_readers == 1, "reader count must be one");
    }

    expect(
        manager.releaseRequest(1) == KvCacheError::None,
        "request release may race with a device reader"
    );
    metadata = manager.pageMetadata(handle);
    expect(metadata.has_value(), "retiring page metadata must exist");
    if (metadata.has_value()) {
        expect(metadata->state == PageState::Retiring, "page must retire");
        expect(metadata->ref_count == 0, "logical reference must be gone");
        expect(metadata->inflight_readers == 1, "lease must preserve reader");
    }
    expect(
        manager.snapshot().micro_pool.allocated_slots == 1,
        "retiring page must not return to the pool"
    );
    expect(manager.checkInvariants(), "retiring reader invariants");

    expect(
        manager.releasePageLease(lease.lease_id) == KvCacheError::None,
        "releasing final reader must succeed"
    );
    metadata = manager.pageMetadata(handle);
    expect(metadata.has_value(), "freed generation metadata is queryable");
    if (metadata.has_value()) {
        expect(metadata->state == PageState::Free, "page must become free");
        expect(metadata->inflight_readers == 0, "reader count must reset");
    }
    expect(
        manager.snapshot().micro_pool.allocated_slots == 0,
        "page must return after Event completion"
    );
    expect(manager.checkInvariants(), "post-reader invariants");
}

void testPromotionLeaseProtectsSourcesAndTarget()
{
    KvCacheManager manager(8, 1);

    expect(manager.createRequest(2) == KvCacheError::None, "create promotion");
    expect(manager.append(2, 64) == KvCacheError::None, "append promotion run");

    PromotionPrepareResult const promotion = manager.preparePromotion(2, 0);
    expect(promotion.ok(), "Prepare must succeed");

    PageLeaseAcquireResult const lease =
        manager.acquirePromotionIoLease(promotion.promotion_id);
    expect(lease.ok(), "promotion I/O lease must succeed");
    expect(lease.handles.size() == 9, "8 sources plus target expected");
    expect(manager.checkInvariants(), "active promotion lease invariants");

    expect(
        manager.releaseRequest(2) == KvCacheError::None,
        "request cancellation must roll back metadata without freeing I/O"
    );
    expect(
        manager.snapshot().promotion_count == 0,
        "cancellation must close transaction"
    );
    expect(
        manager.snapshot().micro_pool.allocated_slots == 8,
        "source slots must remain until CUDA Event"
    );
    expect(
        manager.snapshot().extent_pool.allocated_slots == 1,
        "target slot must remain until CUDA Event"
    );

    for (PageHandle const handle : lease.handles) {
        auto const metadata = manager.pageMetadata(handle);
        expect(metadata.has_value(), "leased promotion metadata must exist");
        if (metadata.has_value()) {
            expect(
                metadata->state == PageState::Retiring,
                "cancelled promotion pages must retire"
            );
            expect(metadata->ref_count == 0, "retiring ref count must be zero");
            expect(
                metadata->inflight_readers == 1,
                "promotion I/O must remain pinned"
            );
        }
    }
    expect(manager.checkInvariants(), "cancelled promotion I/O invariants");

    expect(
        manager.releasePageLease(lease.lease_id) == KvCacheError::None,
        "Event completion must release all promotion pages"
    );
    auto const snapshot = manager.snapshot();
    expect(snapshot.request_count == 0, "no request remains");
    expect(snapshot.page_lease_count == 0, "no lease remains");
    expect(snapshot.micro_pool.allocated_slots == 0, "all sources free");
    expect(snapshot.extent_pool.allocated_slots == 0, "target free");
    expect(manager.checkInvariants(), "promotion lease cleanup invariants");
}

void testSharedReadersAndEmptyLease()
{
    KvCacheManager manager(1, 1);

    expect(manager.createRequest(3) == KvCacheError::None, "create source");
    expect(manager.append(3, 8) == KvCacheError::None, "append source");
    expect(manager.forkRequest(3, 4) == KvCacheError::None, "fork source");

    PageLeaseAcquireResult const first =
        manager.acquireRequestReadLease(3);
    PageLeaseAcquireResult const second =
        manager.acquireRequestReadLease(4);
    expect(first.ok() && second.ok(), "both shared readers acquire leases");

    PageHandle const handle = first.handles[0];
    auto metadata = manager.pageMetadata(handle);
    if (metadata.has_value()) {
        expect(metadata->ref_count == 2, "shared refs must be two");
        expect(metadata->inflight_readers == 2, "shared readers must be two");
    } else {
        expect(false, "shared metadata must exist");
    }

    expect(manager.releaseRequest(4) == KvCacheError::None, "release child");
    expect(manager.releaseRequest(3) == KvCacheError::None, "release parent");
    expect(manager.releasePageLease(first.lease_id) == KvCacheError::None,
        "release first shared reader");
    metadata = manager.pageMetadata(handle);
    if (metadata.has_value()) {
        expect(metadata->state == PageState::Retiring, "one reader retires");
        expect(metadata->inflight_readers == 1, "one reader remains");
    }
    expect(manager.releasePageLease(second.lease_id) == KvCacheError::None,
        "release final shared reader");
    expect(manager.snapshot().micro_pool.allocated_slots == 0, "shared free");

    expect(manager.createRequest(5) == KvCacheError::None, "create empty");
    PageLeaseAcquireResult const empty =
        manager.acquireRequestReadLease(5);
    expect(empty.ok(), "empty request lease must be valid");
    expect(empty.handles.empty(), "empty request has no physical pages");
    expect(manager.releaseRequest(5) == KvCacheError::None, "release empty");
    expect(manager.releasePageLease(empty.lease_id) == KvCacheError::None,
        "release empty lease");
    expect(manager.checkInvariants(), "empty lease invariants");
}

void testLeaseErrors()
{
    KvCacheManager manager(1, 1);

    expect(
        manager.acquireRequestReadLease(kInvalidRequestId).error
            == KvCacheError::InvalidArgument,
        "invalid request id must be rejected"
    );
    expect(
        manager.acquireRequestReadLease(999).error
            == KvCacheError::RequestNotFound,
        "unknown request must be rejected"
    );
    expect(
        manager.acquirePromotionIoLease(kInvalidPromotionId).error
            == KvCacheError::InvalidArgument,
        "invalid promotion id must be rejected"
    );
    expect(
        manager.acquirePromotionIoLease(999).error
            == KvCacheError::PromotionNotFound,
        "unknown promotion must be rejected"
    );
    expect(
        manager.releasePageLease(kInvalidPageLeaseId)
            == KvCacheError::InvalidArgument,
        "invalid lease id must be rejected"
    );
    expect(
        manager.releasePageLease(999) == KvCacheError::InvalidState,
        "unknown lease must be rejected"
    );
    expect(manager.checkInvariants(), "error paths preserve invariants");
}

} // namespace

int main()
{
    testRequestLeaseDefersPhysicalRelease();
    testPromotionLeaseProtectsSourcesAndTarget();
    testSharedReadersAndEmptyLease();
    testLeaseErrors();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All PageLease/K4 lifecycle tests passed\n";
    return 0;
}
