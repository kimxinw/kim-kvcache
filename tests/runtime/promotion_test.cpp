#include "kim-kv/runtime/kv_cache_manager.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <thread>

namespace {

using kimkvcache::BlockTable;
using kimkvcache::KvCacheError;
using kimkvcache::KvCacheManager;
using kimkvcache::PageHandle;
using kimkvcache::PageKind;
using kimkvcache::PageState;
using kimkvcache::PromotionPrepareResult;
using kimkvcache::RequestId;
using kimkvcache::kExtentPageTokenCapacity;
using kimkvcache::kInvalidPromotionId;
using kimkvcache::kInvalidRequestId;
using kimkvcache::kMicroPageTokenCapacity;
using kimkvcache::kPromotionSourcePageCount;

constexpr std::uint32_t kPromotionTokenCount =
    static_cast<std::uint32_t>(
        kPromotionSourcePageCount * kMicroPageTokenCapacity
    );

static_assert(
    kPromotionTokenCount == kExtentPageTokenCapacity,
    "one promotion must cover exactly one extent"
);

int failures = 0;

void expect(bool condition, char const* message)
{
    if (!condition) {
        std::cerr << "[FAILED] " << message << '\n';
        ++failures;
    }
}

bool sameTable(BlockTable const& lhs, BlockTable const& rhs)
{
    return lhs.version() == rhs.version()
        && lhs.entries() == rhs.entries();
}

bool createFullPromotionRun(
    KvCacheManager& manager,
    RequestId request_id)
{
    if (manager.createRequest(request_id) != KvCacheError::None) {
        expect(false, "promotion request creation must succeed");
        return false;
    }

    if (manager.append(request_id, kPromotionTokenCount)
        != KvCacheError::None) {
        expect(false, "64-token append must succeed");
        return false;
    }

    return true;
}

void expectSources(
    KvCacheManager const& manager,
    std::array<PageHandle, kPromotionSourcePageCount> const& sources,
    PageState expected_state,
    std::uint32_t expected_refs,
    std::uint32_t expected_pins,
    char const* context)
{
    for (PageHandle const source : sources) {
        auto const metadata = manager.pageMetadata(source);

        expect(metadata.has_value(), context);

        if (!metadata.has_value()) {
            continue;
        }

        expect(metadata->state == expected_state, context);
        expect(metadata->valid_tokens == (
            expected_state == PageState::Free
                ? 0
                : kMicroPageTokenCapacity
        ), context);
        expect(metadata->ref_count == expected_refs, context);
        expect(metadata->promotion_pins == expected_pins, context);
        expect(metadata->mutable_owner == kInvalidRequestId, context);
    }
}

void testPrepareAndCommit()
{
    KvCacheManager manager(16, 2);

    if (!createFullPromotionRun(manager, 1)) {
        return;
    }

    auto const before_optional = manager.blockTable(1);
    expect(before_optional.has_value(), "pre-promotion table must exist");

    if (!before_optional.has_value()) {
        return;
    }

    BlockTable const before = *before_optional;
    expect(before.version() == 1, "initial table version must be 1");
    expect(before.entries().size() == 8, "eight Micro entries expected");

    PromotionPrepareResult const prepared =
        manager.preparePromotion(1, 0);

    expect(prepared.ok(), "promotion Prepare must succeed");
    expect(
        prepared.promotion_id != kInvalidPromotionId,
        "Prepare must return a valid transaction id"
    );
    expect(
        prepared.target_handle.kind == PageKind::Extent,
        "Prepare target must be an Extent page"
    );

    for (std::size_t index = 0;
         index < kPromotionSourcePageCount;
         ++index) {
        expect(
            prepared.source_handles[index]
                == before.entries()[index].handle,
            "Prepare must snapshot the exact source handles"
        );
    }

    auto const during_optional = manager.blockTable(1);
    auto const target_during =
        manager.pageMetadata(prepared.target_handle);
    auto const transaction =
        manager.promotionTransaction(prepared.promotion_id);

    expect(during_optional.has_value(), "table must exist during Prepare");
    expect(target_during.has_value(), "CopyTarget metadata must exist");
    expect(transaction.has_value(), "transaction snapshot must exist");

    if (during_optional.has_value()) {
        expect(
            sameTable(before, *during_optional),
            "Prepare must not change the visible BlockTable"
        );
    }

    if (target_during.has_value()) {
        expect(
            target_during->state == PageState::CopyTarget,
            "prepared Extent must be CopyTarget"
        );
        expect(
            target_during->valid_tokens == kExtentPageTokenCapacity,
            "CopyTarget must describe the full 64-token range"
        );
        expect(target_during->ref_count == 0, "CopyTarget must be invisible");
    }

    if (transaction.has_value()) {
        expect(
            transaction->prepared_table_version == before.version(),
            "transaction must snapshot the table version"
        );
        expect(
            transaction->target_handle == prepared.target_handle,
            "transaction must own the returned target"
        );
    }

    expectSources(
        manager,
        prepared.source_handles,
        PageState::Sealed,
        1,
        1,
        "prepared source must stay Sealed, referenced, and pinned"
    );
    expect(
        manager.snapshot().promotion_count == 1,
        "one active promotion must be reported"
    );
    expect(
        manager.checkInvariants(),
        "invariants must allow a prepared transaction"
    );

    expect(
        manager.commitPromotion(prepared.promotion_id)
            == KvCacheError::None,
        "promotion Commit must succeed"
    );

    auto const after_optional = manager.blockTable(1);
    auto const target_after = manager.pageMetadata(prepared.target_handle);

    expect(after_optional.has_value(), "committed table must exist");
    expect(target_after.has_value(), "committed target metadata must exist");
    expect(
        !manager.promotionTransaction(prepared.promotion_id).has_value(),
        "committed transaction must be removed"
    );

    if (after_optional.has_value()) {
        BlockTable const& after = *after_optional;

        expect(after.version() == 2, "Commit must increment version once");
        expect(after.tokenCount() == 64, "Commit must preserve token count");
        expect(after.entries().size() == 1, "Commit must create one entry");

        if (after.entries().size() == 1) {
            auto const& entry = after.entries()[0];

            expect(entry.logical_token_begin == 0, "Extent must begin at 0");
            expect(entry.valid_tokens == 64, "Extent must cover 64 tokens");
            expect(entry.kind == PageKind::Extent, "entry must be Extent");
            expect(
                entry.handle == prepared.target_handle,
                "entry must reference the prepared target"
            );
        }
    }

    if (target_after.has_value()) {
        expect(
            target_after->state == PageState::Sealed,
            "committed target must be Sealed"
        );
        expect(target_after->ref_count == 1, "Extent ref count must be 1");
        expect(target_after->promotion_pins == 0, "Extent must not be pinned");
    }

    expectSources(
        manager,
        prepared.source_handles,
        PageState::Free,
        0,
        0,
        "committed source pages must return to MicroPool"
    );
    expect(manager.snapshot().promotion_count == 0, "Commit must close txn");
    expect(
        manager.snapshot().micro_pool.allocated_slots == 0,
        "Commit must release all eight Micro pages"
    );
    expect(
        manager.snapshot().extent_pool.allocated_slots == 1,
        "Commit must retain the visible Extent page"
    );
    expect(manager.checkInvariants(), "committed invariants must hold");

    expect(
        manager.append(1, 1) == KvCacheError::None,
        "append after an Extent mapping must create a Micro tail"
    );

    auto const appended = manager.blockTable(1);
    expect(appended.has_value(), "post-promotion append table must exist");

    if (appended.has_value()) {
        expect(appended->entries().size() == 2, "Extent plus tail expected");
        expect(
            appended->entries()[0].handle == prepared.target_handle,
            "post-promotion append must retain the Extent"
        );
        expect(
            appended->entries()[1].logical_token_begin == 64,
            "new Micro tail must begin at token 64"
        );
    }

    expect(
        manager.releaseRequest(1) == KvCacheError::None,
        "promoted request release must succeed"
    );
    expect(
        manager.snapshot().extent_pool.allocated_slots == 0,
        "request release must return the Extent"
    );
    expect(manager.checkInvariants(), "post-release invariants must hold");
}

void testExplicitRollbackAndDuplicatePrepare()
{
    KvCacheManager manager(8, 1);

    if (!createFullPromotionRun(manager, 10)) {
        return;
    }

    BlockTable const before = *manager.blockTable(10);
    PromotionPrepareResult const prepared =
        manager.preparePromotion(10, 0);

    expect(prepared.ok(), "first Prepare must succeed");
    expect(
        manager.preparePromotion(10, 0).error
            == KvCacheError::PromotionNotEligible,
        "already pinned sources must reject duplicate Prepare"
    );
    expect(
        manager.snapshot().extent_pool.allocated_slots == 1,
        "duplicate Prepare must not leak an Extent target"
    );
    expect(
        manager.rollbackPromotion(prepared.promotion_id)
            == KvCacheError::None,
        "explicit Rollback must succeed"
    );

    auto const after = manager.blockTable(10);
    auto const target = manager.pageMetadata(prepared.target_handle);

    expect(after.has_value(), "table must remain after Rollback");
    expect(target.has_value(), "rolled-back target metadata must remain queryable");

    if (after.has_value()) {
        expect(
            sameTable(before, *after),
            "Rollback must preserve mappings and version"
        );
    }

    if (target.has_value()) {
        expect(target->state == PageState::Free, "target must return to pool");
        expect(target->valid_tokens == 0, "freed target must reset tokens");
    }

    expectSources(
        manager,
        prepared.source_handles,
        PageState::Sealed,
        1,
        0,
        "Rollback must preserve and unpin every source"
    );
    expect(
        manager.rollbackPromotion(prepared.promotion_id)
            == KvCacheError::PromotionNotFound,
        "closed transaction must reject a second Rollback"
    );
    expect(
        manager.commitPromotion(prepared.promotion_id)
            == KvCacheError::PromotionNotFound,
        "closed transaction must reject Commit"
    );
    expect(
        manager.rollbackPromotion(kInvalidPromotionId)
            == KvCacheError::InvalidArgument,
        "invalid transaction id must be rejected"
    );
    expect(manager.checkInvariants(), "rollback invariants must hold");
    expect(
        manager.releaseRequest(10) == KvCacheError::None,
        "rollback request release must succeed"
    );
}

void testExtentOutOfMemoryRollback()
{
    KvCacheManager manager(16, 1);

    if (!createFullPromotionRun(manager, 20)
        || !createFullPromotionRun(manager, 21)) {
        return;
    }

    PromotionPrepareResult const first = manager.preparePromotion(20, 0);
    expect(first.ok(), "first request must reserve the only Extent");

    BlockTable const second_before = *manager.blockTable(21);
    PromotionPrepareResult const exhausted =
        manager.preparePromotion(21, 0);

    expect(
        exhausted.error == KvCacheError::ResourceExhausted,
        "Extent exhaustion must be reported as ResourceExhausted"
    );
    expect(
        exhausted.promotion_id == kInvalidPromotionId,
        "failed Prepare must not publish a transaction id"
    );
    expect(
        !exhausted.target_handle.isStructurallyValid(),
        "failed Prepare must not publish a target"
    );

    auto const second_after = manager.blockTable(21);
    expect(second_after.has_value(), "OOM request table must remain");

    if (second_after.has_value()) {
        expect(
            sameTable(second_before, *second_after),
            "Extent OOM must preserve the old BlockTable"
        );
    }

    std::array<PageHandle, kPromotionSourcePageCount> second_sources{};
    for (std::size_t index = 0;
         index < kPromotionSourcePageCount;
         ++index) {
        second_sources[index] = second_before.entries()[index].handle;
    }

    expectSources(
        manager,
        second_sources,
        PageState::Sealed,
        1,
        0,
        "Extent OOM must not pin candidate sources"
    );
    expect(
        manager.snapshot().promotion_count == 1,
        "only the successful Prepare may remain active"
    );
    expect(manager.checkInvariants(), "Extent OOM invariants must hold");

    expect(
        manager.rollbackPromotion(first.promotion_id) == KvCacheError::None,
        "first transaction cleanup must succeed"
    );
    expect(manager.releaseRequest(21) == KvCacheError::None, "release 21");
    expect(manager.releaseRequest(20) == KvCacheError::None, "release 20");
    expect(manager.checkInvariants(), "OOM cleanup invariants must hold");
}

void testEligibilityAndSharedPageRejection()
{
    {
        KvCacheManager manager(8, 1);

        expect(
            manager.preparePromotion(kInvalidRequestId, 0).error
                == KvCacheError::InvalidArgument,
            "invalid request id must be rejected"
        );
        expect(
            manager.preparePromotion(999, 0).error
                == KvCacheError::RequestNotFound,
            "unknown request must be rejected"
        );
        expect(manager.createRequest(30) == KvCacheError::None, "create 30");
        expect(manager.append(30, 56) == KvCacheError::None, "append 56");
        expect(
            manager.preparePromotion(30, 0).error
                == KvCacheError::PromotionNotEligible,
            "fewer than eight pages must be ineligible"
        );
        expect(
            manager.preparePromotion(30, 1).error
                == KvCacheError::PromotionNotEligible,
            "logical begin must identify an entry boundary"
        );
        expect(manager.releaseRequest(30) == KvCacheError::None, "release 30");
    }

    {
        KvCacheManager manager(8, 1);

        if (!createFullPromotionRun(manager, 31)) {
            return;
        }

        expect(
            manager.forkRequest(31, 32) == KvCacheError::None,
            "full-page fork must succeed"
        );

        auto const table = manager.blockTable(31);
        expect(table.has_value(), "shared source table must exist");

        if (!table.has_value()) {
            return;
        }

        for (auto const& entry : table->entries()) {
            auto const metadata = manager.pageMetadata(entry.handle);
            expect(metadata.has_value(), "shared metadata must exist");

            if (metadata.has_value()) {
                expect(metadata->ref_count == 2, "source must be shared");
            }
        }

        expect(
            manager.preparePromotion(31, 0).error
                == KvCacheError::PromotionNotEligible,
            "shared Micro pages must not be promoted"
        );
        expect(
            manager.snapshot().extent_pool.allocated_slots == 0,
            "ineligible shared pages must not allocate a target"
        );
        expect(manager.checkInvariants(), "shared rejection invariants");
        expect(manager.releaseRequest(32) == KvCacheError::None, "release 32");
        expect(manager.releaseRequest(31) == KvCacheError::None, "release 31");
    }
}

void testVersionConflictAutoRollback()
{
    KvCacheManager manager(10, 1);

    if (!createFullPromotionRun(manager, 40)) {
        return;
    }

    PromotionPrepareResult const prepared =
        manager.preparePromotion(40, 0);
    expect(prepared.ok(), "conflict test Prepare must succeed");

    expect(
        manager.append(40, 1) == KvCacheError::None,
        "append during Copy must succeed and change version"
    );

    auto const changed_optional = manager.blockTable(40);
    expect(changed_optional.has_value(), "changed table must exist");

    if (!changed_optional.has_value()) {
        return;
    }

    BlockTable const changed = *changed_optional;
    expect(changed.version() == 2, "append must advance table version");
    expect(
        manager.checkInvariants(),
        "stale but active transaction must remain structurally valid"
    );
    expect(
        manager.commitPromotion(prepared.promotion_id)
            == KvCacheError::PromotionConflict,
        "version mismatch must reject Commit"
    );

    auto const after_optional = manager.blockTable(40);
    auto const target = manager.pageMetadata(prepared.target_handle);

    expect(after_optional.has_value(), "table must remain after conflict");
    expect(target.has_value(), "conflicted target metadata must exist");

    if (after_optional.has_value()) {
        expect(
            sameTable(changed, *after_optional),
            "conflict Rollback must preserve the newer table"
        );
    }

    if (target.has_value()) {
        expect(target->state == PageState::Free, "conflict must free target");
    }

    expectSources(
        manager,
        prepared.source_handles,
        PageState::Sealed,
        1,
        0,
        "conflict must preserve and unpin source pages"
    );
    expect(manager.snapshot().promotion_count == 0, "conflict must close txn");
    expect(manager.checkInvariants(), "conflict rollback invariants");
    expect(manager.releaseRequest(40) == KvCacheError::None, "release 40");
}

void testSharingConflictAfterPrepare()
{
    KvCacheManager manager(8, 1);

    if (!createFullPromotionRun(manager, 50)) {
        return;
    }

    PromotionPrepareResult const prepared =
        manager.preparePromotion(50, 0);
    expect(prepared.ok(), "sharing conflict Prepare must succeed");
    expect(
        manager.forkRequest(50, 51) == KvCacheError::None,
        "Fork may race after Prepare"
    );
    expect(
        manager.checkInvariants(),
        "pinned sources may become shared before Commit"
    );
    expect(
        manager.commitPromotion(prepared.promotion_id)
            == KvCacheError::PromotionConflict,
        "new sharing must reject Commit and auto-Rollback"
    );

    expectSources(
        manager,
        prepared.source_handles,
        PageState::Sealed,
        2,
        0,
        "sharing conflict must retain both logical references"
    );
    expect(manager.checkInvariants(), "sharing conflict invariants");
    expect(manager.releaseRequest(51) == KvCacheError::None, "release 51");
    expect(manager.releaseRequest(50) == KvCacheError::None, "release 50");
}

void testRequestReleaseRollsBackPromotion()
{
    KvCacheManager manager(8, 1);

    if (!createFullPromotionRun(manager, 60)) {
        return;
    }

    PromotionPrepareResult const prepared =
        manager.preparePromotion(60, 0);
    expect(prepared.ok(), "release test Prepare must succeed");

    expect(
        manager.releaseRequest(60) == KvCacheError::None,
        "request release must Rollback its active promotion"
    );

    auto const target = manager.pageMetadata(prepared.target_handle);
    expect(target.has_value(), "released target metadata must be queryable");

    if (target.has_value()) {
        expect(target->state == PageState::Free, "release must free target");
    }

    expectSources(
        manager,
        prepared.source_handles,
        PageState::Free,
        0,
        0,
        "request release must free every source after Rollback"
    );
    expect(
        manager.commitPromotion(prepared.promotion_id)
            == KvCacheError::PromotionNotFound,
        "released request transaction must no longer exist"
    );

    auto const snapshot = manager.snapshot();
    expect(snapshot.request_count == 0, "no request may remain");
    expect(snapshot.promotion_count == 0, "no promotion may remain");
    expect(snapshot.micro_pool.allocated_slots == 0, "all Micro pages free");
    expect(snapshot.extent_pool.allocated_slots == 0, "all Extent pages free");
    expect(manager.checkInvariants(), "release rollback invariants");
}

void testDisjointTransactionsConflictByVersion()
{
    KvCacheManager manager(16, 2);

    expect(manager.createRequest(70) == KvCacheError::None, "create 70");
    expect(manager.append(70, 128) == KvCacheError::None, "append 128");

    PromotionPrepareResult const first = manager.preparePromotion(70, 0);
    PromotionPrepareResult const second = manager.preparePromotion(70, 64);

    expect(first.ok(), "first disjoint Prepare must succeed");
    expect(second.ok(), "second disjoint Prepare must succeed");
    expect(
        manager.snapshot().promotion_count == 2,
        "both disjoint transactions may be prepared"
    );
    expect(manager.checkInvariants(), "two prepared transaction invariants");

    expect(
        manager.commitPromotion(first.promotion_id) == KvCacheError::None,
        "first disjoint Commit must succeed"
    );
    expect(
        manager.checkInvariants(),
        "other transaction must remain valid after an earlier Commit"
    );
    expect(
        manager.commitPromotion(second.promotion_id)
            == KvCacheError::PromotionConflict,
        "second transaction must detect the changed table version"
    );

    auto const table = manager.blockTable(70);
    expect(table.has_value(), "disjoint result table must exist");

    if (table.has_value()) {
        expect(table->version() == 2, "only one Commit may advance version");
        expect(table->tokenCount() == 128, "token count must stay 128");
        expect(table->entries().size() == 9, "one Extent plus eight Micro");
        expect(table->entries()[0].kind == PageKind::Extent, "first is Extent");
    }

    expectSources(
        manager,
        second.source_handles,
        PageState::Sealed,
        1,
        0,
        "conflicted disjoint sources must remain visible and unpinned"
    );
    expect(manager.snapshot().promotion_count == 0, "all txns must close");
    expect(manager.checkInvariants(), "disjoint conflict invariants");
    expect(manager.releaseRequest(70) == KvCacheError::None, "release 70");
    expect(manager.checkInvariants(), "disjoint cleanup invariants");
}

void testConcurrentCommitAndRequestRelease()
{
    constexpr std::uint32_t iteration_count = 32;

    for (std::uint32_t iteration = 0;
         iteration < iteration_count;
         ++iteration) {
        KvCacheManager manager(8, 1);
        RequestId const request_id = 1000 + iteration;

        if (!createFullPromotionRun(manager, request_id)) {
            return;
        }

        PromotionPrepareResult const prepared =
            manager.preparePromotion(request_id, 0);

        expect(prepared.ok(), "concurrent race Prepare must succeed");

        if (!prepared.ok()) {
            return;
        }

        std::atomic<bool> start{false};
        KvCacheError commit_error = KvCacheError::InvalidState;
        KvCacheError release_error = KvCacheError::InvalidState;

        std::thread commit_thread([&]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            commit_error = manager.commitPromotion(
                prepared.promotion_id
            );
        });

        std::thread release_thread([&]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            release_error = manager.releaseRequest(request_id);
        });

        start.store(true, std::memory_order_release);
        commit_thread.join();
        release_thread.join();

        expect(
            release_error == KvCacheError::None,
            "racing request release must always succeed"
        );
        expect(
            commit_error == KvCacheError::None
                || commit_error == KvCacheError::PromotionNotFound,
            "racing Commit must either win or observe release Rollback"
        );

        auto const snapshot = manager.snapshot();
        expect(snapshot.request_count == 0, "race must remove request");
        expect(snapshot.promotion_count == 0, "race must close transaction");
        expect(
            snapshot.micro_pool.allocated_slots == 0,
            "race must return every Micro page"
        );
        expect(
            snapshot.extent_pool.allocated_slots == 0,
            "race must return the Extent page"
        );
        expect(manager.checkInvariants(), "race invariants must hold");
    }
}

} // namespace

int main()
{
    testPrepareAndCommit();
    testExplicitRollbackAndDuplicatePrepare();
    testExtentOutOfMemoryRollback();
    testEligibilityAndSharedPageRejection();
    testVersionConflictAutoRollback();
    testSharingConflictAfterPrepare();
    testRequestReleaseRollsBackPromotion();
    testDisjointTransactionsConflictByVersion();
    testConcurrentCommitAndRequestRelease();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All Transactional Promotion/K3 contract tests passed\n";
    return 0;
}
