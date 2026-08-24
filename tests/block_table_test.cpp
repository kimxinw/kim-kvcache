#include "heteropage_kv/kv_cache_manager.h"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

namespace {

using kimkvcache::BlockTable;
using kimkvcache::KvCacheError;
using kimkvcache::KvCacheManager;
using kimkvcache::PageHandle;
using kimkvcache::PageState;
using kimkvcache::RequestId;
using kimkvcache::kInvalidRequestId;

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

void testAppendAndLookup()
{
    KvCacheManager manager(8, 2);

    expect(
        manager.createRequest(1) == KvCacheError::None,
        "request creation must succeed"
    );
    expect(
        manager.append(1, 3) == KvCacheError::None,
        "append 3 tokens must succeed"
    );

    auto first_table = manager.blockTable(1);

    expect(first_table.has_value(), "request table must exist");

    if (!first_table.has_value()) {
        return;
    }

    expect(first_table->version() == 1, "first append version must be 1");
    expect(first_table->tokenCount() == 3, "token count must be 3");
    expect(first_table->entries().size() == 1, "one entry expected");
    expect(first_table->entries()[0].valid_tokens == 3, "tail must hold 3");
    expect(first_table->find(0) != nullptr, "token 0 must resolve");
    expect(first_table->find(2) != nullptr, "token 2 must resolve");
    expect(first_table->find(3) == nullptr, "token 3 must not resolve");

    PageHandle const first_handle = first_table->entries()[0].handle;
    auto first_metadata = manager.pageMetadata(first_handle);

    expect(first_metadata.has_value(), "metadata must exist");

    if (first_metadata.has_value()) {
        expect(
            first_metadata->state == PageState::Mutable,
            "partial tail must be Mutable"
        );
        expect(first_metadata->ref_count == 1, "ref count must be 1");
        expect(first_metadata->mutable_owner == 1, "owner must be request 1");
    }

    expect(
        manager.append(1, 5) == KvCacheError::None,
        "append filling the first page must succeed"
    );

    auto sealed_table = manager.blockTable(1);
    auto sealed_metadata = manager.pageMetadata(first_handle);

    expect(sealed_table.has_value(), "sealed table must exist");
    expect(sealed_metadata.has_value(), "sealed metadata must exist");

    if (!sealed_table.has_value() || !sealed_metadata.has_value()) {
        return;
    }

    expect(
        sealed_table->entries().size() == 1,
        "filling the tail must not allocate a new page"
    );
    expect(
        sealed_table->entries()[0].handle == first_handle,
        "filling must preserve the handle"
    );
    expect(
        sealed_table->entries()[0].valid_tokens == 8,
        "first page must contain 8 tokens"
    );
    expect(
        sealed_metadata->state == PageState::Sealed,
        "full page must be Sealed"
    );
    expect(
        sealed_metadata->mutable_owner == kInvalidRequestId,
        "sealed page must not have a mutable owner"
    );

    expect(
        manager.append(1, 9) == KvCacheError::None,
        "append crossing two pages must succeed"
    );

    auto final_table = manager.blockTable(1);

    expect(final_table.has_value(), "final table must exist");

    if (!final_table.has_value()) {
        return;
    }

    expect(final_table->version() == 3, "version must be 3");
    expect(final_table->tokenCount() == 17, "token count must be 17");
    expect(final_table->entries().size() == 3, "three entries expected");

    if (final_table->entries().size() == 3) {
        expect(
            final_table->entries()[0].logical_token_begin == 0
                && final_table->entries()[0].valid_tokens == 8,
            "first mapping must be [0, 8)"
        );
        expect(
            final_table->entries()[1].logical_token_begin == 8
                && final_table->entries()[1].valid_tokens == 8,
            "second mapping must be [8, 16)"
        );
        expect(
            final_table->entries()[2].logical_token_begin == 16
                && final_table->entries()[2].valid_tokens == 1,
            "third mapping must be [16, 17)"
        );
    }

    expect(final_table->checkInvariants(), "table invariants must hold");
    expect(manager.checkInvariants(), "manager invariants must hold");

    expect(
        manager.releaseRequest(1) == KvCacheError::None,
        "request release must succeed"
    );

    auto const snapshot = manager.snapshot();

    expect(snapshot.request_count == 0, "no request may remain");
    expect(
        snapshot.micro_pool.free_slots == snapshot.micro_pool.total_slots,
        "all micro pages must be returned"
    );
}

void testPartialForkAndCopyOnWrite()
{
    KvCacheManager manager(4, 1);

    expect(
        manager.createRequest(10) == KvCacheError::None,
        "source creation must succeed"
    );
    expect(
        manager.append(10, 3) == KvCacheError::None,
        "source append must succeed"
    );

    auto source_before_fork = manager.blockTable(10);

    expect(source_before_fork.has_value(), "source table must exist");

    if (!source_before_fork.has_value()) {
        return;
    }

    PageHandle const shared_handle =
        source_before_fork->entries()[0].handle;

    expect(
        manager.forkRequest(10, 11) == KvCacheError::None,
        "fork must succeed"
    );

    auto source_after_fork = manager.blockTable(10);
    auto child_after_fork = manager.blockTable(11);
    auto shared_metadata = manager.pageMetadata(shared_handle);

    expect(source_after_fork.has_value(), "source table must remain");
    expect(child_after_fork.has_value(), "child table must exist");
    expect(shared_metadata.has_value(), "shared metadata must exist");

    if (!source_after_fork.has_value()
        || !child_after_fork.has_value()
        || !shared_metadata.has_value()) {
        return;
    }

    expect(
        source_after_fork->entries()[0].handle == shared_handle,
        "source must retain shared handle"
    );
    expect(
        child_after_fork->entries()[0].handle == shared_handle,
        "child must receive shared handle"
    );
    expect(
        shared_metadata->state == PageState::Sealed,
        "fork must seal partial tail"
    );
    expect(shared_metadata->ref_count == 2, "shared ref count must be 2");
    expect(shared_metadata->shared(), "page must report shared");

    expect(
        manager.append(11, 2) == KvCacheError::None,
        "child append must perform COW"
    );

    auto source_after_child_cow = manager.blockTable(10);
    auto child_after_cow = manager.blockTable(11);

    expect(source_after_child_cow.has_value(), "source table must remain");
    expect(child_after_cow.has_value(), "child table must remain");

    if (!source_after_child_cow.has_value()
        || !child_after_cow.has_value()) {
        return;
    }

    PageHandle const child_handle = child_after_cow->entries()[0].handle;

    expect(
        source_after_child_cow->entries()[0].handle == shared_handle,
        "child COW must not alter source mapping"
    );
    expect(
        child_handle != shared_handle,
        "child COW must allocate a new handle"
    );
    expect(
        child_after_cow->entries()[0].valid_tokens == 5,
        "child COW page must contain copied and appended tokens"
    );

    auto old_metadata = manager.pageMetadata(shared_handle);
    auto child_metadata = manager.pageMetadata(child_handle);

    expect(old_metadata.has_value(), "old metadata must exist");
    expect(child_metadata.has_value(), "child metadata must exist");

    if (!old_metadata.has_value() || !child_metadata.has_value()) {
        return;
    }

    expect(old_metadata->ref_count == 1, "old page ref count must fall to 1");
    expect(
        old_metadata->state == PageState::Sealed,
        "old page must remain Sealed"
    );
    expect(
        child_metadata->state == PageState::Mutable,
        "new child tail must be Mutable"
    );
    expect(
        child_metadata->mutable_owner == 11,
        "child must own its COW target"
    );

    expect(
        manager.append(10, 1) == KvCacheError::None,
        "source append must also perform COW"
    );

    auto source_after_cow = manager.blockTable(10);

    expect(source_after_cow.has_value(), "source table must remain after COW");

    if (!source_after_cow.has_value()) {
        return;
    }

    PageHandle const source_handle =
        source_after_cow->entries()[0].handle;

    expect(
        source_handle != shared_handle,
        "source COW must allocate a new handle"
    );
    expect(
        source_after_cow->entries()[0].valid_tokens == 4,
        "source page must contain 4 tokens"
    );

    auto freed_old = manager.pageMetadata(shared_handle);

    expect(freed_old.has_value(), "freed generation metadata must remain queryable");

    if (freed_old.has_value()) {
        expect(
            freed_old->state == PageState::Free,
            "old shared page must become Free after the last reference"
        );
        expect(freed_old->ref_count == 0, "freed page ref count must be 0");
    }

    expect(manager.checkInvariants(), "COW invariants must hold");

    expect(
        manager.releaseRequest(11) == KvCacheError::None,
        "child release must succeed"
    );
    expect(
        manager.releaseRequest(10) == KvCacheError::None,
        "source release must succeed"
    );
    expect(manager.checkInvariants(), "post-release invariants must hold");
}

void testFullPageFork()
{
    KvCacheManager manager(3, 1);

    expect(
        manager.createRequest(20) == KvCacheError::None,
        "source creation must succeed"
    );
    expect(
        manager.append(20, 8) == KvCacheError::None,
        "full page append must succeed"
    );

    auto source = manager.blockTable(20);

    expect(source.has_value(), "source table must exist");

    if (!source.has_value()) {
        return;
    }

    PageHandle const shared = source->entries()[0].handle;

    expect(
        manager.forkRequest(20, 21) == KvCacheError::None,
        "full page fork must succeed"
    );
    expect(
        manager.append(21, 1) == KvCacheError::None,
        "child append after full page must succeed"
    );

    auto child = manager.blockTable(21);

    expect(child.has_value(), "child table must exist");

    if (!child.has_value()) {
        return;
    }

    expect(
        child->entries().size() == 2,
        "full shared page append must create a second entry"
    );
    expect(
        child->entries()[0].handle == shared,
        "full shared page must remain mapped"
    );
    expect(
        child->entries()[1].logical_token_begin == 8,
        "new tail must begin at token 8"
    );

    auto shared_metadata = manager.pageMetadata(shared);

    expect(shared_metadata.has_value(), "shared metadata must exist");

    if (shared_metadata.has_value()) {
        expect(shared_metadata->ref_count == 2, "full page must stay shared");
    }

    expect(manager.checkInvariants(), "full fork invariants must hold");

    expect(
        manager.releaseRequest(21) == KvCacheError::None,
        "child release must succeed"
    );
    expect(
        manager.releaseRequest(20) == KvCacheError::None,
        "source release must succeed"
    );
}

void testCowOutOfMemoryRollback()
{
    KvCacheManager manager(1, 1);

    expect(
        manager.createRequest(30) == KvCacheError::None,
        "source creation must succeed"
    );
    expect(
        manager.append(30, 3) == KvCacheError::None,
        "source append must succeed"
    );
    expect(
        manager.forkRequest(30, 31) == KvCacheError::None,
        "fork must succeed"
    );

    auto before_optional = manager.blockTable(31);

    expect(before_optional.has_value(), "child table must exist before OOM");

    if (!before_optional.has_value()) {
        return;
    }

    BlockTable const before = *before_optional;
    PageHandle const shared = before.entries()[0].handle;

    expect(
        manager.append(31, 1) == KvCacheError::ResourceExhausted,
        "COW must report ResourceExhausted"
    );

    auto after_optional = manager.blockTable(31);
    auto metadata = manager.pageMetadata(shared);

    expect(after_optional.has_value(), "child table must remain after OOM");
    expect(metadata.has_value(), "shared metadata must remain after OOM");

    if (!after_optional.has_value() || !metadata.has_value()) {
        return;
    }

    expect(
        sameTable(before, *after_optional),
        "failed COW must not change child BlockTable"
    );
    expect(metadata->ref_count == 2, "failed COW must preserve ref count");
    expect(
        metadata->state == PageState::Sealed,
        "failed COW must preserve Sealed state"
    );
    expect(
        manager.snapshot().micro_pool.allocated_slots == 1,
        "failed COW must not leak pages"
    );
    expect(manager.checkInvariants(), "OOM invariants must hold");

    expect(
        manager.releaseRequest(31) == KvCacheError::None,
        "child release must succeed"
    );
    expect(
        manager.releaseRequest(30) == KvCacheError::None,
        "source release must succeed"
    );
}

void testExplicitSealAndErrors()
{
    KvCacheManager manager(2, 1);

    expect(
        manager.createRequest(kInvalidRequestId) == KvCacheError::InvalidArgument,
        "request id zero must be rejected"
    );
    expect(
        manager.createRequest(40) == KvCacheError::None,
        "request creation must succeed"
    );
    expect(
        manager.createRequest(40) == KvCacheError::RequestAlreadyExists,
        "duplicate request must be rejected"
    );
    expect(
        manager.sealTail(40) == KvCacheError::InvalidState,
        "empty request cannot seal a tail"
    );
    expect(
        manager.append(40, 0) == KvCacheError::InvalidArgument,
        "zero-token append must be rejected"
    );
    expect(
        manager.append(40, 2) == KvCacheError::None,
        "append must succeed"
    );

    auto before_seal_optional = manager.blockTable(40);

    expect(before_seal_optional.has_value(), "table must exist before seal");

    if (!before_seal_optional.has_value()) {
        return;
    }

    BlockTable const before_seal = *before_seal_optional;
    PageHandle const old_handle = before_seal.entries()[0].handle;

    expect(
        manager.sealTail(40) == KvCacheError::None,
        "explicit seal must succeed"
    );

    auto after_seal = manager.blockTable(40);
    auto sealed_metadata = manager.pageMetadata(old_handle);

    expect(after_seal.has_value(), "table must remain after seal");
    expect(sealed_metadata.has_value(), "sealed metadata must exist");

    if (!after_seal.has_value() || !sealed_metadata.has_value()) {
        return;
    }

    expect(
        after_seal->version() == before_seal.version(),
        "seal without mapping change must not increment version"
    );
    expect(
        sealed_metadata->state == PageState::Sealed,
        "explicit seal must change page state"
    );

    expect(
        manager.append(40, 1) == KvCacheError::None,
        "append after explicit seal must perform COW"
    );

    auto after_append = manager.blockTable(40);

    expect(after_append.has_value(), "table must exist after COW");

    if (after_append.has_value()) {
        expect(
            after_append->entries()[0].handle != old_handle,
            "sealed partial tail must not be modified in place"
        );
        expect(
            after_append->entries()[0].valid_tokens == 3,
            "COW result must contain 3 tokens"
        );
    }

    expect(
        manager.forkRequest(40, 40) == KvCacheError::InvalidArgument,
        "self fork must be rejected"
    );
    expect(
        manager.releaseRequest(999) == KvCacheError::RequestNotFound,
        "unknown release must be rejected"
    );
    expect(
        manager.releaseRequest(40) == KvCacheError::None,
        "request release must succeed"
    );
}

void testConcurrentRequests()
{
    constexpr std::uint32_t thread_count = 4;
    constexpr std::uint32_t tokens_per_request = 17;

    KvCacheManager manager(thread_count * 3, 1);
    std::atomic<std::uint32_t> concurrent_failures{0};

    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (std::uint32_t index = 0; index < thread_count; ++index) {
        threads.emplace_back([&manager, &concurrent_failures, index]() {
            RequestId const request_id = 100 + index;

            if (manager.createRequest(request_id) != KvCacheError::None) {
                ++concurrent_failures;
                return;
            }

            if (manager.append(request_id, tokens_per_request)
                != KvCacheError::None) {
                ++concurrent_failures;
            }
        });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    expect(
        concurrent_failures.load() == 0,
        "concurrent requests must succeed"
    );
    expect(
        manager.snapshot().request_count == thread_count,
        "all concurrent requests must remain registered"
    );
    expect(
        manager.checkInvariants(),
        "concurrent manager invariants must hold"
    );

    for (std::uint32_t index = 0; index < thread_count; ++index) {
        expect(
            manager.releaseRequest(100 + index) == KvCacheError::None,
            "concurrent request cleanup must succeed"
        );
    }

    expect(
        manager.snapshot().micro_pool.allocated_slots == 0,
        "concurrent cleanup must return all pages"
    );
    expect(
        manager.checkInvariants(),
        "post-concurrency invariants must hold"
    );
}

} // namespace

int main()
{
    testAppendAndLookup();
    testPartialForkAndCopyOnWrite();
    testFullPageFork();
    testCowOutOfMemoryRollback();
    testExplicitSealAndErrors();
    testConcurrentRequests();

    if (failures != 0) {
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }

    std::cout << "All BlockTable/K2 contract tests passed\n";
    return 0;
}
