#include "heteropage_kv/fixed/fixed_page_manager.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using kimkvcache::FixedPageManager;
using kimkvcache::FixedPageManagerSnapshot;
using kimkvcache::KvCacheError;
using kimkvcache::kInvalidRequestId;

int failures = 0;

void expect(bool condition, char const* message)
{
    if (!condition) {
        std::cerr << "[FAILED] " << message << '\n';
        ++failures;
    }
}

void testAppendPackingAndLookup()
{
    FixedPageManager manager(8, 16);

    expect(manager.tokensPerPage() == 8, "configured page size is exposed");

    expect(
        manager.createRequest(1) == KvCacheError::None,
        "request creation must succeed"
    );
    expect(
        manager.createRequest(1)
            == KvCacheError::RequestAlreadyExists,
        "duplicate creation must be rejected"
    );
    expect(
        manager.append(kInvalidRequestId, 1)
            == KvCacheError::InvalidArgument,
        "invalid request id must be rejected"
    );
    expect(
        manager.append(2, 1) == KvCacheError::RequestNotFound,
        "append to unknown request must fail"
    );
    expect(
        manager.append(1, 0) == KvCacheError::InvalidArgument,
        "zero-token append must be rejected"
    );

    expect(
        manager.append(1, 20) == KvCacheError::None,
        "append 20 tokens must succeed"
    );

    auto table = manager.blockTable(1);
    expect(table.has_value(), "request table must exist");

    if (table.has_value()) {
        expect(table->version() == 1, "first append version must be 1");
        expect(table->tokenCount() == 20, "token count must be 20");
        expect(table->entries().size() == 3, "three pages expected");
        expect(
            table->entries()[0].valid_tokens == 8
                && table->entries()[1].valid_tokens == 8
                && table->entries()[2].valid_tokens == 4,
            "page packing must follow the fixed capacity"
        );
        expect(table->find(0) != nullptr, "token 0 must resolve");
        expect(table->find(19) != nullptr, "token 19 must resolve");
        expect(table->find(20) == nullptr, "token 20 must not resolve");
    }

    // 剩余槽位填满第 3 页，该页转为 Sealed。
    expect(
        manager.append(1, 4) == KvCacheError::None,
        "append filling the tail page must succeed"
    );
    expect(manager.sealTail(1) == KvCacheError::None, "seal idempotent");

    auto sealed_table = manager.blockTable(1);
    expect(sealed_table.has_value(), "sealed table must exist");

    if (sealed_table.has_value()) {
        expect(
            sealed_table->entries().size() == 3,
            "full tail must not add an empty page"
        );
        expect(sealed_table->find(23) != nullptr, "token 23 must resolve");
        expect(sealed_table->find(24) == nullptr, "token 24 must not resolve");
    }

    expect(manager.checkInvariants(), "invariants must hold");
}

void testCowOnSealedPartialTail()
{
    FixedPageManager manager(8, 16);

    static_cast<void>(manager.createRequest(1));
    static_cast<void>(manager.append(1, 5));

    expect(
        manager.forkRequest(1, 2) == KvCacheError::None,
        "fork of partial tail must succeed"
    );
    expect(
        manager.append(2, 0) == KvCacheError::InvalidArgument,
        "zero append on child must be rejected"
    );

    // Fork 会先 Seal 尾页；对父请求的写入必须触发 Partial Tail COW。
    expect(
        manager.append(1, 3) == KvCacheError::None,
        "append onto sealed tail must copy-on-write"
    );

    auto parent = manager.blockTable(1);
    auto child = manager.blockTable(2);
    expect(parent.has_value() && child.has_value(), "both tables exist");

    if (parent.has_value()) {
        expect(
            parent->tokenCount() == 8 && parent->entries().size() == 1,
            "cow tail must fill into one page"
        );
    }
    if (child.has_value()) {
        expect(
            child->tokenCount() == 5 && child->entries().size() == 1,
            "child table must stay unaffected"
        );
        expect(child->version() == 1, "child keeps its own version");
    }

    expect(manager.checkInvariants(), "invariants after cow must hold");
}

void testAppendOomRollback()
{
    FixedPageManager manager(8, 2);

    expect(
        manager.createRequest(1) == KvCacheError::None,
        "request creation must succeed"
    );
    expect(
        manager.append(1, 24) == KvCacheError::ResourceExhausted,
        "append beyond pool must report exhaustion"
    );

    FixedPageManagerSnapshot const snapshot = manager.snapshot();

    expect(snapshot.pool.free_slots == 2, "staged pages must be rolled back");
    expect(
        snapshot.pool.allocated_slots == 0,
        "no page may survive a failed append"
    );
    expect(
        snapshot.peak_allocated_pages == 2,
        "peak allocation must retain rolled-back staging pressure"
    );
    expect(
        manager.checkInvariants(),
        "invariants must hold after oom rollback"
    );

    auto table = manager.blockTable(1);
    expect(table.has_value(), "request must survive oom");

    if (table.has_value()) {
        expect(
            table->tokenCount() == 0 && table->entries().empty(),
            "failed append must not change mapping"
        );
    }

    expect(
        manager.append(1, 16) == KvCacheError::None,
        "exact-fit append must succeed after rollback"
    );
}

void testForkReleaseConservation()
{
    FixedPageManager manager(8, 8);

    static_cast<void>(manager.createRequest(1));
    static_cast<void>(manager.append(1, 16));
    static_cast<void>(manager.forkRequest(1, 2));
    static_cast<void>(manager.forkRequest(1, 3));

    FixedPageManagerSnapshot const shared = manager.snapshot();
    expect(shared.pool.allocated_slots == 2, "forks must share pages");

    expect(
        manager.releaseRequest(1) == KvCacheError::None,
        "releasing the source must keep children alive"
    );
    expect(
        manager.snapshot().pool.allocated_slots == 2,
        "shared pages must not be freed early"
    );

    static_cast<void>(manager.releaseRequest(2));

    FixedPageManagerSnapshot const last_child = manager.snapshot();
    expect(
        last_child.pool.allocated_slots == 2,
        "pages stay until the final owner leaves"
    );
    expect(last_child.request_count == 1, "one request remains");

    static_cast<void>(manager.releaseRequest(3));

    FixedPageManagerSnapshot const drained = manager.snapshot();
    expect(drained.pool.allocated_slots == 0, "pool must drain");
    expect(
        drained.pool.capacityBalanced(),
        "pool accounting must balance"
    );
    expect(drained.request_count == 0, "no requests may remain");
    expect(manager.checkInvariants(), "final invariants must hold");
}

void testLargePageAndRejectPath()
{
    FixedPageManager manager(64, 4);

    static_cast<void>(manager.createRequest(1));
    expect(
        manager.append(1, 100) == KvCacheError::None,
        "100-token append must support a 64-token page size"
    );

    auto table = manager.blockTable(1);
    expect(table.has_value(), "large-page table exists");

    if (table.has_value()) {
        expect(
            table->entries().size() == 2,
            "100 tokens over 64-capacity pages needs two entries"
        );
        expect(
            table->checkInvariants(64),
            "fixed-capacity BlockTable invariants must accept Fixed-64"
        );
        if (table->entries().size() >= 2) {
            expect(
                table->entries()[1].valid_tokens == 36,
                "tail carries the remainder"
            );
        }
    }

    expect(
        manager.sealTail(9) == KvCacheError::RequestNotFound,
        "seal on unknown request must fail"
    );
    expect(
        manager.releaseRequest(9) == KvCacheError::RequestNotFound,
        "release on unknown request must fail"
    );

    static_cast<void>(manager.createRequest(2));
    expect(
        manager.append(1, 500) == KvCacheError::ResourceExhausted,
        "append exceeding total budget must fail atomically"
    );
    expect(manager.checkInvariants(), "reject path keeps invariants");
}

void testInvalidConstruction()
{
    bool rejected = false;
    try {
        FixedPageManager invalid(0, 1);
        static_cast<void>(invalid);
    } catch (std::invalid_argument const&) {
        rejected = true;
    }
    expect(rejected, "zero-token page capacity must be rejected");
}

} // namespace

int main()
{
    testAppendPackingAndLookup();
    testCowOnSealedPartialTail();
    testAppendOomRollback();
    testForkReleaseConservation();
    testLargePageAndRejectPath();
    testInvalidConstruction();

    if (failures != 0) {
        return 1;
    }

    std::cout << "K6 fixed_page_contract passed\n";
    return 0;
}
