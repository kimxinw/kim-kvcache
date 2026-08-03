#include "heteropage_kv/page_pool.h"

  #include <atomic>
  #include <cstdint>
  #include <iostream>
  #include <stdexcept>
  #include <thread>
  #include <type_traits>
  #include <vector>

  namespace {

  using kimkvcache::PageHandle;
  using kimkvcache::PageKind;
  using kimkvcache::PagePool;
  using kimkvcache::PagePoolError;

  int failures = 0;

  void expect(bool condition, char const* message)
  {
      if (!condition) {
          std::cerr << "[FAILED] " << message << '\n';
          ++failures;
      }
  }

  void testInvalidConfiguration()
  {
      bool zero_capacity_rejected = false;

      try {
          PagePool pool(PageKind::Micro, 0);
      } catch (std::invalid_argument const&) {
          zero_capacity_rejected = true;
      }

      expect(
          zero_capacity_rejected,
          "zero capacity must be rejected"
      );

      bool unknown_kind_rejected = false;

      try {
          PagePool pool(static_cast<PageKind>(255), 1);
      } catch (std::invalid_argument const&) {
          unknown_kind_rejected = true;
      }

      expect(
          unknown_kind_rejected,
          "unknown PageKind must be rejected"
      );
  }

  void testAllocateReleaseAndExhaustion()
  {
      PagePool pool(PageKind::Micro, 2);

      auto const initial = pool.snapshot();

      expect(initial.total_slots == 2, "initial total must be 2");
      expect(initial.free_slots == 2, "initial free must be 2");
      expect(initial.allocated_slots == 0, "initial allocated must be 0");
      expect(initial.capacityBalanced(), "initial capacity must balance");

      auto const first = pool.allocate();
      auto const second = pool.allocate();
      auto const exhausted = pool.allocate();

      expect(first.ok(), "first allocation must succeed");
      expect(second.ok(), "second allocation must succeed");
      expect(
          exhausted.error == PagePoolError::Exhausted,
          "third allocation must report Exhausted"
      );
      expect(
          !exhausted.handle.isStructurallyValid(),
          "failed allocation must return invalid handle"
      );

      auto const full = pool.snapshot();

      expect(full.free_slots == 0, "full pool must have no free slots");
      expect(full.allocated_slots == 2, "full pool must have 2 allocations");
      expect(full.failed_allocations == 1, "failed allocation must be counted");
      expect(full.capacityBalanced(), "full capacity must balance");

      if (first.ok()) {
          expect(
              pool.release(first.handle) == PagePoolError::None,
              "first release must succeed"
          );
      }

      if (second.ok()) {
          expect(
              pool.release(second.handle) == PagePoolError::None,
              "second release must succeed"
          );
      }

      auto const released = pool.snapshot();

      expect(released.free_slots == 2, "all slots must be returned");
      expect(released.allocated_slots == 0, "no allocation must remain");
      expect(released.capacityBalanced(), "released capacity must balance");
      expect(pool.checkInvariants(), "pool invariants must hold");
  }

  void testGenerationAndInvalidRelease()
  {
      PagePool pool(PageKind::Micro, 1);

      auto const first = pool.allocate();

      expect(first.ok(), "initial allocation must succeed");

      if (!first.ok()) {
          return;
      }

      expect(
          pool.release(first.handle) == PagePoolError::None,
          "initial release must succeed"
      );

      expect(
          pool.release(first.handle) == PagePoolError::AlreadyFree,
          "double free must be rejected"
      );

      auto const second = pool.allocate();

      expect(second.ok(), "reallocation must succeed");

      if (!second.ok()) {
          return;
      }

      expect(
          second.handle.slot == first.handle.slot,
          "single-slot pool must reuse the same slot"
      );

      expect(
          second.handle.generation != first.handle.generation,
          "reallocated slot must have a new generation"
      );

      expect(
          pool.validate(first.handle) == PagePoolError::StaleGeneration,
          "old handle must be stale after reallocation"
      );

      expect(
          pool.release(first.handle) == PagePoolError::StaleGeneration,
          "stale handle release must be rejected"
      );

      PageHandle wrong_kind = second.handle;
      wrong_kind.kind = PageKind::Extent;

      expect(
          pool.release(wrong_kind) == PagePoolError::WrongKind,
          "wrong PageKind must be rejected"
      );

      PageHandle out_of_range{
          PageKind::Micro,
          pool.capacity(),
          second.handle.generation,
      };

      expect(
          pool.release(out_of_range) == PagePoolError::OutOfRange,
          "out-of-range slot must be rejected"
      );

      expect(
          pool.release(PageHandle::invalid()) == PagePoolError::InvalidHandle,
          "invalid handle must be rejected"
      );

      expect(
          pool.release(second.handle) == PagePoolError::None,
          "current handle release must succeed"
      );

      expect(pool.checkInvariants(), "pool invariants must hold");
  }

  void testConcurrentAllocateRelease()
  {
      constexpr std::uint32_t thread_count = 4;
      constexpr std::uint32_t iterations = 1000;

      PagePool pool(PageKind::Micro, thread_count);
      std::atomic<std::uint32_t> concurrent_failures{0};

      std::vector<std::thread> threads;
      threads.reserve(thread_count);

      for (std::uint32_t thread = 0; thread < thread_count; ++thread) {
          threads.emplace_back([&pool, &concurrent_failures]() {
              for (std::uint32_t iteration = 0;
                   iteration < iterations;
                   ++iteration) {
                  auto const allocation = pool.allocate();

                  if (!allocation.ok()) {
                      ++concurrent_failures;
                      continue;
                  }

                  if (pool.validate(allocation.handle) != PagePoolError::None) {
                      ++concurrent_failures;
                  }

                  if (pool.release(allocation.handle) != PagePoolError::None) {
                      ++concurrent_failures;
                  }
              }
          });
      }

      for (auto& thread : threads) {
          thread.join();
      }

      auto const snapshot = pool.snapshot();
      auto const expected_operations =
          static_cast<std::uint64_t>(thread_count) * iterations;

      expect(
          concurrent_failures.load() == 0,
          "concurrent operations must not fail"
      );
      expect(
          snapshot.successful_allocations == expected_operations,
          "all concurrent allocations must be counted"
      );
      expect(
          snapshot.successful_releases == expected_operations,
          "all concurrent releases must be counted"
      );
      expect(
          snapshot.free_slots == thread_count,
          "all slots must be returned after concurrent test"
      );
      expect(
          snapshot.allocated_slots == 0,
          "no concurrent allocation may remain"
      );
      expect(
          snapshot.capacityBalanced(),
          "concurrent capacity must balance"
      );
      expect(
          pool.checkInvariants(),
          "concurrent pool invariants must hold"
      );
  }

void testExtentPool()
{
    PagePool pool(PageKind::Extent, 2);

    expect(
        pool.kind() == PageKind::Extent,
        "extent pool must report Extent kind"
    );

    auto const allocation = pool.allocate();

    expect(
        allocation.ok(),
        "extent allocation must succeed"
    );

    if (!allocation.ok()) {
        return;
    }

    expect(
        allocation.handle.kind == PageKind::Extent,
        "extent allocation must return an Extent handle"
    );

    expect(
        pool.validate(allocation.handle) == PagePoolError::None,
        "extent handle must be valid"
    );

    auto const allocated = pool.snapshot();

    expect(
        allocated.kind == PageKind::Extent,
        "extent snapshot must report Extent kind"
    );

    expect(
        allocated.total_slots == 2,
        "extent pool total must be 2"
    );

    expect(
        allocated.free_slots == 1,
        "extent pool must have one free slot"
    );

    expect(
        allocated.allocated_slots == 1,
        "extent pool must have one allocated slot"
    );

    expect(
        allocated.capacityBalanced(),
        "extent pool capacity must balance"
    );

    expect(
        pool.release(allocation.handle) == PagePoolError::None,
        "extent release must succeed"
    );

    auto const released = pool.snapshot();

    expect(
        released.free_slots == 2,
        "extent slot must be returned"
    );

    expect(
        released.allocated_slots == 0,
        "extent pool must have no remaining allocation"
    );

    expect(
        pool.checkInvariants(),
        "extent pool invariants must hold"
    );
}

  } // namespace

  static_assert(!std::is_copy_constructible_v<PagePool>);
  static_assert(!std::is_copy_assignable_v<PagePool>);
  static_assert(!std::is_move_constructible_v<PagePool>);
  static_assert(!std::is_move_assignable_v<PagePool>);

  int main()
  {
      testInvalidConfiguration();
      testAllocateReleaseAndExhaustion();
      testExtentPool();
      testGenerationAndInvalidRelease();
      testConcurrentAllocateRelease();

      if (failures != 0) {
          std::cerr << failures << " PagePool test(s) failed\n";
          return 1;
      }

      std::cout << "All PagePool tests passed\n";
      return 0;
  }