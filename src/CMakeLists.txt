add_library(
    kim_kv
    STATIC
        core/block_table.cpp
        core/page_pool.cpp
        fixed/fixed_page_manager.cpp
        reference/kv_reference.cpp
        runtime/kv_cache_append.cpp
        runtime/kv_cache_inspection.cpp
        runtime/kv_cache_invariants.cpp
        runtime/kv_cache_lifecycle.cpp
        runtime/kv_cache_manager.cpp
        runtime/page_lease.cpp
        runtime/promotion.cpp
)

target_include_directories(
    kim_kv
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(
    kim_kv
    PRIVATE
        kim_kv_warnings
    PUBLIC
        Threads::Threads
        kim_kv_sanitizers
)

if(KIM_KV_ENABLE_CUDA)
    add_subdirectory(cuda)
endif()
