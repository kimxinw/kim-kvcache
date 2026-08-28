function(add_heteropage_kv_contract target_name test_name source_file timeout_seconds)
    add_executable(
        ${target_name}
        ${source_file}
    )

    target_link_libraries(
        ${target_name}
        PRIVATE
            heteropage_kv_warnings
            ${ARGN}
    )

    add_test(
        NAME ${test_name}
        COMMAND ${target_name}
    )

    set_tests_properties(
        ${test_name}
        PROPERTIES
            TIMEOUT ${timeout_seconds}
    )
endfunction()

add_heteropage_kv_contract(
    public_headers_test
    public_headers_contract
    api/public_headers_test.cpp
    10
    kim_kv
)

add_heteropage_kv_contract(
    page_pool_test
    page_pool_contract
    core/page_pool_test.cpp
    10
    kim_kv
)

add_heteropage_kv_contract(
    block_table_test
    block_table_contract
    runtime/block_table_test.cpp
    10
    kim_kv
)

add_heteropage_kv_contract(
    fixed_page_test
    fixed_page_contract
    runtime/fixed_page_test.cpp
    10
    kim_kv
)

add_heteropage_kv_contract(
    promotion_test
    promotion_contract
    runtime/promotion_test.cpp
    10
    kim_kv
)

add_heteropage_kv_contract(
    page_lease_test
    page_lease_contract
    runtime/page_lease_test.cpp
    10
    kim_kv
)

add_heteropage_kv_contract(
    kv_reference_test
    kv_reference_contract
    reference/kv_reference_test.cpp
    10
    kim_kv
)

if(HETEROPAGE_KV_ENABLE_CUDA)
    add_heteropage_kv_contract(
        cuda_kv_test
        cuda_kv_contract
        cuda/cuda_kv_test.cpp
        60
        kim_kv_cuda
        CUDA::cudart
    )

    if(HETEROPAGE_KV_BUILD_BENCHMARKS)
        foreach(fixed_page_tokens IN ITEMS 8 16 32 64)
            add_test(
                NAME fixed_cuda_benchmark_smoke_${fixed_page_tokens}
                COMMAND
                    kim_kv_cuda_benchmark
                    --workload short
                    --fixed-page-tokens ${fixed_page_tokens}
                    --warmup 0
                    --iterations 1
                    --max-sequence-length 65
                    --layers 2
                    --kv-heads 2
                    --head-dimension 8
                    --output-dir
                    ${CMAKE_CURRENT_BINARY_DIR}/fixed_cuda_${fixed_page_tokens}
            )
            set_tests_properties(
                fixed_cuda_benchmark_smoke_${fixed_page_tokens}
                PROPERTIES
                    TIMEOUT 30
            )
        endforeach()
    endif()
endif()

if(HETEROPAGE_KV_BUILD_BENCHMARKS)
    add_heteropage_kv_contract(
        benchmark_test
        benchmark_contract
        benchmark/benchmark_test.cpp
        60
        kim_kv_benchmark
    )
endif()
