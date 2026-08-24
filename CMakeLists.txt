cmake_minimum_required(VERSION 3.18)

project(kim_kv LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Sanitizer options
#
# ASan 和 UBSan 可以同时启用。
# TSan 必须使用独立构建目录，不能与 ASan/UBSan 同时启用。
# ---------------------------------------------------------------------------

option(
    HETEROPAGE_KV_ENABLE_ASAN
    "Enable AddressSanitizer"
    OFF
)

option(
    HETEROPAGE_KV_ENABLE_UBSAN
    "Enable UndefinedBehaviorSanitizer"
    OFF
)

option(
    HETEROPAGE_KV_ENABLE_TSAN
    "Enable ThreadSanitizer"
    OFF
)

option(
    HETEROPAGE_KV_ENABLE_CUDA
    "Build K4 CUDA storage and correctness contracts"
    OFF
)

if(
    HETEROPAGE_KV_ENABLE_TSAN
    AND
    (
        HETEROPAGE_KV_ENABLE_ASAN
        OR
        HETEROPAGE_KV_ENABLE_UBSAN
    )
)
    message(
        FATAL_ERROR
        "TSan must use a separate build from ASan/UBSan"
    )
endif()

set(HETEROPAGE_KV_SANITIZERS "")

if(HETEROPAGE_KV_ENABLE_ASAN)
    list(APPEND HETEROPAGE_KV_SANITIZERS address)
endif()

if(HETEROPAGE_KV_ENABLE_UBSAN)
    list(APPEND HETEROPAGE_KV_SANITIZERS undefined)
endif()

if(HETEROPAGE_KV_ENABLE_TSAN)
    list(APPEND HETEROPAGE_KV_SANITIZERS thread)
endif()

# Interface Target 统一保存 Sanitizer 的编译和链接参数。
# 链接到该 Target 的库和测试程序会自动获得相同配置。
add_library(heteropage_kv_sanitizers INTERFACE)

if(NOT "${HETEROPAGE_KV_SANITIZERS}" STREQUAL "")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(
            FATAL_ERROR
            "Sanitizers currently require GCC or Clang"
        )
    endif()

    list(
        JOIN
        HETEROPAGE_KV_SANITIZERS
        ","
        HETEROPAGE_KV_SANITIZER_LIST
    )

    message(
        STATUS
        "Enabled sanitizers: ${HETEROPAGE_KV_SANITIZER_LIST}"
    )

    target_compile_options(
        heteropage_kv_sanitizers
        INTERFACE
            "-fsanitize=${HETEROPAGE_KV_SANITIZER_LIST}"
            -fno-omit-frame-pointer
            -fno-sanitize-recover=all
    )

    target_link_options(
        heteropage_kv_sanitizers
        INTERFACE
            "-fsanitize=${HETEROPAGE_KV_SANITIZER_LIST}"
    )
endif()

# ---------------------------------------------------------------------------
# HeteroPageKV library
# ---------------------------------------------------------------------------

add_library(
    kim_kv
    STATIC
        src/page_pool.cpp
        src/block_table.cpp
        src/kv_cache_manager.cpp
        src/promotion.cpp
        src/page_lease.cpp
        src/kv_reference.cpp
)

target_include_directories(
    kim_kv
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_options(
    kim_kv
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_link_libraries(
    kim_kv
    PUBLIC
        Threads::Threads
        heteropage_kv_sanitizers
)

# ---------------------------------------------------------------------------
# Optional K4 CUDA data path
# ---------------------------------------------------------------------------

if(HETEROPAGE_KV_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(NOT CMAKE_CUDA_COMPILER)
        message(
            FATAL_ERROR
            "HETEROPAGE_KV_ENABLE_CUDA=ON requires a CUDA compiler"
        )
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(
        kim_kv_cuda
        STATIC
            cuda/cuda_kv_storage.cu
            cuda/cuda_kv_cache.cpp
    )

    set_target_properties(
        kim_kv_cuda
        PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(
        kim_kv_cuda
        PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

    target_compile_options(
        kim_kv_cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
    )

    target_link_libraries(
        kim_kv_cuda
        PUBLIC
            kim_kv
            CUDA::cudart
    )
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

if(BUILD_TESTING)
    add_executable(
        page_pool_test
        tests/page_pool_test.cpp
    )

    target_compile_options(
        page_pool_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_link_libraries(
        page_pool_test
        PRIVATE
            kim_kv
    )

    add_test(
        NAME page_pool_contract
        COMMAND page_pool_test
    )

    set_tests_properties(
        page_pool_contract
        PROPERTIES
            TIMEOUT 10
    )

    add_executable(
        block_table_test
        tests/block_table_test.cpp
    )

    target_compile_options(
        block_table_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_link_libraries(
        block_table_test
        PRIVATE
            kim_kv
    )

    add_test(
        NAME block_table_contract
        COMMAND block_table_test
    )

    set_tests_properties(
        block_table_contract
        PROPERTIES
            TIMEOUT 10
    )

    add_executable(
        promotion_test
        tests/promotion_test.cpp
    )

    target_compile_options(
        promotion_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_link_libraries(
        promotion_test
        PRIVATE
            kim_kv
    )

    add_test(
        NAME promotion_contract
        COMMAND promotion_test
    )

    set_tests_properties(
        promotion_contract
        PROPERTIES
            TIMEOUT 10
    )

    add_executable(
        page_lease_test
        tests/page_lease_test.cpp
    )

    target_compile_options(
        page_lease_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_link_libraries(
        page_lease_test
        PRIVATE
            kim_kv
    )

    add_test(
        NAME page_lease_contract
        COMMAND page_lease_test
    )

    set_tests_properties(
        page_lease_contract
        PROPERTIES
            TIMEOUT 10
    )

    add_executable(
        kv_reference_test
        tests/kv_reference_test.cpp
    )

    target_compile_options(
        kv_reference_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_link_libraries(
        kv_reference_test
        PRIVATE
            kim_kv
    )

    add_test(
        NAME kv_reference_contract
        COMMAND kv_reference_test
    )

    set_tests_properties(
        kv_reference_contract
        PROPERTIES
            TIMEOUT 10
    )

    if(HETEROPAGE_KV_ENABLE_CUDA)
        add_executable(
            cuda_kv_test
            tests/cuda_kv_test.cpp
        )

        target_compile_options(
            cuda_kv_test
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )

        target_link_libraries(
            cuda_kv_test
            PRIVATE
                kim_kv_cuda
                CUDA::cudart
        )

        add_test(
            NAME cuda_kv_contract
            COMMAND cuda_kv_test
        )

        set_tests_properties(
            cuda_kv_contract
            PROPERTIES
                TIMEOUT 60
        )
    endif()
endif()
