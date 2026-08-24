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
endif()
