cmake_minimum_required(VERSION 3.18)

project(kim_kv LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)

add_library(
    kim_kv
    STATIC
        src/page_pool.cpp
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
)

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
endif()
