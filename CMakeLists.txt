cmake_minimum_required(VERSION 3.18)

project(kim_kv LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(HeteroPageKvOptions)
include(HeteroPageKvWarnings)
include(HeteroPageKvSanitizers)

find_package(Threads REQUIRED)

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
endif()

add_subdirectory(src)

if(HETEROPAGE_KV_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
