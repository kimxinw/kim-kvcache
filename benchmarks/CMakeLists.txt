set(HETEROPAGE_KV_GIT_COMMIT "unknown")
find_package(Git QUIET)

if(GIT_FOUND)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        OUTPUT_VARIABLE HETEROPAGE_KV_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE HETEROPAGE_KV_GIT_RESULT
    )

    if(NOT HETEROPAGE_KV_GIT_RESULT EQUAL 0)
        set(HETEROPAGE_KV_GIT_COMMIT "unknown")
    endif()
endif()

add_library(
    kim_kv_benchmark
    STATIC
        common/benchmark.cpp
        common/capacity.cpp
        common/command_line.cpp
        common/cpu_benchmark.cpp
        common/cpu_benchmark_fault.cpp
        common/cpu_benchmark_independent.cpp
        common/cpu_benchmark_sharing.cpp
        common/cpu_benchmark_support.cpp
        common/latency.cpp
        common/report.cpp
        common/workload.cpp
)

target_include_directories(
    kim_kv_benchmark
    PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
)

target_compile_definitions(
    kim_kv_benchmark
    PRIVATE
        HETEROPAGE_KV_GIT_COMMIT="${HETEROPAGE_KV_GIT_COMMIT}"
)

target_link_libraries(
    kim_kv_benchmark
    PRIVATE
        heteropage_kv_warnings
    PUBLIC
        kim_kv
)

add_executable(
    kim_kv_cpu_benchmark
    cpu/main.cpp
)

target_link_libraries(
    kim_kv_cpu_benchmark
    PRIVATE
        heteropage_kv_warnings
        kim_kv_benchmark
)

if(HETEROPAGE_KV_ENABLE_CUDA)
    add_executable(
        kim_kv_cuda_benchmark
        cuda/benchmark.cpp
        cuda/benchmark_fault.cpp
        cuda/benchmark_independent.cpp
        cuda/benchmark_sharing.cpp
        cuda/benchmark_support.cpp
        cuda/benchmark_workload.cpp
        cuda/main.cpp
    )

    target_link_libraries(
        kim_kv_cuda_benchmark
        PRIVATE
            heteropage_kv_warnings
            kim_kv_benchmark
            kim_kv_cuda
            CUDA::cudart
    )
endif()
