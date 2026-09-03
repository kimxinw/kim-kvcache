add_executable(
    kim_kv_tinyllama_validate
    tinyllama_validate.cpp
)

target_link_libraries(
    kim_kv_tinyllama_validate
    PRIVATE
        kim_kv_warnings
        kim_kv_cuda
        CUDA::cudart
)

add_executable(
    kim_kv_tinyllama_generate
    tinyllama_generate.cpp
)

target_link_libraries(
    kim_kv_tinyllama_generate
    PRIVATE
        kim_kv_warnings
        kim_kv_cuda
        CUDA::cudart
)

add_executable(
    kim_kv_tinyllama_e2e_benchmark
    tinyllama_e2e_benchmark.cpp
)

target_link_libraries(
    kim_kv_tinyllama_e2e_benchmark
    PRIVATE
        kim_kv_warnings
        kim_kv_cuda
        CUDA::cudart
)
