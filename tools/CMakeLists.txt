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
