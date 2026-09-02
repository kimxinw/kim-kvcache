add_library(
    kim_kv_cuda
    STATIC
        cuda_kv_cache.cpp
        cuda_engine_transaction.cpp
        cuda_engine_kv_backend.cpp
        cuda_kv_append.cpp
        cuda_kv_attention.cpp
        cuda_kv_gather.cpp
        cuda_kv_kernels.cu
        cuda_kv_promotion.cpp
        cuda_kv_storage.cpp
        cuda_submission.cpp
        fixed_cuda_kv_cache.cpp
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
        "${PROJECT_SOURCE_DIR}/include"
)

target_link_libraries(
    kim_kv_cuda
    PRIVATE
        kim_kv_warnings
    PUBLIC
        kim_kv
        CUDA::cudart
)
