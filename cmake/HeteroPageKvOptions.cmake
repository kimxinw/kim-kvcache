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
    "Build the CUDA storage and correctness contracts"
    OFF
)

option(
    HETEROPAGE_KV_BUILD_BENCHMARKS
    "Build deterministic benchmark harnesses"
    ON
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
