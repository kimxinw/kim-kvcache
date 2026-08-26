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
