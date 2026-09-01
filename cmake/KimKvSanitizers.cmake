set(KIM_KV_SANITIZERS "")

if(KIM_KV_ENABLE_ASAN)
    list(APPEND KIM_KV_SANITIZERS address)
endif()

if(KIM_KV_ENABLE_UBSAN)
    list(APPEND KIM_KV_SANITIZERS undefined)
endif()

if(KIM_KV_ENABLE_TSAN)
    list(APPEND KIM_KV_SANITIZERS thread)
endif()

add_library(kim_kv_sanitizers INTERFACE)

if(NOT "${KIM_KV_SANITIZERS}" STREQUAL "")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(
            FATAL_ERROR
            "Sanitizers currently require GCC or Clang"
        )
    endif()

    list(
        JOIN
        KIM_KV_SANITIZERS
        ","
        KIM_KV_SANITIZER_LIST
    )

    message(
        STATUS
        "Enabled sanitizers: ${KIM_KV_SANITIZER_LIST}"
    )

    target_compile_options(
        kim_kv_sanitizers
        INTERFACE
            "-fsanitize=${KIM_KV_SANITIZER_LIST}"
            -fno-omit-frame-pointer
            -fno-sanitize-recover=all
    )

    target_link_options(
        kim_kv_sanitizers
        INTERFACE
            "-fsanitize=${KIM_KV_SANITIZER_LIST}"
    )
endif()
