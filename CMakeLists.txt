cmake_minimum_required(VERSION 3.18)

project(kim_kv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


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
