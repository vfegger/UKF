cmake_minimum_required(VERSION 3.24.0)

set(target Timer)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE source ${source_path}/*.cpp)

add_library(${target} ${source})

target_include_directories(${target} PUBLIC ${include_path})