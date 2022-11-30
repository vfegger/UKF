cmake_minimum_required(VERSION 3.24.0)

set(target Graph)
set(target_exe Graph_UKF)

set(include_path "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")

file(GLOB_RECURSE source ${source_path}/*.cpp)
list(FILTER source EXCLUDE REGEX ".*Graph\\.cpp$")

add_library(${target} ${source})

add_executable(${target_exe} src/Graph.cpp ${source})

target_include_directories(${target} PUBLIC ${include_path})
target_link_libraries(${target} PUBLIC Parser)

target_include_directories(${target_exe} PUBLIC ${include_path})
target_link_libraries(${target_exe} PUBLIC Parser)
