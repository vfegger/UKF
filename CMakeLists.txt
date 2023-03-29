cmake_minimum_required(VERSION 3.24.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 61 86)

project(UKF_1 VERSION 0.1)

# Add Libraries
add_subdirectory(src/structure)
add_subdirectory(src/parser)
add_subdirectory(src/ukf)
add_subdirectory(src/math)
add_subdirectory(src/timer)
add_subdirectory(src/hfe)
add_subdirectory(src/hfe-crc)
add_subdirectory(src/graph)
add_subdirectory(src/cuda)


# Add the executable
add_executable(${PROJECT_NAME} mainTest.cpp)
add_executable(${PROJECT_NAME}_CRC mainCRC.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC Structure Math Parser Timer UKF HFE Graph)
target_link_libraries(${PROJECT_NAME}_CRC PUBLIC Structure Math Parser Timer UKF HFE_CRC Graph)