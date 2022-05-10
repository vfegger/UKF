cmake_minimum_required(VERSION 3.16.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(UKF_1 VERSION 0.1)

# Add Libraries
add_subdirectory(structure)
add_subdirectory(parser)
add_subdirectory(ukf)
add_subdirectory(math)
add_subdirectory(timer)
add_subdirectory(hfe)


# Add the executable
add_executable(${PROJECT_NAME} mainTest.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC Structure Math Parser Timer UKF HeatFluxEstimation)

