cmake_minimum_required(VERSION 3.16.3)

project(UKF_1 VERSION 0.1)

# Add Libraries
add_subdirectory(structure)
add_subdirectory(parser)
add_subdirectory(ukf)

# Add the executable
add_executable(${PROJECT_NAME} mainTest.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC Structure Parser UKF)

