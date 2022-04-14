# UKF
Repository for the Project UKF

## Installation

This project uses the following libraries or applications:

> - WSL* (Windows Subsystem for Linux)
> - CMake
> - Visual Studio Code
> - C++ Compiler

\* Observation: While is not required to be built in WSL, it was develop using it. The integration with CMake should make this project a universal platform project.

## Build

The following commands are used to run this project at the main project folder:

> cmake -S src/ -B build/ \
> make -C build \
> build/UKF_1
> valgrind build/UKF_1

Or

> cmake -S src/ -B build-debug/ -DCMAKE_BUILD_TYPE=Debug \
> cmake --build build-debug \
> build-debug/UKF_1
> valgrind --leak-check=full build-debug/UKF_1
>

## Execution

This project is still ongoing and it is not expected to give any results yet. The only thing that is capable of is memory control.