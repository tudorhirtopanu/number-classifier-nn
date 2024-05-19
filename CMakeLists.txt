cmake_minimum_required(VERSION 3.27)
project(NumberClassifierNN)

set(CMAKE_CXX_STANDARD 20)
set(EIGEN_ROOT_DIR "/Path/to/eigen-3.4.0/")

include_directories(
        ${EIGEN_ROOT_DIR}
)

# Set compiler optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(NumberClassifierNN src/main.cpp
        src/dataset_utils.cpp
        src/helpers.cpp
        src/neural_network.cpp
        src/activation_functions.cpp
        src/parameter_handler.cpp)
