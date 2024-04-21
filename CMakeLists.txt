cmake_minimum_required(VERSION 3.27)
project(NumberClassifierNN)

set(CMAKE_CXX_STANDARD 20)
set(EIGEN_ROOT_DIR "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3/")

include_directories(
        ${EIGEN_ROOT_DIR}
)

add_executable(NumberClassifierNN src/main.cpp
        src/dataset_utils.cpp)
