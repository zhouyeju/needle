cmake_minimum_required(VERSION 3.14)
project(needle_ops LANGUAGES CXX CUDA)
enable_language(CUDA)

find_package(pybind11 REQUIRED)

set(SOURCES
    src/framework/pybind.cpp
    src/framework/tensor.cpp
)

set(HEADER_DIRS
    $ENV{CUDA_HOME}/include
    src/framework
)

set(LIB_DIRS
    $ENV{CUDA_HOME}/lib64
)

pybind11_add_module(ops ${SOURCES})

target_link_directories(ops PRIVATE ${LIB_DIRS})
target_include_directories(ops PRIVATE ${HEADER_DIRS})
target_link_libraries(ops PRIVATE cudart)