cmake_minimum_required(VERSION 3.16)

# 设置cuda 架构
#if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#    set(CMAKE_CUDA_ARCHITECTURES 75)
#endif()

project(AIOperation CXX CUDA)
#message(STATUS ${CMAKE_CUDA_COMPILER_ID})
#message(STATUS ${CMAKE_C_FLAGS})
#message(STATUS ${CMAKE_CXX_FLAGS})
#message(STATUS ${CMAKE_CUDA_FLAGS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_COMPILER  /usr/local/cuda-11.7/bin/nvcc)
# 使用nvcc单独编译cu文件
set(CUDA_SEPARABLE_COMPILATION ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(config)

include_directories("include")
add_subdirectory(src)

