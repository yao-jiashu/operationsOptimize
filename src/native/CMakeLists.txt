#set(CMAKE_CXX_FLAGS -O2)
message(STATUS  ${CMAKE_CXX_FLAGS})
add_executable(matmul_native matmul.cpp)
target_compile_options(matmul_native INTERFACE -O3)
