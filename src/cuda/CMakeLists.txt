file(GLOB MATMUL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cu)
file(GLOB MEM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/memcpy.cu)
file(GLOB VEC_ADD_SRC ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.cu)
file(GLOB MAT_TRANS ${CMAKE_CURRENT_SOURCE_DIR}/mat_transpose.cu)
file(GLOB REDUCE_SRC ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cu)
file(GLOB WARP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/warp_primitives.cu)
file(GLOB STREAM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream.cu)

#list(APPEND MATMUL_SRC utils.cu)
#list(APPEND MEM_SRC utils.cu)
#list(APPEND VEC_ADD_SRC utils.cu)


add_executable(matmul ${MATMUL_SRC})
set_target_properties(matmul PROPERTIES CUDA_ARCHITECTURES "70")
#target_compile_options(
#        matmul_cuda PUBLIC -arch=compute_70 -code=sm_70 )

add_executable(memcpy ${MEM_SRC})
set_target_properties(memcpy PROPERTIES CUDA_ARCHITECTURES 70)
#target_compile_options(
#        mem PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
#        -G -arch=compute_70 -code=sm_70 >)

add_executable(vector_add ${VEC_ADD_SRC})
set_property(TARGET vector_add PROPERTY CUDA_ARCHITECTURES 70-real 70-virtual)
# output the count of registers that used by a kernel function
target_compile_options( vector_add PUBLIC  --ptxas-options=-v)
#target_compile_options(
#        vector_add PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
#        -G -arch=compute_70 -code=sm_70 >)
add_executable(transpose ${MAT_TRANS})
set_property(TARGET transpose PROPERTY CUDA_ARCHITECTURES 70-real 70-virtual)

add_executable(reduce ${REDUCE_SRC})
set_property(TARGET reduce PROPERTY CUDA_ARCHITECTURES 70-real 70-virtual)

add_executable(warp ${WARP_SRC})
set_property(TARGET warp PROPERTY CUDA_ARCHITECTURES 70-real 70-virtual)

add_executable(stream ${STREAM_SRC})
set_property(TARGET stream PROPERTY CUDA_ARCHITECTURES 70-real 70-virtual)