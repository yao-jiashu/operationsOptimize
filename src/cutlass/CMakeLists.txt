add_executable(cutlass_basic_gemm basic_gemm.cu)
set_target_properties(cutlass_basic_gemm PROPERTIES CUDA_ARCHITECTURES "70")
target_include_directories(cutlass_basic_gemm  PUBLIC ${CUTLASS_HEADER_PATH})