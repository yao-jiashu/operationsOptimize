//
// Created by root on 7/30/22.
//
#include <iostream>
#include <memory>
#include <cstdint>
#include <cassert>
#include "utils.h"
#include "cuda_common.cuh"
const uint32_t sm = 80;
const uint32_t smCore = 64;
const uint32_t tWorkload = 30;
const uint32_t m = sm * smCore * tWorkload * 10;
const uint32_t bs = 256;
const float epsilon = 1.0e-15;
const uint32_t warmup = 10;
const uint32_t repeat = 20;

__global__ void vector_add(float* A, float* B, float* C, uint32_t m) {

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < m) {
        C[tid] = A[tid] + B[tid];
    }

}
// tile and shared memory
__global__ void vector_add_v1(float* A, float* B, float* C, uint32_t m) {

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float a;
    if (tid < m) {
        C[tid] = A[tid] + B[tid];
    }

}

void setVectorValueFast(float* vec, float value, uint32_t len) {
#pragma omp parallel for
    for(uint32_t i = 0; i < len; i++) {
        vec[i] = value;
    }
}
void setVectorValue(float* vec, float value, uint32_t len) {
    for(uint32_t i = 0; i < len; i++) {
        vec[i] = value;
    }
}
void checkVectorValue(float* vec, float value, uint32_t len) {
#pragma omp parallel for
    for(uint32_t i = 0; i < len; i++) {
        if (std::abs(vec[i]-value ) > epsilon) assert(false);
    }

}


int main() {
    init_cuda(1);
    checkCUDA(cudaFree(0));
    std::unique_ptr<float[]> h_A(new float[m ]);
    std::unique_ptr<float[]> h_B(new float[m]);
    std::unique_ptr<float[]> h_C(new float[m]);

    Timing( "sequential", setVectorValue(h_A.get(), 1.0f, m) );
    Timing( "parallel", setVectorValueFast(h_A.get(), 1.0f, m) );
    setVectorValueFast(h_B.get(), 2.0f, m);
    setVectorValueFast(h_C.get(), 0.0f, m);

    float  *d_A, *d_B, *d_C;
    checkCUDA(cudaMalloc(&d_A, sizeof(float) * m));
    checkCUDA(cudaMalloc(&d_B, sizeof(float) * m));
    checkCUDA(cudaMalloc(&d_C, sizeof(float) * m));

    checkCUDA(cudaMemcpy(d_A, h_A.get(), sizeof(float) * m, cudaMemcpyDefault));
    checkCUDA(cudaMemcpy(d_B, h_B.get(), sizeof(float) * m, cudaMemcpyDefault));

    dim3 blockSize(bs);
    dim3 gridSize( (m + bs - 1) / bs);

    cudaEvent_t start, end;
    checkCUDA(cudaEventCreate(&start, 0) );
    checkCUDA(cudaEventCreate(&end, 0) );

    // check kernel error
    vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, m); //elapsed time 0.02688 ms
    checkCUDA(cudaGetLastError());
    checkCUDA(cudaDeviceSynchronize());

    for (uint32_t i = 0; i < warmup + repeat; i++) {
        if (i == warmup) {
            checkCUDA(cudaEventRecord(start));
            cudaEventQuery(start);
        }
        vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, m); //elapsed time 0.02688 ms
    }
    checkCUDA(cudaEventRecord(end));
    checkCUDA(cudaEventSynchronize(end));
    float et = 0.0f;
    checkCUDA(cudaEventElapsedTime(&et, start, end));
    std::cout << "elapsed time " << et/repeat << " ms" << std::endl;
    checkCUDA(cudaEventDestroy(start));
    checkCUDA(cudaEventDestroy(end));
    checkCUDA(cudaMemcpy(h_C.get(), d_C, sizeof(float) * m, cudaMemcpyDefault));
    checkCUDA(cudaDeviceSynchronize());

    checkVectorValue(h_C.get(), 3.0f, m);

    std::cout << std::hex << reinterpret_cast<int*>(d_A) << std::endl;
    std::cout << std::hex << reinterpret_cast<int*>(d_B) << std::endl;
    std::cout << std::hex << reinterpret_cast<int*>(d_C) << std::endl;
    std::cout << std::dec << 16 << std::endl;

    for (int i = 0; i < 32; i++)
        std::cout << (i^0x1) << " ";

    checkCUDA(cudaFree(d_A));
    checkCUDA(cudaFree(d_B));
    checkCUDA(cudaFree(d_C));

    return 0;
}