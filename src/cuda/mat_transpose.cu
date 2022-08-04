//
// Created by yaojiashu on 7/31/22.
//
#include <memory>
#include <cassert>
#include "cuda_common.cuh"

//================
//#define V0
//#define V1
#define V2
//#define WA_IB
//#define IA_WB
//#define RA_WB
#define WA_RB
//================
//#define Config1
//================

// 只能进行单个实验
static void singleTest() {
    int count = 0;
#ifdef V0
#ifdef WA_IB
    count += 1;
#endif
#ifdef IA_WB
    count += 1;
#endif
#ifdef RA_WB
    count += 1;
#endif
#ifdef WA_RB
    count += 1;
#endif
#elif defined(V1)
    count += 1;
#elif defined(V2)
    count += 1;
#endif // V0
    if (count != 1) assert(false);
}

#ifdef Config1
// 80(SM count) * 2048(max threads can parallel per SM)
const uint32_t m = 1000 * 64; // 640
const uint32_t n = 400 * 64; // 256
// block size
const uint32_t tileX = 32; // 40
const uint32_t tileY = 32; // 16
#else
// 80(SM count) * 2048(max threads can parallel per SM)
const uint32_t m = 400 * 64;
const uint32_t n = 1000 * 64;
// block size
const uint32_t tileX = 32;
const uint32_t tileY = 32;
#endif

// 结果
/* Config1
 * WA_IB : 7.30731 ms
 * IA_WB : 70.4914 ms
 * RA_WB : 71.3172 ms
 * WA_RB : 27.7881 ms
*/
/* Config2
 * WA_IB : 7.29748 ms
 * IA_WB : 68.6451 ms
 * RA_WB : 69.4099 ms
 * WA_RB : 26.3853 ms
 * V1(RA_WB) : 27.6331 ms
 * V2(RA_WB) : 22.7165 ms
*/

// 对A的读写是合并访存，对B的读写是非合并访存，编译器会将判断出来是常量的变量，缓存到cache中

const float epsilon = 1.0e-15;
const uint32_t warmup = 10;
const uint32_t repeat = 20;

//const  float*  __restrict__ B
__global__ void mat_transpose( float* A,  float* B, const uint32_t m, const uint32_t n) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x < m and y < n ) {
#ifdef WA_IB
        A[y * m + x] = 1.0f;
#elif defined(IA_WB)
        B[x * n + y] = 1.0f;
#elif defined(RA_WB)
        B[x * n + y] = A[y * m + x];
#elif defined(WA_RB)
        A[y * m + x] = B[x * n + y];
#endif
    }
}

// 对A和B都是合并访存，但是对共享内存访问存在 bank conflict
// 通过共享内存来优化访存模式
__global__ void mat_transpose_v1(const float*  __restrict__ A, float* B) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float s_A[tileY][tileX];
    if (x < m and y < n)
        s_A[threadIdx.y][threadIdx.x] = A[y * m + x];
    __syncthreads();
    const uint32_t x1 = blockIdx.y * blockDim.y + threadIdx.x;
    const uint32_t y1 = blockIdx.x * blockDim.x + threadIdx.y;
    if (x1 < n and y1 < m)
        B[y1 * n + x1] = s_A[threadIdx.x][threadIdx.y];

}
// 合并访存，通过共享内存来优化对显存的访问， 消除 bank 冲突
__global__ void mat_transpose_v2(const float*  __restrict__ A, float* B) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float s_A[tileY][tileX + 1]; // 必须是最低维加一，但是最后一个单元我不用
    if (x < m and y < n)
        s_A[threadIdx.y][threadIdx.x] = A[y * m + x];
    __syncthreads();
    const uint32_t x1 = blockIdx.y * blockDim.y + threadIdx.x;
    const uint32_t y1 = blockIdx.x * blockDim.x + threadIdx.y;
    if (x1 < n and y1 < m)
        B[y1 * n + x1] = s_A[threadIdx.x][threadIdx.y];

}
void setVectorValue(float* vec, uint32_t len) {
#pragma omp parallel for
    for(uint32_t i = 0; i < len; i++) {
        vec[i] = static_cast<float>(i);
    }
}
// A<n, m> B<m, n>
void checkTranspose(float* B, float* A, uint32_t m, uint32_t n) {
#pragma omp parallel for
    for(uint32_t i = 0; i < m; i++) {
        for(uint32_t j = 0; j < n; j++) {
            if (std::abs(B[i * n + j] - A[j * m + i]) > epsilon) assert(false);
        }
    }
}
void printMatrix(float* A, uint32_t n, uint32_t m) {
    for(uint32_t i = 0; i < n; i++) {
        for(uint32_t j = 0; j < m; j++) {
            std::cout << A[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
}
void checkMatrix(float* A, float value, uint32_t n, uint32_t m) {
#pragma omp parallel for
    for(uint32_t i = 0; i < n; i++) {
        for(uint32_t j = 0; j < m; j++) {
            if (std::abs(A[i * m + j] - value) > epsilon) assert(false);
        }
    }
}

int main(int argc, char* argv[]) {
    singleTest();
    init_cuda(1);
    checkCUDA(cudaFree(0));
    std::unique_ptr<float[]> h_A(new float[m * n]);
    std::unique_ptr<float[]> h_B(new float[m * n]);

    setVectorValue(h_A.get(), m * n);

    float  *d_A, *d_B;
    checkCUDA(cudaMalloc(&d_A, sizeof(float) * m * n));
    checkCUDA(cudaMalloc(&d_B, sizeof(float) * m * n));

    checkCUDA(cudaMemcpy(d_A, h_A.get(), sizeof(float) * m * n, cudaMemcpyDefault));
    checkCUDA(cudaMemcpy(d_B, h_B.get(), sizeof(float) * m * n, cudaMemcpyDefault));

    dim3 blockSize(tileX, tileY);
    assert(m % tileX == 0 and n % tileY == 0);
    dim3 gridSize(m / tileX, n / tileY);

    cudaEvent_t start, end;
    checkCUDA(cudaEventCreate(&start, 0) );
    checkCUDA(cudaEventCreate(&end, 0) );

    // check kernel error
#ifdef V0
    mat_transpose<<<gridSize, blockSize>>>(d_A, d_B, m, n); //elapsed time 0.0116736 ms 0.0070656 ms
#elif defined(V1)
    mat_transpose_v1<<<gridSize, blockSize>>>(d_A, d_B);
#elif defined(V2)
    mat_transpose_v2<<<gridSize, blockSize>>>(d_A, d_B);
#endif
    checkCUDA(cudaGetLastError());
    checkCUDA(cudaDeviceSynchronize());

    for (uint32_t i = 0; i < warmup + repeat; i++) {
        if (i == warmup) {
            checkCUDA(cudaEventRecord(start));
            cudaEventQuery(start);
        }
#ifdef V0
        mat_transpose<<<gridSize, blockSize>>>(d_A, d_B, m, n); //elapsed time 0.0116736 ms 0.0070656 ms
#elif defined(V1)
        mat_transpose_v1<<<gridSize, blockSize>>>(d_A, d_B);
#elif defined(V2)
        mat_transpose_v2<<<gridSize, blockSize>>>(d_A, d_B);
#endif
    }
    checkCUDA(cudaEventRecord(end));
    checkCUDA(cudaEventSynchronize(end));
    float et = 0.0f;
    checkCUDA(cudaEventElapsedTime(&et, start, end));
    std::cout << "elapsed time " << et/repeat << " ms" << std::endl;
    checkCUDA(cudaEventDestroy(start));
    checkCUDA(cudaEventDestroy(end));
    checkCUDA(cudaMemcpy(h_B.get(), d_B, sizeof(float) * m * n, cudaMemcpyDefault));
    checkCUDA(cudaMemcpy(h_A.get(), d_A, sizeof(float) * m * n, cudaMemcpyDefault));
    checkCUDA(cudaDeviceSynchronize());

//    std::cout << "============================================A" << std::endl;
//    printMatrix(h_A.get(), n, m);
//    std::cout << "============================================B" << std::endl;
//    printMatrix(h_B.get(), m, n);

#ifdef V0
#ifdef WA_IB
    checkMatrix(h_A.get(), 1.0f, n, m);
#elif defined(IA_WB)
    checkMatrix(h_B.get(), 1.0f, m, n);
#elif defined(RA_WB)
    checkTranspose(h_B.get(), h_A.get(), m, n);
#elif defined(WA_RB)
    checkTranspose(h_A.get(), h_B.get(), n, m);
#endif
#elif defined(V1)
    checkTranspose(h_B.get(), h_A.get(), m, n);
#elif defined(V2)
    checkTranspose(h_B.get(), h_A.get(), m, n);
#endif

    checkCUDA(cudaFree(d_A));
    checkCUDA(cudaFree(d_B));

    return 0;
}