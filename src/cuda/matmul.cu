//
// Created by root on 7/13/22.
//
#include <iostream>
#include <cstdlib>
#include "cuda_common.cuh"
#include "utils.h"

const float epsilon = 1.0e-5;

template<typename DType>
void matmul_host_memory(DType** A, DType** B, DType** C, uint32_t M, uint32_t N, uint32_t K) {
    (*A) = static_cast<DType*>(malloc(M * K * sizeof(DType) ) );
    (*B) = static_cast<DType*>(malloc(K * N * sizeof(DType) ) );
    (*C) = static_cast<DType*>(malloc(M * N * sizeof(DType) ) );
#pragma omp parallel for
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < K; j++)
            (*A)[i * M + j] = 2.0;
#pragma omp parallel for
    for (uint32_t i = 0; i < K; i++)
        for (uint32_t j = 0; j < N; j++)
            (*B)[i * K + j] = -2.0;
#pragma omp parallel for
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = 0; j < N; j++)
            (*C)[i * M + j] = 0.0;
}

template<typename DType>
void matmul_device_memory(DType** d_A, DType** d_B, DType** d_C,
                          DType* h_A, DType* h_B,
                          uint32_t M, uint32_t N, uint32_t K) {
    checkCUDA(cudaMalloc(&(*d_A), M * K * sizeof(DType) ) );
    checkCUDA(cudaMalloc(&(*d_B), K * N * sizeof(DType) ) );
    checkCUDA(cudaMalloc(&(*d_C), M * N * sizeof(DType) ) );
    checkCUDA(cudaMemcpy((*d_A), h_A, M * K * sizeof(DType), cudaMemcpyHostToDevice) );
    checkCUDA(cudaMemcpy((*d_B), h_B, K * N * sizeof(DType), cudaMemcpyHostToDevice) );
}

template<typename DType>
bool data_check(const DType* ptr, DType value, const uint32_t L) {
    if (ptr == nullptr) {
        std::cout << "Pointer nullptr" << std::endl;
        return false;
    }
    bool res = true;
    uint32_t num {0};
#pragma omp parallel for
    for (uint32_t i = 0; i < L; i ++) {
        if ( std::abs(ptr[i] - value) > epsilon) {
#pragma omp critical
            num ++;
            res = false;
        }
    }
    if (!res) {
        std::cout << "Value error : " << num << "/" << L << " times" << std::endl;
        std::cout << "--" << ptr[0] << "/" << value << std::endl;
        std::cout << "--" << ptr[1] << "/" << value << std::endl;
        std::cout << "--" << ptr[2] << "/" << value << std::endl;

    }
    return res;
}

template <typename DType>
__global__ void matmul_00(const DType* A, const DType* B, DType* C,
                          uint32_t M, uint32_t N, uint32_t K) {
    uint32_t bx = blockIdx.x;
    uint32_t by = blockIdx.y;
    uint32_t tx = threadIdx.x;
    uint32_t ty = threadIdx.y;

    uint32_t row = by * blockDim.y + ty;
    uint32_t column = bx * blockDim.x + tx;

     if ( row < M and column < N ) {
        DType c = 0;
        for (uint32_t i = 0; i < K; ++i) {
            c += A[row * K + i] * B[i * N + column];
        }
         C[row * N + column] = c;
//#ifdef __CUDA_ARCH__
//         C[row * N + column] =  __CUDA_ARCH__;
//#endif

     }
}
//核函数（静态共享内存版）
//__global__ void matrixMultiplyShared(double *A, double *B, double *C,
//                                     int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
//{
//    //分配共享内存
//    __shared__ int sharedM[32][32];
//    __shared__ int sharedN[32][32];
//
//    int bx = blockIdx.x;
//    int by = blockIdx.y;
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//
//    int row = by * blockDim.y + ty;
//    int col = bx * blockDim.x + tx;
//
//    int Csub = 0.0;
//
//    //核心：下面将保存在全局内存中的矩阵M&N分块存放到共享内存中
//    for (int i = 0; i < (int)(ceil((float)numAColumns / blockDim.x)); i++)//如上图，将一个红框矩形分成多个正方形
//    {
//        if (i*blockDim.x + tx < numAColumns && row < numARows)//分割M矩阵，边界确定方式结合上图蓝色正方形内数据的位置理解
//            sharedM[ty][tx] = A[row*numAColumns + i * blockDim.x + tx];
//        else
//            sharedM[ty][tx] = 0.0;
//
//        if (i*blockDim.y + ty < numBRows && col < numBColumns)//分割N矩阵
//            sharedN[ty][tx] = B[(i*blockDim.y + ty)*numBColumns + col];
//        else
//            sharedN[ty][tx] = 0.0;
//        __syncthreads();//同一线程块中所有线程必须到达运行 __synctrheads()之后才可以做其余操作
//        //此操作可以防止当只有部分数据拷贝到共享内存后就提前进行下列计算。
//
//        for (int j = 0; j < blockDim.x; j++)//分块后的矩阵相乘
//            Csub += sharedM[ty][j] * sharedN[j][tx];
//        __syncthreads();
//    }
//
//    if (row < numCRows && col < numCColumns)//将计算后的矩阵块放到结果矩阵C中
//        C[row*numCColumns + col] = Csub;
//}

//const uint32_t BM = 128;
//const uint32_t BN = 128;
//const uint32_t BK = 8;
//const uint32_t TM = 8;
//const uint32_t TN = 8;
//#define BM 16
//#define BN 16
//#define BK 16
//#define TM 1
//#define TN 1
//#define THREAD_NUM 256
//template<typename DType>
//__global__ static void matmul_0(const DType* a, const DType* b, DType* c, int n) {
//    const int tid = threadIdx.x;
//    const int bid = blockIdx.x;
//    //从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
//    const int idx = bid * THREAD_NUM + tid;
//    const int row = idx / n;
//    const int column = idx % n;
//    int i;
//    //计算矩阵乘法
//    if (row < n && column < n) {
//        DType t = 0;
//        for (i = 0; i < n; i++) {
//            t += a[row * n + i] * b[i * n + column];
//        }
//        c[row * n + column] = t;
//    }
//}
//template<typename DType>
//__global__ static void matmul_1(const DType* A, const DType* B, DType* C, int n) {
//    const int tid = threadIdx.x;
//    const int bid = blockIdx.x;
//    //从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
//    const int idx = bid * THREAD_NUM + tid;
//    const int row = idx / n;
//    const int column = idx % n;
//    int i;
//    //计算矩阵乘法
//    if (row < n && column < n) {
//        __shared__ DType shared_A[256];
//        __shared__ DType shared_B[256];
//        DType register_C  {0.0};
//        for (int bk = 0; bk < (n + 16 - 1) / 16; bk++) {
//            shared_A[tid] = A[row * n + 16 * bk + tid % 16];
//            shared_B[tid] = B[(16 * bk + tid / 16 ) * n + column];
//            __syncthreads();
//#pragma unroll
//            for (int k = 0; k < 16; k++) {
//                register_C += shared_A[ty][k] * shared_B[k][tx];
//            }
//            __syncthreads();
//        }
//        C[row * n + column] = register_C;
//    }
//}

//template <typename DType, typename Size_T>
//__global__ void matmul_kernel1(const DType* A, const DType* B, DType* C,
//                               Size_T M, Size_T N, Size_T K) {
//
//    const Size_T x = blockDim.x * blockIdx.x + threadIdx.x;
//    const Size_T y = blockDim.y * blockIdx.y + threadIdx.y;
//    if (x < N or y < M  ) {
//        const Size_T bx = blockIdx.x;
//        const Size_T by = blockIdx.y;
//        const Size_T tx = threadIdx.x;
//        const Size_T ty = threadIdx.y;
//
//        __shared__ DType shared_A[16][16];
//        __shared__ DType shared_B[16][16];
//
//        DType register_C  {0.0};
//
//        for (int bk = 0; bk < (K + 16 - 1) / 16; bk++) {
//            shared_A[ty][tx] = A[(blockDim.y * by + ty ) * K + blockDim.x * bk + tx ];
//            shared_B[ty][tx] = B[(blockDim.y * bk + ty ) * N + blockDim.x * bx + tx];
//            __syncthreads();
//#pragma unroll
//            for (int k = 0; k < 16; k++) {
//                register_C += shared_A[ty][k] * shared_B[k][tx];
//            }
//            __syncthreads();
//        }
//        C[(blockDim.y * by + ty ) * N + blockDim.x * bx + tx] = register_C;
//    }
//}
//



const uint32_t m = 2048;
const uint32_t n = 2048;
const uint32_t k = 2048;
const uint32_t n_trials = 100;
template<typename DType>
void test_matmul(dim3 dim_g, dim3 dim_b) {
    DType* h_A {nullptr};
    DType* h_B {nullptr};
    DType* h_C {nullptr};
    DType* d_A {nullptr};
    DType* d_B {nullptr};
    DType* d_C {nullptr};

    std::cout << "Matrix size : " << std::endl
        << " --- m : " << m << std::endl
        << " --- n : " << n << std::endl
        << " --- k : " << k << std::endl;

    matmul_host_memory<DType, uint32_t>(&h_A, &h_B, &h_C, m, n, k);
    data_check<DType, uint32_t>(h_A, 2.0, m * k);
    data_check<DType, uint32_t>(h_B, -2.0, k * n);
    data_check<DType, uint32_t>(h_C, 0, m * n);
    init_cuda(0);
    matmul_device_memory<DType, uint32_t>(&d_A, &d_B, &d_C, h_A, h_B, m, n, k);


    // warmup
    for (uint32_t i = 0; i < n_trials ; i++) {
        matmul_00<<< dim_g, dim_b >>> (d_A, d_B, d_C, m, n, k);
        cudaDeviceSynchronize();
    }
    cudaEvent_t start, stop;
    check_cudaError( cudaEventCreate(&start) );
    check_cudaError( cudaEventCreate(&stop) );
    check_cudaError( cudaEventRecord(start, 0) );

    for (uint32_t i = 0; i < n_trials ; i++) {
        matmul_00<DType><<< dim_g, dim_b >>> (d_A, d_B, d_C, m, n, k);
        cudaDeviceSynchronize();
    }
    check_cudaError( cudaEventRecord(stop, 0) );
    check_cudaError( cudaEventSynchronize(stop));
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time = " << elapsedTime/n_trials << " ms .\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(h_C, d_C, m * n * sizeof(DType), cudaMemcpyDeviceToHost);
    data_check<DType, uint32_t>(h_C, -4.0 * static_cast<DType>(k), m * n);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}
int main(int argc, char* argv[]) {


//    long threadsPerBlock = 256;
//    long blocksPerGrid =
//            (m * n + threadsPerBlock - 1) / threadsPerBlock;
    dim3 dim_g = (128, 128, 1);
    dim3 dim_b = (16, 16, 1);
    test_matmul<double>(dim_g,dim_b);

    return 0;
}




