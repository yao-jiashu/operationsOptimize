//
// Created by yaojiashu on 8/1/22.
//
#include <memory>
#include <cassert>
#include <cooperative_groups.h>
#include "cuda_common.cuh"
using namespace cooperative_groups;

#define Config1

//#define V0
//#define V1
//#define V2
//#define V3
//#define V4
//#define V5
//#define V6
//#define V7
//#define V8
#define V9

#ifdef USE_DB
using real = double;
#else
using real = float;
#endif

#ifdef Config1
const uint32_t N = 1000000000;
const uint32_t BS = 256;                     //block size
const uint32_t GS = (N + BS - 1) / BS;       //grid size
const uint32_t GS1 = 5120;
#else
#endif


const real epsilon = 1.0e-15;
const uint32_t warmup = 10;
const uint32_t repeat = 20;

static void legalTest() {
    uint32_t count = 0;
#ifdef V0
    count += 1;
    assert(N % BS == 0);
#endif
#ifdef V1
    count += 1;
#endif
#ifdef V2
    count += 1;
#endif
#ifdef V3
    count += 1;
#endif
#ifdef V4
    count += 1;
#endif
#ifdef V5
    count += 1;
#endif
#ifdef V6
    count += 1;
#endif
#ifdef V7
    count += 1;
#endif
#ifdef V8
    count += 1;
#endif
#ifdef V9
    count += 1;
#endif
    if (count != 1) assert(false);
}

// 要求N是blockDim.x的整数倍， 要求blockDim.x是 2 的整数次方
__global__ void reduce_v0(real* A,  real* res) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    // block start address of A
    real* X  = A + bid * blockDim.x;
    for (uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset)
            X[tid] += X[tid + offset];
        __syncthreads();
    }
    if (tid == 0)
        res[bid] = X[0];
}

// 不要求数据量是blockDim.x的整数倍，但是要求blockDim.x是 2 的整数次方
__global__ void reduce_v1(const real* A, real* res) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t n = bid * blockDim.x + tid;
    __shared__ real s_A[256];
    s_A[tid] = n < N ? A[n] : 0;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset];
        __syncthreads();
    }
    if (tid == 0)
        res[bid] = s_A[0];
}

// 动态共享内存
__global__ void reduce_v2(const real* A, real* res) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t n = bid * blockDim.x + tid;
    extern __shared__ real s_A[];
    s_A[tid] = n < N ? A[n] : 0;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset];
        __syncthreads();
    }
    if (tid == 0)
        res[bid] = s_A[0];
}

// 动态共享内存，原子函数，使得完全在GPU中做归约操作
__global__ void reduce_v3(const real* A, real* res) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t n = bid * blockDim.x + tid;
    extern __shared__ real s_A[];
    s_A[tid] = n < N ? A[n] : 0;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(&res[0], s_A[0]);
}

// 动态共享内存，原子函数，使得完全在GPU中做归约操作， warp内线程间同步
__global__ void reduce_v4(const real* A, real* res) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t n = bid * blockDim.x + tid;
    extern __shared__ real s_A[];
    s_A[tid] = n < N ? A[n] : 0;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1; offset > 16; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset];
        __syncthreads();
    }
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset]; // 无bank冲突
        __syncwarp();
    }
    if (tid == 0)
        atomicAdd(&res[0], s_A[0]);
}

// 动态共享内存，原子函数，使得完全在GPU中做归约操作， warp内线程间同步,warp内先读后写
__global__ void reduce_v5(const real* A, real* res) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t n = bid * blockDim.x + tid;
    extern __shared__ real s_A[];
    s_A[tid] = n < N ? A[n] : 0;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1; offset > 16; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset]; // 无bank冲突
        __syncthreads();
    }
    real v;
    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        if (tid < offset) { //限制访问越界
            v = s_A[tid + offset]; // 无bank冲突
            __syncwarp();
            s_A[tid] += v;  // 无bank冲突
            __syncwarp();
        }
    }
    if (tid == 0)
        atomicAdd(&res[0], s_A[0]);
}

// 动态共享内存，原子函数，使得完全在GPU中做归约操作， warp内线程间同步
// 利用线程束洗牌函数进行归约计算
__global__ void reduce_v6(const real* A, real* res) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t n = bid * blockDim.x + tid;
    extern __shared__ real s_A[];
    s_A[tid] = n < N ? A[n] : 0;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1; offset > 16; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset];
        __syncthreads();
    }
    real reg = s_A[tid];
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        reg += __shfl_down_sync(0xffff, reg, offset, offset * 2);
    }
    if (tid == 0)
        atomicAdd(&res[0], reg);
}

// 动态共享内存，原子函数，使得完全在GPU中做归约操作， warp内线程间同步
// 使用协作组进行归约运算
__global__ void reduce_v7(const real* A, real* res) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t n = bid * blockDim.x + tid;
    extern __shared__ real s_A[];
    s_A[tid] = n < N ? A[n] : 0;
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset];
        __syncthreads();
    }
    real reg = s_A[tid];
    thread_block_tile<32> g32 = tiled_partition<32>(this_thread_block());
    for (int32_t offset = g32.size() >> 1; offset > 0; offset >>= 1) {
        // 这一行代码，后面的函数得出结果是同步的，结果加到reg上不保证同步，但对结果没有影响，因为后面结果的数已经取到了
        reg += g32.shfl_down(reg, offset);
    }
    if (tid == 0)
        atomicAdd(&res[0], reg);
}

// 一个线程进行更多的访存运算，从而提高线程的利用率，减少了线程的个数（但也不能减太多，保证有足够多的线程来隐藏延迟）
// 动态共享内存， warp内洗牌函数进行归约
// 调用两次核函数，两次都是折半归约，不存在大数吃小数的情况，所以比使用原子函数更准确
__global__ void reduce_v8(const real* A, real* res, const uint32_t M) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    real y = 0.0;
    const uint32_t stride = blockDim.x * gridDim.x; // total data size > total threads count, so stride = total threads count
    for (uint32_t n = bid * blockDim.x + tid; n < M; n += stride) {
        y += A[n]; // 相邻的线程访问相邻地数据，因此合并访存
    }
    extern __shared__ real s_A[];
    s_A[tid] = y; // 不存在线程没有对应数据的情况
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset]; // 没有bank冲突
        __syncthreads();
    }
    real reg = s_A[tid];
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        reg += __shfl_down_sync(0xffff, reg, offset, offset * 2);
    }
    if (tid == 0)
        res[bid] = reg;
}

// 一个线程进行更多的访存运算，从而提高线程的利用率，减少了线程的个数（但也不能减太多，保证有足够多的线程来隐藏延迟）
// 动态共享内存， warp内洗牌函数进行归约
// 调用两次核函数，两次都是折半归约，不存在大数吃小数的情况，所以比使用原子函数更准确
__global__ void reduce_v9(const real* A, real* res, const uint32_t M) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    real y = 0.0;
    const uint32_t stride = blockDim.x * gridDim.x; // total data size > total threads count, so stride = total threads count
    for (uint32_t n = bid * blockDim.x + tid; n < M; n += stride) {
        y += A[n]; // 相邻的线程访问相邻地数据，因此合并访存
    }
    extern __shared__ real s_A[];
    s_A[tid] = y; // 不存在线程没有对应数据的情况
    __syncthreads();
    for (uint32_t offset = blockDim.x >> 1; offset >= 32; offset >>= 1) {
        if (tid < offset)
            s_A[tid] += s_A[tid + offset]; // 没有bank冲突
        __syncthreads();
    }
    real reg = s_A[tid];
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        reg += __shfl_down_sync(0xffff, reg, offset, offset * 2);
    }
    if (tid == 0)
        atomicAdd(&res[0], reg);
}

void setVectorValue(real* vec, uint32_t len) {
#pragma omp parallel for
    for(uint32_t i = 0; i < len; i++) {
        vec[i] = static_cast<real>(1);
    }
}

void checkVector(real* A, double value, uint32_t n) {
#pragma omp parallel for
    for(uint32_t i = 0; i < n; i++) {
            if (std::abs(A[i] - value) > epsilon) assert(false);
    }
}

int main(/*int argc, char* argv[]*/) {
    legalTest();
    init_cuda(1);
    checkCUDA(cudaFree(nullptr));
    dim3 blockSize(BS);
#ifdef V8
    dim3 gridSize(GS1);
#elif defined(V9)
    dim3 gridSize(GS1);
#else
    dim3 gridSize(GS);
#endif
    std::unique_ptr<real[]> h_A(new real[N]);
    std::unique_ptr<real[]> h_res(new real[GS]);
    h_res[0] = 0;

    setVectorValue(h_A.get(), N);

    float  *d_A, *d_res;
    checkCUDA(cudaMalloc(&d_A, sizeof(real) * N));
    checkCUDA(cudaMalloc(&d_res, sizeof(real) * GS));
    checkCUDA(cudaMemcpy(d_A, h_A.get(), sizeof(real) * N, cudaMemcpyDefault));
    checkCUDA(cudaMemcpy(d_res, h_res.get(), sizeof(real) * GS, cudaMemcpyDefault));

    cudaEvent_t start, end;
    checkCUDA(cudaEventCreate(&start, 0) );
    checkCUDA(cudaEventCreate(&end, 0) );

    // check kernel error
#ifdef V0
    reduce_v0<<<gridSize, blockSize>>>(d_A, d_res); //elapsed time
#elif defined(V1)
    reduce_v1<<<gridSize, blockSize>>>(d_A, d_res); //elapsed time
#elif defined(V2)
    reduce_v2<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
#elif defined(V3)
    reduce_v3<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
#elif defined(V4)
    reduce_v4<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
#elif defined(V5)
    reduce_v5<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
#elif defined(V6)
    reduce_v6<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
#elif defined(V7)
    reduce_v7<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
#elif defined(V8)
    reduce_v8<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res, N);
    reduce_v8<<<1, 1024, sizeof(real) * 1024>>>(d_res, d_res, GS1);
#elif defined(V9)
    reduce_v9<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res, N);
#endif
    checkCUDA(cudaGetLastError());
    checkCUDA(cudaDeviceSynchronize());
    checkCUDA(cudaMemcpy(h_res.get(), d_res, sizeof(real) * GS, cudaMemcpyDefault));
#ifdef V3
    checkVector(h_res.get(), N, 1);
#elif defined(V4)
    checkVector(h_res.get(), N, 1);
#elif defined(V5)
    checkVector(h_res.get(), N, 1);
#elif defined(V6)
    checkVector(h_res.get(), N, 1);
#elif defined(V7)
    checkVector(h_res.get(), N, 1);
#elif defined(V8)
    checkVector(h_res.get(), N, 1);
#elif defined(V9)
    checkVector(h_res.get(), N, 1);
#else
    checkVector(h_res.get(), BS, GS);
#endif

    for (uint32_t i = 0; i < warmup + repeat; i++) {
        if (i == warmup) {
            checkCUDA(cudaEventRecord(start));
            cudaEventQuery(start);
        }
#ifdef V0
        reduce_v0<<<gridSize, blockSize>>>(d_A, d_res);                             //elapsed time 12.1649 ms
#elif defined(V1)
        reduce_v1<<<gridSize, blockSize>>>(d_A, d_res);                             //elapsed time 11.054 ms
#elif defined(V2)
        reduce_v2<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
                                                                                    //elapsed time 10.4593 ms
#elif defined(V3)
        reduce_v3<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
                                                                                    //elapsed time 11.0357 ms
#elif defined(V4)
        reduce_v4<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
                                                                                    //elapsed time 9.7026 ms
#elif defined(V5)
        reduce_v5<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
                                                                                    //elapsed time 13.5011 ms
#elif defined(V6)
        reduce_v6<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
                                                                                    //elapsed time 8.93763 ms
#elif defined(V7)
        reduce_v7<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res);
                                                                                    //elapsed time 8.99999 ms
#elif defined(V8)
        reduce_v8<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res, N);
        reduce_v8<<<1, 1024, sizeof(real) * 1024>>>(d_res, d_res, GS1);
                                                                                    //elapsed time 5.71904 ms
#elif defined(V9)
        reduce_v9<<<gridSize, blockSize, sizeof(real) * BS>>>(d_A, d_res, N);
                                                                                    //elapsed time 5.70767 ms
#endif
    }
    checkCUDA(cudaEventRecord(end));
    checkCUDA(cudaEventSynchronize(end));
    float et = 0.0f;
    checkCUDA(cudaEventElapsedTime(&et, start, end));
    std::cout << "elapsed time " << et/repeat << " ms" << std::endl;
    checkCUDA(cudaEventDestroy(start));
    checkCUDA(cudaEventDestroy(end));
    checkCUDA(cudaDeviceSynchronize());

    checkCUDA(cudaFree(d_A));
    checkCUDA(cudaFree(d_res));

    return 0;
}