//
// Created by root on 7/13/22.
//

#ifndef AIOPERATIONS_CUDA_COMMON_CUH
#define AIOPERATIONS_CUDA_COMMON_CUH

#include <sstream>
#include <iostream>


#define checkCUDA(call) do {                                                    \
  cudaError_t status = call;                                                    \
  std::stringstream serr;                                                       \
  if ((status) != cudaSuccess) {                                                \
    serr << "Cuda Error: " << (status) << " : " << cudaGetErrorString(status);  \
    serr << "\n" << __FILE__ << " : " << __LINE__;                              \
    std::cerr << serr.str() <<  "\nAborting...\n";                              \
    exit(1);                                                                    \
  }                                                                             \
} while(0)

inline void printDeviceProp(const cudaDeviceProp &prop) {
    printf("[ Device Name : %s. ]\n", prop.name);
    printf("[ Compute Capability : %d.%d ]\n", prop.major, prop.minor);
    printf("[ Total Global Memory : %ld GB ]\n", prop.totalGlobalMem/ (1024 * 1024 * 1024) );
    printf("[ Total Const Memory : %ld KB ]\n", prop.totalConstMem/ 1024 );
    printf("[ SM counts : %d. ]\n", prop.multiProcessorCount);
    printf("[ Max Block counts Per SM : %d. ]\n", prop.maxBlocksPerMultiProcessor);
    printf("[ Max GridSize[0 - 2] : %d %d %d. ]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("[ Max ThreadsDim[0 - 2] : %d %d %d. ]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("[ Max Shared Memory Per SM : %ld KB ]\n", prop.sharedMemPerMultiprocessor/1024);
    printf("[ Max Shared Memory Per Block : %ld KB ]\n", prop.sharedMemPerBlock/1024);
    printf("[ Max Registers Number Per SM : %d. ]\n", prop.regsPerMultiprocessor);
    printf("[ Max Registers Number Per Block : %d. ]\n", prop.regsPerBlock);
    printf("[ Max Threads Number Per SM : %d. ]\n", prop.maxThreadsPerMultiProcessor);
    printf("[ Max Threads Number Per Block : %d. ]\n", prop.maxThreadsPerBlock);

    printf("[ warpSize : %d. ]\n", prop.warpSize);
    printf("[ memPitch : %ld. ]\n", prop.memPitch);
    printf("[ clockRate : %d. ]\n", prop.clockRate);
    printf("[ textureAlignment : %ld. ]\n", prop.textureAlignment);
    printf("[ deviceOverlap : %d. ]\n\n", prop.deviceOverlap);

}

/*Obtain computing device information and initialize the computing device*/
inline bool init_cuda(int verbose)
{
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        std::cout << "There is no device.\n";
        return false;
    }
    else {
        std::cout << "Find the device successfully.\n";
    }
    //set its value between 0 and n - 1 if there are n GPUS
    cudaSetDevice(count -1);
    if (verbose == 1) {
        cudaDeviceProp prop {} ;
        cudaGetDeviceProperties(&prop, count -1);
        printDeviceProp(prop);
    }
    return true;
}
#endif //AIOPERATIONS_CUDA_COMMON_CUH
