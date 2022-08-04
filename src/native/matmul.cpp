//
// Created by root on 7/14/22.
//
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <omp.h>

void matrix(const double* a, const double* b, double* c, long n) {
    long i, j, k;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            for (k= 0; k < n; k++)
                c[i * n + j] += a[i * n + k] * b[k * n + j];
}
const long n = 2048;
int main(int argc, char* argv[]) {
    std::cout << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << std::endl;
#ifdef __cplusplus
    std::cout << "__cplusplus = " << __cplusplus << std::endl;
#endif // __cplusplus
#ifdef __CUDACC__
    std::cout << "__CUDACC__ " << __CUDACC__ << std::endl;
#endif
#ifdef __CUDA_ARCH__
    std::cout << "__CUDA_ARCH__" << __CUDA_ARCH__ << std::endl;
#endif

    double* ptr_a;
    double* ptr_b;
    double* ptr_c;
    struct timeval tv1, tv2;
    long i, j, sec, usec;

    std::cout << "Input matrix size : " << n << std::endl;
//    std::cin >> n;
    ptr_a = static_cast<double*>(malloc(n * n * sizeof(double) ) );
    ptr_b = static_cast<double*>(malloc(n * n * sizeof(double) ) );
    ptr_c = static_cast<double*>(malloc(n * n * sizeof(double) ) );

#pragma omp parallel for
    for (i = 0; i < n; i++)
        for (j =0; j < n; j++) {
            ptr_a[i * n + j] = 2.0;
            ptr_b[i * n + j] = -2.0;
            ptr_c[i * n + j] = 0.0;
        }
    gettimeofday(&tv1, nullptr);
    matrix(ptr_a, ptr_b, ptr_c, n);
    gettimeofday(&tv2, nullptr);
    usec = (tv2.tv_sec - tv1.tv_sec)*1000000 + (tv2.tv_usec - tv1.tv_usec);
    sec = usec / 1000000;
    usec = usec - sec*1000000;
    printf("time elapse %ld s: %ld us\n", sec, usec);

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            if (ptr_c[i * n + j] != -4.0 * static_cast<double>(n) ) printf("error!");

    return 0;
}