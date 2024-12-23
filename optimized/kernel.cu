#include "cuda_runtime.h"
#include "omp.h"
#include <vector>
#include <cassert>
#include <iostream>

#define CHECK_ERROR(x) assert(x == cudaError_t::cudaSuccess);

template<int block_size>
__global__ void mmul(const double* A, std::size_t A_m, const double* B, std::size_t B_m, double* C) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    unsigned int wA = A_m;
    unsigned int wB = B_m;

    int aBegin = A_m * block_size * by;

    int aEnd = aBegin + wA - 1;

    int aStep = block_size;

    int bBegin = block_size * bx;
    int bStep = block_size * wB;
    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ float As[block_size][block_size];
        __shared__ float Bs[block_size][block_size];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        __syncthreads();

#pragma unroll
        for (int k = 0; k < block_size; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }
    int c = wB * block_size * by + block_size * bx;
    C[c + wB * ty + tx] = Csub;

}

void launch_cuda_mmul(const double* A, std::size_t A_n, std::size_t A_m, const double* B, std::size_t B_n, std::size_t B_m, double* C) {
    const std::size_t block_size = 32;

    double* gpuA, * gpuB, * gpuC;

    // MEMORY ALLOC
    CHECK_ERROR(cudaMalloc((void**)&gpuA, A_n * A_m * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&gpuB, B_n * B_m * sizeof(double)));
    CHECK_ERROR(cudaMalloc((void**)&gpuC, A_n * B_m * sizeof(double)));

    // MEMORY COPY H to D
    CHECK_ERROR(cudaMemcpy(gpuA, A, A_n * A_m * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(gpuB, B, B_n * B_m * sizeof(double), cudaMemcpyHostToDevice));


    dim3 blocks(block_size, block_size);
    dim3 grid(B_m / block_size, A_n / block_size);

    mmul<block_size> <<<grid, blocks>>> (gpuA, A_m, gpuB, B_m, gpuC);
    cudaDeviceSynchronize();
    // MEMORY COPY D to H
    CHECK_ERROR(cudaMemcpy(C, gpuC, A_n * B_m * sizeof(double), cudaMemcpyDeviceToHost));

    // FREE
    CHECK_ERROR(cudaFree(gpuA));
    CHECK_ERROR(cudaFree(gpuB));
    CHECK_ERROR(cudaFree(gpuC));
}


int main() {
    std::vector<double> A(4000000, 1);
    std::vector<double> B(4000000, 2);
    std::vector<double> C(4000000);

    launch_cuda_mmul(A.data(), 2000, 2000, B.data(), 2000, 2000, C.data());
    launch_cuda_mmul(A.data(), 2000, 2000, B.data(), 2000, 2000, C.data());
    launch_cuda_mmul(A.data(), 2000, 2000, B.data(), 2000, 2000, C.data());

    for (auto i : C) {
        assert(C[i] == 4000);
    }
    return 0;
}
