#include <stdio.h>
#include <cuda_runtime.h>

#define MASK_WIDTH 3

// CUDA Kernel for 1D Convolution
__global__ void convolution_1D(float *N, float *M, float *P, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (i < width) {
        for (int j = 0; j < MASK_WIDTH; j++) {
            int index = i - MASK_WIDTH / 2 + j;
            if (index >= 0 && index < width) {
                sum += N[index] * M[j];
            }
        }
        P[i] = sum;
    }
}

int main() {
    const int width = 10;
    float h_N[width] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float h_M[MASK_WIDTH] = {0.2, 0.5, 0.2};
    float h_P[width];

    float *d_N, *d_M, *d_P;

    cudaMalloc((void **)&d_N, width * sizeof(float));
    cudaMalloc((void **)&d_M, MASK_WIDTH * sizeof(float));
    cudaMalloc((void **)&d_P, width * sizeof(float));

    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (width + blockSize - 1) / blockSize;
    convolution_1D<<<gridSize, blockSize>>>(d_N, d_M, d_P, width);

    cudaMemcpy(h_P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output array P:\n");
    for (int i = 0; i < width; i++) {
        printf("%f ", h_P[i]);
    }
    printf("\n");

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}