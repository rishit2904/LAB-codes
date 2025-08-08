#include <iostream>
#include <cuda_runtime.h>

#define N 16  // Size of the matrix

// Kernel to perform matrix multiplication
__global__ void matrixMulCUDA(int* A, int* B, int* C, int width) {
    // Get the row and column index of the element to compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        int value = 0;
        for (int k = 0; k < width; k++) {
            value += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = value;
    }
}

int main() {
    int *A, *B, *C;  // Host matrices
    int *d_A, *d_B, *d_C;  // Device matrices

    // Allocate memory for host matrices
    A = (int*)malloc(N * N * sizeof(int));
    B = (int*)malloc(N * N * sizeof(int));
    C = (int*)malloc(N * N * sizeof(int));

    // Initialize matrices A and B with random values
    for (int i = 0; i < N * N; i++) {
        A[i] = rand() % 10;
        B[i] = rand() % 10;
    }

    // Allocate memory on the device
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));
    cudaMalloc(&d_C, N * N * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define the dimensions of the grid and block
    dim3 blockDim(4, 4);  // 4x4 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);  // Grid size

    // Launch the kernel
    matrixMulCUDA<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix C
    std::cout << "Matrix C (Result of A * B):" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
