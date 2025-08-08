#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel to create matrix B with 1's complement for non-border elements
__global__ void onesComplementNonBorder(float *d_A, float *d_B, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int idx = row * N + col;
        float value = d_A[idx];

        // Check if element is non-border
        bool isBorder = (row == 0 || row == M-1 || col == 0 || col == N-1);

        if (!isBorder) {
            // Apply 1's complement (assuming binary values: 0->1, 1->0)
            int val= value;
            int binVal = 0;
            int size = 0;
            while(val>0)
            {
                int rem = val%2==0?1:0;
                binVal = rem*(int)(pow(10, size)) + binVal;
                size++;
                val/=2;
            }
            d_B[idx] = binVal;
        } else {
            // Keep border elements unchanged
            d_B[idx] = value;
        }
    }
}

// Function to print matrix
void printMatrix(float *matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.0f\t", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Matrix dimensions
    int M = 4;  // rows
    int N = 4;  // columns

    // Host matrices
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_B = (float*)malloc(M * N * sizeof(float));

    // Initialize matrix A with sample binary values
    float values[M][N] = {
        {1, 2, 3, 4},
        {6, 5, 8, 3},
        {2, 4, 10, 1},
        {9, 1, 2, 5}
    };

    // Copy values to 1D array
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = values[i][j];
        }
    }

    printf("Input Matrix A:\n");
    printMatrix(h_A, M, N);

    // Device matrices
    float *d_A, *d_B;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, M * N * sizeof(float));

    // Copy input matrix to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch kernel
    onesComplementNonBorder<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);

    // Copy result back to host
    cudaMemcpy(h_B, d_B, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output matrix
    printf("Output Matrix B:\n");
    printMatrix(h_B, M, N);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
