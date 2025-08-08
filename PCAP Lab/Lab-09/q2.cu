#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void modifyMatrixRows(float *d_A, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int idx = row * N + col;
        float value = d_A[idx];


        if (row > 0) {
            float result = value;
            for (int p = 1; p < row + 1; p++) {
                result *= value;
            }
            d_A[idx] = result;
        }
    }
}


void printMatrix(float *matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f\t", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {

    int M = 4;  
    int N = 3;  


    float *h_A = (float*)malloc(M * N * sizeof(float));


    float values[M][N] = {
        {1.0, 2.0, 3.0},
        {1.0, 2.0, 3.0},
        {1.0, 2.0, 3.0},
        {1.0, 2.0, 3.0}
    };


    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_A[i * N + j] = values[i][j];
        }
    }

    printf("Original Matrix:\n");
    printMatrix(h_A, M, N);


    float *d_A;
    cudaMalloc(&d_A, M * N * sizeof(float));


    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);


    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );


    modifyMatrixRows<<<blocksPerGrid, threadsPerBlock>>>(d_A, M, N);


    cudaMemcpy(h_A, d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);


    printf("Modified Matrix:\n");
    printMatrix(h_A, M, N);


    cudaFree(d_A);
    free(h_A);

    return 0;
}
