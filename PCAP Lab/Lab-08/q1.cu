#include <stdio.h>
#include <cuda.h>

#define N 3  // Matrix size

// Kernel: Each row computed by one thread
__global__ void matrixAddRow(int *A, int *B, int *C, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        for (int col = 0; col < n; col++) {
            C[row * n + col] = A[row * n + col] + B[row * n + col];
        }
    }
}

// Kernel: Each column computed by one thread
__global__ void matrixAddCol(int *A, int *B, int *C, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        for (int row = 0; row < n; row++) {
            C[row * n + col] = A[row * n + col] + B[row * n + col];
        }
    }
}

// Kernel: Each element computed by one thread
__global__ void matrixAddElement(int *A, int *B, int *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

// Function to print matrix
void printMatrix(const char* msg, int C[N][N]) {
    printf("%s\n", msg);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", C[i][j]);
        printf("\n");
    }
    printf("\n");
}

int main() {
    int size = N * N * sizeof(int);
    int h_A[N][N], h_B[N][N], h_C[N][N];

    // Taking user input
    printf("Enter elements of Matrix A (%dx%d):\n", N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            scanf("%d", &h_A[i][j]);

    printf("Enter elements of Matrix B (%dx%d):\n", N, N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            scanf("%d", &h_B[i][j]);

    int *d_A, *d_B, *d_C_row, *d_C_col, *d_C_element;

    // Allocate memory on GPU
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C_row, size);
    cudaMalloc((void**)&d_C_col, size);
    cudaMalloc((void**)&d_C_element, size);

    // Copy input matrices to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 1. Row-wise computation
    matrixAddRow<<<N, 1>>>(d_A, d_B, d_C_row, N);
    cudaMemcpy(h_C, d_C_row, size, cudaMemcpyDeviceToHost);
    printMatrix("Matrix Addition (Row-wise Computation):", h_C);

    // 2. Column-wise computation
    matrixAddCol<<<N, 1>>>(d_A, d_B, d_C_col, N);
    cudaMemcpy(h_C, d_C_col, size, cudaMemcpyDeviceToHost);
    printMatrix("Matrix Addition (Column-wise Computation):", h_C);

    // 3. Element-wise computation
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixAddElement<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_element, N);
    cudaMemcpy(h_C, d_C_element, size, cudaMemcpyDeviceToHost);
    printMatrix("Matrix Addition (Element-wise Computation):", h_C);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_row);
    cudaFree(d_C_col);
    cudaFree(d_C_element);

    return 0;
}
