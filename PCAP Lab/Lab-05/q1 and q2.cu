#include <stdio.h>
#include <cuda.h>

__global__ void vectorAdd(int *A, int *B, int *C, int N, int useSingleBlock) {
    int i;
    if (useSingleBlock) {
        i = threadIdx.x;
    } else {
        i = threadIdx.x + blockIdx.x * blockDim.x;
    }
    if (i < N)
        C[i] = A[i] + B[i];
}

int main() {
    int N;
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    printf("Enter the size of vectors (N): ");
    scanf("%d", &N);

    A = (int*) malloc(N * sizeof(int));
    B = (int*) malloc(N * sizeof(int));
    C = (int*) malloc(N * sizeof(int));

    printf("Enter %d elements for vector A: ", N);
    for (int i = 0; i < N; i++) scanf("%d", &A[i]);

    printf("Enter %d elements for vector B: ", N);
    for (int i = 0; i < N; i++) scanf("%d", &B[i]);

    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    printf("\nMethod 1: Single Block, N Threads\n");
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N, 1);
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: ");
    for (int i = 0; i < N; i++) printf("%d ", C[i]);
    printf("\n");

    printf("\nMethod 2: N Threads Total (Multiple Blocks)\n");
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N, 0);
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: ");
    for (int i = 0; i < N; i++) printf("%d ", C[i]);
    printf("\n");

    printf("\nMethod 3: Fixed 256 Threads Per Block\n");
    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N, 0);
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Result: ");
    for (int i = 0; i < N; i++) printf("%d ", C[i]);
    printf("\n");

    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
