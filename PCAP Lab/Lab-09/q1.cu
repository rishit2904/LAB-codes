#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>


__global__ void spmv_csr_kernel(
    int num_rows,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    const float *x,
    float *y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;  
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = csrRowPtr[row];
        int row_end = csrRowPtr[row + 1];
        for (int i = row_start; i < row_end; i++) {
            sum += csrVal[i] * x[csrColInd[i]];
        }
        y[row] = sum;
    }
}
void spmv_csr(
    int num_rows,
    int num_cols,
    int nnz,              
    float *csrVal,        
    int *csrRowPtr,       
    int *csrColInd,       
    float *x,             
    float *y)             
{

    float *d_csrVal, *d_x, *d_y;
    int *d_csrRowPtr, *d_csrColInd;
    cudaMalloc(&d_csrVal, nnz * sizeof(float));
    cudaMalloc(&d_csrRowPtr, (num_rows + 1) * sizeof(int));
    cudaMalloc(&d_csrColInd, nnz * sizeof(int));
    cudaMalloc(&d_x, num_cols * sizeof(float));
    cudaMalloc(&d_y, num_rows * sizeof(float));
    cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtr, csrRowPtr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, num_cols * sizeof(float), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    spmv_csr_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        num_rows, d_csrVal, d_csrRowPtr, d_csrColInd, d_x, d_y);


    cudaMemcpy(y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_csrVal);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() { 
    int num_rows = 4;
    int num_cols = 4;
    int nnz = 7;  
    float csrVal[] = {5, 2, 3, 1, 4, 6};           
    int csrRowPtr[] = {0, 2, 3, 5, 6, 7};         
    int csrColInd[] = {0, 2, 1, 0, 3, 2};          
    float x[] = {1, 2, 3, 4};
    float *y = (float*)malloc(num_rows * sizeof(float));
    spmv_csr(num_rows, num_cols, nnz, csrVal, csrRowPtr, csrColInd, x, y);
    printf("Result vector y:\n");
    for (int i = 0; i < num_rows; i++) {
        printf("y[%d] = %.2f\n", i, y[i]);
    }

    free(y);
    return 0;
}
