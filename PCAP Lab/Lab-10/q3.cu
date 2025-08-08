#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 256 

__global__ void tiledConvolution1D(const float* N, const float* M, float* O, int width, int mask_width) {
    extern __shared__ float shared_mem[]; // Dynamically allocated shared memory

    int tx = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory with padding for boundaries
    int start = blockIdx.x * blockDim.x - (mask_width / 2);
    int end = start + blockDim.x + mask_width - 1;

    // Handle boundary conditions
    if (start + tx >= 0 && start + tx < width) {
        shared_mem[tx] = N[start + tx];
    } else {
        shared_mem[tx] = 0.0f; // Padding with zeros
    }

    __syncthreads(); // Synchronize threads to ensure all data is loaded into shared memory
  
    if (idx < width) {
        float result = 0.0f;
        for (int i = 0; i < mask_width; ++i) {
            result += shared_mem[tx + i] * M[i];
        }
        O[idx] = result;
    }
}

// Host function to launch the kernel
void convolution1D(const float* h_N, const float* h_M, float* h_O, int width, int mask_width) {
    float *d_N, *d_M, *d_O;

    // Allocate device memory
    cudaMalloc((void**)&d_N, width * sizeof(float));
    cudaMalloc((void**)&d_M, mask_width * sizeof(float));
    cudaMalloc((void**)&d_O, width * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_N, h_N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, mask_width * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int blockDim = TILE_WIDTH;
    int gridDim = (width + blockDim - 1) / blockDim;

    // Launch the kernel with dynamically allocated shared memory
    size_t shared_mem_size = (blockDim + mask_width - 1) * sizeof(float);
    tiledConvolution1D<<<gridDim, blockDim, shared_mem_size>>>(d_N, d_M, d_O, width, mask_width);

    cudaMemcpy(h_O, d_O, width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_O);
}

int main() {
    int width = 10; 
    int mask_width = 3; 

    float h_N[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float h_M[] = {0.25, 0.5, 0.25};
    float h_O[width];

    convolution1D(h_N, h_M, h_O, width, mask_width);

    // Print the result
    std::cout << "Output Array: ";
    for (int i = 0; i < width; ++i) {
        std::cout << h_O[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
