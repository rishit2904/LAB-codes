#include <iostream>
#include <cuda_runtime.h>

#define N 1024   // Length of the input signal
#define K 5      // Length of the kernel

// Declare constant memory for the kernel
__constant__ float d_kernel[K];

// Kernel for 1D convolution using constant memory
__global__ void convolution1D(float* d_input, float* d_output, int signal_size, int kernel_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < signal_size) {
        float result = 0.0f;
        // Perform convolution by iterating over the kernel
        for (int i = 0; i < kernel_size; i++) {
            // Handle the boundary conditions
            if (index - i >= 0)
                result += d_input[index - i] * d_kernel[i];
        }
        d_output[index] = result;
    }
}

int main() {
    // Host arrays
    float h_input[N], h_kernel[K], h_output[N];
    float *d_input, *d_output;

    // Initialize the input signal and kernel with some values
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % 100;
    }
    for (int i = 0; i < K; i++) {
        h_kernel[i] = rand() % 10;
    }

    // Allocate device memory
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    // Copy input and kernel data to the device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, K * sizeof(float));

    // Define grid and block size
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the kernel
    convolution1D<<<gridSize, blockSize>>>(d_input, d_output, N, K);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the output (first 10 values for brevity)
    std::cout << "Convolution Output (first 10 values):" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
