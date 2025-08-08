#include <stdio.h>
#include <cuda.h>
#include <math.h>

__global__ void computeSine(float *input, float *output, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N)
        output[i] = sinf(input[i]);  
}

int main() {
    int N;
    float *h_input, *h_output;
    float *d_input, *d_output;

    printf("Enter the number of elements (N): ");
    scanf("%d", &N);

    h_input = (float*) malloc(N * sizeof(float));
    h_output = (float*) malloc(N * sizeof(float));

    printf("Enter %d angles in radians: ", N);
    for (int i = 0; i < N; i++) scanf("%f", &h_input[i]);

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    computeSine<<<blocks, threadsPerBlock>>>(d_input, d_output, N);

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sine values:\n");
    for (int i = 0; i < N; i++) printf("sin(%.4f) = %.4f\n", h_input[i], h_output[i]);

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);

    return 0;
}
