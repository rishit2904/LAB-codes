#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include <stdio.h>
#include <string.h>

__global__ void appendStringAtomic(char *output, char *input, int inputLength, int *pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < inputLength) {

        int sublen=inputLength-idx;
        int p=atomicAdd(pos,sublen);
        for (int i = 0; i <sublen; i++)
        {
            output[p+i]=input[i];
        }
    }
}

int main() {
    char *input = "PCAP";
    int inputLength = strlen(input);
    int outputLength = inputLength * (inputLength+1)/2;
    printf("Input String: %s\n",input);

    char *d_input, *d_output;
    int *d_pos, *d_subsize;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    cudaMalloc(&d_input, inputLength * sizeof(char));
    cudaMalloc(&d_output, outputLength * sizeof(char));  
    cudaMalloc(&d_pos, sizeof(int));  
    cudaMalloc(&d_subsize, sizeof(int));

    cudaMemcpy(d_input, input, inputLength * sizeof(char), cudaMemcpyHostToDevice);

    cudaError_t error=cudaGetLastError();
    if(error!=cudaSuccess)
    printf("CUDA Error 1: %s\n",cudaGetErrorString(error));

    cudaMemset(d_output, 0, outputLength * sizeof(char)); 
    cudaMemset(d_pos, 0, sizeof(int));  

    appendStringAtomic<<<1, 256>>>(d_output, d_input, inputLength, d_pos);
    

    error=cudaGetLastError();
    if(error!=cudaSuccess)
    printf("CUDA Error 2: %s\n",cudaGetErrorString(error));

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed,start,stop);

    char *h_output = new char[outputLength + 1];
    cudaMemcpy(h_output, d_output, outputLength * sizeof(char), cudaMemcpyDeviceToHost);  // Copy output string

    h_output[outputLength] = '\0';  // Null-terminate the string

    printf("Result: %s\n",h_output);
    printf("Time taken = %f\n",elapsed);

    // Clean up
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_subsize);
    cudaFree(d_pos);
    
    return 0;
}
