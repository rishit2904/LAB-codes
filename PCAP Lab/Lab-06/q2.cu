#include <stdio.h>
#include <cuda_runtime.h>
#define N 10  

__global__ void selectionSort(int *arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }
        __syncthreads();

        if (minIndex != i) {
            int temp = arr[i];
            arr[i] = arr[minIndex];
            arr[minIndex] = temp;
        }
        __syncthreads();
    }
}

int main() {
    int h_arr[N];
    int *d_arr;

    printf("Enter %d elements: ", N);
    for (int i = 0; i < N; i++) {
        scanf("%d", &h_arr[i]);
    }

    cudaMalloc((void **)&d_arr, N * sizeof(int));
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    selectionSort<<<1, 1>>>(d_arr, N);
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted array: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    cudaFree(d_arr);
    return 0;
}
