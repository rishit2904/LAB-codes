#include <stdio.h>
#include <mpi.h>
#define N 3  

int count_occurrences(int arr[], int size, int target) {
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] == target) {
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int rank, size, matrix[N][N], sub_matrix[N], element, local_count, global_count;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != N) {
        if (rank == 0) {
            printf("Please run with %d processes (each handling one row).\n", N);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Enter a %dx%d matrix (row by row):\n", N, N);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }

        printf("Enter the element to search: ");
        scanf("%d", &element);
    }

    MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrix, N, MPI_INT, sub_matrix, N, MPI_INT, 0, MPI_COMM_WORLD);
    local_count = count_occurrences(sub_matrix, N, element);
    printf("Process %d searched row %d and found %d occurrences.\n", rank, rank, local_count);
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Element %d appears %d times in the matrix.\n", element, global_count);
    }

    MPI_Finalize();
    return 0;
}
