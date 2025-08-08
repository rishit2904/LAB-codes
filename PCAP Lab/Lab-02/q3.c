#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size, *array = NULL;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        array = (int *)malloc(size * sizeof(int));
        printf("Enter %d elements: ", size);
        for (int i = 0; i < size; i++) {
            scanf("%d", &array[i]);
        }
        for (int i = 1; i < size; i++) {
            MPI_Send(&array[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        int element;
        MPI_Recv(&element, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank % 2 == 0) {
            printf("Process %d squared value: %d\n", rank, element * element);
        } else {
            printf("Process %d cubed value: %d\n", rank, element * element * element);
        }
    }

    if (rank == 0) {
        free(array);
    }
    MPI_Finalize();
    return 0;
}
