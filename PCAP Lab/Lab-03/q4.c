#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int rank, size;
    char *S1 = NULL, *S2 = NULL;
    int length;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter the first string: ");
        scanf("%ms", &S1);
        printf("Enter the second string: ");
        scanf("%ms", &S2);
        length = strlen(S1);
    }

    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    char* local_S1 = malloc((length / size + 1) * sizeof(char));
    char* local_S2 = malloc((length / size + 1) * sizeof(char));
    char* local_result = malloc((length / size + 1) * sizeof(char));

    MPI_Scatter(S1, length / size, MPI_CHAR, local_S1, length / size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, length / size, MPI_CHAR, local_S2, length / size, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < length / size; i++) {
        local_result[i] = (i % 2 == 0) ? local_S1[i] : local_S2[i];
    }

    char* result = NULL;
    if (rank == 0) {
        result = malloc((length + 1) * sizeof(char));
    }
    MPI_Gather(local_result, length / size, MPI_CHAR, result, length / size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        result[length] = '\0';
        printf("Resultant String: %s\n", result);
        free(S1);
        free(S2);
        free(result);
    }

    free(local_S1);
    free(local_S2);
    free(local_result);
    MPI_Finalize();
    return 0;
}