#include <stdio.h>
#include <mpi.h>

int factorial(int n) {
    int fact = 1;
    for (int i = 1; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

int main(int argc, char *argv[]) {
    int rank, size, local_fact, partial_sum, total_sum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_fact = factorial(rank + 1);
    printf("Process %d computed %d! = %d\n", rank, rank + 1, local_fact);

    MPI_Scan(&local_fact, &partial_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Reduce(&partial_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sum of factorials up to %d! = %d\n", size, total_sum);
    }

    MPI_Finalize();
    return 0;
}
