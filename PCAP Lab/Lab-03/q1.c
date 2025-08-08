#include <stdio.h>
#include <mpi.h>

int fac(int x){
    int r = 1;
    while(x>1){
        r = r*x;
        x = x-1;
    }
    return r;
}

int main(int argc, char *argv[]){
    int rank, size, n, recv[1];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int arr[size], finrecv;

    if(rank == 0){
        for (int i = 0; i<size; i++){
            scanf("%d", &arr[i]);
        }
    }
    MPI_Scatter(arr, 1, MPI_INT, recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    recv[0] = fac(recv[0]);
    MPI_Reduce(recv, &finrecv, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Final Result: %d\n", finrecv);
    }

    MPI_Finalize();
    return 0;
}