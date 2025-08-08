#include <stdio.h>
#include <mpi.h>

float avg(int x[], int n){
    int s = 0;
    for (int i = 0; i<n; i++){
        s += x[i];
    }
    return s/n;
}


int main(int argc, char *argv[]){
    int rank, size, n, recv[1000], m;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int arr[1000], finrecv;

    if(rank == 0){
        printf("Enter M: ");
        scanf("%d", &m);
        for (int i = 0; i<m*size; i++){
            scanf("%d", &arr[i]);
        }
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(arr, m, MPI_INT, recv, m, MPI_INT, 0, MPI_COMM_WORLD);
    int a = avg(recv, m);
    MPI_Reduce(&a, &finrecv, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Final Result: %f\n", (float) finrecv/size);
    }

    MPI_Finalize();
    return 0;
}