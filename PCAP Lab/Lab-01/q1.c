//pow(x, rank) where x is integer constant
#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char* argv[]){
	int rank,size;
	int x = 5;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	printf("The value of pow(%d,%d) is %f\n",x,rank,pow(x,rank));
	MPI_Finalize();
	return 0;
}