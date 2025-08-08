#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
	int rank,size;
	int x=5,y=2;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(rank==0)
		printf("Process %d Addition of (%d,%d) is %d\n",rank, x,y,(x+y));
	if(rank==1)
		printf("Process %d Subtraction of (%d,%d) is %d\n",rank, x,y,(x-y));
	if(rank==2)
		printf("Process %d Multiplication of (%d,%d) is %d\n",rank,x,y,(x*y));
	if(rank==3)
		printf("Process %d Division of (%d,%d) is %f\n",rank,x,y,(float)x/y);


	MPI_Finalize();
	return 0;
}