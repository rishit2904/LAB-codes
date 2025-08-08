#include <stdio.h>
#include <mpi.h>
#include<ctype.h>

int main(int argc, char *argv[])
{
	int rank,size;
	char word[]="HeLlo";

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if(isupper(word[rank]))
		word[rank]=tolower(word[rank]);
	else
		word[rank]=toupper(word[rank]);

	printf("Rank %d Modified word: %s\n",rank,word);

	MPI_Finalize();

	return 0;
}