#include <stdio.h>
#include<mpi.h>

int fact(int n){
	if (n==1|| n==0){return 1;}
	else{ return n*fact(n-1);}
}

int fib(int n){
	if (n==0 || n==1) return n;
	else return fib(n-1)+fib(n-2);
}

int main(int argc, char* argv[]){
	int rank,size;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank%2==0) printf("Rank: %d, Fact: %d\n",rank,fact(rank));
	else printf("Rank: %d, Fib:%d\n",rank,fib(rank));
	MPI_Finalize();
	return 0;
}