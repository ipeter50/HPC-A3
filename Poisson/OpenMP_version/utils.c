#include "utils.h"
#include <math.h>

void
init_matrices(double **Uk, double **Uk1, double **F, int N){
	int i, j;
	for(i=0;i<N+2;i++){
		for(j=0;j<N+2;j++){
			Uk[i][j] = 0;
			Uk1[i][j] = 0;
			F[i][j] = 0;
		}
	}

	for(i=0;i<N+2;i++){
		Uk[i][0]=20; //left wall
		Uk[i][N+1]=20; // right wall
		Uk[0][i]=20; // top wall
		Uk1[i][0]=20; //left wall
		Uk1[i][N+1]=20; // right wall
		Uk1[0][i]=20; // top wall
	
	}
	int rad_start_i, rad_start_j, rad_end_i, rad_end_j;

	rad_start_i = round((N+2)*4/6);
	rad_end_i = round((N+2)*5/6);
	rad_start_j = round((N+2)*3/6);
	rad_end_j = round((N+2)*4/6);	

	for(i=rad_start_i; i< rad_end_i; i++){
		for(j=rad_start_j; j< rad_end_j; j++){
			F[i][j] = 200;		
		}
	}
}

void
display_mat(double **M, int N){
	int i, j;
	printf("\n");
	for(i=0;i<N+2;i++){
		for(j=0;j<N+2;j++){
			printf("%f ", M[i][j]);
		}
		printf("\n");
	}

}


double
norm_difference(double ** A, double **B, int N){
	int i,j;
	double norm = 0.0;
	double tmp;
	for(i=0;i<N+2;i++){
		for(j=0;j<N+2;j++){
			tmp = A[i][j] - B[i][j];
			norm += tmp*tmp;
		}
		
	}
	return norm;
}
