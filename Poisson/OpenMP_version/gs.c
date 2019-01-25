#include "gs.h"
#include "utils.h"

void
gs(double **Uk, double **F, int N, int max_iter, double threshold){

	int k=0;

	double d = 1000; 

	double delta_squared = 1.0/(N+2) * 1.0/(N+2);
	double h = 1.0/4;

	while(k<max_iter && d>threshold){
		d = update_gs(Uk, F, N, delta_squared, h); 
		k = k+1;
	}	
	printf("\nNorm: %f \nk: %d", d, k);

}


double 
update_gs(double **Uk, double **F, int N, double delta_squared, double h){
	int i,j;

	double tmp, tmp1;
	double norm=0.0;
	for(j=1;j<N+1;j++){
		for(i=1;i<N+1;i++){
			tmp = h*(Uk[i+1][j] + Uk[i-1][j] + Uk[i][j+1] + Uk[i][j-1] +  delta_squared * F[i][j]);
			tmp1 = tmp - Uk[i][j];
			norm +=  tmp1 * tmp1;
			Uk[i][j] = tmp; 
		}
	}
	return norm;

}
