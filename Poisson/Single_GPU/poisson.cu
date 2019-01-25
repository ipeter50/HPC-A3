#include <stdio.h>
#include <math.h>

__global__ void
update_jacobi_gpu(double *d_Uk, double *d_Uk1, double *d_F, int N, double delta_squared, double h){
	
	int i,j;	
	double tmp, tmp1;
	double norm=0.0;
	int up, down, left, right;
	int loc = 0;
	for(j=1;j<N+1;j++){
		for(i=1;i<N+1;i++){
			up = ((N+2) * (i - 1)) + j;
			down = ((N+2) * (i + 1)) + j;
			left = ((N+2) * i) + (j -1);
			right = ((N+2) * i) + (j + 1);
			loc = ((N+2) * i) + j;
			tmp = h*(d_Uk[down] + d_Uk[up] + d_Uk[right] + d_Uk[left] +  delta_squared * d_F[loc]);
			tmp1 = tmp - d_Uk[loc];
			norm +=  tmp1 * tmp1;
			d_Uk1[loc] = tmp;
		}
	}
}

__global__ void
update_jacobi_gpu2(double *d_Uk, double *d_Uk1, double *d_F, int N, double delta_squared, double h){
	
	int i,j;
	j = threadIdx.x + blockIdx.x * blockDim.x +1;
	i = threadIdx.y + blockIdx.y * blockDim.y +1;
	//printf("i = %d, j = %d \n",i,j);	
	double tmp, tmp1;
	double norm=0.0;
	int up, down, left, right;
	int loc = 0;
	if (j < N+1 && i < N+1) {
		up = ((N+2) * (i - 1)) + j;
		down = ((N+2) * (i + 1)) + j;
		left = ((N+2) * i) + (j -1);
		right = ((N+2) * i) + (j + 1);
		loc = ((N+2) * i) + j;
		tmp = h*(d_Uk[down] + d_Uk[up] + d_Uk[right] + d_Uk[left] +  delta_squared * d_F[loc]);
		tmp1 = tmp - d_Uk[loc];
		norm +=  tmp1 * tmp1;
		d_Uk1[loc] = tmp;
	}
}

void
init_matrices(double *h_Uk, double *h_Uk1, double *h_F, int N){
	int i, j;
	int loc = 0;
	for(i=0;i<N+2;i++){
		for(j=0;j<N+2;j++){
			//initialize matrix with 0s or 20 for walls.
			int value = 0;
			if (i == 0 || j ==0 || j == N+1) value = 20;
			loc = ((N+2) * i) + j;
			h_Uk[loc] = value;
			h_Uk1[loc] = value;
			h_F[loc] = value;
		}
	}

	
	int rad_start_i, rad_start_j, rad_end_i, rad_end_j;

	rad_start_i = round((N+2)*4.0/6);
	rad_end_i = round((N+2)*5.0/6);
	rad_start_j = round((N+2)*3.0/6);
	rad_end_j = round((N+2)*4.0/6);	

	for(i=rad_start_i; i< rad_end_i; i++){
		for(j=rad_start_j; j< rad_end_j; j++){
			h_F[((N+2) * i) + j] = 2000;		
		}
	}
}

void
display_mat(double *M, int N){
	int i, j;
	printf("\n");
	for(i=0;i<N+2;i++){
		for(j=0;j<N+2;j++){
			printf("%f ", M[((N+2) * i) + j]);
		}
		printf("\n");
	}
}




