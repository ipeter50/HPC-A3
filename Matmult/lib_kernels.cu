extern "C"{


#include "lib_kernels.h"


__global__ void
kernel_gpu1(double *d_A, double *d_B, double *d_C, int m, int n, int k){

	int i, j, l;
	for(i = 0; i < m; i++){
		for(j = 0; j < n; j++){
			d_C[i * m + j] = 0;
	    	}
	    }    


	for(j = 0; j < n; j++){
		for(i = 0; i < m; i++){
			double tmp = 0.0;
			for(l = 0; l < k; l++){
				tmp += d_A[i*m +l] * d_B[l*k+j];
			}
			d_C[i*m +j] = tmp;
		}
	}

}


__global__ void
kernel_gpu2(double *d_A, double *d_B, double *d_C, int m, int n, int k){
	int i,j;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	j = threadIdx.y + blockIdx.y * blockDim.y;
	double tmp = 0.0;
	for(int l = 0; l < k; l++){
		tmp += d_A[i*m +l] * d_B[l*k+j];
	}
	d_C[i*m +j] = tmp;

}


}
