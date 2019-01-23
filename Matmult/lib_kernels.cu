extern "C"{

#include <stdio.h>
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
	j = threadIdx.x + blockIdx.x * blockDim.x;
	i = threadIdx.y + blockIdx.y * blockDim.y;
	if(j<n && i<m){
		double tmp = 0.0;
		for(int l = 0; l < k; l++){
			tmp += d_A[i*k+l] * d_B[j+n*l];
		}
		d_C[i*n + j] = tmp;
	}
}


__global__ void
kernel_gpu3(double *d_A, double *d_B, double *d_C, int m, int n, int k){
	int i,j;
	j = 1 * (threadIdx.x + blockIdx.x * blockDim.x);
	i = 2 * (threadIdx.y + blockIdx.y * blockDim.y);
	if(j<n && i<m){
		double tmp = 0.0;
		double tmp2 = 0.0;
		for(int l = 0; l < k; l++){
			tmp += d_A[i*k+l] * d_B[j+n*l];
			tmp2 += d_A[(i+1)*k + l] * d_B[j+n*l];
		}
		d_C[i*n + j] = tmp;
		d_C[(i+1)*n + j] = tmp2;
	}
}


}
