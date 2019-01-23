#include <cublas_v2.h>

extern "C"
{
#include <stdio.h>
#include <cblas.h>
#include "lib_kernels.h"
#include <math.h>



void
matmult_lib(int m, int n, int k, double *A, double *B, double *C) {
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m ,m , n, 1.0, A, k, B, n, 0.0, C, n);
}


void
matmult_nat(int m, int n, int k, double *A, double *B, double *C) {
    
    int i, j, l;
	for(i = 0; i < m; i++){
		for(j = 0; j < n; j++){
			C[i * m + j] = 0;
	    	}
	    }    


    for(j = 0; j < n; j++){
        for(i = 0; i < m; i++){
	    double tmp = 0.0;
            for(l = 0; l < k; l++){
                tmp += A[i*m +l] * B[l*k+j];
            }
            C[i*m +j] = tmp;
        }
    }
}


void
matmult_gpu1(int m, int n, int k, double *h_A, double *h_B, double *h_C){

	double *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_C, m * n * sizeof(double));

	kernel_gpu1<<<1,1>>>(d_A, d_B, d_C, m, n, k);
	
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
}

void
matmult_gpu2(int m, int n, int k, double *h_A, double *h_B, double *h_C){

	double *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_C, m * n * sizeof(double)); 

	int dimGridX = (int)ceil(1.0*n/16);
	int dimGridY = (int)ceil(1.0*m/16);
	
	kernel_gpu2<<<dim3(dimGridX, dimGridY),dim3(16,16)>>>(d_A, d_B, d_C, m, n, k);
	
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}



void
matmult_gpu3(int m, int n, int k, double *h_A, double *h_B, double *h_C){

	double *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_C, m * n * sizeof(double)); 

	int dimGridX = (int)ceil(1.0*n/16);
	int dimGridY = (int)ceil(1.0*m/32);
	
	kernel_gpu3<<<dim3(dimGridX, dimGridY),dim3(16,16)>>>(d_A, d_B, d_C, m, n, k);
	
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}


void
matmult_gpulib(int m, int n, int k, double *h_A, double *h_B, double *h_C){
	double *d_A, *d_B, *d_C;

	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	cudaMalloc((void **)&d_A, m * k * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_C, m * n * sizeof(double));
	
	cublasHandle_t handle;
	cublasCreate(&handle);

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, d_B, n, d_A, k, beta, d_C, m);
	

	cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

}




