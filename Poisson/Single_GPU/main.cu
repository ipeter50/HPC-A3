//extern "C" {

#include <stdio.h>
#include <stdlib.h>
#include "poisson.h"


int
main(int argc, char *argv[]) {
		
	double *h_Uk, *h_Uk1, *h_F;
	double *d_Uk, *d_Uk1, *d_F;
	double *d_Uktop, *d_Uk1top, *d_Ftop;
	double *d_Ukbot, *d_Uk1bot, *d_Fbot;
	double threshold = 0.000000001;
	int N = 200;
	int num_gpu = 1;	
	int max_iter = 10000;
	
	int JACOBI = 1;
	if ( argc >=2 ) N = atoi(argv[1]);
	if ( argc >=3 ) JACOBI = atoi(argv[2]);
	size_t size = (N + 2) * (N + 2) * sizeof(double);	
	printf("Matrix size = %d , Memory = %d\n", N, size);
	
	cudaMallocHost((void **)&h_Uk, size);
	cudaMallocHost((void **)&h_Uk1, size);
	cudaMallocHost((void **)&h_F, size);
	
	init_matrices(h_Uk, h_Uk1, h_F, N);
	//display_mat(h_Uk, N);
	
		
	cudaMalloc((void **)&d_Uk, size);	
	cudaMalloc((void **)&d_Uk1, size);
	cudaMalloc((void **)&d_F, size);
	cudaMemcpy(d_Uk, h_Uk, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Uk1, h_Uk1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, h_F, size, cudaMemcpyHostToDevice);
	

	int k = 0;
	double d = 1000;
	double delta_squared = 1.0/(N+2) * 1.0/(N+2);
	double h = 1.0/4;
	int dimGridX, dimGridY;
	dimGridX = (int)ceil(1.0*N/16);
	dimGridY = dimGridX;
	
	
	while(k<max_iter){
		update_jacobi_gpu<<<1,1>>>(d_Uk, d_Uk1, d_F, N, delta_squared, h);
		//update_jacobi_gpu2<<<dim3(dimGridX, dimGridY),dim3(16,16)>>>(d_Uk, d_Uk1, d_F, N, 					delta_squared, h);
		cudaDeviceSynchronize();
		double *tmp = d_Uk;
		d_Uk = d_Uk1;
		d_Uk1 = tmp;
		k = k+1;
	}
	
	
	printf("jacobi_gpu%d\n", JACOBI);
	printf("k = %d\n", k);
	
	
	cudaMemcpy(h_Uk, d_Uk, size, cudaMemcpyDeviceToHost);
	display_mat(h_Uk, N);
    
		
	cudaFreeHost(h_Uk);
	cudaFreeHost(h_Uk1);
	cudaFreeHost(h_F);
	
	cudaFree(d_Uk);
	cudaFree(d_Uk1);
	cudaFree(d_F);
	
	//display_mat(h_F, N);
	
	
	
	return 0;
}




