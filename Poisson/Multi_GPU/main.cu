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
	
	int JACOBI = 2;
	if ( argc >=2 ) N = atoi(argv[1]);
	if ( argc >=3 ) JACOBI = atoi(argv[2]);
	size_t size = (N + 2) * (N + 2) * sizeof(double);	
	printf("Matrix size = %d , Memory = %d\n", N, size);

	cudaMallocHost((void **)&h_Uk, size);
	cudaMallocHost((void **)&h_Uk1, size);
	cudaMallocHost((void **)&h_F, size);
	
	init_matrices(h_Uk, h_Uk1, h_F, N);
	//display_mat(h_Uk, N);

	int top_size = (N + 2) * (N + 2)/2;//+N+2;
	//printf("Top: %d\n", top_size);
	//printf("%f\n", h_Uk[top_size]);
		
	cudaSetDevice(0);
	cudaMalloc((void **)&d_Uktop, size/2);	
	cudaMalloc((void **)&d_Uk1top, size/2);
	cudaMalloc((void **)&d_Ftop, size/2);		

	cudaMemcpy(d_Uktop, h_Uk, size/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Uk1top, h_Uk1, size/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Ftop, h_F, size/2, cudaMemcpyHostToDevice);

	
	cudaSetDevice(1);

	cudaMalloc((void **)&d_Ukbot, size/2);	
	cudaMalloc((void **)&d_Uk1bot, size/2);
	cudaMalloc((void **)&d_Fbot, size/2);

	cudaMemcpy(d_Ukbot, &h_Uk[top_size], size/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Uk1bot, &h_Uk1[top_size], size/2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Fbot, &h_F[top_size], size/2, cudaMemcpyHostToDevice);
	

	int k = 0;
	double d = 1000;
	double delta_squared = 1.0/(N+2) * 1.0/(N+2);
	double h = 1.0/4;

	int dimGridX, dimGridY;
	dimGridX = (int)ceil(1.0*N/16);
	dimGridY = (int)ceil(1.0*N/32);
	
	while(k<max_iter){
		//update_jacobi_gpu<<<1,1>>>(d_Uk, d_Uk1, d_F, N, delta_squared, h);
		//update_jacobi_gpu2<<<dim3(dimGridX, dimGridY),dim3(16,16)>>>(d_Uk, d_Uk1, d_F, N, delta_squared, h);
		cudaSetDevice(0);
		cudaDeviceEnablePeerAccess( 1, 0 );
		update_jacobi_2gpu0<<<dim3(dimGridX, dimGridY),dim3(16,16)>>>(d_Uktop, d_Ukbot, d_Uk1top, d_Ftop, N, delta_squared, h);
		cudaDeviceSynchronize();
		double *tmptop = d_Uktop;
		d_Uktop = d_Uk1top;
		d_Uk1top = tmptop;

		cudaSetDevice(1);
		cudaDeviceEnablePeerAccess( 0, 0 );
		update_jacobi_2gpu1<<<dim3(dimGridX, dimGridY),dim3(16,16)>>>(d_Ukbot, d_Uktop, d_Uk1bot, d_Fbot, N, delta_squared, h);
		cudaDeviceSynchronize();
		double *tmpbot = d_Ukbot;
		d_Ukbot = d_Uk1bot;
		d_Uk1bot = tmpbot;
		
		
		
		k = k+1;
	}
	
	
	printf("jacobi_gpu%d\n", JACOBI);
	printf("k = %d\n", k);
	
	cudaDeviceDisablePeerAccess(0);
	cudaDeviceDisablePeerAccess(1);

	cudaSetDevice(0);
	cudaMemcpy(h_Uk, d_Uktop, size/2, cudaMemcpyDeviceToHost);
	cudaSetDevice(1);
	cudaMemcpy(&h_Uk[top_size], d_Ukbot, size/2, cudaMemcpyDeviceToHost);
	//display_mat(h_Uk, N);
    
		
	cudaFreeHost(h_Uk);
	cudaFreeHost(h_Uk1);
	cudaFreeHost(h_F);
	
	cudaFree(d_Uk);
	cudaFree(d_Uk1);
	cudaFree(d_F);
	
	//display_mat(h_F, N);
	
	
	
	return 0;
}




