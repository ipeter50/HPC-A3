#include <cublas_v2.h>

extern "C"
{
#include <stdio.h>
#include <cblas.h>
#include "lib_kernels.h"
#include <math.h>
#include "omp.h"



#define BLOCK_SIZE 4 //only for GPU5


// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    double *elements;
} Matrix;



/***********************
 	  CPU LIB
************************/

void
matmult_lib(int m, int n, int k, double *A, double *B, double *C) {
	double te, ts;
	ts = omp_get_wtime();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m ,m , n, 1.0, A, k, B, n, 0.0, C, n);
	te = omp_get_wtime() - ts;
	//printf("Time:%f\n", te);
}


/***********************
 	  NAT CPU
************************/

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


/***********************
 	  GPU 1
************************/

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


/***********************
 	  GPU 2
************************/

void
matmult_gpu2(int m, int n, int k, double *h_A, double *h_B, double *h_C){

	double *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_C, m * n * sizeof(double)); 
	int bs = 32;
	int dimGridX = (int)ceil(1.0*n/bs);
	int dimGridY = (int)ceil(1.0*m/bs);
	
	kernel_gpu2<<<dim3(dimGridX, dimGridY),dim3(bs,bs)>>>(d_A, d_B, d_C, m, n, k);
	
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}


/***********************
 	  GPU 3
************************/


void
matmult_gpu3(int m, int n, int k, double *h_A, double *h_B, double *h_C){

	double *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_C, m * n * sizeof(double)); 
	int bs = 32;
	int dimGridX = (int)ceil(1.0*n/(1*bs));
	int dimGridY = (int)ceil(1.0*m/(2*bs));
	
	kernel_gpu3<<<dim3(dimGridX, dimGridY),dim3(bs,bs)>>>(d_A, d_B, d_C, m, n, k);
	
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}


/***********************
 	  GPU 4
************************/


void
matmult_gpu4(int m, int n, int k, double *h_A, double *h_B, double *h_C){

	double *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, m * k * sizeof(double));
	cudaMalloc((void **)&d_B, k * n * sizeof(double));
	cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_C, m * n * sizeof(double)); 

	int blockSize = 16;
	int elemPerThread = 8;
	int dimGridX = (int)ceil(1.0*n/blockSize);
	int dimGridY = (int)ceil(1.0*m/(elemPerThread*blockSize));
	
	kernel_gpu4<<<dim3(dimGridX, dimGridY),dim3(blockSize,blockSize)>>>(d_A, d_B, d_C, m, n, k);
	
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}



/***********************
 	  GPU 5
************************/

__global__ void kernel_gpu5(const Matrix, const Matrix, Matrix);


void
matmult_gpu5(int m, int n, int k, double *h_A, double *h_B, double *h_C){

	Matrix A, B, C; 

	A.width = k;
	A.height = m;
	//A.stride = k;
	A.elements = h_A;
	B.width = n;
	B.height = k;
	//B.stride = n;
	B.elements = h_B;
	C.width = n;
	C.height = m;
	//C.stride = n;
	C.elements = h_C;
	


	// Load A and B to device memory
	Matrix d_A;
	d_A.width = d_A.stride = A.width; d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(double);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = d_B.stride = B.width; d_B.height = B.height;
	size = B.width * B.height * sizeof(double);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size,
	cudaMemcpyHostToDevice);

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = d_C.stride = C.width; d_C.height = C.height;
	size = C.width * C.height * sizeof(double);
	cudaMalloc(&d_C.elements, size);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	kernel_gpu5<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	// Read C from device memory
	cudaMemcpy(h_C, d_C.elements, size, cudaMemcpyDeviceToHost);
	

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}


// Get a matrix element
__device__ double GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           double value)
{
    A.elements[row * A.stride + col] = value;
}



// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Thread block size




 __global__ void 
kernel_gpu5(const Matrix A, const Matrix B, Matrix C){
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    double Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}



/***********************
 	  GPU LIB
************************/

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




