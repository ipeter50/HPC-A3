void display_mat(double *M, int N);
__global__ void update_jacobi_gpu(double *d_Uk, double *d_Uk1, double *d_F, int N, double delta_squared, double h);
__global__ void update_jacobi_gpu2(double *d_Uk, double *d_Uk1, double *d_F, int N, double delta_squared, double h);
void init_matrices(double *h_Uk, double *h_Uk1, double *h_F, int N);
