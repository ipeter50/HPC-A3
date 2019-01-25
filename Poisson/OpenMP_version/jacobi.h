void jacobi(double **Uk, double **Uk1, double **F, int N, int max_iter, double threshold);
double update_jacobi(double **Uk, double **Uk1, double **F, int N, double delta_squared, double h);
