#ifndef MATRIX_MULTIPLICATION_CUH
#define MATRIX_MULTIPLICATION_CUH

// #include <nvfunctional>
// #include <thrust/functional.h>
#include <cuda_runtime.h>
#include <iostream>

typedef void (*FunctionPtr)(double* input, double* output);

__global__ void MatrixMultiplication(double* d_matrix, double* d_vectors, double* d_result, int rows, int cols, int vec_num);

__global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* mu, int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, FunctionPtr func_ptr, void* context, int type);
  
__global__ void obtain_res(double* pts, double* weights, double* results, int sigma_rows, int res_rows, int res_cols);

__global__ void func_Vmu(const double* vec_x, double* pt, const double* mu, int dim, int index, double& func_value);

__global__ void func_Vmumu(const double* vec_x, double* pt, const double* mu, int dim, int index, double& func_value);

__host__ __device__ double cost_function1(const double* vec_x, int dim);

void MatrixMul(double* d_matrix, double* d_vectors, double* d_result, int rows, int cols, int vec_num);

void CudaIntegration(FunctionPtr function, double* d_sigmapts, double* d_weights, double* d_results, double* d_mu, int sigma_rows, int sigma_cols, int res_rows, int res_cols, void* context, double* d_pts1, double* d_pts2, int type);

void CudaIntegration1(double* d_pts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols);

#endif // MATRIX_MULTIPLICATION_CUH


// extern template void CudaIntegration<nvstd::function<double* (double*, int)>>(nvstd::function<double* (double*, int)>&, double*, double*, double*, int, int, int, int, void*, double*, double*);
