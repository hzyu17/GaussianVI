#ifndef MATRIX_MULTIPLICATION_CUH
#define MATRIX_MULTIPLICATION_CUH

// #include <nvfunctional>
// #include <thrust/functional.h>
#include <cuda_runtime.h>
#include <iostream>

typedef void (*FunctionPtr)(double* input, double* output, int size, void* context);

// template <typename Function>
__global__ void Sigma_function(double* sigmapts, double* pts, int sigma_rows, int sigma_cols, int res_rows, int res_cols, FunctionPtr function, void* context);
  
__global__ void obtain_res(double* pts, double* weights, double* results, int sigma_rows, int res_rows, int res_cols);

void MatrixMul(double* d_matrix, double* d_vectors, double* d_result, int rows, int cols, int vec_num);

// template <typename Function>
void CudaIntegration(FunctionPtr function, double* d_sigmapts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols, void* context, double* d_pts1, double* d_pts2);

// extern template void CudaIntegration<nvstd::function<double* (double*, int)>>(nvstd::function<double* (double*, int)>&, double*, double*, double*, int, int, int, int, void*, double*, double*);


void CudaIntegration1(double* d_pts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols);

#endif // MATRIX_MULTIPLICATION_CUH


// void CudaIntegration(FunctionPtr func_ptr, double* d_sigmapts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols, void* context);
// void CudaIntegration(FunctionPtr func_ptr, double* d_sigmapts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols, void* context, double* d_pts1, double* d_pts2);