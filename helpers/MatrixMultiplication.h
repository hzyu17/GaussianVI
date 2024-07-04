#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

#include <iostream>
// #include <vector>
// #include <chrono>

typedef void (*FunctionPtr)(double* input, double* output, int size, void* context);

void MatrixMul(double* d_matrix, double* d_vectors, double* d_result, int rows, int cols, int vec_num);

// void CudaIntegration(FunctionPtr func_ptr, double* d_sigmapts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols, void* context);
void CudaIntegration(FunctionPtr func_ptr, double* d_sigmapts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols, void* context, double* d_pts1, double* d_pts2);

void CudaIntegration1(double* d_pts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols);

#endif // MATRIX_MULTIPLICATION_H