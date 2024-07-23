#ifndef CUDA_OPERATION_H
#define CUDA_OPERATION_H

#include <cuda_runtime.h>
#include <iostream>
#include <Eigen/Dense>
// #include "ngd/NGDFactorizedBaseGH_Cuda.h"

using namespace Eigen;

using GHFunction = std::function<MatrixXd(const VectorXd&)>;

namespace gvi{

template <typename CostClass>
class CudaOperation{

public:
    CudaOperation(){}

    void CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type);

    // void CudaIntegration1(MatrixXd& d_pts, const MatrixXd& d_weights, MatrixXd& d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols);

    __host__ __device__ inline double cost_function1(const VectorXd& vec_x){
        double x = vec_x(0);
        double mu_p = 20, f = 400, b = 0.1, sig_r_sq = 0.09;
        double sig_p_sq = 9;

        // y should be sampled. for single trial just give it a value.
        double y = f*b/mu_p - 0.8;

        return ((x - mu_p)*(x - mu_p) / sig_p_sq / 2 + (y - f*b/x)*(y - f*b/x) / sig_r_sq / 2); 
    }

    // __host__ __device__ inline MatrixXd sigma_func(const VectorXd& vec_x){
    //     return _func(vec_x);
    // }

    // GHFunction _func;

};

// __global__ void MatrixMultiplication(double* d_matrix, double* d_vectors, double* d_result, int rows, int cols, int vec_num);

// __global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* mu, int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, FunctionPtr func_ptr, void* context, int type);
  
// __global__ void obtain_res(double* pts, double* weights, double* results, int sigma_rows, int res_rows, int res_cols);

// __global__ void func_Vmu(const double* vec_x, double* pt, const double* mu, int dim, int index, double& func_value);

// __global__ void func_Vmumu(const double* vec_x, double* pt, const double* mu, int dim, int index, double& func_value);

}


#endif // CUDA_OPERATION_H