#include <cuda_runtime.h>
#include <optional>
#include "ngd/NGDFactorizedBaseGH_Cuda.h"
// #include "helpers/CudaOperation.h"

using namespace Eigen;
using GHFunction = std::function<MatrixXd(const VectorXd&)>;

template <typename CostClass>
__global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* mu, int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, int type, gvi::NGDFactorizedBaseGH_Cuda<CostClass>* pointer, const CostClass& cost_class){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sigmapts_rows){
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts, sigmapts_rows, sigmapts_cols);
        Eigen::Map<MatrixXd> pts(d_sigmapts, res_rows, sigmapts_rows*res_cols);
        Eigen::Map<VectorXd> mean(mu, sigmapts_cols);

        double function_value = pointer -> cost_obstacle_planar(sigmapts.row(idx));
        printf("function value: %lf\n", function_value);
        // double function_value = 1;
        // double function_value = pointer -> _function(sigmapts.row(idx), pointer -> _cost_class);

        if (type == 0)
            d_pts[idx] = function_value;
        else if (type == 1){
            for (int i=0; i<sigmapts_cols; i++)
                d_pts[idx*sigmapts_cols + i] = (d_sigmapts[idx + sigmapts_cols * i] - mu[i]) * function_value;
        }
        else{
            for (int i=0; i<sigmapts_cols; i++)
                for (int j=0; j<sigmapts_cols; j++)
                    d_pts[idx*sigmapts_cols *sigmapts_cols+ i*sigmapts_cols +j] = (d_sigmapts[idx*sigmapts_cols + i] - mu[i]) * (d_sigmapts[idx*sigmapts_cols + j] - mu[j]) * function_value;

        }
    }
}

__global__ void obtain_res(double* d_pts, double* d_weights, double* d_result, int sigmapts_rows, int res_rows, int res_cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < res_rows && col < res_cols){
        double sum = 0;
        for(int i = 0; i < sigmapts_rows; i++){
            sum += d_pts[i*res_rows*res_cols + col*res_rows + row] * d_weights[i];
        }
        d_result[col*res_rows + row] = sum;
    }
}


namespace gvi{

template <typename CostClass>
void NGDFactorizedBaseGH_Cuda<CostClass>::CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type, MatrixXd& pts)
{
    double *sigmapts_gpu, *pts_gpu, *weight_gpu, *result_gpu, *mu_gpu;

    NGDFactorizedBaseGH_Cuda<CostClass>* class_gpu;
    cudaMalloc(&class_gpu, sizeof(NGDFactorizedBaseGH_Cuda<CostClass>));
    cudaMemcpy(class_gpu, this, sizeof(NGDFactorizedBaseGH_Cuda<CostClass>), cudaMemcpyHostToDevice);

    // std::cout << sizeof(NGDFactorizedBaseGH<CostClass>) << std::endl;

    cudaMalloc(&sigmapts_gpu, sigmapts.size() * sizeof(double));
    cudaMalloc(&pts_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&weight_gpu, sigmapts.rows() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMalloc(&mu_gpu, sigmapts.cols() * sizeof(double));

    cudaMemcpy(sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_gpu, weights.data(), sigmapts.rows() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_gpu, mean.data(), sigmapts.cols() * sizeof(double), cudaMemcpyHostToDevice);

    // Dimension for the first kernel function
    dim3 blockSize1(3);
    dim3 threadperblock1((sigmapts.rows() + blockSize1.x - 1) / blockSize1.x);

    // std::cout << "Kernel 1" << std::endl;
    // Kernel 1: Obtain the result of function 
    Sigma_function<<<blockSize1, threadperblock1>>>(sigmapts_gpu, pts_gpu, mu_gpu, sigmapts.rows(), sigmapts.cols(), results.rows(), results.cols(), type, class_gpu, this -> _cost_class);
    cudaDeviceSynchronize();
    // std::cout << "After Kernel 1" << std::endl;

    cudaMemcpy(pts.data(), pts_gpu, sigmapts.rows() * results.size() * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(pts_gpu, d_pts1, sigma_rows * res_rows * res_cols * sizeof(double), cudaMemcpyHostToDevice);
    
    // Dimension for the second kernel function
    dim3 blockSize2(3, 3);
    dim3 threadperblock2((results.cols() + blockSize2.x - 1) / blockSize2.x, (results.rows() + blockSize2.y - 1) / blockSize2.y);

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    obtain_res<<<blockSize2, threadperblock2>>>(pts_gpu, weight_gpu, result_gpu, sigmapts.rows(), results.rows(), results.cols());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(sigmapts_gpu);
    cudaFree(pts_gpu);
    cudaFree(weight_gpu);
    cudaFree(result_gpu);
    
}

// template class NGDFactorizedBaseGH_Cuda<NoneType>;
template class NGDFactorizedBaseGH_Cuda<gpmp2::ObstaclePlanarSDFFactor<gpmp2::PointRobotModel>>;

}