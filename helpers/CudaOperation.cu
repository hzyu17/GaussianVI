#include <cuda_runtime.h>
#include <optional>
// #include "ngd/NGDFactorizedBaseGH_Cuda.h"
// #include <gpmp2/obstacle/ObstaclePlanarSDFFactor.h>
#include "helpers/CudaOperation.h"

using namespace Eigen;
using GHFunction = std::function<MatrixXd(const VectorXd&)>;

template <typename CostClass>
__global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* mu,
                               int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, int type, 
                               gvi::CudaOperation<CostClass>* pointer, double* d_data){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < res_rows && col < res_cols*sigmapts_rows){
        int idx = col / res_cols;

        Eigen::Map<MatrixXd> sigmapts(d_sigmapts, sigmapts_rows, sigmapts_cols);
        // Eigen::Map<MatrixXd> pts(d_sigmapts, res_rows, sigmapts_rows*res_cols);
        // Eigen::Map<MatrixXd> data_map(d_data, pointer->_sdf.field_rows_, pointer->_sdf.field_cols_);
        // Eigen::Map<VectorXd> mean(mu, sigmapts_cols);

        (pointer->_sdf).data_array = d_data;

        double function_value = pointer -> cost_obstacle_planar(sigmapts.row(idx), pointer->_sdf);

        if (type == 0)
            d_pts[idx*res_rows + row] = function_value;
        else if (type == 1)
            d_pts[idx*res_rows + row] = (d_sigmapts[idx + sigmapts_rows * row] - mu[row]) * function_value;
        else{
            int r = col % res_cols;
            d_pts[idx*sigmapts_cols*sigmapts_cols+ r*sigmapts_cols + row] = (d_sigmapts[idx + sigmapts_rows * row] - mu[row]) * (d_sigmapts[idx + sigmapts_rows * r] - mu[r]) * function_value;
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
void CudaOperation<CostClass>::CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type, MatrixXd& pts)
{
    double *sigmapts_gpu, *pts_gpu, *weight_gpu, *result_gpu, *mu_gpu, *data_gpu;

    CudaOperation<CostClass>* class_gpu;
    cudaMalloc(&class_gpu, sizeof(CudaOperation<CostClass>));
    cudaMemcpy(class_gpu, this, sizeof(CudaOperation<CostClass>), cudaMemcpyHostToDevice);


    cudaMalloc(&sigmapts_gpu, sigmapts.size() * sizeof(double));
    cudaMalloc(&pts_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&weight_gpu, sigmapts.rows() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMalloc(&mu_gpu, sigmapts.cols() * sizeof(double));
    cudaMalloc(&data_gpu, _sdf.data_.size() * sizeof(double));


    cudaMemcpy(sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_gpu, weights.data(), sigmapts.rows() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_gpu, mean.data(), sigmapts.cols() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data_gpu, _sdf.data_.data(), _sdf.data_.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Dimension for the first kernel function
    dim3 blockSize1(16, 4);
    dim3 threadperblock1((results.cols()*sigmapts.rows() + blockSize1.x - 1) / blockSize1.x, (results.rows() + blockSize1.y - 1) / blockSize1.y);

    // Kernel 1: Obtain the result of function 
    Sigma_function<<<blockSize1, threadperblock1>>>(sigmapts_gpu, pts_gpu, mu_gpu, sigmapts.rows(), sigmapts.cols(), results.rows(), results.cols(), type, class_gpu, data_gpu);
    cudaDeviceSynchronize();

    cudaMemcpy(pts.data(), pts_gpu, sigmapts.rows() * results.size() * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Dimension for the second kernel function
    dim3 blockSize2(4, 4);
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
template class CudaOperation<gpmp2::ObstaclePlanarSDFFactor<gpmp2::PointRobotModel>>;

}