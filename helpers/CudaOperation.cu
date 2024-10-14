#include <cuda_runtime.h>
#include <optional>
#include "helpers/CudaOperation.h"

using namespace Eigen;

__global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* mu,
                               int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, int type, 
                               gvi::CudaOperation_PlanarPR* pointer, double* d_data){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < res_rows && col < res_cols*sigmapts_rows){
        int idx = col / res_cols;
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts, sigmapts_rows, sigmapts_cols);

        (pointer->_sdf).data_array_ = d_data;

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

__global__ void Sigma_function_quad(double* d_sigmapts, double* d_pts, double* mu,
                               int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, int type, 
                               gvi::CudaOperation_Quad* pointer, double* d_data){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < res_rows && col < res_cols*sigmapts_rows){
        int idx = col / res_cols;
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts, sigmapts_rows, sigmapts_cols);

        (pointer->_sdf).data_array_ = d_data;

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

__global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* mu,
                               int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, int type, 
                               gvi::CudaOperation_3dpR* pointer, double* d_data){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < res_rows && col < res_cols*sigmapts_rows){
        int idx = col / res_cols;
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts, sigmapts_rows, sigmapts_cols);

        (pointer->_sdf).data_array_ = d_data;

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


__global__ void cost_function(double* d_sigmapts, double* d_pts, int sigmapts_rows, int sigmapts_cols, 
                                int n_states, gvi::CudaOperation_PlanarPR* pointer, double* d_data){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sigmapts_rows && col < n_states){
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts + col*sigmapts_rows*sigmapts_cols, sigmapts_rows, sigmapts_cols);

        (pointer->_sdf).data_array_ = d_data;

        double function_value = pointer -> cost_obstacle_planar(sigmapts.row(row), pointer->_sdf);

        d_pts[col*sigmapts_rows + row] = function_value;
    }
}

__global__ void cost_function(double* d_sigmapts, double* d_pts, int sigmapts_rows, int sigmapts_cols, 
                                int n_states, gvi::CudaOperation_Quad* pointer, double* d_data){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sigmapts_rows && col < n_states){
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts + col*sigmapts_rows*sigmapts_cols, sigmapts_rows, sigmapts_cols);

        (pointer->_sdf).data_array_ = d_data;

        double function_value = pointer -> cost_obstacle_planar(sigmapts.row(row), pointer->_sdf);

        d_pts[col*sigmapts_rows + row] = function_value;
    }
}

__global__ void cost_function(double* d_sigmapts, double* d_pts, int sigmapts_rows, int sigmapts_cols, 
                                int n_states, gvi::CudaOperation_3dpR* pointer, double* d_data){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sigmapts_rows && col < n_states){
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts + col*sigmapts_rows*sigmapts_cols, sigmapts_rows, sigmapts_cols);

        (pointer->_sdf).data_array_ = d_data;

        double function_value = pointer -> cost_obstacle_planar(sigmapts.row(row), pointer->_sdf);

        d_pts[col*sigmapts_rows + row] = function_value;
    }
}

__global__ void dmu_function(double* d_sigmapts, double* d_mu, double* d_pts, double* d_vec, int sigmapts_rows, int dim_state, int n_states){
// __global__ void dmu_function(int sigmapts_rows, int dim_state, int n_states){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sigmapts_rows && col < n_states*dim_state){
        int idx = col / dim_state;
        int r = col % dim_state;

        d_vec[col*sigmapts_rows + row] = (d_sigmapts[col*sigmapts_rows + row] - d_mu[idx*dim_state + r]) * d_pts[idx * sigmapts_rows + row];
    }
}

__global__ void ddmu_function(double* d_sigmapts, double* d_mu, double* d_pts, double* d_vec, int sigmapts_rows, int sigmapts_cols, int n_states){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sigmapts_rows*sigmapts_cols && col < n_states*sigmapts_cols){
        int idx_x = col / sigmapts_cols; // Use which mu
        int idx_y = row / sigmapts_cols; // use which sigma
        int mat_x = col % sigmapts_cols; // 
        int mat_y = row % sigmapts_cols;

        d_vec[col*sigmapts_rows*sigmapts_cols + row] = (d_sigmapts[(idx_x*sigmapts_cols + mat_x) * sigmapts_rows + idx_y] - d_mu[idx_x*sigmapts_cols + mat_x]) 
                                                     * (d_sigmapts[(idx_x*sigmapts_cols + mat_y) * sigmapts_rows + idx_y] - d_mu[idx_x*sigmapts_cols + mat_y]) 
                                                     * d_pts[idx_x * sigmapts_rows + idx_y];
    }
}

__global__ void obtain_cost(double* d_pts, double* d_weights, double* d_result, int sigmapts_rows, int n_states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_states){
        double sum = 0;
        for(int i = 0; i < sigmapts_rows; i++){
            sum += d_pts[idx*sigmapts_rows + i]* d_weights[i];
        }
        d_result[idx] = sum;
    }
}

__global__ void obtain_dmu(double* d_vec, double* d_weights, double* d_result, int sigmapts_rows, int res_cols){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < res_cols){
        double sum = 0;
        for(int i = 0; i < sigmapts_rows; i++){
            sum += d_vec[idx*sigmapts_rows + i]* d_weights[i];
        }
        d_result[idx] = sum;
    }
}

__global__ void obtain_ddmu(double* d_vec, double* d_weights, double* d_result, int sigmapts_rows, int res_rows, int res_cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < res_rows && col < res_cols){
        double sum = 0;
        for(int i = 0; i < sigmapts_rows; i++){
            sum += d_vec[col*sigmapts_rows*res_rows + row + i*res_rows] * d_weights[i];
        }
        d_result[col*res_rows + row] = sum;
    }
}


namespace gvi{

void CudaOperation_PlanarPR::CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type)
{
    double *sigmapts_gpu, *mu_gpu, *pts_gpu, *result_gpu;
    int n_balls = 1;

    cudaMalloc(&sigmapts_gpu, sigmapts.size() * sizeof(double));
    cudaMalloc(&mu_gpu, sigmapts.cols() * sizeof(double));
    cudaMalloc(&pts_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    cudaMemcpy(sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_gpu, mean.data(), sigmapts.cols() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.cols()*sigmapts.rows() + blockSize1.x - 1) / blockSize1.x, (results.rows() + blockSize1.y - 1) / blockSize1.y);

    Sigma_function<<<blockSize1, threadperblock1>>>(sigmapts_gpu, pts_gpu, mu_gpu, sigmapts.rows(), sigmapts.cols(), results.rows(), results.cols(), type, _class_gpu, _data_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(sigmapts_gpu);
    cudaFree(mu_gpu);
    

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16, 16);
    dim3 threadperblock2((results.cols() + blockSize2.x - 1) / blockSize2.x, (results.rows() + blockSize2.y - 1) / blockSize2.y);

    obtain_res<<<blockSize2, threadperblock2>>>(pts_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.rows(), results.cols());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(pts_gpu);
    cudaFree(result_gpu);
}

void CudaOperation_PlanarPR::costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){
    double *result_gpu;
    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    cudaMemcpy(_sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.size() + blockSize1.x - 1) / blockSize1.x, (sigmapts.rows() + blockSize1.y - 1) / blockSize1.y);

    cost_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _func_value_gpu, sigmapts.rows(), sigmapts_cols, results.size(), _class_gpu, _data_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16);
    dim3 threadperblock2((results.size() + blockSize2.x - 1) / blockSize2.x);

    obtain_cost<<<blockSize2, threadperblock2>>>(_func_value_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(result_gpu);
}

void CudaOperation_PlanarPR::dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols){

    double *vec_gpu, *result_gpu;
    cudaMalloc(&vec_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMalloc(&_mu_gpu, mu.size() * sizeof(double));

    cudaMemcpy(_mu_gpu, mu.data(), mu.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.size() + blockSize1.x - 1) / blockSize1.x, (_sigmapts_rows + blockSize1.y - 1) / blockSize1.y);

    dmu_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _mu_gpu, _func_value_gpu, vec_gpu, _sigmapts_rows, _dim_state, _n_states);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16);
    dim3 threadperblock2((results.size() + blockSize2.x - 1) / blockSize2.x);

    obtain_dmu<<<blockSize2, threadperblock2>>>(vec_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(vec_gpu);
    cudaFree(result_gpu);
}

void CudaOperation_PlanarPR::ddmuIntegration(MatrixXd& results){
    // Reuse the sigmapts and pts passed into gpu before
    double *vec_gpu, *result_gpu;
    cudaMalloc(&vec_gpu, _sigmapts_rows * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.cols() + blockSize1.x - 1) / blockSize1.x, (_sigmapts_rows * results.rows() + blockSize1.y - 1) / blockSize1.y);

    ddmu_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _mu_gpu, _func_value_gpu, vec_gpu, _sigmapts_rows, _dim_state, _n_states);
    cudaDeviceSynchronize();

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16, 16);
    dim3 threadperblock2((results.cols() + blockSize2.x - 1) / blockSize2.x, (results.rows() + blockSize2.y - 1) / blockSize2.y);

    obtain_ddmu<<<blockSize2, threadperblock2>>>(vec_gpu, _weight_gpu, result_gpu, _sigmapts_rows, results.rows(), results.cols());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(vec_gpu);
    cudaFree(result_gpu);
    cudaFree(_mu_gpu);
}


void CudaOperation_Quad::CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type)
{
    double *sigmapts_gpu, *pts_gpu, *result_gpu, *mu_gpu;
    int n_balls = 5;

    cudaMalloc(&sigmapts_gpu, sigmapts.size() * sizeof(double));
    cudaMalloc(&pts_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMalloc(&mu_gpu, sigmapts.cols() * sizeof(double));

    cudaMemcpy(sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_gpu, mean.data(), sigmapts.cols() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.cols()*sigmapts.rows() + blockSize1.x - 1) / blockSize1.x, (results.rows() + blockSize1.y - 1) / blockSize1.y);

    Sigma_function_quad<<<blockSize1, threadperblock1>>>(sigmapts_gpu, pts_gpu, mu_gpu, sigmapts.rows(), sigmapts.cols(), results.rows(), results.cols(), type, _class_gpu, _data_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(sigmapts_gpu);
    cudaFree(mu_gpu);

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16, 16);
    dim3 threadperblock2((results.cols() + blockSize2.x - 1) / blockSize2.x, (results.rows() + blockSize2.y - 1) / blockSize2.y);

    obtain_res<<<blockSize2, threadperblock2>>>(pts_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.rows(), results.cols());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(pts_gpu);
    cudaFree(result_gpu);
}

void CudaOperation_Quad::costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){
    double *result_gpu;
    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    cudaMemcpy(_sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.size() + blockSize1.x - 1) / blockSize1.x, (sigmapts.rows() + blockSize1.y - 1) / blockSize1.y);

    cost_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _func_value_gpu, sigmapts.rows(), sigmapts_cols, results.size(), _class_gpu, _data_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16);
    dim3 threadperblock2((results.size() + blockSize2.x - 1) / blockSize2.x);

    obtain_cost<<<blockSize2, threadperblock2>>>(_func_value_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // cudaFree(_func_value_gpu);
    cudaFree(result_gpu);
}

void CudaOperation_Quad::dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols){

    double *vec_gpu, *result_gpu;
    cudaMalloc(&vec_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMalloc(&_mu_gpu, mu.size() * sizeof(double));

    cudaMemcpy(_mu_gpu, mu.data(), mu.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.size() + blockSize1.x - 1) / blockSize1.x, (_sigmapts_rows + blockSize1.y - 1) / blockSize1.y);

    dmu_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _mu_gpu, _func_value_gpu, vec_gpu, _sigmapts_rows, _dim_state, _n_states);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16);
    dim3 threadperblock2((results.size() + blockSize2.x - 1) / blockSize2.x);

    obtain_dmu<<<blockSize2, threadperblock2>>>(vec_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(vec_gpu);
    cudaFree(result_gpu);
}

void CudaOperation_Quad::ddmuIntegration(MatrixXd& results){
    // Reuse the sigmapts and pts passed into gpu before
    double *vec_gpu, *result_gpu;
    cudaMalloc(&vec_gpu, _sigmapts_rows * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.cols() + blockSize1.x - 1) / blockSize1.x, (_sigmapts_rows * results.rows() + blockSize1.y - 1) / blockSize1.y);

    ddmu_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _mu_gpu, _func_value_gpu, vec_gpu, _sigmapts_rows, _dim_state, _n_states);
    cudaDeviceSynchronize();

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16, 16);
    dim3 threadperblock2((results.cols() + blockSize2.x - 1) / blockSize2.x, (results.rows() + blockSize2.y - 1) / blockSize2.y);

    obtain_ddmu<<<blockSize2, threadperblock2>>>(vec_gpu, _weight_gpu, result_gpu, _sigmapts_rows, results.rows(), results.cols());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(vec_gpu);
    cudaFree(result_gpu);
    cudaFree(_mu_gpu);
}


void CudaOperation_3dpR::CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type)
{
    double *sigmapts_gpu, *mu_gpu, *pts_gpu, *result_gpu;
    int n_balls = 1;

    cudaMalloc(&sigmapts_gpu, sigmapts.size() * sizeof(double));
    cudaMalloc(&mu_gpu, sigmapts.cols() * sizeof(double));
    cudaMalloc(&pts_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    cudaMemcpy(sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_gpu, mean.data(), sigmapts.cols() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.cols()*sigmapts.rows() + blockSize1.x - 1) / blockSize1.x, (results.rows() + blockSize1.y - 1) / blockSize1.y);

    Sigma_function<<<blockSize1, threadperblock1>>>(sigmapts_gpu, pts_gpu, mu_gpu, sigmapts.rows(), sigmapts.cols(), results.rows(), results.cols(), type, _class_gpu, _data_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(sigmapts_gpu);
    cudaFree(mu_gpu);
    

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16, 16);
    dim3 threadperblock2((results.cols() + blockSize2.x - 1) / blockSize2.x, (results.rows() + blockSize2.y - 1) / blockSize2.y);

    obtain_res<<<blockSize2, threadperblock2>>>(pts_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.rows(), results.cols());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(pts_gpu);
    cudaFree(result_gpu);
}

void CudaOperation_3dpR::costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){
    double *result_gpu;

    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    cudaMemcpy(_sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.size() + blockSize1.x - 1) / blockSize1.x, (sigmapts.rows() + blockSize1.y - 1) / blockSize1.y);

    cost_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _func_value_gpu, sigmapts.rows(), sigmapts_cols, results.size(), _class_gpu, _data_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16);
    dim3 threadperblock2((results.size() + blockSize2.x - 1) / blockSize2.x);

    obtain_cost<<<blockSize2, threadperblock2>>>(_func_value_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(result_gpu);
}

void CudaOperation_3dpR::dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols){

    double *vec_gpu, *result_gpu;
    cudaMalloc(&vec_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMalloc(&_mu_gpu, mu.size() * sizeof(double));

    cudaMemcpy(_mu_gpu, mu.data(), mu.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.size() + blockSize1.x - 1) / blockSize1.x, (_sigmapts_rows + blockSize1.y - 1) / blockSize1.y);

    dmu_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _mu_gpu, _func_value_gpu, vec_gpu, _sigmapts_rows, _dim_state, _n_states);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(16);
    dim3 threadperblock2((results.size() + blockSize2.x - 1) / blockSize2.x);

    obtain_dmu<<<blockSize2, threadperblock2>>>(vec_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(vec_gpu);
    cudaFree(result_gpu);
}

void CudaOperation_3dpR::ddmuIntegration(MatrixXd& results){
    // Reuse the sigmapts and pts passed into gpu before
    double *vec_gpu, *result_gpu;
    cudaMalloc(&vec_gpu, _sigmapts_rows * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    // Kernel 1: Obtain the result of function 
    dim3 blockSize1(64, 64);
    dim3 threadperblock1((results.cols() + blockSize1.x - 1) / blockSize1.x, (_sigmapts_rows * results.rows() + blockSize1.y - 1) / blockSize1.y);

    ddmu_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _mu_gpu, _func_value_gpu, vec_gpu, _sigmapts_rows, _dim_state, _n_states);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(64, 64);
    dim3 threadperblock2((results.cols() + blockSize2.x - 1) / blockSize2.x, (results.rows() + blockSize2.y - 1) / blockSize2.y);

    obtain_ddmu<<<blockSize2, threadperblock2>>>(vec_gpu, _weight_gpu, result_gpu, _sigmapts_rows, results.rows(), results.cols());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(vec_gpu);
    cudaFree(result_gpu);
    cudaFree(_mu_gpu);
}

}





// void CudaOperation_PlanarPR::funcValueIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){

//     double *result_gpu;
//     cudaMalloc(&result_gpu, results.size() * sizeof(double));

//     cudaMemcpy(_sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);

//     // Kernel 1: Obtain the result of function 
//     dim3 blockSize1(16, 16);
//     dim3 threadperblock1((_n_states + blockSize1.x - 1) / blockSize1.x, (sigmapts.rows() + blockSize1.y - 1) / blockSize1.y);

//     cost_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _func_value_gpu, sigmapts.rows(), sigmapts_cols, _n_states, _class_gpu, _data_gpu);
//     cudaDeviceSynchronize();

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
//     }

//     // Kernel 2: Obtain the result by multiplying the pts and the weights
//     dim3 blockSize2(16);
//     dim3 threadperblock2((results.size() + blockSize2.x - 1) / blockSize2.x);

//     obtain_cost<<<blockSize2, threadperblock2>>>(_func_value_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
//     cudaDeviceSynchronize();
//     cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

//     err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
//     }

//     cudaFree(result_gpu);
// }