#include <cuda_runtime.h>
#include "helpers/CudaOperation.h"

using namespace Eigen;

template class CudaOperation_Base<PlanarSDF>;
template class CudaOperation_Base<SignedDistanceField>;

template <typename RobotType>
__global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* mu,
                               int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, int type, 
                               RobotType* pointer, double* d_data){
    
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
                               gvi::CudaOperation_3dArm* pointer, double* d_data){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < res_rows && col < res_cols*sigmapts_rows){
        int idx = col / res_cols;
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts, sigmapts_rows, sigmapts_cols);

        (pointer->_sdf).data_array_ = d_data;

        double function_value = pointer -> cost_obstacle(sigmapts.row(idx), pointer->_sdf, pointer->_fk);

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

template <typename RobotType>
__global__ void cost_function(double* d_sigmapts, double* d_pts, int sigmapts_rows, int sigmapts_cols, 
                                int n_states, RobotType* pointer, double* d_data){
    
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
                                int n_states, gvi::CudaOperation_3dArm* pointer, double* sdf_data, 
                                double* a_data, double* alpha_data, double* d_data, double* theta_data,
                                double* rad_data, int* frames_data, double* centers_data){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sigmapts_rows && col < n_states){
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts + col*sigmapts_rows*sigmapts_cols, sigmapts_rows, sigmapts_cols);

        (pointer->_sdf).data_array_ = sdf_data;

        (pointer->_fk)._a_data = a_data;
        (pointer->_fk)._alpha_data = alpha_data;
        (pointer->_fk)._d_data = d_data;
        (pointer->_fk)._theta_bias_data = theta_data;

        pointer->_radii_data = rad_data;
        (pointer->_fk)._frames_data = frames_data;
        (pointer->_fk)._centers_data = centers_data;

        double function_value = pointer -> cost_obstacle(sigmapts.row(row), pointer->_sdf, pointer->_fk);

        d_pts[col*sigmapts_rows + row] = function_value;
    }
}

__global__ void dmu_function(double* d_sigmapts, double* d_mu, double* d_pts, double* d_vec, int sigmapts_rows, int dim_conf, int n_states){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sigmapts_rows && col < n_states*dim_conf){
        int idx = col / dim_conf;
        int r = col % dim_conf;

        d_vec[col*sigmapts_rows + row] = (d_sigmapts[col*sigmapts_rows + row] - d_mu[idx*dim_conf + r]) * d_pts[idx * sigmapts_rows + row];
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


__global__ void sqrtKernel(double* d_vals, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_vals[idx] = sqrt(d_vals[idx]);
    }
}

__global__ void addMeanKernel(double* sigmaPts, const double* mean, int num_rows, int dim_state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * dim_state;
    if (idx < total)
    {
        int j = idx / num_rows;
        sigmaPts[idx] += mean[j];
    }
}



namespace gvi{

template <typename SDFType>
void CudaOperation_Base<SDFType>::update_sigmapts(const MatrixXd& covariance, const MatrixXd& mean, int dim_conf, int num_states, MatrixXd& sigmapts){
    cusolverDnHandle_t cusolverH;
    cublasHandle_t cublasH;

    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);

    int state_idx = 0;

    // Compute the Cholesky decomposition of the covariance matrix
    double *covariance_gpu, *mean_gpu, *d_eigen_values, *d_work, *d_sqrtP, *d_sigmapts;
    int *d_info, Lwork;
    cudaMalloc(&covariance_gpu, dim_conf * dim_conf * num_states * sizeof(double));
    cudaMalloc(&d_info, sizeof(int));
    cudaMalloc(&d_eigen_values, dim_conf * sizeof(double));
    cudaMalloc(&d_sqrtP, dim_conf * dim_conf * sizeof(double));
    cudaMalloc(&d_sigmapts, _sigmapts_rows * dim_conf * sizeof(double));

    cudaMalloc(&mean_gpu, mean.size() * sizeof(double));

    cudaMemcpy(covariance_gpu, covariance.data(), dim_conf * dim_conf * num_states * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mean_gpu, mean.data(), mean.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* d_Pi = covariance_gpu + state_idx * dim_conf * dim_conf;
    double* d_mi = mean_gpu + 2 * (state_idx+1) * dim_conf;
    
    cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                            dim_conf, d_Pi, dim_conf, d_eigen_values, &Lwork);
    cudaMalloc(&d_work, Lwork * sizeof(double));

    cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                    dim_conf, d_Pi, dim_conf, d_eigen_values, d_work, Lwork, d_info);

    // Check if the Cholesky decomposition was successful
    int h_info;
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        std::cerr << "Cholesky decomposition failed with info: " << h_info << std::endl;
    }

    int threadsPerBlock = 16;
    int blocks = (dim_conf + threadsPerBlock - 1) / threadsPerBlock;
    sqrtKernel<<<blocks, threadsPerBlock>>>(d_eigen_values, dim_conf);
    cudaDeviceSynchronize();

    double* d_V_scaled;
    cudaMalloc(&d_V_scaled, dim_conf * dim_conf * sizeof(double));

    cublasStatus_t stat = cublasDdgmm(cublasH, CUBLAS_SIDE_RIGHT, dim_conf, dim_conf, d_Pi, dim_conf, d_eigen_values, 1, d_V_scaled, dim_conf);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasDdgmm failed" << std::endl;
    }

    const double alpha = 1.0, beta = 0.0;
    stat = cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, dim_conf, dim_conf, dim_conf, &alpha, d_V_scaled, dim_conf,
                       d_Pi, dim_conf, &beta, d_sqrtP, dim_conf);

    stat = cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, _sigmapts_rows, dim_conf, dim_conf, &alpha,
                        _zeromean_gpu, _sigmapts_rows, d_sqrtP, dim_conf, &beta, d_sigmapts, _sigmapts_rows); 

    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasDgemm failed" << std::endl;
    }

    threadsPerBlock = 256;
    blocks = (_sigmapts_rows * dim_conf + threadsPerBlock - 1) / threadsPerBlock;
    addMeanKernel<<<blocks, threadsPerBlock>>>(d_sigmapts, d_mi, _sigmapts_rows, dim_conf);
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
         std::cerr << "cudaDeviceSynchronize error: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    MatrixXd sigma(_sigmapts_rows, dim_conf);
    cudaMemcpy(sigma.data(), d_sigmapts, _sigmapts_rows * dim_conf * sizeof(double), cudaMemcpyDeviceToHost);
    sigmapts = sigma;
    

    // MatrixXd sqrtP(dim_conf, dim_conf);
    // cudaMemcpy(sqrtP.data(), d_sqrtP, dim_conf * dim_conf * sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << "Cuda result: " << std::endl << sqrtP << std::endl;

    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_V_scaled);
    cusolverDnDestroy(cusolverH);
    // cublasDestroy(cublasH);
}

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
    dim3 blockSize1(256, 256);
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
    dim3 blockSize2(256, 256);
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
    dim3 threadperblock1(32, 32);
    dim3 blockSize1((results.size() + threadperblock1.x - 1) / threadperblock1.x, (sigmapts.rows() + threadperblock1.y - 1) / threadperblock1.y);

    cost_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _func_value_gpu, sigmapts.rows(), sigmapts_cols, results.size(), _class_gpu, _data_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 threadperblock2(256);
    dim3 blockSize2((results.size() + threadperblock2.x - 1) / threadperblock2.x);

    obtain_cost<<<blockSize2, threadperblock2>>>(_func_value_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(result_gpu);
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
    dim3 blockSize1(64, 64);
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
    dim3 blockSize2(64, 64);
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
    dim3 blockSize1(64, 64);
    dim3 threadperblock1((results.size() + blockSize1.x - 1) / blockSize1.x, (sigmapts.rows() + blockSize1.y - 1) / blockSize1.y);

    cost_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _func_value_gpu, sigmapts.rows(), sigmapts_cols, results.size(), _class_gpu, _data_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2(64);
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
    dim3 blockSize1(64, 64);
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
    dim3 blockSize2(64, 64);
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
    dim3 threadperblock1(32, 32);
    dim3 blockSize1((results.size() + threadperblock1.x - 1) / threadperblock1.x, (sigmapts.rows() + threadperblock1.y - 1) / threadperblock1.y);

    cost_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _func_value_gpu, sigmapts.rows(), sigmapts_cols, results.size(), _class_gpu, _data_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 threadperblock2(256);
    dim3 blockSize2((results.size() + threadperblock2.x - 1) / threadperblock2.x);

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

template <typename SDFType>
void CudaOperation_Base<SDFType>::dmuIntegration(const MatrixXd& sigmapts, const MatrixXd& mu, VectorXd& results, const int sigmapts_cols){

    double *vec_gpu, *result_gpu;
    cudaMalloc(&vec_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMalloc(&_mu_gpu, mu.size() * sizeof(double));

    cudaMemcpy(_mu_gpu, mu.data(), mu.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 threadperblock1(32, 32);
    dim3 blockSize1((results.size() + threadperblock1.x - 1) / threadperblock1.x, (_sigmapts_rows + threadperblock1.y - 1) / threadperblock1.y);

    dmu_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _mu_gpu, _func_value_gpu, vec_gpu, _sigmapts_rows, _dim_conf, _n_states);
    cudaDeviceSynchronize();

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 threadperblock2(256);
    dim3 blockSize2((results.size() + threadperblock2.x - 1) / threadperblock2.x);

    obtain_dmu<<<blockSize2, threadperblock2>>>(vec_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(vec_gpu);
    cudaFree(result_gpu);
}

template <typename SDFType>
void CudaOperation_Base<SDFType>::ddmuIntegration(MatrixXd& results){
    // Reuse the sigmapts and pts passed into gpu before
    double *vec_gpu, *result_gpu;
    cudaMalloc(&vec_gpu, _sigmapts_rows * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    // Kernel 1: Obtain the result of function 
    dim3 threadperblock(32, 32); //1024
    dim3 blockSize1((results.cols() + threadperblock.x - 1) / threadperblock.x, (_sigmapts_rows * results.rows() + threadperblock.y - 1) / threadperblock.y);

    ddmu_function<<<blockSize1, threadperblock>>>(_sigmapts_gpu, _mu_gpu, _func_value_gpu, vec_gpu, _sigmapts_rows, _dim_conf, _n_states);
    cudaDeviceSynchronize();

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2((results.cols() + threadperblock.x - 1) / threadperblock.x, (results.rows() + threadperblock.y - 1) / threadperblock.y);

    obtain_ddmu<<<blockSize2, threadperblock>>>(vec_gpu, _weight_gpu, result_gpu, _sigmapts_rows, results.rows(), results.cols());
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

void CudaOperation_3dArm::CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type)
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
    dim3 blockSize1(1024, 1024);
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
    dim3 blockSize2(1024, 1024);
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

void CudaOperation_3dArm::costIntegration(const MatrixXd& sigmapts, VectorXd& results, const int sigmapts_cols){
    double *result_gpu;

    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMemcpy(_sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Kernel 1: Obtain the result of function 
    dim3 threadperblock1(32, 32);
    dim3 blockSize1((results.size() + threadperblock1.x - 1) / threadperblock1.x, (sigmapts.rows() + threadperblock1.y - 1) / threadperblock1.y);

    cost_function<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _func_value_gpu, sigmapts.rows(), sigmapts_cols, results.size(), 
                                                    _class_gpu, _data_gpu, _a_gpu, _alpha_gpu, _d_gpu, _theta_gpu,
                                                    _rad_gpu, _frames_gpu, _centers_gpu);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 threadperblock2(256);
    dim3 blockSize2((results.size() + threadperblock2.x - 1) / threadperblock2.x);

    obtain_cost<<<blockSize2, threadperblock2>>>(_func_value_gpu, _weight_gpu, result_gpu, sigmapts.rows(), results.size());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(result_gpu);
}



// set m, l, J as input
__host__ __device__ void function_value(const VectorXd& sigmapt, VectorXd& function_value){
    double px = sigmapt(0);       // p_x
    double pz = sigmapt(1);       // p_z
    double phi = sigmapt(2);      // ϕ
    double vx = sigmapt(3);       // v_x
    double vz = sigmapt(4);       // v_z
    double phi_dot = sigmapt(5);  // ϕ_dot
    const double g = 9.81;

    function_value(0) = vx * cos(phi) - vz * sin(phi); // \(\dot{p_x}\)
    function_value(1) = vx * sin(phi) + vz * cos(phi); // \(\dot{p_z}\)
    function_value(2) = phi_dot;                       // \(\dot{\phi}\)
    function_value(3) = vz * phi_dot - g * sin(phi);   // \(\dot{v_x}\)
    function_value(4) = -vx * phi_dot - g * cos(phi);  // \(\dot{v_z}\)
    function_value(5) = 0.0;                           // \(\ddot{\phi}\)
}

__global__ void obtain_y_sigma(double* d_sigmapts, double* d_y_sigmapts, int sigmapts_rows, int sigmapts_cols, int n_states){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sigmapts_rows && col < n_states){
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts + col*sigmapts_rows*sigmapts_cols, sigmapts_rows, sigmapts_cols);
        VectorXd y_sigmapt(sigmapts_cols);
        function_value(sigmapts.row(row), y_sigmapt);
        for (int i = 0; i < sigmapts_cols; i++)
            d_y_sigmapts[col*sigmapts_rows*sigmapts_cols + i * sigmapts_rows + row] = y_sigmapt(i);
    }
}

__global__ void covariance_function(double* d_sigmapts, double* d_x_bar, double* d_y_sigmapts, double* d_y_bar, double* d_vec, int sigmapts_rows, int sigmapts_cols, int n_states){
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < sigmapts_rows*sigmapts_cols && col < n_states*sigmapts_cols){
        int idx_x = col / sigmapts_cols; // Use which x_bar and y_bar
        int idx_y = row / sigmapts_cols; // use which sigma point
        int mat_x = col % sigmapts_cols; // which element in x
        int mat_y = row % sigmapts_cols;

        d_vec[col*sigmapts_rows*sigmapts_cols + row] = (d_sigmapts[(idx_x*sigmapts_cols + mat_y) * sigmapts_rows + idx_y] - d_x_bar[idx_x*sigmapts_cols + mat_y]) 
                                                     * (d_y_sigmapts[(idx_x*sigmapts_cols + mat_x) * sigmapts_rows + idx_y] - d_y_bar[idx_x*sigmapts_cols + mat_x]) ;
    }
}

__global__ void obtain_covariance(double* d_vec, double* d_weights, double* d_result, int sigmapts_rows, int res_rows, int res_cols){
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

__global__ void obtain_y_bar(double* d_pts, double* d_weights, double* d_result, int sigmapts_rows, int n_states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_states){
        double sum = 0;
        for(int i = 0; i < sigmapts_rows; i++){
            sum += d_pts[idx*sigmapts_rows + i]* d_weights[i];
        }
        d_result[idx] = sum;
    }
}

void CudaOperation_SLR::expectationIntegration(MatrixXd& y_bar){

    // Kernel 1: Obtain the result of function 
    dim3 threadperblock1(32, 32);
    dim3 blockSize1((_n_states + threadperblock1.x - 1) / threadperblock1.x, (_sigmapts_rows + threadperblock1.y - 1) / threadperblock1.y);

    obtain_y_sigma<<<blockSize1, threadperblock1>>>(_sigmapts_gpu, _y_sigmapts_gpu, _sigmapts_rows, _dim_state, _n_states);
    cudaDeviceSynchronize();

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 threadperblock2(256);
    dim3 blockSize2((_dim_state*_n_states + threadperblock2.x - 1) / threadperblock2.x);

    obtain_y_bar<<<blockSize2, threadperblock2>>>(_y_sigmapts_gpu, _weights_gpu, _y_bar_gpu, _sigmapts_rows, _dim_state*_n_states);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(y_bar.data(), _y_bar_gpu, y_bar.size() * sizeof(double), cudaMemcpyDeviceToHost);
}

void CudaOperation_SLR::covarianceIntegration(MatrixXd& results){
    // result is the matrix of P_xy (dim_state, dim_state*n_states)
    double *vec_gpu, *result_gpu;
    cudaMalloc(&vec_gpu, _sigmapts_rows * results.size() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));

    // Kernel 1: Obtain the result of function 
    dim3 threadperblock(32, 32);
    dim3 blockSize1((results.cols() + threadperblock.x - 1) / threadperblock.x, (_sigmapts_rows * results.rows() + threadperblock.y - 1) / threadperblock.y);

    covariance_function<<<blockSize1, threadperblock>>>(_sigmapts_gpu, _x_bar_gpu, _y_sigmapts_gpu, _y_bar_gpu, vec_gpu, _sigmapts_rows, _dim_state, _n_states);
    cudaDeviceSynchronize();

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    dim3 blockSize2((results.cols() + threadperblock.x - 1) / threadperblock.x, (results.rows() + threadperblock.y - 1) / threadperblock.y);

    obtain_covariance<<<blockSize2, threadperblock>>>(vec_gpu, _weights_gpu, result_gpu, _sigmapts_rows, _dim_state, _dim_state*_n_states);
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(vec_gpu);
    cudaFree(result_gpu);
}





__global__ void compute_AT_B_A_kernel(const double* d_Mat_A, const double* d_Mat_B, double* d_result, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < cols && col < cols) {
        double sum = 0.0;
        for (int k = 0; k < rows; k++) {
            for (int l = 0; l < rows; l++) {
                sum += d_Mat_A[k * cols + row] * d_Mat_B[k * rows + l] * d_Mat_A[l * cols + col];
            }
        }
        d_result[row * cols + col] = sum;
    }
}

MatrixXd compute_AT_B_A(MatrixXd& Matrix_A, MatrixXd& Matrix_B){
    int rows = Matrix_A.rows();
    int cols = Matrix_A.cols();

    double *d_Mat_A, *d_Mat_B, *d_result;
    cudaMalloc(&d_Mat_A, Matrix_A.size() * sizeof(double));
    cudaMalloc(&d_Mat_B, Matrix_B.size() * sizeof(double));
    cudaMalloc(&d_result, cols * cols * sizeof(double));

    cudaMemcpy(d_Mat_A, Matrix_A.data(), Matrix_A.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mat_B, Matrix_B.data(), Matrix_B.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, cols * cols * sizeof(double));

    dim3 blockSize(16, 16);
    dim3 threadperblock((cols + blockSize.x - 1) / blockSize.x, (cols + blockSize.y - 1) / blockSize.y);

    compute_AT_B_A_kernel<<<blockSize, threadperblock>>>(d_Mat_A, d_Mat_B, d_result, rows, cols);

    MatrixXd result = MatrixXd::Zero(cols, cols);
    cudaMemcpy(result.data(), d_result, cols * cols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_Mat_A);
    cudaFree(d_Mat_B);
    cudaFree(d_result);

    return result;
}

__global__ void computeTmpKernel(double* tmp, const double* covariance, const double* AT_precision_A, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < dim && j < dim) {
        double sum = 0.0;
        for (int k = 0; k < dim; k++) {
            for (int l = 0; l < dim; l++) {
                sum += (covariance[i * dim + j] * covariance[k * dim + l] +
                        covariance[i * dim + k] * covariance[j * dim + l] +
                        covariance[i * dim + l] * covariance[j * dim + k]) * AT_precision_A[k * dim + l];
            }
        }
        tmp[i * dim + j] = sum;
    }
}



void computeTmp_CUDA(Eigen::MatrixXd& tmp, const Eigen::MatrixXd& covariance, const Eigen::MatrixXd& AT_precision_A){
    int dim = covariance.rows();
    double *d_tmp, *d_covariance, *d_AT_precision_A;
    cudaMalloc(&d_tmp, dim * dim * sizeof(double));
    cudaMalloc(&d_covariance, dim * dim * sizeof(double));
    cudaMalloc(&d_AT_precision_A, dim * dim * sizeof(double));

    cudaMemcpy(d_covariance, covariance.data(), dim * dim * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AT_precision_A, AT_precision_A.data(), dim * dim * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((dim + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (dim + threadsPerBlock.y - 1) / threadsPerBlock.y);

    computeTmpKernel<<<blocksPerGrid, threadsPerBlock>>>(d_tmp, d_covariance, d_AT_precision_A, dim);
    cudaDeviceSynchronize();

    cudaMemcpy(tmp.data(), d_tmp, dim * dim * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_tmp);
    cudaFree(d_covariance);
    cudaFree(d_AT_precision_A);
}


}


// cudaDeviceProp prop;
// cudaGetDeviceProperties(&prop, 0);
// std::cout << "Double precision performance ratio: " << prop.singleToDoublePrecisionPerfRatio << std::endl;