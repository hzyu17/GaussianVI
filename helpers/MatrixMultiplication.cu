#include <cuda_runtime.h>
#include "helpers/CudaOperation.h"

using namespace Eigen;
using GHFunction = std::function<MatrixXd(const VectorXd&)>;
// using CudaFunction = nvstd::function<double*(double*, int)>;

// CUDA kernel for matrix-vector multiplication
__global__ void MatrixMultiplication(double* d_matrix, double* d_vectors, double* d_result, int rows, int cols, int vec_num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < vec_num) {
        double sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            sum += d_matrix[i * rows + row] * d_vectors[col * cols + i];
        }
        d_result[col * rows + row] = sum;
    }
}

// __global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* input_vector, double* pts_vector, int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, FunctionPtr func_ptr, void* context){
__global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* mu, int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, int type, gvi::CudaOperation<GHFunction>* pointer){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sigmapts_rows){
        Eigen::Map<MatrixXd> sigmapts(d_sigmapts, sigmapts_rows, sigmapts_cols);
        Eigen::Map<MatrixXd> pts(d_sigmapts, res_rows, sigmapts_rows*res_cols);
        Eigen::Map<VectorXd> mean(mu, sigmapts_cols);
        // printf("idx=%d:%lf\n", idx, sigmapts(idx));
        // printf("rows:%d, cols:%d\n", sigmapts_rows, sigmapts_cols);
        double function_value = pointer -> cost_function1(sigmapts.row(idx), sigmapts_cols);
        // printf("Thread%d: %lf  \n", idx, function_value);
        // double function_value = cost_function1(d_sigmapts + idx*sigmapts_cols, sigmapts_cols);

        if (type == 0) //res.size = 1*1
            d_pts[idx] = function_value;
        else if (type == 1){
            // pts.block(0, idx * res_cols, res_cols, res_rows) = (sigmapts.row(idx) - mean).transpose() * function_value;
            for (int i=0; i<sigmapts_cols; i++)
                d_pts[idx*sigmapts_cols + i] = (d_sigmapts[idx + sigmapts_cols * i] - mu[i]) * function_value;
            // dim3 blockSize1(3);
            // dim3 threadperblock1((sigmapts_cols + blockSize1.x - 1) / blockSize1.x);
            // func_Vmu<<<blockSize1, threadperblock1>>>(d_sigmapts, d_pts, mu, sigmapts_cols, idx, function_value);
        }
        else{
            // pts.block(0, idx * res.cols(), res.cols(), res.rows()) = (sigmapts.row(idx) - mean)* (sigmapts.row(idx) - mean).transpose() * function_value;

            for (int i=0; i<sigmapts_cols; i++)
                for (int j=0; j<sigmapts_cols; j++)
                    d_pts[idx*sigmapts_cols *sigmapts_cols+ i*sigmapts_cols +j] = (d_sigmapts[idx*sigmapts_cols + i] - mu[i]) * (d_sigmapts[idx*sigmapts_cols + j] - mu[j]) * function_value;

            // dim3 blockSize2(3, 3);
            // dim3 threadperblock2((sigmapts_cols + blockSize2.x - 1) / blockSize2.x, (sigmapts_cols + blockSize2.y - 1) / blockSize2.y);
            // func_Vmu<<<blockSize2, threadperblock2>>>(d_sigmapts + idx*sigmapts_cols, d_pts + idx*res_rows*res_cols, mu, sigmapts_cols, function_value);
        }
    }
}

__global__ void obtain_res(double* d_pts, double* d_weights, double* d_result, int sigmapts_rows, int res_rows, int res_cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < res_rows && col < res_cols){
        double sum = 0;
        for(int i = 0; i < sigmapts_rows; i++){
            // printf("row = %d, col = %d, i = %d, value = %lf\n", row, col, i, d_pts[i*res_rows*res_cols + col*res_rows + row]);
            sum += d_pts[i*res_rows*res_cols + col*res_rows + row] * d_weights[i];
        }
        d_result[col*res_rows + row] = sum;
    }
    // printf("sigma:%lf \n",d_result[0]);
}

__global__ void func_Vmu(const double* vec_x, double* pt, const double* mu, int dim, int index, double& func_value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim){
        pt[index*dim + idx] = (vec_x[index*dim + idx] - mu[idx]) * func_value;
    }
}

__global__ void func_Vmumu(const double* vec_x, double* pt, const double* mu, int dim, int index, double& func_value) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < dim && col < dim) {
        pt[row * dim + col] = (vec_x[row] - mu[row]) * (vec_x[col] - mu[col]) * func_value;
    }
}

// __device__ void calculate_derivative(const double* vec_x, double* pt, const double* mu, int dim, double& func_value, int& type){
//     dim3 blockSize(3, 3);
//     dim3 threadperblock((dim + blockSize.x - 1) / blockSize.x, (dim + blockSize.y - 1) / blockSize.y);
//     if (type == 1)

// }



namespace gvi{

void MatrixMul(double* matrix, double* vectorMatrix, double* result, int rows, int cols, int vec_num)
{   
    double *matrix_gpu, *vectorMatrix_gpu, *result_gpu;  

    cudaMalloc(&matrix_gpu, rows * cols * sizeof(double));
    cudaMalloc(&vectorMatrix_gpu, cols * vec_num * sizeof(double));
    cudaMalloc(&result_gpu, rows * vec_num * sizeof(double));

    // Copy the data from host to device
    cudaMemcpy(matrix_gpu, matrix, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vectorMatrix_gpu, vectorMatrix, cols * vec_num * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(4, 4);
    dim3 threadperblock((vec_num + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

        
    MatrixMultiplication<<<threadperblock, blockSize>>>(matrix_gpu, vectorMatrix_gpu, result_gpu, rows, cols, vec_num);
    cudaDeviceSynchronize();
    cudaMemcpy(result, result_gpu, rows * vec_num * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(matrix_gpu);
    cudaFree(vectorMatrix_gpu);
    cudaFree(result_gpu);
}

// void CudaIntegration(FunctionPtr func_ptr, double* d_sigmapts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols, void* context)
template <typename Function>
void CudaOperation<Function>::CudaIntegration(Function function, const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int sigma_rows, int sigma_cols, int res_rows, int res_cols, double* d_pts1, double* d_pts2, int type)
{
    double *sigmapts_gpu, *pts_gpu, *weight_gpu, *result_gpu, *mu_gpu;

    // std::cout << "Mean:" << std::endl << mean.transpose() << std::endl << std::endl;
    // std::cout << "Sigma rows and cols" << std::endl << sigmapts.rows() << sigmapts.cols() << std::endl << std::endl;
    // std::cout << "Sigma rows and cols1" << std::endl << sigma_rows << sigma_cols << std::endl << std::endl;

    CudaOperation<Function>* class_gpu;
    // std::cout << sizeof(*this) << std::endl;
    cudaMalloc(&class_gpu, sizeof(CudaOperation<Function>));
    cudaMemcpy(class_gpu, this, sizeof(CudaOperation<Function>), cudaMemcpyHostToDevice);

    cudaMalloc(&sigmapts_gpu, sigmapts.size() * sizeof(double));
    cudaMalloc(&pts_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&weight_gpu, sigmapts.rows() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMalloc(&mu_gpu, sigmapts.cols() * sizeof(double));

    cudaMemcpy(sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_gpu, weights.data(), sigma_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_gpu, mean.data(), sigma_cols * sizeof(double), cudaMemcpyHostToDevice);

    // Dimension for the first kernel function
    dim3 blockSize1(3);
    dim3 threadperblock1((sigma_rows + blockSize1.x - 1) / blockSize1.x);

    // Kernel 1: Obtain the result of function 
    Sigma_function<<<blockSize1, threadperblock1>>>(sigmapts_gpu, pts_gpu, mu_gpu, sigma_rows, sigma_cols, res_rows, res_cols, type, class_gpu);
    cudaDeviceSynchronize();

    cudaMemcpy(d_pts2, pts_gpu, sigma_rows * res_rows * res_cols * sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(pts_gpu, d_pts1, sigma_rows * res_rows * res_cols * sizeof(double), cudaMemcpyHostToDevice);
    
    // Dimension for the second kernel function
    dim3 blockSize2(3, 3);
    dim3 threadperblock2((res_cols + blockSize2.x - 1) / blockSize2.x, (res_rows + blockSize2.y - 1) / blockSize2.y);

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    obtain_res<<<blockSize2, threadperblock2>>>(pts_gpu, weight_gpu, result_gpu, sigma_rows, res_rows, res_cols);
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, res_rows * res_cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(sigmapts_gpu);
    cudaFree(pts_gpu);
    cudaFree(weight_gpu);
    cudaFree(result_gpu);
    
}


template <typename Function>
void CudaOperation<Function>::CudaIntegration1(MatrixXd& d_pts, const MatrixXd& d_weights, MatrixXd& d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols)
{
    double *pts_gpu, *weight_gpu, *result_gpu; 

    cudaMalloc(&pts_gpu, sigma_rows * res_rows * res_cols * sizeof(double));
    cudaMalloc(&weight_gpu, sigma_rows * sizeof(double));
    cudaMalloc(&result_gpu, res_rows * res_cols * sizeof(double));

    cudaMemcpy(pts_gpu, d_pts.data(), sigma_rows * res_rows * res_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_gpu, d_weights.data(), sigma_rows * sizeof(double), cudaMemcpyHostToDevice);

    // Dimension for the second kernel function
    dim3 blockSize(3, 3);
    dim3 threadperblock((res_cols + blockSize.x - 1) / blockSize.x, (res_rows + blockSize.y - 1) / blockSize.y);

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    obtain_res<<<blockSize, threadperblock>>>(pts_gpu, weight_gpu, result_gpu, sigma_rows, res_rows, res_cols);
    cudaDeviceSynchronize();
    cudaMemcpy(d_results.data(), result_gpu, res_rows * res_cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(pts_gpu);
    cudaFree(weight_gpu);
    cudaFree(result_gpu);

}

template class CudaOperation<GHFunction>;
// template __global__ void Sigma_function<GHFunction>(double*, double*, double*, int, int, int, int, FunctionPtr, int, CudaOperation<GHFunction>*);
}