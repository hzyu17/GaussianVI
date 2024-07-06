#include <cuda_runtime.h>
#include "helpers/MatrixMultiplication.cuh"
// #include <nvfunctional>


// using CudaFunction = nvstd::function<double*(double*, int)>;

// CUDA kernel for matrix-vector multiplication
__global__ void MatrixMultiplication(double* d_matrix, double* d_vectors, double* d_result, int rows, int cols, int vec_num) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < vec_num) {
        double sum = 0.0;
        for (int i = 0; i < cols; ++i) {
            sum += d_matrix[row * cols + i] * d_vectors[i * vec_num + col];
        }
        d_result[row * vec_num + col] = sum;
    }
} 

// __global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* input_vector, double* pts_vector, int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, FunctionPtr func_ptr, void* context){

// template <typename Function>
__global__ void Sigma_function(double* d_sigmapts, double* d_pts, int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, FunctionPtr func_ptr, void* context){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // double* input_vector = (double*)malloc(sigmapts_cols * sizeof(double));
    double* pts_vector = (double*)malloc(res_rows * res_cols * sizeof(double));

    if (idx < sigmapts_rows){
        
        for(int i = 0; i < sigmapts_cols; ++i) {
            printf("dix=%d, sigma=%lf\n",idx, d_sigmapts[idx * sigmapts_cols + i]);
        }

        func_ptr(d_sigmapts+idx * sigmapts_cols, pts_vector, sigmapts_cols, context);

        // for(int i = 0; i < res_rows * res_cols; ++i) {
        //     d_pts[idx * res_rows * res_cols + i] = pts_vector[i];
        // }
    }
    d_pts[idx] = 1;
    // free(input_vector);
    free(pts_vector);

}

__global__ void obtain_res(double* d_pts, double* d_weights, double* d_result, int sigmapts_rows, int res_rows, int res_cols){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < res_rows && col < res_cols){
        double sum = 0;
        for(int i = 0; i < sigmapts_rows; i++){
            sum += d_pts[i*res_rows*res_cols + row*res_cols + col] * d_weights[i];
        }
        d_result[row*res_cols + col] = sum;
    }
    // printf("sigma:%lf \n",d_result[0]);
}


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
// template <typename Function>
void CudaIntegration(FunctionPtr function, double* d_sigmapts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols, void* context, double* d_pts1, double* d_pts2)
{
    double *sigmapts_gpu, *pts_gpu, *weight_gpu, *result_gpu; 

    cudaMalloc(&sigmapts_gpu, sigma_rows * sigma_cols * sizeof(double));
    cudaMalloc(&pts_gpu, sigma_rows * res_rows * res_cols * sizeof(double));
    cudaMalloc(&weight_gpu, sigma_rows * sizeof(double));
    cudaMalloc(&result_gpu, res_rows * res_cols * sizeof(double));

    cudaMemcpy(sigmapts_gpu, d_sigmapts, sigma_rows * sigma_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_gpu, d_weights, sigma_rows * sizeof(double), cudaMemcpyHostToDevice);

    // Dimension for the first kernel function
    dim3 blockSize1(3);
    dim3 threadperblock1((sigma_rows + blockSize1.x - 1) / blockSize1.x);

    // Kernel 1: Obtain the result of function 
    // Sigma_function<<<threadperblock1, blockSize1>>>(sigmapts_gpu, pts_gpu, input_vector, pts_vector, sigma_rows, sigma_cols, res_rows, res_cols, func_ptr, context);
    Sigma_function<<<threadperblock1, blockSize1>>>(sigmapts_gpu, pts_gpu, sigma_rows, sigma_cols, res_rows, res_cols, *function, context);
    cudaDeviceSynchronize();
    cudaMemcpy(d_pts2, pts_gpu, sigma_rows * res_rows * res_cols * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(pts_gpu, d_pts1, sigma_rows * res_rows * res_cols * sizeof(double), cudaMemcpyHostToDevice);
    
    // Dimension for the second kernel function
    dim3 blockSize2(3, 3);
    dim3 threadperblock2((res_cols + blockSize2.x - 1) / blockSize2.x, (res_rows + blockSize2.y - 1) / blockSize2.y);

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    obtain_res<<<threadperblock2, blockSize2>>>(pts_gpu, weight_gpu, result_gpu, sigma_rows, res_rows, res_cols);
    cudaDeviceSynchronize();
    cudaMemcpy(d_results, result_gpu, res_rows * res_cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(sigmapts_gpu);
    cudaFree(pts_gpu);
    cudaFree(weight_gpu);
    cudaFree(result_gpu);
    
}


void CudaIntegration1(double* d_pts, double* d_weights, double* d_results, int sigma_rows, int sigma_cols, int res_rows, int res_cols)
{
    double *pts_gpu, *weight_gpu, *result_gpu; 

    cudaMalloc(&pts_gpu, sigma_rows * res_rows * res_cols * sizeof(double));
    cudaMalloc(&weight_gpu, sigma_rows * sizeof(double));
    cudaMalloc(&result_gpu, res_rows * res_cols * sizeof(double));

    cudaMemcpy(pts_gpu, d_pts, sigma_rows * res_rows * res_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_gpu, d_weights, sigma_rows * sizeof(double), cudaMemcpyHostToDevice);

    // Dimension for the second kernel function
    dim3 blockSize(3, 3);
    dim3 threadperblock((res_cols + blockSize.x - 1) / blockSize.x, (res_rows + blockSize.y - 1) / blockSize.y);

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    obtain_res<<<threadperblock, blockSize>>>(pts_gpu, weight_gpu, result_gpu, sigma_rows, res_rows, res_cols);
    cudaDeviceSynchronize();
    cudaMemcpy(d_results, result_gpu, res_rows * res_cols * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(pts_gpu);
    cudaFree(weight_gpu);
    cudaFree(result_gpu);

}