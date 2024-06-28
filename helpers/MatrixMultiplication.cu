// #include <cuda_runtime.h>
#include "helpers/MatrixMultiplication.h"
  
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
    cudaFree(matrix);
    cudaFree(vectorMatrix);
    cudaFree(result);
}