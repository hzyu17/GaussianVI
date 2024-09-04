#include <cuda_runtime.h>
#include <optional>
// #include <gpmp2/obstacle/ObstaclePlanarSDFFactor.h>
#include "helpers/CudaOperation.h"

using namespace Eigen;

// __host__ __device__ void gvi::invert_matrix(double* A, double* A_inv, int dim) {
//     for (int i = 0; i < dim; i++) {
//         for (int j = 0; j < dim; j++) {
//             A_inv[i + j * dim] = (i == j) ? 1 : 0;
//         }
//     }

//     // Gaussian Elimination
//     for (int i = 0; i < dim; i++) {
//         double pivot = A[i + i * dim];
//         for (int j = 0; j < dim; j++) {
//             A[i + j * dim] /= pivot;
//             A_inv[i + j * dim] /= pivot;
//         }

//         for (int j = 0; j < dim; j++) {
//             if (j != i) {
//                 double factor = A[i + j * dim];
//                 for (int k = 0; k < dim; k++) {
//                     A[k + j * dim] -= factor * A[k + i * dim];
//                     A_inv[k + j * dim] -= factor * A_inv[k + i * dim];
//                 }
//             }
//         }
//     }
// }



__global__ void Sigma_function(double* d_sigmapts, double* d_pts, double* mu,
                               int sigmapts_rows, int sigmapts_cols, int res_rows, int res_cols, int type, 
                               gvi::CudaOperation* pointer, double* d_data){
    
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

// __global__ void belief_update(double* d_joint_factor, double* d_message, double* d_message1, double* d_sigma, int dim_state, int num_state, int joint_cols, int message_cols){
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if(idx < num_state - 1){
//         Eigen::Map<MatrixXd> joint_factor(d_joint_factor, 2*dim_state, joint_cols);
//         Eigen::Map<MatrixXd> message(d_message, dim_state, message_cols);
//         Eigen::Map<MatrixXd> message1(d_message1, dim_state, message_cols);

//         MatrixXd lam_joint = joint_factor.block(0, idx*2*dim_state, 2*dim_state, 2*dim_state);
//         MatrixXd factor_message = message.block(0, idx*dim_state, dim_state, dim_state);
//         MatrixXd factor_message1 = message1.block(0, idx*dim_state, dim_state, dim_state);

//         lam_joint.block(0, 0, dim_state, dim_state) += factor_message;
//         lam_joint.block(dim_state, dim_state, dim_state, dim_state) += factor_message1;
//         // MatrixXd variance_joint = lam_joint.inverse();
//         MatrixXd variance_joint = MatrixXd::Identity(lam_joint.rows(), lam_joint.cols());
//         gvi::invert_matrix(lam_joint.data(), variance_joint.data(), lam_joint.rows());

//         // printf("idx = %d, norm of invert is %.10lf\n", idx, variance_joint.norm());

//         for (int i = 0; i < variance_joint.rows(); i++){
//             for (int j = 0; j < variance_joint.cols(); j++){
//                 int row = idx*dim_state + i;
//                 int col = idx*dim_state + j;
//                 d_sigma[row + col * dim_state * num_state] = variance_joint(i, j);
//             }
//         }

//     }
// }

namespace gvi{

void CudaOperation::CudaIntegration(const MatrixXd& sigmapts, const MatrixXd& weights, MatrixXd& results, const MatrixXd& mean, int type)
{
    double *sigmapts_gpu, *pts_gpu, *weight_gpu, *result_gpu, *mu_gpu, *data_gpu;
    CudaOperation* class_gpu;

    cudaMalloc(&sigmapts_gpu, sigmapts.size() * sizeof(double));
    cudaMalloc(&pts_gpu, sigmapts.rows() * results.size() * sizeof(double));
    cudaMalloc(&weight_gpu, sigmapts.rows() * sizeof(double));
    cudaMalloc(&result_gpu, results.size() * sizeof(double));
    cudaMalloc(&mu_gpu, sigmapts.cols() * sizeof(double));
    cudaMalloc(&data_gpu, _sdf.data_.size() * sizeof(double));
    cudaMalloc(&class_gpu, sizeof(CudaOperation));

    cudaMemcpy(sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(weight_gpu, weights.data(), sigmapts.rows() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(mu_gpu, mean.data(), sigmapts.cols() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data_gpu, _sdf.data_.data(), _sdf.data_.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(class_gpu, this, sizeof(CudaOperation), cudaMemcpyHostToDevice);

    // Dimension for the first kernel function
    dim3 blockSize1(16, 16);
    dim3 threadperblock1((results.cols()*sigmapts.rows() + blockSize1.x - 1) / blockSize1.x, (results.rows() + blockSize1.y - 1) / blockSize1.y);

    // Kernel 1: Obtain the result of function 
    Sigma_function<<<blockSize1, threadperblock1>>>(sigmapts_gpu, pts_gpu, mu_gpu, sigmapts.rows(), sigmapts.cols(), results.rows(), results.cols(), type, class_gpu, data_gpu);
    cudaDeviceSynchronize();

    cudaFree(sigmapts_gpu);
    cudaFree(mu_gpu);

    // cudaMemcpy(pts.data(), pts_gpu, sigmapts.rows() * results.size() * sizeof(double), cudaMemcpyDeviceToHost);
    
    // Dimension for the second kernel function
    dim3 blockSize2(16, 16);
    dim3 threadperblock2((results.cols() + blockSize2.x - 1) / blockSize2.x, (results.rows() + blockSize2.y - 1) / blockSize2.y);

    // Kernel 2: Obtain the result by multiplying the pts and the weights
    obtain_res<<<blockSize2, threadperblock2>>>(pts_gpu, weight_gpu, result_gpu, sigmapts.rows(), results.rows(), results.cols());
    cudaDeviceSynchronize();
    cudaMemcpy(results.data(), result_gpu, results.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(pts_gpu);
    cudaFree(weight_gpu);
    cudaFree(result_gpu);
    cudaFree(data_gpu);
    cudaFree(class_gpu);
}


// template class NGDFactorizedBaseGH_Cuda<NoneType>;
// template class CudaOperation<gpmp2::ObstaclePlanarSDFFactor<gpmp2::PointRobotModel>>;


// MatrixXd GBP_Cuda::obtain_cov(std::vector<MatrixXd> joint_factor, std::vector<MatrixXd> factor_message, std::vector<MatrixXd> factor_message1){
//     int dim_state = factor_message[0].rows();
//     int num_state = factor_message.size();
//     int dim = dim_state * num_state;

//     MatrixXd joint_factor_matrix(2*dim_state, (num_state-1)*2*dim_state);
//     MatrixXd factor_message_matrix(dim_state, dim);
//     MatrixXd factor_message1_matrix(dim_state, dim);
//     MatrixXd Sigma(dim, dim);
//     Sigma.setZero();
     
//     for (int i = 0; i < num_state-1; i++) {
//         joint_factor_matrix.block(0, i * 2*dim_state, 2*dim_state, 2*dim_state) = joint_factor[i];
//     }

//     for (int i = 0; i < num_state; i++) {
//         factor_message_matrix.block(0, i * dim_state, dim_state, dim_state) = factor_message[i];
//         factor_message1_matrix.block(0, i * dim_state, dim_state, dim_state) = factor_message1[i];
//     }

//     double *joint_factor_gpu, *factor_message_gpu, *factor_message1_gpu, *sigma_gpu;

//     cudaMalloc(&joint_factor_gpu, joint_factor_matrix.size() * sizeof(double));
//     cudaMalloc(&factor_message_gpu, factor_message_matrix.size() * sizeof(double));
//     cudaMalloc(&factor_message1_gpu, factor_message1.size() * sizeof(double));
//     cudaMalloc(&sigma_gpu, Sigma.size() * sizeof(double));

//     cudaMemcpy(joint_factor_gpu, joint_factor_matrix.data(), joint_factor_matrix.size() * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(factor_message_gpu, factor_message_matrix.data(), factor_message_matrix.size() * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(factor_message1_gpu, factor_message1_matrix.data(), factor_message1_matrix.size() * sizeof(double), cudaMemcpyHostToDevice);

//     dim3 blockSize(4);
//     dim3 threadperblock((num_state - 1 + blockSize.x - 1) / blockSize.x);

//     belief_update<<<blockSize, threadperblock>>>(joint_factor_gpu, factor_message_gpu, factor_message1_gpu, sigma_gpu, dim_state, num_state, joint_factor_matrix.cols(), factor_message_matrix.cols());

//     cudaMemcpy(Sigma.data(), sigma_gpu, Sigma.size() * sizeof(double), cudaMemcpyDeviceToHost);

//     cudaFree(joint_factor_gpu);
//     cudaFree(factor_message_gpu);
//     cudaFree(factor_message1_gpu);
//     cudaFree(sigma_gpu);

//     return Sigma;
// }

}