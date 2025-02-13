#include <cuda_runtime.h>
#include "helpers/CudaOperation.h"
#include "helpers/timer.h"

using namespace Eigen;

template class CudaOperation_Base<PlanarSDF>;
template class CudaOperation_Base<SignedDistanceField>;

void printGPUMemoryInfo() {
    size_t free_mem = 0, total_mem = 0;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    std::cout << "GPU Memory: free = " << free_mem / (1024.0 * 1024.0) << " MB" << std::endl;
}


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

__global__ void addMeanKernel_batched(double* d_sigmapts, const double* d_mean, int num_rows, int dim, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_rows * dim;
    if (idx < total) {
    int state = idx / (num_rows * dim);
    int col   = (idx % (num_rows * dim)) / num_rows;
    d_sigmapts[idx] += d_mean[state * dim + col];
    }
}


// Combined kernel to compute the square root of eigenvalues and scale eigenvectors accordingly.
// Assumes matrices are stored in column-major order.
__global__ void sqrtAndScaleEigenvectorsKernel(const double* d_eigenvalues, const double* d_eigvec,
                                               double* d_scaledEigvec, int dim, int batch) {
    // Each block handles one matrix in the batch.
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch) return;

    // Pointers for current batch's eigenvalues and eigenvector matrix.
    const double* eigenvals = d_eigenvalues + batch_idx * dim;
    const double* eigvec    = d_eigvec + batch_idx * dim * dim;
    double* scaledEigvec    = d_scaledEigvec + batch_idx * dim * dim;

    // Allocate shared memory to store sqrt(eigenvalues) for the current matrix.
    extern __shared__ double s_sqrtEig[];

    // Each thread computes part of the square root.
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    s_sqrtEig[i] = sqrt(eigenvals[i]);
    }
    __syncthreads();

    // Scale eigenvector matrix columns by corresponding sqrt eigenvalues.
    int totalElements = dim * dim;
    for (int idx = threadIdx.x; idx < totalElements; idx += blockDim.x) {
    int col = idx / dim;  // In column-major, column index = idx / dim.
    scaledEigvec[idx] = eigvec[idx] * s_sqrtEig[col];
    }
}


namespace gvi{

// template <typename SDFType>
// void CudaOperation_Base<SDFType>::update_sigmapts(const MatrixXd& covariance, const MatrixXd& mean, int dim_conf, int num_states, MatrixXd& sigmapts){
//     // Compute the Cholesky decomposition of the covariance matrix
//     double *covariance_gpu, *mean_gpu, *d_sigmapts;
//     cudaMalloc(&covariance_gpu, dim_conf * dim_conf * num_states * sizeof(double));
//     cudaMalloc(&d_sigmapts, _sigmapts_rows * dim_conf * num_states * sizeof(double));
//     cudaMalloc(&mean_gpu, mean.size() * sizeof(double));

//     cudaMemcpy(covariance_gpu, covariance.data(), dim_conf * dim_conf * num_states * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(mean_gpu, mean.data(), mean.size() * sizeof(double), cudaMemcpyHostToDevice);

//     const int num_streams = 20;

//     // Create a CUDA stream, cuSOLVER, and cuBLAS handle for each state
//     std::vector<cudaStream_t> streams(num_streams);
//     std::vector<cusolverDnHandle_t> cusolver_handles(num_streams);
//     std::vector<cublasHandle_t> cublas_handles(num_streams);

//     // Containers to store temporary device memory for each state
//     std::vector<double*> d_eigen_values_vec(num_states, nullptr);
//     std::vector<int*>    d_info_vec(num_states, nullptr);
//     std::vector<double*> d_work_vec(num_states, nullptr);
//     std::vector<int>     Lwork_vec(num_states, 0);
//     std::vector<double*> d_V_scaled_vec(num_states, nullptr);
//     std::vector<double*> d_sqrtP_vec(num_states, nullptr);

//     const double alpha = 1.0, beta = 0.0;

//     std::cout << "Beginning: ";
//     printGPUMemoryInfo();

//     for (int i = 0; i < num_streams; i++) {
//         cudaStreamCreate(&streams[i]);
//         cusolverDnCreate(&cusolver_handles[i]);
//         cusolverDnSetStream(cusolver_handles[i], streams[i]);
//         cublasCreate(&cublas_handles[i]);
//         cublasSetStream(cublas_handles[i], streams[i]);
//     }

//     std::cout << "Finished creating streams and handles: ";
//     printGPUMemoryInfo();

//     // Submit tasks for each state to its corresponding stream
//     for (int state = 0; state < num_states; state++) {        
//         // Each state's covariance matrix is stored at an offset in covariance_gpu
//         double* d_Pi = covariance_gpu + state * dim_conf * dim_conf;
//         // Assuming mean data occupies two rows per state; adjust offset as necessary
//         double* d_mi = mean_gpu + 2 * (state + 1) * dim_conf;
        
//         // 3.1 Allocate temporary memory for eigenvalues and info
//         cudaMalloc(&d_eigen_values_vec[state], dim_conf * sizeof(double));
//         cudaMalloc(&d_info_vec[state], sizeof(int));

//         int stream_idx = state % num_streams;  // Assign to one of the 10 streams
        
//         // Query workspace size required for eigen decomposition
//         cusolverDnDsyevd_bufferSize(cusolver_handles[stream_idx],
//                                     CUSOLVER_EIG_MODE_VECTOR,
//                                     CUBLAS_FILL_MODE_LOWER,
//                                     dim_conf,
//                                     d_Pi,
//                                     dim_conf,
//                                     d_eigen_values_vec[state],
//                                     &Lwork_vec[state]);
//         cudaMalloc(&d_work_vec[state], Lwork_vec[state] * sizeof(double));
        
//         // 3.2 Perform symmetric eigenvalue decomposition on d_Pi (d_Pi will store eigenvectors after the operation)
//         cusolverDnDsyevd(cusolver_handles[stream_idx],
//                          CUSOLVER_EIG_MODE_VECTOR,
//                          CUBLAS_FILL_MODE_LOWER,
//                          dim_conf,
//                          d_Pi,
//                          dim_conf,
//                          d_eigen_values_vec[state],
//                          d_work_vec[state],
//                          Lwork_vec[state],
//                          d_info_vec[state]);
//         // cusolverDnDsyevjBatched
        
//         int h_info;
//         cudaMemcpy(&h_info, d_info_vec[state], sizeof(int), cudaMemcpyDeviceToHost);
//         if (h_info != 0) {
//             std::cerr << "Eigen decomposition failed for state " << state << " with info " << h_info << std::endl;
//         }
        
//         // 3.3 Apply square root to the eigenvalues using a custom kernel (executed on the corresponding stream)
//         int threadsPerBlock = 16;
//         int blocks = (dim_conf + threadsPerBlock - 1) / threadsPerBlock;
//         sqrtKernel<<<blocks, threadsPerBlock, 0, streams[stream_idx]>>>(d_eigen_values_vec[state], dim_conf);

//         // 3.4 Use cublasDdgmm to scale the eigenvector matrix by the square-rooted eigenvalues
//         cudaMalloc(&d_V_scaled_vec[state], dim_conf * dim_conf * sizeof(double));
//         cublasStatus_t stat = cublasDdgmm(cublas_handles[stream_idx],
//                                           CUBLAS_SIDE_RIGHT,
//                                           dim_conf,
//                                           dim_conf,
//                                           d_Pi,
//                                           dim_conf,
//                                           d_eigen_values_vec[state],
//                                           1,
//                                           d_V_scaled_vec[state],
//                                           dim_conf);
//         if (stat != CUBLAS_STATUS_SUCCESS) {
//             std::cerr << "cublasDdgmm failed for state " << state << ": " << stat << std::endl;
//         }
        
//         // 3.5 Compute sqrtP = d_V_scaled * (d_Pi)^T
//         cudaMalloc(&d_sqrtP_vec[state], dim_conf * dim_conf * sizeof(double));
//         stat = cublasDgemm(cublas_handles[stream_idx],
//                            CUBLAS_OP_N,
//                            CUBLAS_OP_T,
//                            dim_conf,
//                            dim_conf,
//                            dim_conf,
//                            &alpha,
//                            d_V_scaled_vec[state],
//                            dim_conf,
//                            d_Pi,
//                            dim_conf,
//                            &beta,
//                            d_sqrtP_vec[state],
//                            dim_conf);
//         if (stat != CUBLAS_STATUS_SUCCESS) {
//             std::cerr << "cublasDgemm for sqrtP failed for state " << state << ": " << stat << std::endl;
//         }
        
//         // 3.6 Compute the current state's portion of sigmapts.
//         // d_sigmapts is allocated contiguously on the device; each state occupies a block of _sigmapts_rows x dim_conf.
//         double* d_sigmapts_state = d_sigmapts + state * _sigmapts_rows * dim_conf;
//         stat = cublasDgemm(cublas_handles[stream_idx], CUBLAS_OP_N, CUBLAS_OP_T, _sigmapts_rows, dim_conf, dim_conf, &alpha, _zeromean_gpu, _sigmapts_rows, d_sqrtP_vec[state], dim_conf, &beta, d_sigmapts_state, _sigmapts_rows);
//         if (stat != CUBLAS_STATUS_SUCCESS) {
//             std::cerr << "cublasDgemm for sigmapts failed for state " << state << ": " << stat << std::endl;
//         }
        
//         // 3.7 Add the mean to sigmapts (for the current state's block)
//         threadsPerBlock = 256;
//         blocks = (_sigmapts_rows * dim_conf + threadsPerBlock - 1) / threadsPerBlock;
//         addMeanKernel<<<blocks, threadsPerBlock, 0, streams[stream_idx]>>>(d_sigmapts_state, d_mi, _sigmapts_rows, dim_conf);
//     }

//     std::cout << "Finished submitting tasks to streams: ";
//     printGPUMemoryInfo();

//     for (int i = 0; i < num_streams; i++) {
//         cudaStreamSynchronize(streams[i]);
//     }

//     // 5. Copy all states' sigmapts from device to host.
//     // The output matrix sigmapts should have dimensions _sigmapts_rows x (dim_conf * num_states)
//     MatrixXd sigma(_sigmapts_rows, dim_conf * num_states);
//     cudaMemcpy(sigma.data(), d_sigmapts, _sigmapts_rows * dim_conf * num_states * sizeof(double), cudaMemcpyDeviceToHost);
//     sigmapts = sigma;
    
//     // 6. Free temporary memory for each state and destroy handles and streams
//     for (int state = 0; state < num_states; state++) {
//         cudaFree(d_eigen_values_vec[state]);
//         cudaFree(d_info_vec[state]);
//         cudaFree(d_work_vec[state]);
//         cudaFree(d_V_scaled_vec[state]);
//         cudaFree(d_sqrtP_vec[state]);
        
//         // cusolverDnDestroy(cusolver_handles[state]);
//         // // cublasDestroy(cublas_handles[state]);
//         // cudaStreamDestroy(streams[state]);
//     }

//     for (int i = 0; i < num_streams; i++) {
//         cusolverDnDestroy(cusolver_handles[i]);
//         // cublasDestroy(cublas_handles[i]);
//         cudaStreamDestroy(streams[i]);
//     }

//     std::cout << "Finished freeing temporary memory: ";
//     printGPUMemoryInfo();
    
//     // Free global data
//     cudaFree(covariance_gpu);
//     cudaFree(mean_gpu);
//     cudaFree(d_sigmapts);
// }


template <typename SDFType>
void CudaOperation_Base<SDFType>::update_sigmapts(const MatrixXd& covariance, const MatrixXd& mean,
                                                  int dim_conf, int num_states, MatrixXd& sigmapts) {
    Timer timer;
    timer.start();
    // 1. Allocate device memory for covariance, mean, and sigmapts.
    double alpha = 1.0, beta = 0.0;
    double *d_covariance, *d_mean, *d_sigmapts;
    size_t covarianceSize = num_states * dim_conf * dim_conf * sizeof(double);
    size_t meanSize       = num_states * dim_conf * sizeof(double);
    size_t sigmaptsSize   = _sigmapts_rows * dim_conf * num_states * sizeof(double);

    cudaMalloc(&d_covariance, covarianceSize);
    cudaMalloc(&d_mean, meanSize);
    cudaMalloc(&d_sigmapts, sigmaptsSize);
    
    cudaMemcpy(d_covariance, covariance.data(), covarianceSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, mean.data(), meanSize, cudaMemcpyHostToDevice);

    std::cout << "Time to allocate memory: " << timer.end_mus_output() << " us" << std::endl;

    // // 2. Setup cusolver handle and create syevj parameter object.
    // timer.start();
    // cusolverDnHandle_t cusolverH = nullptr;
    // cusolverDnCreate(&cusolverH);

    // cublasHandle_t cublasH = nullptr;
    // cublasCreate(&cublasH);
    
    // syevjInfo_t syevj_params = nullptr;
    // cusolverDnCreateSyevjInfo(&syevj_params);
    // std::cout << "Time to create handles: " << timer.end_mus_output() << " us" << std::endl;

    // 3. Allocate memory for eigenvalues and info.
    timer.start();
    double* d_eigenvalues = nullptr;
    size_t eigenvaluesSize = dim_conf * num_states * sizeof(double);
    cudaMalloc(&d_eigenvalues, eigenvaluesSize);
    
    int* d_info = nullptr;
    cudaMalloc(&d_info, num_states * sizeof(int));
    
    // 4. Query workspace size for batched eigen decomposition.
    int lwork = 0;
    cusolverDnDsyevjBatched_bufferSize(_cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                       dim_conf, d_covariance, dim_conf, d_eigenvalues, &lwork, _syevj_params, num_states);
    
    // 5. Allocate workspace memory.
    double* work = nullptr;
    cudaMalloc(&work, lwork * sizeof(double));
    
    // 6. Perform batched symmetric eigen decomposition.
    cusolverStatus_t status = cusolverDnDsyevjBatched(_cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                           dim_conf, d_covariance, dim_conf, d_eigenvalues, work, lwork, d_info, _syevj_params, num_states);


    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cusolverDnDsyevjBatched failed with status " << status << std::endl;
    }
    std::cout << "Time to perform eigen decomposition: " << timer.end_mus_output() << " us" << std::endl;
    
    // 7. Copy eigenvector matrices from d_covariance to d_eigvec (Preserve original eigenvectors)
    double* d_eigvec = nullptr;
    cudaMalloc(&d_eigvec, covarianceSize);
    cudaMemcpy(d_eigvec, d_covariance, covarianceSize, cudaMemcpyDeviceToDevice);
    
    // 8. Launch combined kernel to compute sqrt of eigenvalues and scale eigenvectors.
    timer.start();
    double* d_scaledEigvec = nullptr;
    cudaMalloc(&d_scaledEigvec, covarianceSize);
    int threadsPerBlock = 256;
    int blocksPerGrid = num_states;
    size_t sharedMemSize = dim_conf * sizeof(double);
    sqrtAndScaleEigenvectorsKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_eigenvalues, d_eigvec, d_scaledEigvec, dim_conf, num_states);

    cudaDeviceSynchronize();
    std::cout << "Time to compute sqrt and scale eigenvectors: " << timer.end_mus_output() << " us" << std::endl;
    
    // 9. Use cublasDgemmBatched to compute sqrtP = d_scaledEigvec * (d_eigvec)^T for each state.
    timer.start();
    double* d_sqrtP = nullptr;
    size_t sqrtPSize = num_states * dim_conf * dim_conf * sizeof(double);
    cudaMalloc(&d_sqrtP, sqrtPSize);
    
    std::vector<const double*> h_A_gemm(num_states);
    std::vector<const double*> h_B_gemm(num_states);
    std::vector<double*> h_C_gemm(num_states);
    for (int i = 0; i < num_states; i++) {
        h_A_gemm[i] = d_scaledEigvec + i * dim_conf * dim_conf; // scaled eigenvectors
        h_B_gemm[i] = d_eigvec + i * dim_conf * dim_conf;         // original eigenvectors
        h_C_gemm[i] = d_sqrtP + i * dim_conf * dim_conf;           // output sqrtP
    }
    const double** d_A_gemm = nullptr;
    const double** d_B_gemm = nullptr;
    double** d_C_gemm = nullptr;
    cudaMalloc(&d_A_gemm, num_states * sizeof(const double*));
    cudaMalloc(&d_B_gemm, num_states * sizeof(const double*));
    cudaMalloc(&d_C_gemm, num_states * sizeof(double*));
    cudaMemcpy(d_A_gemm, h_A_gemm.data(), num_states * sizeof(const double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_gemm, h_B_gemm.data(), num_states * sizeof(const double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_gemm, h_C_gemm.data(), num_states * sizeof(double*), cudaMemcpyHostToDevice);    
    
    // Compute sqrtP = d_scaledEigvec * (d_eigvec)^T for each state.
    cublasDgemmBatched(_cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                       dim_conf, dim_conf, dim_conf,
                       &alpha,
                       d_A_gemm, dim_conf,
                       d_B_gemm, dim_conf,
                       &beta,
                       d_C_gemm, dim_conf,
                       num_states);
    std::cout << "Time to compute sqrtP: " << timer.end_mus_output() << " us" << std::endl;
    
    // 10. Compute sigmapts for each state: sigmapts = _zeromean_gpu * (sqrtP)^T.
    timer.start();
    std::vector<const double*> h_A2_gemm(num_states);
    std::vector<const double*> h_B2_gemm(num_states);
    std::vector<double*> h_C2_gemm(num_states);
    for (int i = 0; i < num_states; i++) {
        h_A2_gemm[i] = _zeromean_gpu; 
        h_B2_gemm[i] = d_sqrtP + i * dim_conf * dim_conf; // sqrtP for current state
        h_C2_gemm[i] = d_sigmapts + i * _sigmapts_rows * dim_conf;
    }
    const double** d_A2_gemm = nullptr;
    const double** d_B2_gemm = nullptr;
    double** d_C2_gemm = nullptr;
    cudaMalloc(&d_A2_gemm, num_states * sizeof(const double*));
    cudaMalloc(&d_B2_gemm, num_states * sizeof(const double*));
    cudaMalloc(&d_C2_gemm, num_states * sizeof(double*));
    cudaMemcpy(d_A2_gemm, h_A2_gemm.data(), num_states * sizeof(const double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2_gemm, h_B2_gemm.data(), num_states * sizeof(const double*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_gemm, h_C2_gemm.data(), num_states * sizeof(double*), cudaMemcpyHostToDevice);
    
    // Compute sigmapts = _zeromean_gpu * (sqrtP)^T.
    cublasDgemmBatched(_cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                       _sigmapts_rows, dim_conf, dim_conf,
                       &alpha,
                       d_A2_gemm, _sigmapts_rows,
                       d_B2_gemm, dim_conf,
                       &beta,
                       d_C2_gemm, _sigmapts_rows,
                       num_states);
    std::cout << "Time to compute sigmapts: " << timer.end_mus_output() << " us" << std::endl;
    
    // 11. Add the mean to each state's sigmapts block. (Several milliseconds)
    int totalElements = num_states * _sigmapts_rows * dim_conf;
    int threads = 256;
    int blocks = (totalElements + threads - 1) / threads;
    addMeanKernel_batched<<<blocks, threads>>>(d_sigmapts, d_mean, _sigmapts_rows, dim_conf, num_states);
    
    // // 12. Copy the final sigmapts from device to host.
    // timer.start();
    // Eigen::MatrixXd sigma(_sigmapts_rows, dim_conf * num_states);
    // cudaMemcpy(sigma.data(), d_sigmapts, sigmaptsSize, cudaMemcpyDeviceToHost);
    // // sigmapts.resize(_sigmapts_rows, dim_conf * num_states);
    // sigmapts = sigma;
    // std::cout << "Time to copy sigmapts to host: " << timer.end_mus_output() << " us" << std::endl;
    
    // 13. Free all allocated device memory and destroy handles.
    timer.start();
    cudaFree(d_covariance);
    cudaFree(d_mean);
    cudaFree(d_sigmapts);
    cudaFree(d_eigenvalues);
    cudaFree(d_info);
    cudaFree(work);
    cudaFree(d_eigvec);
    cudaFree(d_scaledEigvec);
    cudaFree(d_sqrtP);
    
    cudaFree(d_A_gemm);
    cudaFree(d_B_gemm);
    cudaFree(d_C_gemm);
    cudaFree(d_A2_gemm);
    cudaFree(d_B2_gemm);
    cudaFree(d_C2_gemm);
    
    // cusolverDnDestroySyevjInfo(_syevj_params);
    // cusolverDnDestroy(_cusolverH);
    // // cublasDestroy(_cublasH);
    std::cout << "Time to free memory and destroy handles: " << timer.end_mus_output() << " us" << std::endl;
}



template <typename SDFType>
void CudaOperation_Base<SDFType>::initializeSigmaptsResources(int dim_conf, int num_states, int sigmapts_rows){
    printGPUMemoryInfo();
    // Allocate global device memory for covariance, mean, and sigmapts.
    std::cout << "sigmaPts rows: " << sigmapts_rows << " , sigmarows: " << _sigmapts_rows << std::endl;
    cudaMalloc(&covariance_gpu, dim_conf * dim_conf * num_states * sizeof(double));
    cudaMalloc(&d_sigmapt_cuda, sigmapts_rows * dim_conf * num_states * sizeof(double));
    cudaMalloc(&mean_gpu, 2 * dim_conf * (num_states+2) * sizeof(double));

    num_streams = 5;
    
    // Resize vectors to hold per-state resources.
    streams.resize(num_streams);
    cusolver_handles.resize(num_streams);
    cublas_handles.resize(num_streams);

    for (int i = 0; i < num_streams; i++) {
        // Create CUDA stream.
        cudaStreamCreate(&streams[i]);
        // Create cuSOLVER and cuBLAS handles and bind them with the stream.
        cusolverDnCreate(&cusolver_handles[i]);
        cublasCreate(&cublas_handles[i]);
        cusolverDnSetStream(cusolver_handles[i], streams[i]);
        cublasSetStream(cublas_handles[i], streams[i]);
    }

    // // Try one thread first
    // cusolverDnCreate(&cusolver_handle);
    // cublasCreate(&cublas_handle);

    d_eigen_values_vec.resize(num_states, nullptr);
    d_info_vec.resize(num_states, nullptr);
    d_work_vec.resize(num_states, nullptr);
    Lwork_vec.resize(num_states, 0);
    d_V_scaled_vec.resize(num_states, nullptr);
    d_sqrtP_vec.resize(num_states, nullptr);

    double* d_dummy;
    cudaMalloc(&d_dummy, dim_conf * dim_conf * sizeof(double));
    
    // For each state, create stream, create handles, and allocate temporary memory.
    for (int state = 0; state < num_states; state++) {
        int stream_idx = state % num_streams;
        // Allocate memory for eigenvalues and info.
        cudaMalloc(&d_eigen_values_vec[state], dim_conf * sizeof(double));
        cudaMalloc(&d_info_vec[state], sizeof(int));
        
        // Query workspace size for eigen decomposition.
        cusolverDnDsyevd_bufferSize(cusolver_handles[stream_idx],
                                    CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_LOWER,
                                    dim_conf,
                                    d_dummy,
                                    dim_conf,
                                    d_eigen_values_vec[state],
                                    &Lwork_vec[state]);
        // Allocate workspace memory.
        cudaMalloc(&d_work_vec[state], Lwork_vec[state] * sizeof(double));
        
        // Allocate memory for V_scaled and sqrtP matrices (each size: dim_conf x dim_conf).
        cudaMalloc(&d_V_scaled_vec[state], dim_conf * dim_conf * sizeof(double));
        cudaMalloc(&d_sqrtP_vec[state], dim_conf * dim_conf * sizeof(double));
    }
    cudaFree(d_dummy);

    printGPUMemoryInfo();
}


template <typename SDFType>
void CudaOperation_Base<SDFType>::update_sigmapts_separate(const MatrixXd& covariance, const MatrixXd& mean, int dim_conf, int num_states, MatrixXd& sigmapts){
    // printGPUMemoryInfo();
    const double alpha = 1.0, beta = 0.0;

    Timer timer;
    
    // Copy new covariance and mean data into the pre-allocated device memory.
    size_t covariance_size = dim_conf * dim_conf * num_states * sizeof(double);
    size_t mean_size = mean.size() * sizeof(double);
    cudaMemcpy(covariance_gpu, covariance.data(), covariance_size, cudaMemcpyHostToDevice);
    cudaMemcpy(mean_gpu, mean.data(), mean_size, cudaMemcpyHostToDevice);
    
    // For each state, perform the computation.
    for (int state = 0; state < num_states; state++) {
        // Each state's covariance matrix is stored at an offset in covariance_gpu.
        double* d_Pi = covariance_gpu + state * dim_conf * dim_conf;
        // Assuming mean data occupies two rows per state; adjust the offset as needed.
        double* d_mi = mean_gpu + 2 * (state + 1) * dim_conf;

        int stream_idx = state % num_streams;
        
        if (state % 1000 == 0){
            timer.start();
        }
        
        // 2.1 Perform symmetric eigenvalue decomposition on d_Pi.
        cusolverDnDsyevd(cusolver_handles[stream_idx],
                         CUSOLVER_EIG_MODE_VECTOR,
                         CUBLAS_FILL_MODE_LOWER,
                         dim_conf,
                         d_Pi,
                         dim_conf,
                         d_eigen_values_vec[state],
                         d_work_vec[state],
                         Lwork_vec[state],
                         d_info_vec[state]);

        if (state % 1000 == 0){
            std::cout << "Eigen decomposition time: " << timer.end_mus_output() << " us" << std::endl;
            timer.start();
        }
        
        int h_info;
        cudaMemcpyAsync(&h_info, d_info_vec[state], sizeof(int), cudaMemcpyDeviceToHost);
        if (h_info != 0) {
            std::cerr << "Eigen decomposition failed for state " << state << " with info " << h_info << std::endl;
        }

        // 2.2 Apply square root to the eigenvalues using a custom kernel.
        int threadsPerBlock = 8;
        int blocks = (dim_conf + threadsPerBlock - 1) / threadsPerBlock;
        sqrtKernel<<<blocks, threadsPerBlock, 0, streams[stream_idx]>>>(d_eigen_values_vec[state], dim_conf);
        cudaStreamSynchronize(streams[stream_idx]);

        if (state % 1000 == 0){
            std::cout << "Sqrt kernel time: " << timer.end_mus_output() << " us" << std::endl;
            timer.start();
        }
        
        // 2.3 Use cublasDdgmm to scale the eigenvector matrix by the square-rooted eigenvalues.
        cublasStatus_t stat = cublasDdgmm(cublas_handles[stream_idx], CUBLAS_SIDE_RIGHT, dim_conf, dim_conf,
                                          d_Pi, dim_conf,
                                          d_eigen_values_vec[state], 1,
                                          d_V_scaled_vec[state], dim_conf);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasDdgmm failed for state " << state << ": " << stat << std::endl;
        }

        if (state % 1000 == 0){
            std::cout << "Ddgmm time: " << timer.end_mus_output() << " us" << std::endl;
            timer.start();
        }
        
        // 2.4 Compute sqrtP = d_V_scaled * (d_Pi)^T.
        stat = cublasDgemm(cublas_handles[stream_idx], CUBLAS_OP_N, CUBLAS_OP_T, 
                           dim_conf, dim_conf, dim_conf,
                           &alpha, d_V_scaled_vec[state], dim_conf,
                           d_Pi, dim_conf,
                           &beta, d_sqrtP_vec[state], dim_conf);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasDgemm for sqrtP failed for state " << state << ": " << stat << std::endl;
        }

        if (state % 1000 == 0){
            std::cout << "Gemm time: " << timer.end_mus_output() << " us" << std::endl;
            timer.start();
        }
        
        // 2.5 Compute the current state's portion of sigmapts.
        // d_sigmapt_cuda is allocated contiguously on the device; each state occupies a block of _sigmapts_rows x dim_conf.
        
        // std::cout << "sigma rows: " << _sigmapts_rows << " dim_conf: " << dim_conf << " state: " << state << std::endl;
        double* d_sigmapts_state = d_sigmapt_cuda + state * _sigmapts_rows * dim_conf;
        // cudaMemcpy(_zeromean_gpu, zeromean.data(), _sigmapts_rows * dim_conf * sizeof(double), cudaMemcpyHostToDevice);
        stat = cublasDgemm(cublas_handles[stream_idx], CUBLAS_OP_N, CUBLAS_OP_T, 
                           _sigmapts_rows, dim_conf, dim_conf,
                           &alpha, _zeromean_gpu, _sigmapts_rows,
                           d_sqrtP_vec[state], dim_conf,
                           &beta, d_sigmapts_state, _sigmapts_rows);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cublasDgemm for sigmapts failed for state " << state << ": " << stat << std::endl;
        }

        if (state % 1000 == 0){
            std::cout << "Zero mean time: " << timer.end_mus_output() << " us" << std::endl;
            timer.start();
        }
        
        // 2.6 Add the mean to sigmapts (for the current state's block).
        threadsPerBlock = 256;
        blocks = (_sigmapts_rows * dim_conf + threadsPerBlock - 1) / threadsPerBlock;
        addMeanKernel<<<blocks, threadsPerBlock, 0, streams[stream_idx]>>>(d_sigmapts_state, d_mi, _sigmapts_rows, dim_conf);

        if (state % 1000 == 0){
            std::cout << "Add mean time: " << timer.end_mus_output() << " us" << std::endl << std::endl;
        }
    }
    
    // Synchronize all streams.
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    // // Copy the computed sigmapts from device to host.
    // MatrixXd sigma(_sigmapts_rows, dim_conf * num_states);
    // size_t sigmapts_size = _sigmapts_rows * dim_conf * num_states * sizeof(double);
    // cudaMemcpy(sigma.data(), d_sigmapt_cuda, sigmapts_size, cudaMemcpyDeviceToHost);
    // sigmapts = sigma;
}

template <typename SDFType>
void CudaOperation_Base<SDFType>::freeSigmaptsResources(int num_states)
{
    // std::cout << "Memory before freeing: ";
    // printGPUMemoryInfo();

    // Free per-state temporary memory and destroy handles/streams.
    for (int state = 0; state < num_states; state++) {
        cudaFree(d_eigen_values_vec[state]);
        cudaFree(d_info_vec[state]);
        cudaFree(d_work_vec[state]);
        cudaFree(d_V_scaled_vec[state]);
        cudaFree(d_sqrtP_vec[state]);
    }
    // Free global device memory.
    cudaFree(covariance_gpu);
    cudaFree(mean_gpu);
    cudaFree(d_sigmapt_cuda);

    for(int i = 0; i < num_streams; i++){
        cusolverDnDestroy(cusolver_handles[i]);
        // cublasDestroy(cublas_handles[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    // Clear the vectors.
    streams.clear();
    cusolver_handles.clear();
    cublas_handles.clear();
    d_eigen_values_vec.clear();
    d_info_vec.clear();
    d_work_vec.clear();
    Lwork_vec.clear();
    d_V_scaled_vec.clear();
    d_sqrtP_vec.clear();

    // std::cout << "Memory after freeing: ";
    // printGPUMemoryInfo();
    
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
    // cudaMemcpy(_sigmapts_gpu, sigmapts.data(), sigmapts.size() * sizeof(double), cudaMemcpyHostToDevice);

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



// // Using MAGMA batched interface to compute the square root of each state's covariance matrix,
// // then multiply with a given matrix (_zeromean_gpu) and finally add a mean vector.
// // - Assume covariance is a (dim_conf x dim_conf*num_states) matrix,
// // - where each state's covariance is stored consecutively (column-major);
// // - mean is a (num_states x dim_conf) matrix.
// // - The output sigmapts has size (_sigmapts_rows x dim_conf*num_states).
// template <typename SDFType>
// void CudaOperation_Base<SDFType>::update_sigmapts_magma_batched(const MatrixXd& covariance,
//                                                                   const MatrixXd& mean,
//                                                                   int dim_conf,
//                                                                   int num_states,
//                                                                   MatrixXd& sigmapts)
// {
//     int sig_rows = _sigmapts_rows;  // Predefined number of sigma points rows

//     // 1. Copy covariance and mean to device memory.
//     double *d_cov, *d_mean, *d_sig;
//     size_t cov_size = num_states * dim_conf * dim_conf * sizeof(double);
//     size_t mean_size = num_states * dim_conf * sizeof(double);
//     size_t sig_size = num_states * sig_rows * dim_conf * sizeof(double);
//     cudaMalloc(&d_cov, cov_size);
//     cudaMalloc(&d_mean, mean_size);
//     cudaMalloc(&d_sig, sig_size);
//     cudaMemcpy(d_cov, covariance.data(), cov_size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_mean, mean.data(), mean_size, cudaMemcpyHostToDevice);

//     // 2. Construct a batched pointer array that points to each state's covariance matrix.
//     double **d_A_array;
//     cudaMalloc(&d_A_array, num_states * sizeof(double*));
//     std::vector<double*> h_A_array(num_states);
//     for (int state = 0; state < num_states; state++) {
//         h_A_array[state] = d_cov + state * dim_conf * dim_conf;
//     }
//     cudaMemcpy(d_A_array, h_A_array.data(), num_states * sizeof(double*), cudaMemcpyHostToDevice);

//     // 3. Allocate device memory for eigenvalues (each state has dim_conf eigenvalues)
//     double *d_w;
//     cudaMalloc(&d_w, num_states * dim_conf * sizeof(double));
//     // Allocate info array for each state.
//     int *d_info;
//     cudaMalloc(&d_info, num_states * sizeof(int));

//     // 4. Create a MAGMA queue (MAGMA uses an internal CUDA stream).
//     magma_queue_t queue;
//     magma_queue_create(0, &queue);

//     // 5. Call MAGMA batched eigenvalue decomposition:
//     // Compute eigenvalue decomposition A = Q and store eigenvalues in d_w.

//     // // There is no such function
//     // magma_dsyevd_batched(MagmaVec, MagmaLower, dim_conf,
//     //                      d_A_array, dim_conf,
//     //                      d_w, d_info, num_states, queue);

//     magma_queue_sync(queue);

//     // 6. Scale each eigenvector column: Q(:,j) = Q(:,j) * sqrt(eigenvalue_j).
//     {
//         dim3 threads(16, 16);
//         dim3 grid((dim_conf + threads.x - 1) / threads.x,
//                   (num_states + threads.y - 1) / threads.y);
//         scale_eigvecs_kernel<<<grid, threads, 0, magma_queue_get_cuda_stream(queue)>>>(
//             d_A_array, d_w, dim_conf, num_states);
//         magma_queue_sync(queue);
//     }

//     // 7. Use batched GEMM to compute the square root of each covariance matrix:
//     // sqrt(P) = Q * Q^T for each state.
//     double **d_sqrtP_array;
//     cudaMalloc(&d_sqrtP_array, num_states * sizeof(double*));
//     std::vector<double*> h_sqrtP_array(num_states);
//     // Allocate memory for sqrtP for each state (size: dim_conf x dim_conf).
//     double *d_sqrtP;
//     cudaMalloc(&d_sqrtP, num_states * dim_conf * dim_conf * sizeof(double));
//     for (int s = 0; s < num_states; s++) {
//         h_sqrtP_array[s] = d_sqrtP + s * dim_conf * dim_conf;
//     }
//     cudaMemcpy(d_sqrtP_array, h_sqrtP_array.data(), num_states * sizeof(double*), cudaMemcpyHostToDevice);

//     double alpha = 1.0, beta = 0.0;
//     magma_dgemm_batched(MagmaNoTrans, MagmaTrans,
//                         dim_conf, dim_conf, dim_conf,
//                         alpha, d_A_array, dim_conf,
//                         d_A_array, dim_conf,
//                         beta, d_sqrtP_array, dim_conf,
//                         num_states, queue);
//     magma_queue_sync(queue);

//     // 8. Compute sigma points for each state using sqrt(P):
//     // For each state, compute: sigmapts_state = (_zeromean) * sqrt(P)
//     // _zeromean_gpu is assumed to be pre-allocated on the device (size: sig_rows x dim_conf).
//     // Construct a batched pointer array pointing to _zeromean (assumed identical for all states).
//     double **d_zeromean_array;
//     cudaMalloc(&d_zeromean_array, num_states * sizeof(double*));
//     std::vector<double*> h_zeromean_array(num_states, _zeromean_gpu);
//     cudaMemcpy(d_zeromean_array, h_zeromean_array.data(), num_states * sizeof(double*), cudaMemcpyHostToDevice);

//     // Construct a batched pointer array for each state's output sigmapts block.
//     double **d_sig_array;
//     cudaMalloc(&d_sig_array, num_states * sizeof(double*));
//     std::vector<double*> h_sig_array(num_states);
//     for (int s = 0; s < num_states; s++) {
//         h_sig_array[s] = d_sig + s * (sig_rows * dim_conf);
//     }
//     cudaMemcpy(d_sig_array, h_sig_array.data(), num_states * sizeof(double*), cudaMemcpyHostToDevice);

//     // Perform batched GEMM: for each state,
//     // sigmapts = _zeromean_gpu (size: sig_rows x dim_conf) multiplied by sqrt(P) (dim_conf x dim_conf).
//     magma_dgemm_batched(MagmaNoTrans, MagmaNoTrans,
//                         sig_rows, dim_conf, dim_conf,
//                         alpha, d_zeromean_array, sig_rows,
//                         d_sqrtP_array, dim_conf,
//                         beta, d_sig_array, sig_rows,
//                         num_states, queue);
//     magma_queue_sync(queue);

//     // 9. For each state, add the mean vector to the corresponding sigmapts (add to every row).
//     {
//         dim3 threads(16, 16);
//         // Total rows: num_states * sig_rows, each row has dim_conf elements.
//         dim3 grid((dim_conf + threads.x - 1) / threads.x,
//                   ((num_states * sig_rows) + threads.y - 1) / threads.y);
//         add_mean_kernel_batched<<<grid, threads, 0, magma_queue_get_cuda_stream(queue)>>>(
//             d_sig, d_mean, sig_rows, dim_conf, num_states);
//         magma_queue_sync(queue);
//     }

//     // 10. Copy the computed sigma points back to host.
//     sigmapts.resize(sig_rows, num_states * dim_conf);
//     cudaMemcpy(sigmapts.data(), d_sig, sig_size, cudaMemcpyDeviceToHost);

//     // 11. Free all device resources.
//     cudaFree(d_cov);
//     cudaFree(d_mean);
//     cudaFree(d_sig);
//     cudaFree(d_A_array);
//     cudaFree(d_w);
//     cudaFree(d_info);
//     cudaFree(d_sqrtP_array);
//     cudaFree(d_sqrtP);
//     cudaFree(d_zeromean_array);
//     cudaFree(d_sig_array);
//     magma_queue_destroy(queue);
// }





// template <typename SDFType>
// void CudaOperation_Base<SDFType>::update_sigmapts(const MatrixXd& covariance, const MatrixXd& mean, int dim_conf, int num_states, MatrixXd& sigmapts){
//     double *covariance_gpu, *mean_gpu, *d_sigmapts;
//     cudaMalloc(&covariance_gpu, dim_conf * dim_conf * num_states * sizeof(double));
//     cudaMalloc(&mean_gpu, mean.size() * sizeof(double));
//     cudaMalloc(&d_sigmapts, _sigmapts_rows * dim_conf * num_states * sizeof(double));

//     cudaMemcpy(covariance_gpu, covariance.data(),
//                dim_conf * dim_conf * num_states * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(mean_gpu, mean.data(),
//                mean.size() * sizeof(double), cudaMemcpyHostToDevice);

//     // Create 10 streams and corresponding cuSOLVER handles for eigen decomposition
//     const int num_streams = 10;
//     std::vector<cudaStream_t> eigen_streams(num_streams);
//     std::vector<cusolverDnHandle_t> cusolver_handles(num_streams);
//     for (int i = 0; i < num_streams; i++) {
//         cudaStreamCreate(&eigen_streams[i]);
//         cusolverDnCreate(&cusolver_handles[i]);
//         cusolverDnSetStream(cusolver_handles[i], eigen_streams[i]);
//     }

//     // Create a global cuBLAS handle for batched operations (using the default stream)
//     cublasHandle_t cublas_handle;
//     cublasCreate(&cublas_handle);

//     // Allocate temporary device memory for each state's operations and record pointers
//     std::vector<double*> d_Pi_array(num_states);
//     std::vector<double*> d_eigen_values_array(num_states);
//     std::vector<double*> d_V_scaled_array(num_states);
//     std::vector<double*> d_sqrtP_array(num_states);
//     std::vector<double*> d_sigmapts_array(num_states);
//     std::vector<double*> d_mean_array(num_states);

//     const double alpha = 1.0, beta = 0.0;
//     for (int state = 0; state < num_states; state++) {
//         // 1. Each state's covariance matrix offset in covariance_gpu
//         double* d_Pi = covariance_gpu + state * dim_conf * dim_conf;
//         d_Pi_array[state] = d_Pi;

//         // 2. Allocate temporary memory for eigen decomposition: eigenvalues and info.
//         double* d_eigen_values;
//         int* d_info;
//         int Lwork = 0;
//         cudaMalloc(&d_eigen_values, dim_conf * sizeof(double));
//         cudaMalloc(&d_info, sizeof(int));
//         d_eigen_values_array[state] = d_eigen_values;

//         int stream_idx = state % num_streams;  // Assign to one of the 10 streams
//         cusolverDnDsyevd_bufferSize(cusolver_handles[stream_idx],
//                                     CUSOLVER_EIG_MODE_VECTOR,
//                                     CUBLAS_FILL_MODE_LOWER,
//                                     dim_conf,
//                                     d_Pi,
//                                     dim_conf,
//                                     d_eigen_values,
//                                     &Lwork);
//         double* d_work;
//         cudaMalloc(&d_work, Lwork * sizeof(double));

//         cusolverDnDsyevd(cusolver_handles[stream_idx],
//                          CUSOLVER_EIG_MODE_VECTOR,
//                          CUBLAS_FILL_MODE_LOWER,
//                          dim_conf,
//                          d_Pi,
//                          dim_conf,
//                          d_eigen_values,
//                          d_work,
//                          Lwork,
//                          d_info);

//         int h_info;
//         cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
//         if (h_info != 0) {
//             std::cerr << "Eigen decomposition failed for state " << state
//                       << " with info " << h_info << std::endl;
//         }
//         cudaFree(d_info);
//         cudaFree(d_work);

//         // 3. Apply square root to eigenvalues using a custom kernel on the corresponding stream
//         int threadsPerBlock = 16;
//         int blocks = (dim_conf + threadsPerBlock - 1) / threadsPerBlock;
//         sqrtKernel<<<blocks, threadsPerBlock, 0, eigen_streams[stream_idx]>>>(d_eigen_values, dim_conf);

//         // 4. Allocate memory for ddgmm result (d_V_scaled)
//         double* d_V_scaled;
//         cudaMalloc(&d_V_scaled, dim_conf * dim_conf * sizeof(double));
//         d_V_scaled_array[state] = d_V_scaled;

//         // 5. Allocate memory for sqrtP: will compute sqrtP = d_V_scaled * (d_Pi)^T
//         double* d_sqrtP;
//         cudaMalloc(&d_sqrtP, dim_conf * dim_conf * sizeof(double));
//         d_sqrtP_array[state] = d_sqrtP;

//         // 6. Each state's sigmapts block stored in d_sigmapts (block size: _sigmapts_rows x dim_conf)
//         double* d_sigmapts_state = d_sigmapts + state * _sigmapts_rows * dim_conf;
//         d_sigmapts_array[state] = d_sigmapts_state;
//         double* d_mi = mean_gpu + 2 * (state + 1) * dim_conf;
//         d_mean_array[state] = d_mi;
//     }
//     // std::cout << "Finished allocating temporary memory" << std::endl;

//     // Synchronize all eigen streams to ensure eigen decomposition and sqrtKernel are completed
//     for (int i = 0; i < num_streams; i++) {
//         cudaStreamSynchronize(eigen_streams[i]);
//     }

//     // 4.1 Batched ddgmm: compute d_V_scaled = d_Pi .* diag( sqrt(eigen_values) )
//     int threads = 256;
//     int blocks = (dim_conf * dim_conf * num_states + threads - 1) / threads;
//     batchedDdgmmKernel<<<blocks, threads>>>(d_Pi_array.data(), d_eigen_values_array.data(), d_V_scaled_array.data(), dim_conf, num_states);
//     cudaDeviceSynchronize();

//     // 4.2 Batched GEMM: compute sqrtP = d_V_scaled * (d_Pi)^T for each state
//     {
//         // cublasDgemmBatched requires arrays of pointesanrs for matrices
//         std::vector<const double*> A_array(num_states);  // from d_V_scaled_array
//         std::vector<const double*> B_array(num_states);  // from d_Pi_array (using transpose)
//         std::vector<double*>      C_array(num_states);  // result sqrtP stored in d_sqrtP_array

//         for (int state = 0; state < num_states; state++) {
//             A_array[state] = d_V_scaled_array[state];
//             B_array[state] = d_Pi_array[state];
//             C_array[state] = d_sqrtP_array[state];
//         }
//         cublasDgemmBatched(cublas_handle,
//                            CUBLAS_OP_N,   // A is not transposed
//                            CUBLAS_OP_T,   // B is transposed
//                            dim_conf,      // m
//                            dim_conf,      // n
//                            dim_conf,      // k
//                            &alpha,
//                            A_array.data(), dim_conf,  // Leading dimension of each A
//                            B_array.data(), dim_conf,  // Leading dimension of each B
//                            &beta,
//                            C_array.data(), dim_conf,  // Leading dimension of each C
//                            num_states);

//     }

//     // 4.3 Batched GEMM: compute sigmapts = _zeromean_gpu * (sqrtP)^T for each state
//     {
//         // Assume _zeromean_gpu stores each state's block of size _sigmapts_rows x dim_conf,
//         // and each sqrtP matrix in d_sqrtP_array is of size dim_conf x dim_conf.
//         std::vector<const double*> zeromean_array(num_states);
//         std::vector<const double*> sqrtP_array(num_states);
//         std::vector<double*>       sigmapts_array(num_states);
//         for (int state = 0; state < num_states; state++) {
//             zeromean_array[state] = _zeromean_gpu;
//             sqrtP_array[state] = d_sqrtP_array[state];
//             sigmapts_array[state] = d_sigmapts_array[state];
//         }
//         cublasDgemmBatched(cublas_handle,
//                            CUBLAS_OP_N,   // _zeromean_gpu is not transposed
//                            CUBLAS_OP_T,   // sqrtP is transposed
//                            _sigmapts_rows,  // m
//                            dim_conf,        // n
//                            dim_conf,        // k
//                            &alpha,
//                            zeromean_array.data(), _sigmapts_rows,
//                            sqrtP_array.data(), dim_conf,
//                            &beta,
//                            sigmapts_array.data(), _sigmapts_rows,
//                            num_states);
//     }

//     // 4.4 Batched add mean: add the corresponding mean to each state's sigmapts block
//     int total_elements = _sigmapts_rows * dim_conf * num_states;
//     threads = 256;
//     blocks = (total_elements + threads - 1) / threads;
//     addMeanKernelBatched<<<blocks, threads>>>(d_sigmapts, d_mean_array.data(), _sigmapts_rows, dim_conf, num_states);
//     cudaDeviceSynchronize();

//     // Copy the final result from device memory back to host
//     MatrixXd sigma(_sigmapts_rows, dim_conf * num_states);
//     cudaMemcpy(sigma.data(), d_sigmapts, _sigmapts_rows * dim_conf * num_states * sizeof(double), cudaMemcpyDeviceToHost);
//     sigmapts = sigma;

    // // Free temporary memory for each state and destroy handles and streams
    // for (int state = 0; state < num_states; state++) {
    //     cudaFree(d_eigen_values_array[state]);
    //     cudaFree(d_V_scaled_array[state]);
    //     cudaFree(d_sqrtP_array[state]);
    // }
    // for (int i = 0; i < num_streams; i++) {
    //     cusolverDnDestroy(cusolver_handles[i]);
    //     cudaStreamDestroy(eigen_streams[i]);
    // }
    // // cublasDestroy(cublas_handle);
    // cudaFree(covariance_gpu);
    // cudaFree(mean_gpu);
    // cudaFree(d_sigmapts);

    // printGPUMemoryInfo();
// }