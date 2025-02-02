#include<iostream>
#include<Eigen/Dense>
#include<benchmark/benchmark.h>
#include<mkl.h>
#include <mkl_vsl.h>
#include <vector>


void printMatrix(const std::vector<double>& mat, const int & n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << mat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}


void EigenToMKL(const Eigen::MatrixXd& eigen_matrix, std::vector<double>& mkl_matrix, const int & N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mkl_matrix[i * N + j] = eigen_matrix(i, j);
        }
    }
}


void SqrtEigenSolverMKL(std::vector<double>& mkl_matrix, std::vector<double>& result, const int & N) {
    char transa = 'N';  // No transpose for M
    char transb = 'T';  // Transpose for M^T
    double alpha = 1.0; // Scaling factor for M * M^T
    double beta = 0.0;  // No scaling for the result matrix (A)
    int lda = N;        // Leading dimension of M (rows of M)
    int ldb = N;        // Leading dimension of M^T (columns of M)
    int ldc = N;        // Leading dimension of result matrix A (rows of A)

    // Workspaces for the eigenvalues, eigenvectors, and intermediate calculations
    // Cholesky decomposition (jobz = 'V' to compute both eigenvalues and eigenvectors)
    char jobz = 'V'; // Compute eigenvectors
    char uplo = 'L'; // Lower triangular part of A is used

    std::vector<double> w(N, 0.0);  // Eigenvalues
    std::vector<double> work(3 * N); // Workspace for intermediate results
    // std::vector<long long int> iwork(N);       // Integer workspace
    // Calling the dsyev_ function from MKL to compute eigenvalues and eigenvectors
    long long int info;
    long long int N_long = static_cast<long long int>(N);
    long long int lwork = 3 * N;

    dsyev_(&jobz, &uplo, &N_long, mkl_matrix.data(), &N_long, w.data(), work.data(), &lwork, &info);

    // Create D^(1/2) as a diagonal matrix (stored in a vector)
    std::vector<double> D_sqrt_matrix(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        D_sqrt_matrix[i * N + i] = std::sqrt(w[i]); // Diagonal matrix with square roots of eigenvalues
    }

    // Multiply V * D^(1/2) to get intermediate result
    // V's rows are the transposed eigen vectors!!
    std::vector<double> temp_result(N * N, 0.0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, mkl_matrix.data(), N, D_sqrt_matrix.data(), N, 0.0, temp_result.data(), N);

    // Multiply the result by V^T to get the final matrix square root
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, temp_result.data(), N, mkl_matrix.data(), N, 0.0, result.data(), N);

}


void squareMatrixMultiplication(const std::vector<double>& m1, const std::vector<double>& m2, std::vector<double>& result, const int N) {
    // Perform matrix multiplication using cblas_dgemm
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, m1.data(), N, m2.data(), N, 0.0, result.data(), N);
}


void AMultiplyBT(const std::vector<double>& m1, const std::vector<double>& m2, std::vector<double>& result, const int N) {
    // Perform matrix multiplication using cblas_dgemm
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, m1.data(), N, m2.data(), N, 0.0, result.data(), N);
}


void LLTDecomposition(std::vector<double>& matrix_L, const int& N){
    
    // Cholesky decomposition using dpotrf (double precision)
    char uplo = 'U';  // 'L' for lower triangular matrix
    long long int info;
    long long int N_long = static_cast<long long int>(N);
    // Call MKL's dpotrf function for Cholesky decomposition
    dpotrf_(&uplo, &N_long, matrix_L.data(), &N_long, &info);

    // Make the upper triangular part of L to be zero
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            matrix_L[i * N + j] = 0.0;
        }
    }
}