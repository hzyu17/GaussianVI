#include<iostream>
#include<Eigen/Dense>
#include<mkl.h>
#include <mkl_vsl.h>
#include <vector>

#ifndef GVIMP_MKL_H
#define GVIMP_MKL_H

void printMatrix_MKL(const std::vector<double>& mat, const int & rows, const int & cols) {
    std::cout.precision(10);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}


void printVector_MKL(const std::vector<double>& vec, const int & rows) {
    std::cout.precision(10);
    for (int i = 0; i < rows; ++i) {
            std::cout << vec[i] << std::endl;
        }
        std::cout << std::endl;
}


void EigenToMKL(const Eigen::MatrixXd& eigen_matrix, std::vector<double>& mkl_matrix, const int & rows, const int & cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mkl_matrix[i * rows + j] = eigen_matrix(i, j);
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
    long long int lwork = 3*N;

    dsyev_(&jobz, &uplo, &N_long, mkl_matrix.data(), &N_long, w.data(), work.data(), &lwork, &info);

    // Create D^(1/2) as a diagonal matrix (stored in a vector)
    std::vector<double> D_sqrt_matrix(N * N, 0.0);
    for (int i = 0; i < N; ++i) {
        D_sqrt_matrix[i * N + i] = std::sqrt(w[i]); // Diagonal matrix with square roots of eigenvalues
    }

    // Multiply V * D^(1/2) to get intermediate result
    // V's rows are the transposed eigen vectors.
    std::vector<double> temp_result(N * N, 0.0);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, mkl_matrix.data(), N, D_sqrt_matrix.data(), N, 0.0, temp_result.data(), N);

    // Multiply the result by V^T to get the final matrix square root
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, temp_result.data(), N, mkl_matrix.data(), N, 0.0, result.data(), N);

}


void squareMatrixMultiplication(const std::vector<double>& m1, const std::vector<double>& m2, std::vector<double>& result, const int rows, const int cols) {
    // Perform matrix multiplication using cblas_dgemm
    cblas_dgemm(CblasRowMajor, 
                CblasNoTrans, CblasNoTrans, 
                rows, cols, cols, 1.0, 
                m1.data(), cols, 
                m2.data(), cols, 
                0.0, 
                result.data(), cols);
}

void matrix_addition(const std::vector<double>& A, 
                    std::vector<double>& B, 
                    int NumElements) {
    
        // Use cblas_daxpy to add B to A (B = \alpha*A + B)
        // N is the number of elements, 1.0 is the scaling factor, B is added to A (no scaling)
        cblas_daxpy(NumElements, 1.0, A.data(), 1, B.data(), 1);
}

void AMultiplyB(const std::vector<double>& m1, 
                const std::vector<double>& m2, 
                std::vector<double>& result, 
                const int& rows_m1, const int& cols_m1, const int& cols_m2) {
    // Perform matrix multiplication using cblas_dgemm
    cblas_dgemm(CblasRowMajor, 
                CblasNoTrans, CblasNoTrans, 
                rows_m1, cols_m2, cols_m1, 
                1.0, 
                m1.data(), cols_m1, 
                m2.data(), cols_m2, 
                0.0, 
                result.data(), cols_m2);
}

void ATMultiplyB(const std::vector<double>& m1, 
                const std::vector<double>& m2, 
                std::vector<double>& result, 
                const int& rows_m1, const int& cols_m1, const int& cols_m2) {
    // Perform matrix multiplication using cblas_dgemm
    cblas_dgemm(CblasRowMajor, 
                CblasTrans, CblasNoTrans, 
                cols_m1, cols_m2, rows_m1, 
                1.0, 
                m1.data(), cols_m1, 
                m2.data(), cols_m2, 
                0.0, 
                result.data(), cols_m2);
}

void AMultiplyBT(const std::vector<double>& m1, 
                const std::vector<double>& m2, 
                std::vector<double>& result, 
                const int& rows_m1, const int& cols_m1, const int& rows_m2) {
    // Perform matrix multiplication using cblas_dgemm
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows_m1, rows_m2, cols_m1, 1.0, m1.data(), cols_m1, m2.data(), cols_m1, 0.0, result.data(), rows_m2);
}

void AMultiplyBTPlusC(const std::vector<double>& m1, const std::vector<double>& m2, std::vector<double>& m3, const int N) {
    // Perform matrix multiplication using cblas_dgemm
    // C <- \alpha AB^T + \beta C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, m1.data(), N, m2.data(), N, 1.0, m3.data(), N);
}

void AddTransposeToRows(std::vector<double>& A, const std::vector<double>& a, int numRows, int numCols) {
    #pragma omp parallel for
    for (int i = 0; i < numRows; ++i) {
        // Use cblas_daxpy to add a to the i-th row of A
        // A[i*N + j] is the element at row i, column j in A
        cblas_daxpy(numCols, 1.0, a.data(), 1, &A[i * numCols], 1);
    }
}

void get_row_i(const std::vector<double>& matrix, 
                const int& row_i, 
                const int& numCols, 
                std::vector<double>& result){
    // Copy elements from the row row_i of 'matrix' into 'result'.
    std::copy(matrix.begin() + row_i * numCols, 
              matrix.begin() + (row_i + 1) * numCols,
              result.begin());
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
        // Zero out the elements above the diagonal in the i-th row.
        // The lower triangular part is at indices i+1 to N-1 of the i-th row.
        std::fill(matrix_L.begin() + i * N + i + 1,
                  matrix_L.begin() + i * N + N,
                  0.0);
    }
}

#endif