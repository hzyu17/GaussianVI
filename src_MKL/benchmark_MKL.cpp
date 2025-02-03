#include "gvimp_mkl.h"
#include<benchmark/benchmark.h>

const int N = 4;

// const int matrix_size = N * N;
// const int vector_size = N;

// Allocate memory for the random matrix
// std::vector<double> matrix(matrix_size);
// std::vector<double> vector(vector_size);

// // Define the range for random numbers
// double lower_bound = 0.0;  // Minimum value
// double upper_bound = 1.0;  // Maximum value

// // Create a VSL random stream
// VSLStreamStatePtr stream;
// int status = vslNewStream(&stream, VSL_BRNG_MT19937, 42); // MT19937 generator with seed 42
// int status_1 = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, matrix_size, matrix.data(), lower_bound, upper_bound);

// int status_2 = vslNewStream(&stream, VSL_BRNG_MT19937, 42); // MT19937 generator with seed 42
// int status_3 = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, vector_size, vector.data(), lower_bound, upper_bound);


static void Eigen_mm(benchmark::State& state) {
  Eigen::MatrixXd m(N,N);
  Eigen::VectorXd v(N);

  for (auto _ : state)
    m = Eigen::MatrixXd::Random(N, N);
    v = Eigen::VectorXd::Random(N);
    Eigen::VectorXd result = m * v;
}
// Register the function as a benchmark
BENCHMARK(Eigen_mm);


static void MKL_mm(benchmark::State& state) {
  const int matrix_size = N * N;
  const int vector_size = N;

  std::vector<double> matrix(matrix_size);
  std::vector<double> vector(vector_size);

  // Define the range for random numbers
  double lower_bound = 0.0;  // Minimum value
  double upper_bound = 1.0;  // Maximum value

  // Create a VSL random stream
  VSLStreamStatePtr stream;
  int status = vslNewStream(&stream, VSL_BRNG_MT19937, 42); // MT19937 generator with seed 42
  int status_1 = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, matrix_size, matrix.data(), lower_bound, upper_bound);

  int status_2 = vslNewStream(&stream, VSL_BRNG_MT19937, 42); // MT19937 generator with seed 42
  int status_3 = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, vector_size, vector.data(), lower_bound, upper_bound);
  
  // Allocate memory for the vector
  double* vector_y = new double[vector_size];

  for (auto _ : state)
    // Initialize the vector to zeros
    for (int i = 0; i < vector_size; ++i) {
        vector_y[i] = 0.0;
    }

    // Perform matrix-vector multiplication: y = alpha * A * x + beta * y
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, matrix.data(), N, vector.data(), 1, 0.0, vector_y, 1);

}
// Register the function as a benchmark
BENCHMARK(MKL_mm);


// Test calculating the square root of a matrix
static void LLT_Eigen(benchmark::State& state) {
  Eigen::MatrixXd m(N,N);
  Eigen::VectorXd v(N);

  m = Eigen::MatrixXd::Random(N, N);
  m = m * m.transpose();  // Make the matrix symmetric positive definite

  Eigen::LLT<Eigen::MatrixXd> lltP;
  
  for (auto _ : state)
    lltP.compute(m);
    Eigen::MatrixXd sig{lltP.matrixL()};
}
// Register the function as a benchmark
BENCHMARK(LLT_Eigen);


static void LLT_MKL(benchmark::State& state) {

  const int matrix_size = N * N;
  const int vector_size = N;

  std::vector<double> matrix(matrix_size);

  // Define the range for random numbers
  double lower_bound = 0.0;  // Minimum value
  double upper_bound = 1.0;  // Maximum value

  // Create a VSL random stream
  VSLStreamStatePtr stream;
  int status = vslNewStream(&stream, VSL_BRNG_MT19937, 42); // MT19937 generator with seed 42
  int status_1 = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, matrix_size, matrix.data(), lower_bound, upper_bound);

  // Convert Eigen matrix to MKL-compatible format
  std::vector<double> matrix_A(N * N, 0.0); 

  // AA^T
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, matrix.data(), N, matrix.data(), N, 0.0, matrix_A.data(), N);
  
  for (auto _ : state)
    LLTDecomposition(matrix_A, N);

}

BENCHMARK(LLT_MKL);

// Eigen value solver Eigen
static void SQRTM_EigenSolver_Eigen(benchmark::State& state) {
  Eigen::MatrixXd m(N,N);
  m = Eigen::MatrixXd::Random(N, N);

  Eigen::MatrixXd _sqrtM(N,N);

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(m);

  for (auto _ : state)
    es.compute(m);
    _sqrtM = es.operatorSqrt();
}
BENCHMARK(SQRTM_EigenSolver_Eigen);


// static void EigenSolver_Eigen_MKL(benchmark::State& state) {
//   for (auto _ : state)

//     m = Eigen::MatrixXd::Random(N, N);
//     Eigen::MatrixXd mmT(N,N);
//     mmT.setZero();
//     mmT = m*m.transpose();

//     Eigen::MatrixXd _sqrtM(N,N);
//     Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(mmT);
    
//     _sqrtM = es.operatorSqrt();
//     std::cout << "Matrix SQRT EIgen" << std::endl << _sqrtM << std::endl;

//     // Convert Eigen matrix to MKL-compatible format
//     std::vector<double> matrix_A(N * N, 0.0); 
//     EigenToMKL(m, matrix_A, N);

//     std::vector<double> matrix_AAT(N * N, 0.0);

//     // Perform matrix multiplication using cblas_dgemm
//     AMultiplyBT(matrix_A, matrix_A, matrix_AAT, N);

//     std::vector<double> result(N * N, 0.0);
//     SqrtEigenSolverMKL(matrix_AAT, result, N);


//     std::cout << "Matrix Sqrt MKL:" << std::endl;
//     printMatrix_MKL(result, N);


// }
// BENCHMARK(EigenSolver_Eigen_MKL);
static void SQRTM_EigenSolver_MKL(benchmark::State& state) {
  
  const int matrix_size = N * N;

  std::vector<double> matrix(matrix_size);
  std::vector<double> matrix_AAT(N * N);

  // Define the range for random numbers
  double lower_bound = 0.0;  // Minimum value
  double upper_bound = 1.0;  // Maximum value

  // Create a VSL random stream
  VSLStreamStatePtr stream;

  int status = vslNewStream(&stream, VSL_BRNG_MT19937, 42); // MT19937 generator with seed 42
  int status_1 = vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, matrix_size, matrix.data(), lower_bound, upper_bound);

  
  for (auto _ : state)    
    AMultiplyBT(matrix, matrix, matrix_AAT, N);    
    std::vector<double> result(N * N, 0.0);
    SqrtEigenSolverMKL(matrix_AAT, result, N);

}

BENCHMARK(SQRTM_EigenSolver_MKL);
BENCHMARK_MAIN();