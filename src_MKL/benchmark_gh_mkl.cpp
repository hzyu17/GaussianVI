/**
 * @file test_GH.cpp
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief use known integrations to test the Gausse-Hermite approximated integrations.
 * @version 0.1
 * @date 2025-02-02
 * 
 * @copyright Copyright (c) 2025
 * 
 */

// #include "quadrature/GaussHermite.h"
#include "quadrature/SparseGaussHermite.h"
#include "quadrature/SparseGaussHermite_MKL.h"
#include <functional>
#include<benchmark/benchmark.h>

using namespace Eigen;
using namespace gvi;

/// integrands used for testing
MatrixXd gx_1d(const VectorXd& x){
    int dim = x.rows();
    MatrixXd precision = MatrixXd::Identity(dim, dim)*10000;
    return MatrixXd::Constant(1, 1, x.transpose() * precision * x);
}

std::vector<double> gx_1d_mkl(const std::vector<double>& x){
    int dim = x.size();
    std::vector<double> precision(dim*dim, 0.0);
    for (int j=0; j<dim; j++){
        precision[j*dim+j] = 10000;
    }
    std::vector<double> result(dim, 0.0);
    ATMultiplyB(x, precision, result, dim);

    return result;
}

MatrixXd gx_2d(const VectorXd& x){
    MatrixXd res(2, 1);
    res.setZero();
    res << 3*x(0)*x(0), 2*x(0)*x(1);
    return MatrixXd{res};
}

MatrixXd gx_3d(const VectorXd& x){
    return MatrixXd{x*x.transpose().eval()};
}


using Function = std::function<MatrixXd(const VectorXd&)>;
using Function_MKL = std::function<std::vector<double>(const std::vector<double>&)>;

std::optional<std::shared_ptr<QuadratureWeightsMap>> weight_sigpts_map_option=std::nullopt;
std::optional<std::shared_ptr<QuadratureWeightsMap_MKL>> weight_sigpts_map_option_mkl=std::nullopt;

/**
 * @brief test the case where the cost function is 1 dimensional.
 */
static void sparseGH(benchmark::State& state){
    int dim = 4;
    VectorXd m = VectorXd::Zero(dim);
    MatrixXd P = MatrixXd::Identity(dim, dim)*0.0001;
    // GaussHermite<Function> gausshermite(3, dim, m, P);
    SparseGaussHermite<Function> gausshermite_sp(3, dim, m, P, weight_sigpts_map_option);

    for (auto _ : state)
        MatrixXd integral1_sp{gausshermite_sp.Integrate(gx_1d)};    
        // MatrixXd integral_expected{MatrixXd::Constant(1, 1, 4.0)};
        // std::cout << "SparseGH Integration value: " << std::endl << integral1_sp[0] << std::endl;

}

BENCHMARK(sparseGH);

static void sparseGH_MKL(benchmark::State& state){
    int dim = 4;
    std::vector<double> m_mkl(dim, 0.0);
    std::vector<double> P_mkl(dim*dim, 0.0);
    for (int j=0; j<dim; j++){
        P_mkl[j*dim+j] = 0.0001;
    }

    SparseGaussHermite_MKL<Function_MKL> gausshermite_mkl(3, dim, m_mkl, P_mkl, weight_sigpts_map_option_mkl);

    std::cout << "GH MKL class created." << std::endl;

    std::vector<double> integral1_sp_mkl(1, 0.0);
    
    for (auto _ : state)        
        integral1_sp_mkl = gausshermite_mkl.Integrate(gx_1d_mkl, 1, 1);
        std::cout << "integration MKL: " << std::endl;
        printMatrix_MKL(integral1_sp_mkl, 1, 1);
}

BENCHMARK(sparseGH_MKL);

BENCHMARK_MAIN();

// TEST(GaussHermite, two_dim){
//     int dim = 2;
//     VectorXd m = VectorXd::Ones(dim);
//     MatrixXd prec(dim, dim);
//     prec << 1.0,-0.74,-0.74,1.0;
//     MatrixXd cov{prec.inverse()};

//     GaussHermite<Function> gausshermite(10, dim, m, cov);

//     MatrixXd integral1{gausshermite.Integrate(gx_1d)};

//     MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.420866489831914e+04)};

//     ASSERT_LE((integral1 - integral_expected).norm(), 1e-5);

// }

// TEST(SparseGaussHermite, two_dim){
//     int dim = 2;
//     VectorXd m = VectorXd::Ones(dim);
//     MatrixXd prec(dim, dim);
//     prec << 1.0,-0.74,-0.74,1.0;
//     MatrixXd cov{prec.inverse()};

//     GaussHermite<Function> gausshermite_sp(10, dim, m, cov);

//     MatrixXd integral1_sp{gausshermite_sp.Integrate(gx_1d)};

//     MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.420866489831914e+04)};

//     ASSERT_LE((integral1_sp - integral_expected).norm(), 1e-5);

// }

// TEST(GaussHermite, two_dim_input_two_dim_output){
//     int dim = 2;
//     int deg = 8;
//     VectorXd m = VectorXd::Ones(dim);
//     MatrixXd prec(dim, dim);
//     prec << 1.0,-0.74,-0.74,1.0;
//     MatrixXd cov{prec.inverse()};

//     GaussHermite<Function> gausshermite(deg, dim, m, cov);

//     // gausshermite.update_integrand(gx_2d);
//     MatrixXd integral2{gausshermite.Integrate(gx_2d)};

//     MatrixXd integral2_expected(2, 1);
//     integral2_expected << 9.6313, 5.27144;

//     ASSERT_LE((integral2 - integral2_expected).norm(), 1e-5);
// }

// TEST(SparseGaussHermite, two_dim_input_two_dim_output){
//     int dim = 2;
//     int deg = 25;
//     VectorXd m = VectorXd::Ones(dim);
//     MatrixXd prec(dim, dim);
//     prec << 1.0,-0.74,-0.74,1.0;
//     MatrixXd cov{prec.inverse()};

//     SparseGaussHermite<Function> gausshermite_sp(deg, dim, m, cov);

//     // gausshermite.update_integrand(gx_2d);
//     MatrixXd integral2_sp{gausshermite_sp.Integrate(gx_2d)};

//     MatrixXd integral2_expected(2, 1);
//     integral2_expected << 9.6313, 5.27144;

//     ASSERT_LE((integral2_sp - integral2_expected).norm(), 1e-5);
// }

// TEST(GaussHermite, three_dim){
//     int dim = 3;
//     int deg = 8;
//     VectorXd m = VectorXd::Ones(dim);
//     MatrixXd P = MatrixXd::Identity(dim, dim);

//     GaussHermite<Function> gausshermite(deg, dim, m, P);

//     // Start time
//     auto start_time = std::chrono::high_resolution_clock::now();

//     MatrixXd integral1{gausshermite.Integrate(gx_1d)};

//     for (int i=0; i<10; i++){
//         integral1 = gausshermite.Integrate(gx_1d);
//     }
    
//     // End time
//     auto end_time = std::chrono::high_resolution_clock::now();
//     // Calculate duration
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//     // Print the duration
//     std::cout << "Time taken for 3dim integration full grid GH: " << duration.count() << " microseconds" << std::endl;

//     MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.00e+4)};

//     ASSERT_LE((integral1 - integral_expected).norm(), 1e-7);

// }

// TEST(SparseGaussHermite, three_dim){
//     int dim = 3;
//     int deg = 8;
//     VectorXd m = VectorXd::Ones(dim);
//     MatrixXd P = MatrixXd::Identity(dim, dim);

//     SparseGaussHermite<Function> gausshermite_sp(deg, dim, m, P);

//     // Start time
//     auto start_time = std::chrono::high_resolution_clock::now();
    
//     MatrixXd integral1_sp{gausshermite_sp.Integrate(gx_1d)};
//     for (int i=0; i<10; i++){
//         integral1_sp = gausshermite_sp.Integrate(gx_1d);
//     }
//     // End time
//     auto end_time = std::chrono::high_resolution_clock::now();
//     // Calculate duration
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//     // Print the duration
//     std::cout << "Time taken for 3dim integration sparse grid GH: " << duration.count() << " microseconds" << std::endl;

//     MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.00e+4)};

//     ASSERT_LE((integral1_sp - integral_expected).norm(), 1e-7);

// }

// TEST(SparseGaussHermite, external_weight_map){
//     int dim = 3;
//     int deg = 8;
//     VectorXd m = VectorXd::Ones(dim);
//     MatrixXd P = MatrixXd::Identity(dim, dim);

//     QuadratureWeightsMap nodes_weights_map;

//     // Read the weight and node map for sparse GH
//     try {
//         std::ifstream ifs(map_file, std::ios::binary);
//         if (!ifs.is_open()) {
//             throw std::runtime_error("Failed to open file for GH weights reading. File: " + map_file);
//         }

//         cereal::BinaryInputArchive archive(ifs); // Use cereal for deserialization
//         archive(nodes_weights_map); // Read and deserialize into nodes_weights_map

//     } catch (const std::exception& e) {
//         std::cerr << "Standard exception: " << e.what() << std::endl;
//     }

//     // Initialize the sparse GH using the read weight map
//     SparseGaussHermite<Function> gausshermite_sp(deg, dim, m, P, nodes_weights_map);

//     // Start time
//     auto start_time = std::chrono::high_resolution_clock::now();
    
//     MatrixXd integral1_sp{gausshermite_sp.Integrate(gx_1d)};
//     for (int i=0; i<10; i++){
//         integral1_sp = gausshermite_sp.Integrate(gx_1d);
//     }
//     // End time
//     auto end_time = std::chrono::high_resolution_clock::now();
//     // Calculate duration
//     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//     // Print the duration
//     std::cout << "Time taken for 3dim integration sparse grid GH: " << duration.count() << " microseconds" << std::endl;

//     MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.00e+4)};

//     ASSERT_LE((integral1_sp - integral_expected).norm(), 1e-7);

// }