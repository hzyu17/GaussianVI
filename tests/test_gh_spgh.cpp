/**
 * @file test_GH.cpp
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief use known integrations to test the Gausse-Hermite approximated integrations.
 * @version 0.1
 * @date 2022-05-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "quadrature/GaussHermite.h"
#include "quadrature/SparseGaussHermite.h"
#include "helpers/EigenWrapper.h"
#include <chrono>
#include <functional>
#include <gtest/gtest.h>

using namespace Eigen;
using namespace gvi;

/// integrands used for testing
MatrixXd gx_1d(const VectorXd& x){
    int dim = x.rows();
    MatrixXd precision = MatrixXd::Identity(dim, dim)*10000;
    return MatrixXd::Constant(1, 1, x.transpose() * precision * x);
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


EigenWrapper ei;

/**
 * @brief Test the permute function helper with replacement.
 */
TEST(GaussHermite, permute){
    int dim = 1;
    VectorXd m = VectorXd::Zero(dim);
    MatrixXd P = MatrixXd::Identity(dim, dim)*0.0001;

    GaussHermite<std::function<MatrixXd(const VectorXd&)>> gausshermite(10, dim, m, P);

    int expected_0[dim] = {3,2,1,0};
    int expected_N[dim] = {0,1,2,3};
    
    int deg = 3;
    std::vector<int> vec = {0,1,2};
    std::vector<int> res = {0};
    std::vector<std::vector<int>> v_res;

    gausshermite.permute_replacing(vec, dim, res, 0, v_res);

    for (std::vector<int>& i_res: v_res){
        for (int& j:i_res){
            std::cout << "--j--" << std::endl << j << std::endl;
        }
    }
}


using Function = std::function<MatrixXd(const VectorXd&)>;

/**
 * @brief test the case where the cost function is 1 dimensional.
 */
TEST(GaussHermite, one_dim){
    int dim = 4;
    VectorXd m = VectorXd::Zero(dim);
    MatrixXd P = MatrixXd::Identity(dim, dim)*0.0001;
    GaussHermite<Function> gausshermite(3, dim, m, P);
    GaussHermite<Function> gausshermite_sp(3, dim, m, P);

    MatrixXd integral1{gausshermite.Integrate(gx_1d)};    
    MatrixXd integral1_sp{gausshermite_sp.Integrate(gx_1d)};    
    MatrixXd integral_expected{MatrixXd::Constant(1, 1, 4.0)};

    ASSERT_LE((integral1 - integral_expected).norm(), 1e-10);
    ASSERT_LE((integral1_sp - integral_expected).norm(), 1e-10);

}

TEST(GaussHermite, two_dim){
    int dim = 2;
    VectorXd m = VectorXd::Ones(dim);
    MatrixXd prec(dim, dim);
    prec << 1.0,-0.74,-0.74,1.0;
    MatrixXd cov{prec.inverse()};

    GaussHermite<Function> gausshermite(10, dim, m, cov);

    MatrixXd integral1{gausshermite.Integrate(gx_1d)};

    MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.420866489831914e+04)};

    ASSERT_LE((integral1 - integral_expected).norm(), 1e-5);

}

TEST(SparseGaussHermite, two_dim){
    int dim = 2;
    VectorXd m = VectorXd::Ones(dim);
    MatrixXd prec(dim, dim);
    prec << 1.0,-0.74,-0.74,1.0;
    MatrixXd cov{prec.inverse()};

    GaussHermite<Function> gausshermite_sp(10, dim, m, cov);

    MatrixXd integral1_sp{gausshermite_sp.Integrate(gx_1d)};

    MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.420866489831914e+04)};

    ASSERT_LE((integral1_sp - integral_expected).norm(), 1e-5);

}

TEST(GaussHermite, two_dim_input_two_dim_output){
    int dim = 2;
    int deg = 8;
    VectorXd m = VectorXd::Ones(dim);
    MatrixXd prec(dim, dim);
    prec << 1.0,-0.74,-0.74,1.0;
    MatrixXd cov{prec.inverse()};

    GaussHermite<Function> gausshermite(deg, dim, m, cov);

    // gausshermite.update_integrand(gx_2d);
    MatrixXd integral2{gausshermite.Integrate(gx_2d)};

    MatrixXd integral2_expected(2, 1);
    integral2_expected << 9.6313, 5.27144;

    ASSERT_LE((integral2 - integral2_expected).norm(), 1e-5);
}

TEST(SparseGaussHermite, two_dim_input_two_dim_output){
    int dim = 2;
    int deg = 25;
    VectorXd m = VectorXd::Ones(dim);
    MatrixXd prec(dim, dim);
    prec << 1.0,-0.74,-0.74,1.0;
    MatrixXd cov{prec.inverse()};

    SparseGaussHermite<Function> gausshermite_sp(deg, dim, m, cov);

    // gausshermite.update_integrand(gx_2d);
    MatrixXd integral2_sp{gausshermite_sp.Integrate(gx_2d)};

    MatrixXd integral2_expected(2, 1);
    integral2_expected << 9.6313, 5.27144;

    ASSERT_LE((integral2_sp - integral2_expected).norm(), 1e-5);
}

TEST(GaussHermite, three_dim){
    int dim = 3;
    int deg = 8;
    VectorXd m = VectorXd::Ones(dim);
    MatrixXd P = MatrixXd::Identity(dim, dim);

    GaussHermite<Function> gausshermite(deg, dim, m, P);

    // Start time
    auto start_time = std::chrono::high_resolution_clock::now();

    MatrixXd integral1{gausshermite.Integrate(gx_1d)};

    for (int i=0; i<10; i++){
        integral1 = gausshermite.Integrate(gx_1d);
    }
    
    // End time
    auto end_time = std::chrono::high_resolution_clock::now();
    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // Print the duration
    std::cout << "Time taken for 3dim integration full grid GH: " << duration.count() << " microseconds" << std::endl;

    MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.00e+4)};

    ASSERT_LE((integral1 - integral_expected).norm(), 1e-7);

}

TEST(SparseGaussHermite, three_dim){
    int dim = 3;
    int deg = 8;
    VectorXd m = VectorXd::Ones(dim);
    MatrixXd P = MatrixXd::Identity(dim, dim);

    SparseGaussHermite<Function> gausshermite_sp(deg, dim, m, P);

    // Start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MatrixXd integral1_sp{gausshermite_sp.Integrate(gx_1d)};
    for (int i=0; i<10; i++){
        integral1_sp = gausshermite_sp.Integrate(gx_1d);
    }
    // End time
    auto end_time = std::chrono::high_resolution_clock::now();
    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // Print the duration
    std::cout << "Time taken for 3dim integration sparse grid GH: " << duration.count() << " microseconds" << std::endl;

    MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.00e+4)};

    ASSERT_LE((integral1_sp - integral_expected).norm(), 1e-7);

}

TEST(SparseGaussHermite, external_weight_map){
    int dim = 3;
    int deg = 8;
    VectorXd m = VectorXd::Ones(dim);
    MatrixXd P = MatrixXd::Identity(dim, dim);

    QuadratureWeightsMap nodes_weights_map;

    // Read the weight and node map for sparse GH
    try {
        std::ifstream ifs(map_file, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open file for GH weights reading. File: " + map_file);
        }

        cereal::BinaryInputArchive archive(ifs); // Use cereal for deserialization
        archive(nodes_weights_map); // Read and deserialize into nodes_weights_map

    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }

    // Initialize the sparse GH using the read weight map
    SparseGaussHermite<Function> gausshermite_sp(deg, dim, m, P, nodes_weights_map);

    // Start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    MatrixXd integral1_sp{gausshermite_sp.Integrate(gx_1d)};
    for (int i=0; i<10; i++){
        integral1_sp = gausshermite_sp.Integrate(gx_1d);
    }
    // End time
    auto end_time = std::chrono::high_resolution_clock::now();
    // Calculate duration
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // Print the duration
    std::cout << "Time taken for 3dim integration sparse grid GH: " << duration.count() << " microseconds" << std::endl;

    MatrixXd integral_expected{MatrixXd::Constant(1, 1, 6.00e+4)};

    ASSERT_LE((integral1_sp - integral_expected).norm(), 1e-7);

}