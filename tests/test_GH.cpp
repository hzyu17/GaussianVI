/**
 * @file test_GH.cpp
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Test the Gauss-Hermite estimator using a known 1d experiment.
 * @version 0.1
 * @date 2023-07-18
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "quadrature/GaussHermite.h"
#include <gtest/gtest.h>

using namespace Eigen;
using namespace gvi;

// define function input-output
using Function = std::function<MatrixXd(const VectorXd&)>;    

MatrixXd phi(const VectorXd& vec_x){
    double x = vec_x(0);
    double mu_p = 20, f = 400, b = 0.1, sig_r_sq = 0.09;
    double sig_p_sq = 9;

    // y should be sampled. for single trial just give it a value.
    double y = f*b/mu_p + 0.05;

    MatrixXd res(1, 1);
    res(0, 0) = ((x - mu_p)*(x - mu_p) / sig_p_sq / 2 + (y - f*b/x)*(y - f*b/x) / sig_r_sq / 2); 

    return res;

}

// phi_22 = [3*x(1)*x(1); 2*x(1)*x(2)]
MatrixXd ph22(const VectorXd& x){

    MatrixXd res(2, 1);
    res(0, 0) = 3.0*x(0)*x(0); 
    res(1, 0) = 2.0*x(0)*x(1); 

    return res;

}


MatrixXd xmu_phi(const VectorXd& vec_x){
    double x = vec_x(0);
    double mu_p = 20, f = 400, b = 0.1, sig_r_sq = 0.09;
    double sig_p_sq = 9;

    // y should be sampled. for single trial just give it a value.
    double y = f*b/mu_p + 0.05;

    MatrixXd res(1, 1);
    res(0, 0) = (x - mu_p) * ((x - mu_p)*(x - mu_p) / sig_p_sq / 2 + (y - f*b/x)*(y - f*b/x) / sig_r_sq / 2); 

    return res;
}

/**
 * @brief Test the weights got from Gauss Hermite integrator
 */
TEST(TestGH, GH_weights){

    VectorXd mean{VectorXd::Constant(1, 20)};
    MatrixXd cov{MatrixXd::Constant(1,1,9)};

    int deg = 10;
    int dim = 1;
    typedef gvi::GaussHermite<Function> GH;
    
    GH gh_inst{deg, dim, mean, cov};
    
    VectorXd weights{gh_inst.weights()};
    VectorXd sigmapoints{gh_inst.sigmapts()};

    VectorXd weights_expected(10);
    weights_expected << 4.310652630718227e-06, 4.310652630718376e-06, 7.580709343122321e-04, 
                        7.580709343121815e-04, 0.344642334932012, 0.344642334932016, 0.135483702980275, 
                        0.135483702980267, 0.019111580500769, 0.019111580500770;

    VectorXd sigmapoints_expected(10);
    sigmapoints_expected << 4.859462828332310, -4.859462828332314, 3.581823483551924, 
                            -3.581823483551934, 0.484935707515505, -0.484935707515517,
                            1.465989094391161, -1.465989094391140, 2.484325841638960, 
                            -2.484325841638965;
    
    ASSERT_LE((sigmapoints - sigmapoints_expected).norm(), 1e-10);
    ASSERT_LE((weights - weights_expected).norm(), 1e-10);

}

TEST(TestGH, gh){
    int deg = 6, dim = 1;
    VectorXd mean(1);
    mean.setZero();
    mean(0) = 20.0;
    MatrixXd cov(1, 1);
    cov.setZero();
    cov(0, 0) = 9.0;

    GaussHermite<Function> gh(deg, dim, mean, cov);

    MatrixXd phi_mu = phi(mean);
    MatrixXd phi_mu_GT(1,1);
    phi_mu_GT(0,0) = 0.013888888888889;
    ASSERT_LE((phi_mu - phi_mu_GT).norm(), 1e-5);
    
    MatrixXd xmu_phi_mu = xmu_phi(mean);
    MatrixXd xmu_phi_mu_GT(1,1);
    xmu_phi_mu_GT(0,0) = 0.0;
    ASSERT_LE((xmu_phi_mu - xmu_phi_mu_GT).norm(), 1e-5);

    // integration of phi
    std::shared_ptr<Function> p_phi = std::make_shared<Function>(phi);
    MatrixXd E_Phi = gh.Integrate(phi);
    double E_Phi_GT = 1.1129;
    ASSERT_LE(abs(E_Phi(0,0) - E_Phi_GT), 1e-4);

    // integration of (x-mu)*phi
    std::shared_ptr<Function> p_xmu_phi = std::make_shared<Function>(xmu_phi);
    MatrixXd E_xmu_phi = gh.Integrate(xmu_phi);
    double E_xmu_phi_GT = -1.2144;
    ASSERT_LE(abs(E_xmu_phi(0,0) - E_xmu_phi_GT), 1e-4);

}


// Test Sparse GH class
#include "quadrature/SparseGaussHermite.h"

TEST(TestGH, sp_gh){

    int deg = 6, dim = 1;
    VectorXd mean(1);
    mean.setZero();
    mean(0) = 20.0;
    MatrixXd cov(1, 1);
    cov.setZero();
    cov(0, 0) = 9.0;

    // Sparse Gauss-Hermite Integrator
    SparseGaussHermite<Function> sp_gh(deg, dim, mean, cov);

    // Integrate
    MatrixXd E_Phi_sp = sp_gh.Integrate(phi);

    // Ground truth
    double E_Phi_GT = 1.1129;

    ASSERT_LE(abs(E_Phi_sp(0,0) - E_Phi_GT), 1e-4);

    // integration of (x-mu)*phi
    std::shared_ptr<Function> p_xmu_phi = std::make_shared<Function>(xmu_phi);
    MatrixXd E_xmu_phi = sp_gh.Integrate(xmu_phi);
    double E_xmu_phi_GT = -1.2144;
    ASSERT_LE(abs(E_xmu_phi(0,0) - E_xmu_phi_GT), 1e-4);

}


TEST(TestGH, sp_gh_multidim){
    // Test for multiple dimension functions
    int deg = 10; 
    int dim = 2;
    VectorXd mean_2d(2);
    mean_2d.setZero();
    mean_2d << 1.0, 1.0;
    MatrixXd cov_2d(2, 2);
    cov_2d.setZero();
    cov_2d << 2.210433244916004, 1.635720601237843, 1.635720601237843, 2.210433244916004;

    SparseGaussHermite<Function> sp_gh22(deg, dim, mean_2d, cov_2d);

    MatrixXd E_phi22 = sp_gh22.Integrate(ph22);

    std::cout << "E_phi22" << std::endl << E_phi22 << std::endl;

    MatrixXd E_phi22_gt(2, 1);
    E_phi22_gt << 9.631450087970276, 5.271519032251217;
    
    ASSERT_LE((E_phi22 - E_phi22_gt).norm(), 1e-3);
}