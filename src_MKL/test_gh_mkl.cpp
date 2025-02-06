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

    
    std::vector<double> temp(dim, 0.0);
    AMultiplyB(precision, x, temp, dim, dim, 1);

    std::vector<double> result(1, 0.0);
    ATMultiplyB(x, temp, result, dim, 1, 1);

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


/**
 * @brief test the case where the cost function is 1 dimensional.
 */
void sparseGH(){
    std::optional<std::shared_ptr<QuadratureWeightsMap>> weight_sigpts_map_option=std::nullopt;
    int dim = 4;
    VectorXd m = VectorXd::Zero(dim);
    MatrixXd P = MatrixXd::Identity(dim, dim)*0.0001;
    // GaussHermite<Function> gausshermite(3, dim, m, P);
    SparseGaussHermite<Function> gausshermite_sp(2, dim, m, P, weight_sigpts_map_option);

    MatrixXd integral1_sp{gausshermite_sp.Integrate(gx_1d)};   
    std::cout << "SparseGH Integration value: " << std::endl << integral1_sp(0,0) << std::endl; 

}

void sparseGH_MKL(){
    std::optional<std::shared_ptr<QuadratureWeightsMap_MKL>> weight_sigpts_map_option_mkl=std::nullopt;
    int dim = 4;
    std::vector<double> m_mkl(dim, 0.0);
    std::vector<double> P_mkl(dim*dim, 0.0);
    for (int j=0; j<dim; j++){
        P_mkl[j*dim+j] = 0.0001;
    }

    SparseGaussHermite_MKL<Function_MKL> gausshermite_mkl(2, dim, m_mkl, P_mkl, weight_sigpts_map_option_mkl);
    
    std::vector<double> integral1_sp_mkl(1, 0.0);
    integral1_sp_mkl = gausshermite_mkl.Integrate(gx_1d_mkl, 1, 1);
    std::cout << "SparseGH MKL Integration value: " << std::endl << integral1_sp_mkl[0] << std::endl; 
    // printMatrix_MKL(integral1_sp_mkl, 1, 1);
}


int main(){
    // sparseGH();
    sparseGH_MKL();

    return 0;
}