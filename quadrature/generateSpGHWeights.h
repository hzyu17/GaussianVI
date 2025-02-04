/**
 * A function to generate sigmapoints for the GH quadratures.
 * Hongzhe Yu
 * 01/27/2024
*/

#pragma once

#include <iostream>
#include "quadrature/libSpGH/for_testing/libSpGH.h"
#include <Eigen/Dense>
#include "mclmcrrt.h"
#include "mclcppclass.h"
#include "quadrature/SparseGHQuadratureWeights.h"
#include "src_MKL/gvimp_mkl.h"

#define STRING(x) #x
#define XSTRING(x) STRING(x)
std::string source_root{XSTRING(SOURCE_ROOT)};

namespace gvi{

PointsWeightsTuple get_sigmapts_weights(double D, double k) {
    
    // Inputs
    mwArray mw_dim(D);
    mwArray mw_k(k);

    std::string gh_type_str = "GQN";
    mwArray gh_type(gh_type_str.c_str());

    mwArray mw_sym(1);

    // Outputs
    mwArray sigmapts;
    mwArray weights;

    try {
        nwspgr(2, sigmapts, weights, gh_type, mw_dim, mw_k, mw_sym);

        auto dims = sigmapts.GetDimensions();

        int numRows = dims.Get(2, 1, 1);
        int numCols = dims.Get(2, 1, 2);

        // std::cout << "dim: " << D << std::endl<<
        // "k: " << k << std::endl;
        
        // std::cout << "numRows: " << numRows << std::endl <<
        // "numCols: " << numCols << std::endl;

        Eigen::MatrixXd sigmapts_eigen(numRows, numCols);
        Eigen::VectorXd weights_eigen(numRows);

        for (mwIndex col = 0; col < numCols; ++col) {
            for (mwIndex row = 0; row < numRows; ++row) {
                sigmapts_eigen(row, col) = sigmapts.Get(2, row+1, col+1);
            }
        }

        for (int row=0; row<numRows; row++) {
            weights_eigen(row) = weights.Get(2, row+1, 1);
        }

        // std::cout << "points_eigen " << std::endl << sigmapts_eigen << std::endl;
        // std::cout << "weights_eigen " << std::endl << weights_eigen << std::endl;

        return std::make_tuple(sigmapts_eigen, weights_eigen);

    } catch (const mwException& e) {
        char * message = "An error occurred while initializing libSpGH";
        auto details = mclGetLastErrorMessage();
        if (details && *details)
        {
            fprintf(stderr, "%s: %s", message, details);
        }
        else
        {
            fprintf(stderr, message);
        }
        std::cerr << "Error calling nwspgr: " << e.what() << std::endl;
    }

}


void get_sigmapts_weights_mkl(const double& D, const double& k, PointsWeightsTuple_MKL& result) {
    
    // Inputs
    mwArray mw_dim(D);
    mwArray mw_k(k);

    std::string gh_type_str = "GQN";
    mwArray gh_type(gh_type_str.c_str());

    mwArray mw_sym(1);

    // Outputs
    mwArray sigmapts;
    mwArray weights;

    try {
        nwspgr(2, sigmapts, weights, gh_type, mw_dim, mw_k, mw_sym);

        auto dims = sigmapts.GetDimensions();

        int numRows = dims.Get(2, 1, 1);
        int numCols = dims.Get(2, 1, 2);

        // std::cout << "dim: " << D << std::endl<<
        // "k: " << k << std::endl;

        // std::cout << "numRows: " << numRows << std::endl <<
        // "numCols: " << numCols << std::endl;
        
        std::vector<double> sigmapts_mkl(numRows*numCols, 0.0);
        std::vector<double> weights_mkl;
        weights_mkl.resize(numRows);
        // std::vector<double> weights_mkl(numRows, 0.0);
        
        nwspgr(2, sigmapts, weights, gh_type, mw_dim, mw_k, mw_sym);

        for (int row=0; row < numRows; row++){
            for (int col=0; col<numCols; col++){
                sigmapts_mkl[row*numCols + col] = sigmapts.Get(2, row+1, col+1);
            }
        }

        for (int ii=0; ii < numRows; ii++){
            // std::cout << "weights.Get(2, row+1, 1) " << weights.Get(2, row+1, 1) << std::endl;
            double weight_i = static_cast<double>(weights.Get(2, ii+1, 1));
            weights_mkl[ii] = weight_i; 
        }

        // for (mwIndex col = 0; col < numCols; ++col) {
        //     for (mwIndex row = 0; row < numRows; ++row) {
        //         sigmapts_eigen(row, col) = sigmapts.Get(2, row+1, col+1);
        //     }
        // }

        // for (int row=0; row<numRows; row++) {
        //     weights_eigen(row) = weights.Get(2, row+1, 1);
        // }

        // std::cout << "points: " << std::endl;
        // printMatrix_MKL(sigmapts_mkl, numRows, numCols);
        // std::cout << std::endl;

        // std::cout << "weights_mkl: " << std::endl;
        // printVector_MKL(weights_mkl, numRows);
        // std::cout << std::endl;

        result = std::make_tuple(sigmapts_mkl, weights_mkl);

    } catch (const mwException& e) {
        char * message = "An error occurred while initializing libSpGH";
        auto details = mclGetLastErrorMessage();
        if (details && *details)
        {
            fprintf(stderr, "%s: %s", message, details);
        }
        else
        {
            fprintf(stderr, message);
        }
        std::cerr << "Error calling nwspgr: " << e.what() << std::endl;
    }

}

} // namespace gvi