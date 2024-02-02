/**
 * A function to generate sigmapoints for the GH quadratures.
 * Hongzhe Yu
 * 01/27/2024
*/

#include <iostream>
#include "quadrature/SparseGH/libSpGH/for_testing/libSpGH.h"
#include<Eigen/Dense>
#include "mclmcrrt.h"
#include "mclcppclass.h"

namespace gvi{


using PointsWeightsTuple = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;
PointsWeightsTuple sigmapts_weights(double D, double k) {
    
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

} // namespace gvi