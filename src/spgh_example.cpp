/**
 * spgh_example.cpp
 * Author: Hongzhe Yu
 * Brief: Example usage of a shared library build from matlab tools.
*/
#include <iostream>
#include "libSpGH.h"
#include<Eigen/Dense>
#include "mclmcrrt.h"
#include "mclcppclass.h"

using namespace Eigen;

void spGHSample() {
    
    mwArray dim(mxDouble(1));
    mwArray k(mxDouble(10));
    std::string gh_type_str = "GQN";
    mwArray gh_type(gh_type_str.c_str());

    mwArray sigmapts;
    mwArray weights;

    int sym = 1;
    mwArray mw_sym(sym);

    try {
        nwspgr(2, sigmapts, weights, gh_type, dim, k, mw_sym);

        auto dims = sigmapts.GetDimensions();

        int numRows = dims.Get(2, 1, 1);
        int numCols = dims.Get(2, 1, 2);

        Eigen::MatrixXd sigmapts_eigen(numRows, numCols);
        Eigen::MatrixXd weights_eigen(numRows, numCols);

        // Copy data from mwArray to Eigen::MatrixXd
        int n_pts = sigmapts.NumberOfElements();
        double sigmapts_copy[n_pts];
        double weights_copy[n_pts];
        sigmapts.GetData(sigmapts_copy, n_pts);
        weights.GetData(weights_copy, n_pts);

        int cnt = 0;
        for (mwIndex col = 0; col < numCols; ++col) {
            for (mwIndex row = 0; row < numRows; ++row) {
                sigmapts_eigen(row, col) = sigmapts_copy[cnt];
                weights_eigen(row, col) = weights_copy[cnt];
                cnt ++;
            }
        }

        std::cout << "sigmapts_eigen " << std::endl << sigmapts_eigen << std::endl;
        std::cout << "weights_eigen " << std::endl << weights_eigen << std::endl;


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

int run_main(int argc, const char** argv) {
    if (!libSpGHInitialize()) {
        std::cerr << "Could not initialize the library properly" << std::endl;
        return 2;
    } else {
        spGHSample();
        // Call the application and library termination routine
        libSpGHTerminate();
    }
    // Note that you should call mclTerminateApplication at the end of
    // your application to shut down all MATLAB Runtime instances.
    mclTerminateApplication();
    return 0;
}

// The main routine. On macOS, the main thread runs the system code, and
// user code must be processed by a secondary thread. On other platforms, 
// the main thread runs both the system code and the user code.
int main(int argc, const char** argv) {
    /* Call the mclInitializeApplication routine. Make sure that the application
     * was initialized properly by checking the return status. This initialization
     * has to be done before calling any MATLAB APIs or MATLAB Compiler SDK
     * generated shared library functions.
     */
    if (!mclInitializeApplication(nullptr, 0)) {
        std::cerr << "Could not initialize the application." << std::endl;
        return 1;
    }
    return mclRunMain(static_cast<mclMainFcnType>(run_main), argc, argv);
}