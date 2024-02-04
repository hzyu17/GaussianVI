/**
 * A function to generate sigmapoints for the GH quadratures.
 * Hongzhe Yu
 * 01/27/2024
*/

#include "quadrature/generateSpGHWeights.h"

int run_main(int argc, const char** argv) {
    if (!libSpGHInitialize()) {
        std::cerr << "Could not initialize the library properly" << std::endl;
        return 2;
    } else {
        std::tuple<Eigen::MatrixXd, Eigen::VectorXd> pt_w;
        pt_w = gvi::get_sigmapts_weights(2, 3);

        std::cout << "sigmapts_eigen " << std::endl << std::get<0>(pt_w) << std::endl;
        std::cout << "weights_eigen " << std::endl << std::get<1>(pt_w) << std::endl;

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