/**
 * @file save_SparseGH_weights.cpp
 * @author Hongzhe Yu
 * @date 02/03/2024
 * @brief Save a map containing the sigma points and the weights for different degrees and dimensions.
*/

#include "quadrature/saveSparseGHWeightMap.h"

int main(){

    if (!libSpGHInitialize()) {
        std::cerr << "Could not initialize the library properly" << std::endl;
        
    } else {
        // save_pointweightmaps(double max_dim, double max_deg)
        double max_dim = 8.0, max_deg = 10.0;
        gvi::save_pointweightmaps(max_dim, max_deg);
    }

    libSpGHTerminate();
    mclTerminateApplication();

    // Read from the saved bin file
    // Load the map from the binary file
    std::unordered_map<gvi::DimDegTuple, gvi::NodesWeightsTuple> loadedWeightMap;

    {
        std::ifstream ifs(source_root+"/quadrature/SparseGHQuadratureWeights.bin", std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> loadedWeightMap;
    }

    gvi::DimDegTuple key = std::make_tuple(8.0, 10.0);

    Eigen::MatrixXd pts = std::get<0>(loadedWeightMap[key]);
    Eigen::VectorXd weights = std::get<1>(loadedWeightMap[key]);
    
    return 0;
}