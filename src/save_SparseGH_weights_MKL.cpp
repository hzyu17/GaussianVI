/**
 * @file save_cereal_SparseGH_weights.cpp
 * @author Zinuo Chang
 * @date 11/05/2024
 * @brief Save a map containing the sigma points and the weights for different degrees and dimensions.
*/

#include "quadrature/saveSparseGHWeightMap.h"

int main(){

    if (!libSpGHInitialize()) {
        std::cerr << "Could not initialize the library properly" << std::endl;
        
    } else {
        gvi::save_pointweightmaps_mkl();
    }

    libSpGHTerminate();
    mclTerminateApplication();

    // // Read from the saved bin file
    // // Load the map from the binary file
    // std::unordered_map<gvi::DimDegTuple, gvi::PointsWeightsTuple> loadedWeightMap;

    // {
    //     std::ifstream ifs(source_root+"/quadrature/SparseGHQuadratureWeights_cereal.bin", std::ios::binary);
    //     cereal::BinaryInputArchive archive(ifs);
    //     archive(loadedWeightMap);
    // }

    // gvi::DimDegTuple key = std::make_tuple(8.0, 10.0);

    // Eigen::MatrixXd pts = std::get<0>(loadedWeightMap[key]);
    // Eigen::VectorXd weights = std::get<1>(loadedWeightMap[key]);
    
    return 0;
}