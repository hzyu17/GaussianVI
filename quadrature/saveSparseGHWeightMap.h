/**
 * @file saveSparseGHWeightMap.h
 * @author Hongzhe Yu
 * @brief Serialize and save the Sparse grid GH quadrature table to a binary file.
*/

#pragma once

#include "quadrature/generateSpGHWeights.h"
#include "quadrature/SparseGHQuadratureWeights.h"

namespace gvi{

void save_pointweightmaps(double max_dim, double max_deg){
    QuadratureWeightsMap map;
    for (double dim = 1.0; dim <= max_dim; dim += 1.0){
        for (double k = 1.0; k <= max_deg; k += 1.0){
            DimDegTuple dim_k= std::make_tuple(dim, k);

            PointsWeightsTuple pt_wts;
            pt_wts = get_sigmapts_weights(dim, k);

            map[dim_k] = pt_wts;
        }
    }

    DimDegTuple dim_k= std::make_tuple(1.0, 2.0);
    PointsWeightsTuple pt_wt = map[dim_k];

    // Save the tuple to a binary file
    {
        std::ofstream ofs(source_root+"/quadrature/SparseGHQuadratureWeights.bin", std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << map;
    }
}
    
} // namespace gvi;