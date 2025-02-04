/**
 * @file saveSparseGHWeightMap.h
 * @author Hongzhe Yu
 * @brief Serialize and save the Sparse grid GH quadrature table to a binary file.
*/

#pragma once

#include "quadrature/generateSpGHWeights.h"
#include "quadrature/SparseGHQuadratureWeights.h"
#include "src_MKL/gvimp_mkl.h"

namespace gvi{

void save_pointweightmaps(){
    QuadratureWeightsMap map;
    std::vector<DimDegTuple> dim_degs{{std::make_tuple(1.0, 25.0), std::make_tuple(2.0, 25.0), std::make_tuple(3.0, 19.0), 
                                        std::make_tuple(4.0, 13.0), std::make_tuple(5.0, 11.0), std::make_tuple(6.0, 9.0),
                                        std::make_tuple(7.0, 8.0), std::make_tuple(8.0, 7.0), std::make_tuple(9.0, 7.0), 
                                        std::make_tuple(10.0, 7.0), std::make_tuple(11.0, 6.0), std::make_tuple(12.0, 6.0), 
                                        std::make_tuple(13.0, 6.0)}};
    for (double i_dim=14.0; i_dim<21.0; i_dim++){
        dim_degs.push_back(std::make_tuple(i_dim, 5.0));
    }

    for (DimDegTuple& dim_maxk: dim_degs){
            // DimDegTuple dim_k= std::make_tuple(dim, k);
            double dim = std::get<0>(dim_maxk);
            double max_k = std::get<1>(dim_maxk);

        for (double k=1.0; k<=max_k; k++){

            std::cout << "Reading dim: " << dim << ", k: " << k <<std::endl;

            PointsWeightsTuple pt_wts;
            pt_wts = get_sigmapts_weights(dim, k);

            DimDegTuple dim_k = std::make_tuple(dim, k);
            map[dim_k] = pt_wts;
        }
    }

    // Save the tuple to a binary file with cereal
    {
        std::string file_name = source_root + "/quadrature/SparseGHQuadratureWeights_cereal.bin";
        std::cout << "Saving the sigma points and weights into the following file: " << std::endl
                << file_name << std::endl;
        std::ofstream ofs(file_name, std::ios::binary);
        cereal::BinaryOutputArchive archive(ofs);
        archive(map);
    }
}

void save_pointweightmaps_mkl(){
    
    QuadratureWeightsMap_MKL map;
    std::vector<DimDegTuple> dim_degs{{std::make_tuple(1.0, 25.0), std::make_tuple(2.0, 25.0), std::make_tuple(3.0, 19.0), 
                                        std::make_tuple(4.0, 13.0), std::make_tuple(5.0, 11.0), std::make_tuple(6.0, 9.0),
                                        std::make_tuple(7.0, 8.0), std::make_tuple(8.0, 7.0), std::make_tuple(9.0, 7.0), 
                                        std::make_tuple(10.0, 7.0), std::make_tuple(11.0, 6.0), std::make_tuple(12.0, 6.0), 
                                        std::make_tuple(13.0, 6.0)}};
    for (double i_dim=14.0; i_dim<21.0; i_dim++){
        dim_degs.push_back(std::make_tuple(i_dim, 5.0));
    }

    // DimDegTuple dim_k= std::make_tuple(dim, k);
    PointsWeightsTuple_MKL pt_wts;
    DimDegTuple dim_k;

    for (DimDegTuple& dim_maxk: dim_degs){
        double dim = std::get<0>(dim_maxk);
        double max_k = std::get<1>(dim_maxk);

        for (double k=1.0; k<=max_k; k++){

            std::cout << "Reading dim: " << dim << ", k: " << k <<std::endl;

            get_sigmapts_weights_mkl(dim, k, pt_wts);
            dim_k = std::make_tuple(dim, k);

            // std::vector<double> pts = std::get<0>(pt_wts);
            // std::vector<double> weights = std::get<1>(pt_wts);

            // int num_points = pts.size() / dim;
            // std::cout << "weights " << std::endl;
            // printMatrix_MKL(weights, num_points, dim);

            map[dim_k] = pt_wts;
        }
    }

    // Save the tuple to a binary file with cereal
    {
        std::string file_name = source_root + "/quadrature/SparseGHQuadratureWeights_MKL_cereal.bin";
        std::cout << "Saving the sigma points and weights into the following file: " << std::endl
                << file_name << std::endl;
        std::ofstream ofs(file_name, std::ios::binary);
        cereal::BinaryOutputArchive archive(ofs);
        archive(map);
    }
}
    
} // namespace gvi;