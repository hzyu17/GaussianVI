#include "helpers/SerializeEigenMaps.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

#define STRING(x) #x
#define XSTRING(x) STRING(x)
std::string source_root{XSTRING(SOURCE_ROOT)};

using PointsWeights = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;
using DegreeDim = std::tuple<double, double>;

TEST(TestGH, data_io){
    // Random Matrix and Vector
    MatrixXd m_rnd{MatrixXd::Random(3, 3)};
    VectorXd v_rnd{VectorXd::Random(3)};

    PointsWeights pts_weights{m_rnd, v_rnd};
    DegreeDim deg_dim{1.0, 2.0};


    MatrixXd m_rnd1{MatrixXd::Random(3, 3)};
    VectorXd v_rnd1{VectorXd::Random(3)};

    PointsWeights pts_weights1{m_rnd1, v_rnd1};
    DegreeDim deg_dim1{3.0, 4.0};
    
    // Create the map
    std::unordered_map<DegreeDim, PointsWeights> testMap
    {
        {deg_dim, pts_weights},
        {deg_dim1, pts_weights1}
    };

    // Save the map to a binary file
    {
        std::ofstream ofs(source_root+"/tests/map_data.bin", std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << testMap;
    }

    // Load the map from the binary file
    std::unordered_map<DegreeDim, PointsWeights> loadedHashMap;

    {
        std::ifstream ifs(source_root+"/tests/map_data.bin", std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> loadedHashMap;
    }

    DegreeDim key = std::make_tuple(1.0, 2.0);

    ASSERT_EQ((std::get<0>(loadedHashMap[key]) - m_rnd).norm(), 0);
    ASSERT_EQ((std::get<1>(loadedHashMap[key]) - v_rnd).norm(), 0);
}


TEST(TestGH, gh_weight_data){
    
    // Load the map from the binary file
    std::unordered_map<DegreeDim, PointsWeights> loadedWeightMap;

    {
        std::ifstream ifs(source_root+"/quadrature/SparseGHQuadratureWeights.bin", std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> loadedWeightMap;
    }

    Eigen::MatrixXd pts_groundtruth_52(11, 5);
    pts_groundtruth_52 << -1.0,	0,	0,	0,	0, 
                            0,	-1.0,	0,	0,	0,
                            0,	0,	-1.0,	0,	0,
                            0,	0,	0,	-1.0,	0,
                            0,	0,	0,	0,	-1.0,
                            0,	0,	0,	0,	0,
                            0,	0,	0,	0,	1.0,
                            0,	0,	0,	1.0,	0,
                            0,	0,	1.0,	0,	0,
                            0,	1.0,	0,	0,	0,
                            1.0,	0,	0,	0,	0;

    Eigen::VectorXd weights_groundtruth_52(11);   
    weights_groundtruth_52 << 0.5, 0.5, 0.5, 0.5, 0.5, -4.0, 0.5, 0.5, 0.5, 0.5, 0.5;
    
    DegreeDim key = std::make_tuple(5.0, 2.0);

    Eigen::MatrixXd pts = std::get<0>(loadedWeightMap[key]);
    Eigen::VectorXd weights = std::get<1>(loadedWeightMap[key]);

    ASSERT_LE((std::get<0>(loadedWeightMap[key]) - pts_groundtruth_52).norm(), 1e-6);
    ASSERT_LE((std::get<1>(loadedWeightMap[key]) - weights_groundtruth_52).norm(), 1e-6);
}