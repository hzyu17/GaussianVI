#include "helpers/SerializeEigen.h"
#include <gtest/gtest.h>
#include <Eigen/Dense>

using PointsWeights = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;
using DigreeDim = std::tuple<double, double>;
using TableGHPoints = std::tuple<DigreeDim, PointsWeights>;

TEST(TestGH, data_io){
    // Random Matrix and Vector
    MatrixXd m_rnd{MatrixXd::Random(3, 3)};
    VectorXd v_rnd{VectorXd::Random(3)};

    PointsWeights pts_weights{m_rnd, v_rnd};
    DigreeDim deg_dim{1.0, 2.0};
    
    // Create the tuple
    std::unordered_map<DigreeDim, PointsWeights> testMap
    {
        {deg_dim, pts_weights}
    };

    // Save the tuple to a binary file
    {
        std::ofstream ofs("map_data.bin", std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << testMap;
    }

    // Load the tuple from the binary file
    std::unordered_map<DigreeDim, PointsWeights> loadedHashMap;

    {
        std::ifstream ifs("map_data.bin", std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> loadedHashMap;
    }

    for (const auto& pair : loadedHashMap) {
        std::cout << "Key 1: " << std::get<0>(pair.first) << std::endl;
        std::cout << "Key 2: " << std::get<1>(pair.first) << std::endl;

        std::cout << "Value 1: " << std::get<0>(pair.second) << std::endl;
        std::cout << "Value 2: " << std::get<1>(pair.second) << std::endl;
    }

    DigreeDim key = std::make_tuple(1.0, 2.0);

    ASSERT_EQ((std::get<0>(loadedHashMap[key]) - m_rnd).norm(), 0);
    ASSERT_EQ((std::get<1>(loadedHashMap[key]) - v_rnd).norm(), 0);
}
