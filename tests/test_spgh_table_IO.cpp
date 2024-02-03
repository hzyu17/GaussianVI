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
    std::tuple<std::tuple<double, double>, std::tuple<MatrixXd, VectorXd>> myTuple{
        deg_dim,
        pts_weights
    };

    // Save the tuple to a binary file
    {
        std::ofstream ofs("tuple_data.bin", std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << myTuple;
    }

    // Load the tuple from the binary file
    std::tuple<std::tuple<double, double>, std::tuple<MatrixXd, VectorXd>> loadedTuple;
    {
        std::ifstream ifs("tuple_data.bin", std::ios::binary);
        boost::archive::binary_iarchive ia(ifs);
        ia >> loadedTuple;
    }

    // Display the loaded tuple components
    std::cout << "Loaded tuple:\n";
    std::cout << "Double values: " << std::get<0>(std::get<0>(loadedTuple)) << ", " << std::get<1>(std::get<0>(loadedTuple)) << "\n";
    std::cout << "Eigen::MatrixXd:\n" << std::get<0>(std::get<1>(loadedTuple)) << "\n";
    std::cout << "Eigen::VectorXd:\n" << std::get<1>(std::get<1>(loadedTuple)) << "\n";

    ASSERT_EQ(std::get<0>(std::get<0>(loadedTuple))-1.0, 0);
    ASSERT_EQ(std::get<1>(std::get<0>(loadedTuple))-2.0, 0);
    ASSERT_EQ((std::get<0>(std::get<1>(loadedTuple)) - m_rnd).norm(), 0);
    ASSERT_EQ((std::get<1>(std::get<1>(loadedTuple)) - v_rnd).norm(), 0);
}
