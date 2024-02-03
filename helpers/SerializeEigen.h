/**
 * SerializeEigen.h
 * @author: Hongzhe Yu
 * @brief: Serilization into binary files of a tuple which includes Eigen:MatrixXd type data.
 * https://stackoverflow.com/questions/18382457/eigen-and-boostserialize
*/

#include <iostream>
#include <fstream>
#include <tuple>
#include <Eigen/Dense>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace Eigen;
namespace boost {
   namespace serialization {
        // Serialization function for Eigen::MatrixXd
        
        template <typename Archive>
        inline void save(Archive & ar, 
                        const Eigen::MatrixXd & g, 
                        const unsigned int version)
        {
            int rows = g.rows();
            int cols = g.cols();

            ar & rows;
            ar & cols;
            ar & boost::serialization::make_array(g.data(), rows * cols);
        }

        template<typename Archive>
        inline void load(
            Archive & ar, 
            Eigen::MatrixXd & g, 
            const unsigned int version)
        {
            int rows, cols;
            ar & rows;
            ar & cols;
            g.resize(rows, cols);
            ar & boost::serialization::make_array(g.data(), rows * cols);
        }
        
        template<class Archive>
        inline void serialize(
            Archive & ar, 
            Eigen::MatrixXd & g, 
            const unsigned int version)
        {
            split_free(ar, g, version);
        }

        // Serialization function for Eigen::VectorXd
        template <typename Archive>
        inline void save(Archive & ar, 
                        const Eigen::VectorXd & g, 
                        const unsigned int version)
        {
            int rows = g.rows();

            ar & rows;
            ar & boost::serialization::make_array(g.data(), rows);
        }

        template<typename Archive>
        inline void load(
            Archive & ar, 
            Eigen::VectorXd & g, 
            const unsigned int version)
        {
            int rows, cols;
            ar & rows;
            g.resize(rows);
            ar & boost::serialization::make_array(g.data(), rows);
        }

        template<class Archive>
        inline void serialize(
            Archive & ar, 
            Eigen::VectorXd & g, 
            const unsigned int version)
        {
            split_free(ar, g, version);
        }

        // Serialization function for the main tuple
        template <typename Archive>
        void serialize(Archive& ar, std::tuple<std::tuple<double, double>, std::tuple<MatrixXd, VectorXd>>& tuple, const unsigned int version) {
            ar & std::get<0>(std::get<0>(tuple)); // Serialize inner tuple
            ar & std::get<1>(std::get<0>(tuple)); // Serialize inner tuple
            ar & std::get<0>(std::get<1>(tuple)); // Serialize inner tuple
            ar & std::get<1>(std::get<1>(tuple)); // Serialize inner tuple
        }
    }
}

// int main() {
//     // Random Matrix and Vector
//     MatrixXd m_rnd{MatrixXd::Random(3, 3)};
//     VectorXd v_rnd{VectorXd::Random(3)};
//     // Create the tuple
//     std::tuple<std::tuple<double, double>, std::tuple<MatrixXd, VectorXd>> myTuple{
//         {1.0, 2.0},
//         {m_rnd, v_rnd}
//     };

//     // Save the tuple to a binary file
//     {
//         std::ofstream ofs("tuple_data.bin", std::ios::binary);
//         boost::archive::binary_oarchive oa(ofs);
//         oa << myTuple;
//     }

//     // Load the tuple from the binary file
//     std::tuple<std::tuple<double, double>, std::tuple<MatrixXd, VectorXd>> loadedTuple;
//     {
//         std::ifstream ifs("tuple_data.bin", std::ios::binary);
//         boost::archive::binary_iarchive ia(ifs);
//         ia >> loadedTuple;
//     }

//     // Display the loaded tuple components
//     std::cout << "Loaded tuple:\n";
//     std::cout << "Double values: " << std::get<0>(std::get<0>(loadedTuple)) << ", " << std::get<1>(std::get<0>(loadedTuple)) << "\n";
//     std::cout << "Eigen::MatrixXd:\n" << std::get<0>(std::get<1>(loadedTuple)) << "\n";
//     std::cout << "Eigen::VectorXd:\n" << std::get<1>(std::get<1>(loadedTuple)) << "\n";

//     return 0;
// }