/**
 * @file SerializeEigenMaps.h
 * @author: Hongzhe Yu
 * @date 02/03/2024
 * @brief: Serilization into binary files of a tuple which includes Eigen:MatrixXd type data.
 * https://stackoverflow.com/questions/18382457/eigen-and-boostserialize
*/

#pragma once

#include <iostream>
#include <fstream>
#include <tuple>
#include <Eigen/Dense>
#include <unordered_map>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace Eigen;

// Define Hash functions for the tuple<double, double> class
namespace std {
    template <>
    struct hash<std::tuple<double, double>> {
        size_t operator()(const std::tuple<double, double>& key) const {
            // Combine the hash values of the tuple elements using a hash function
            size_t hash1 = std::hash<double>{}(std::get<0>(key));
            size_t hash2 = std::hash<double>{}(std::get<1>(key));

            // A simple way to combine hash values
            return hash1 ^ (hash2 << 1);
        }
    };
}

#ifdef GTSAM_ENV // gtsam already defined the serialization for MatrixXd and VectorXd;
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>

#else
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

        // De-serialize
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

        // The serialization is split into save and load here.
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

        // De-serialize
        template<typename Archive>
        inline void load(
            Archive & ar, 
            Eigen::VectorXd & g, 
            const unsigned int version)
        {
            int rows;
            ar & rows;
            g.resize(rows);
            ar & boost::serialization::make_array(g.data(), rows);
        }

        // The serialization is split into save and load here.
        template<class Archive>
        inline void serialize(
            Archive & ar, 
            Eigen::VectorXd & g, 
            const unsigned int version)
        {
            split_free(ar, g, version);
        }

   }
}
#endif

namespace boost {
   namespace serialization {
        typedef std::unordered_map<std::tuple<double, double>, std::tuple<Eigen::MatrixXd, Eigen::VectorXd>> Map;
        
        template<class Archive>
        void save(Archive& ar, 
                  const Map& map, 
                  const unsigned int version) 
        {
            int size = map.size();
            ar & size;
            for (auto & pair: map) { 
                std::tuple<double, double> key = pair.first;
                std::tuple<Eigen::MatrixXd, Eigen::VectorXd> value = pair.second;

                ar & std::get<0>(key); // Serialize first double in the key tuple
                ar & std::get<1>(key); // Serialize second double in the key tuple
                ar & std::get<0>(value); // Serialize first MatrixXd in the value tuple
                ar & std::get<1>(value); // Serialize second VectorXd in the value tuple

            }
        }

        template<class Archive>
        void load(Archive& ar, Map& map, const unsigned int version) {
            int size = 0;
            ar & size;
            for (int i=0; i< size; i++) { 

                double d1, d2;
                MatrixXd mat;
                VectorXd vect;

                ar & d1; // Serialize first double in the key tuple
                ar & d2; // Serialize second double in the key tuple
                ar & mat; // Serialize first double in the key tuple
                ar & vect; // Serialize second double in the key tuple

                std::tuple<double, double> key{d1, d2};
                std::tuple<Eigen::MatrixXd, Eigen::VectorXd> value{mat, vect};

                map[key] = value;

            }
        }

        // The serialization is split into save and load here.
        template<class Archive>
        inline void serialize(
            Archive & ar, 
            Map& map, 
            const unsigned int version)
        {
            split_free(ar, map, version);
        }

    }
}


