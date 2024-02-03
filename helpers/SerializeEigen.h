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
#include <unordered_map>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

using namespace Eigen;

using DoubleTuple = std::tuple<double, double>;
using MatrixVectorTuple = std::tuple<Eigen::MatrixXd, Eigen::VectorXd>;

// Define Hash functions for the tuple<double, double> class
namespace std {
    template <>
    struct hash<DoubleTuple> {
        size_t operator()(const DoubleTuple& key) const {
            // Combine the hash values of the tuple elements using a hash function
            size_t hash1 = std::hash<double>{}(std::get<0>(key));
            size_t hash2 = std::hash<double>{}(std::get<1>(key));

            // A simple way to combine hash values
            return hash1 ^ (hash2 << 1);
        }
    };
}

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


        typedef std::unordered_map<DoubleTuple, MatrixVectorTuple> Map;
        
        template<class Archive>
        void save(Archive& ar, 
                  const Map& map, 
                  const unsigned int version) 
        {
            int size = map.size();
            ar & size;
            for (auto & pair: map) { 
                DoubleTuple key = pair.first;
                MatrixVectorTuple value = pair.second;

                std::cout << "Value 1: " << std::get<0>(value) << std::endl;
                std::cout << "Value 2: " << std::get<1>(value) << std::endl;

                ar & std::get<0>(key); // Serialize first double in the key tuple
                ar & std::get<1>(key); // Serialize second double in the key tuple
                ar & std::get<0>(value); // Serialize first MatrixXd in the value tuple
                ar & std::get<1>(value); // Serialize second VectorXd in the value tuple

                // out << p.first << p.second; 
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

                DoubleTuple key{d1, d2};
                MatrixVectorTuple value{mat, vect};

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
