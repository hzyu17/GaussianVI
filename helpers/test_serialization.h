#include "helpers/CommonDefinitions.h"
#include "helpers/EigenWrapper.h"
#include "helpers/SerializeEigenMaps.h"
#include <vector>

namespace gvi{

class MyData{

public:
    MyData(){}

    MyData(int dim, int nt): _nt{nt}, _dim{dim}, _mat{nt, dim, dim}{}

    MyData(int dim, int nt, const Matrix3D& data): _nt{nt}, _dim{dim}, _mat{data}{}

    void set_dim(int dim){ _dim = dim; }
    void set_nt(int nt){ _nt = nt; }
    void set_mat(const Matrix3D& mat){ _mat = mat; }

    void save_data(const std::string& file_name){
        // Save the tuple to a binary file
        std::ofstream ofs(file_name, std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << _mat;
        ofs.close();
    }

    void load_data(const std::string& file_name){
        try {
            std::ifstream ifs(file_name, std::ios::binary);
            if (!ifs.is_open()) {
                throw std::runtime_error("Failed to open file of iteration data. \n File: " + file_name);
            }

            boost::archive::binary_iarchive ia(ifs);
            ia >> _mat;

            ifs.close();

        } catch (const boost::archive::archive_exception& e) {
            std::cerr << "Boost archive exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Standard exception: " << e.what() << std::endl;
        }
    }

    void update_data(const Matrix3D& mat3d){
        _mat = mat3d;
    }

    void print_data(){
        std::cout << _mat << std::endl;
    }

protected:
    int _nt, _dim;
    Matrix3D _mat;

};



// template<typename DataType>
// class IterationData{
// using IterDataMap = std::unordered_map<int, DataType>;

// public:
//     IterationData(int niters, int dim, int nt): _niters{niters}, _dim{dim}{}

//     IterationData(){}

//     void add_data(int iter, const DataType& data){
//         _map[iter] = data;
//     }

//     int size(){
//         int map_size = _map.size();
//         return map_size;
//     }

//     void save_data(const std::string& file_name){
//         // Save the tuple to a binary file
//         std::ofstream ofs(file_name, std::ios::binary);
//         boost::archive::binary_oarchive oa(ofs);
//         oa << _map;
//     }

//     void print_data(){
//         for(auto & pair:_map){
//             int key = pair.first;
//             DataType data = pair.second;
//             std::cout << "Key: " << key << ", " << std::endl << "Value: " << std::endl;
//             data.print_data();
//         }
//     }

//     void load_data(const std::string& file_name){
//         try {
//             std::ifstream ifs(file_name, std::ios::binary);
//             if (!ifs.is_open()) {
//                 throw std::runtime_error("Failed to open file of iteration data. \n File: " + file_name);
//             }

//             boost::archive::binary_iarchive ia(ifs);
//             ia >> _map;

//         } catch (const boost::archive::archive_exception& e) {
//             std::cerr << "Boost archive exception: " << e.what() << std::endl;
//         } catch (const std::exception& e) {
//             std::cerr << "Standard exception: " << e.what() << std::endl;
//         }
//     }


// protected:
//     IterDataMap _map;
//     int _niters, _dim;

// };

// typedef IterationData<MyData> IterationMydata;

}

using namespace gvi;

// Define the serialization of iteration data type.
namespace boost {
   namespace serialization {        
        template<class Archive>
        void save(Archive& ar, 
                  const MyData& data, 
                  const unsigned int version) 
        {   
            int nt{data._nt};
            int dim{data._dim};
            MatrixXd mat{data._mat};

            ar & nt;
            ar & dim;
            ar & mat;
        }

        template<class Archive>
        void load(Archive& ar, MyData& data, const unsigned int version) {
            int nt = 0;
            int dim = 0;

            ar & nt;
            ar & dim;

            Matrix3D mat(nt, dim, dim);

            ar & mat;

            data.set_nt(nt);
            data.set_dim(dim);
            data.set_mat(mat);

        }

        // The serialization is split into save and load here.
        template<class Archive>
        inline void serialize(
            Archive & ar, 
            MyData& data, 
            const unsigned int version)
        {
            split_free(ar, data, version);
        }

    }
}


// // Define the serialization of iteration data type.
// namespace boost {
//    namespace serialization {  

//         template<class Archive>
//         void save(Archive& ar, 
//                   const IterationMydata& map, 
//                   const unsigned int version) 
//         {
//             int size = map._map.size();
            
//             ar & size;

//             for (auto & pair: map._map) { 
//                 int key = pair.first;
//                 MyData value = pair.second;

//                 ar & key; // Serialize first int key
//                 ar & value; // Serialize MyData value

//             }
//         }

//         template<class Archive>
//         void load(Archive& ar, IterationMydata& map, const unsigned int version) {
//             int size = 0;
//             ar & size;

//             for (int i=0; i< size; i++) { 

//                 int key = 0;
//                 MyData data;

//                 ar & key; // Serialize first double in the key tuple
//                 ar & data; // Serialize first double in the key tuple

//                 map._map[key] = data;

//             }
//         }

//         // The serialization is split into save and load here.
//         template<class Archive>
//         inline void serialize(
//             Archive & ar, 
//             IterationMydata& map, 
//             const unsigned int version)
//         {
//             split_free(ar, map, version);
//         }

//     }
// }

