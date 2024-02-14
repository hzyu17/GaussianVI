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

    int nt() const { return _nt; }
    int dim() const { return _dim; }
    Matrix3D mat() const {return _mat; }

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


template<typename DataType>
class IterationData{
    typedef std::unordered_map<int, DataType> IterDataMap;
public:
    IterationData(int niters, int dim, int nt): _niters{niters}, _dim{dim}{}

    IterationData(){}

    void add_data(int iter, const DataType& data){
        _map[iter] = data;
    }

    int size() const {
        return _map.size();
    }

    int dim() const {
        return _dim;
    }

    int niters() const {
        return _niters;
    }

    IterDataMap map() const {
        return _map;
    }

    void save_data(const std::string& file_name) const {
        // Save the tuple to a binary file
        std::ofstream ofs(file_name, std::ios::binary);
        boost::archive::binary_oarchive oa(ofs);
        oa << _map;
        ofs.close();
    }

    void print_data() const{
        for(auto & pair:_map){
            int key = pair.first;
            DataType data = pair.second;
            std::cout << "Key: " << key << ", " << std::endl << "Value: " << std::endl;
            data.print_data();
        }
    }

    void load_data(const std::string& file_name){
        try {
            std::ifstream ifs(file_name, std::ios::binary);
            if (!ifs.is_open()) {
                throw std::runtime_error("Failed to open file of iteration data. \n File: " + file_name);
            }

            boost::archive::binary_iarchive ia(ifs);
            ia >> _map;

        } catch (const boost::archive::archive_exception& e) {
            std::cerr << "Boost archive exception: " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Standard exception: " << e.what() << std::endl;
        }
    }

    void set_map(const IterDataMap& map){
        _map = map;
    }

    void set_niters(int niters){
        _niters = niters;
    }

    void set_dim(int dim){
        _dim = dim;
    }

protected:
    IterDataMap _map;
    int _niters, _dim;

};

typedef IterationData<MyData> IterationMydata;

}

using namespace gvi;

// Define the serialization of MyData type.
namespace boost {
   namespace serialization {        
        template<class Archive>
        void save(Archive& ar, 
                  const MyData& data, 
                  const unsigned int version) 
        {   
            int nt = data.nt();
            int dim = data.dim();
            MatrixXd mat{data.mat()};

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

            Eigen::MatrixXd mat(dim*dim, nt);

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

// Define the serialization of IterDataMap<DataType> type.
namespace boost {
   namespace serialization {        
        template<class Archive, class DataType>
        void save(Archive& ar, 
                  const std::unordered_map<int, DataType>& map, 
                  const unsigned int version) 
        {   
            int size = 0;
            size = map.size();
            
            ar & size;

            for (auto & pair: map) { 
                int key = pair.first;
                DataType value = pair.second;

                ar & key; // Serialize first int key
                ar & value; // Serialize MyData value

            }
        }

        template<class Archive, class DataType>
        void load(Archive& ar, std::unordered_map<int, DataType>& map, const unsigned int version) {
            int size = 0;
            ar & size;

            for (int i=0; i< size; i++) { 

                int key = 0;
                DataType data;

                ar & key; // Serialize first double in the key tuple
                ar & data; // Serialize first double in the key tuple

                map[key] = data;

            }

        }

        // The serialization is split into save and load here.
        template<class Archive, class DataType>
        inline void serialize(
            Archive & ar, 
            std::unordered_map<int, DataType>& map, 
            const unsigned int version)
        {
            split_free(ar, map, version);
        }

    }
}


// Define the serialization of iteration data type.
namespace boost {
   namespace serialization {  

        template<class Archive, class DataType>
        void save(Archive& ar, 
                  const IterationData<DataType>& iter_data, 
                  const unsigned int version) 
        {
            int niters=0;
            int dim=0;
            
            niters = iter_data.niters();
            dim = iter_data.dim();
            std::unordered_map<int, DataType> map{iter_data.map()};

            ar & niters; // Serialize first int key
            ar & dim; // Serialize MyData value
            ar & map;

        }

        template<class Archive, class DataType>
        void load(Archive& ar, IterationData<DataType>& iter_data, const unsigned int version) {
            int niters=0;
            int dim=0;

            std::unordered_map<int, DataType> map;

            ar & niters; // Serialize first int key
            ar & dim; // Serialize MyData value
            ar & map;

            iter_data.set_niters(niters);
            iter_data.set_dim(dim);
            iter_data.set_map(map);

        }

        // The serialization is split into save and load here.
        template<class Archive, class DataType>
        inline void serialize(
            Archive & ar, 
            IterationData<DataType>& map, 
            const unsigned int version)
        {
            split_free(ar, map, version);
        }

    }
}

