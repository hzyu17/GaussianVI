#include <gtest/gtest.h>
#include "helpers/test_serialization.h"
#include "helpers/Serialization.h"

#define STRING(x) #x
#define XSTRING(x) STRING(x)
std::string source_root{XSTRING(SOURCE_ROOT)};
using namespace gvi;

TEST(Serialization, MyData){
    
    // ======================= Serialize and De-serialize of the data ========================
    MyData d(2, 2);

    Matrix3D mat3d(2,2,2);
    mat3d(0,0) = 1.0;
    mat3d(1,1) = 2.0;
    mat3d(2,1) = 3.0;

    d.update_data(mat3d);

    std::string fname = source_root+"/tests/test_serialize_mydata.bin";

    // Save the data to a binary file
    serialize<MyData>(fname, d);

    // Load data from the just saved file
    MyData loaded_d;
    deserialize<MyData>(fname, loaded_d);

    ASSERT_EQ((d.mat() - loaded_d.mat()).norm(), 0);

}

TEST(Serialization, MapMyData){
// ======================= Serialize and De-serialize of the data map ========================
    // Create a map from int to MyData type
    typedef std::unordered_map<int, MyData> IterDataMap;
    IterDataMap test_map;
    MyData d1(2, 2);

    Matrix3D mat3d1(2,2,2);
    mat3d1(0,1) = 2.0;
    mat3d1(1,1) = 5.0;
    mat3d1(0,0) = 7.0;

    d1.update_data(mat3d1);
    test_map[0] = d1;

    std::string file_name_map = source_root+"/tests/test_serialize_mapdata.bin";

    // Save the map to a binary file
    serialize<IterDataMap>(file_name_map, test_map);

    // Load the map data
    IterDataMap test_loaded_map;
    deserialize<IterDataMap>(file_name_map, test_loaded_map);

    ASSERT_EQ((test_map[0].mat() - test_loaded_map[0].mat()).norm(), 0);
}


TEST(Serialization, IterationMyData){
    // ======================= Serialize and De-serialize of the iteration data ========================
    // Create iteration data
    IterationMydata iter_data;

    // Add iteration data
    int niter = 5;
    for (int i=0; i<niter; i++){
        MyData d(2, 2);

        Matrix3D mat3d(2,2,2);
        mat3d(0,1) = i;
        mat3d(1,1) = i+1;
        mat3d(0,0) = i+2;

        d.update_data(mat3d);

        iter_data.add_data(i, d);
    }

    std::string file_name = source_root+"/tests/test_serialize_iterdata.bin";

    // Save data
    serialize<IterationMydata>(file_name, iter_data);

    // Load data
    IterationMydata data_loaded;
    deserialize<IterationMydata>(file_name, data_loaded);

    std::unordered_map<int, MyData> saved_map = iter_data.map();
    std::unordered_map<int, MyData> load_map = data_loaded.map();

    std::cout << "======== print loaded iteration data ========" << std::endl;
    for (int i=0; i<niter; i++){
        ASSERT_EQ((load_map[i].mat() - saved_map[i].mat()).norm(), 0);
    }
}