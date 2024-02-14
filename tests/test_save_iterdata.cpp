#include <gtest/gtest.h>
#include "helpers/test_serialization.h"

#define STRING(x) #x
#define XSTRING(x) STRING(x)
std::string source_root{XSTRING(SOURCE_ROOT)};

TEST(Serialization, initialize){
using namespace gvi;

MyData d(2, 2);
std::cout << "======== print data ========" << std::endl;
d.print_data();

Matrix3D mat3d(2,2,2);
mat3d(0,0) = 1.0;
mat3d(1,1) = 2.0;
mat3d(2,1) = 3.0;

d.update_data(mat3d);
std::cout << "======== print data ========" << std::endl;
d.print_data();

d.save_data(source_root+"/tests/test_serialize_iterdata.bin");

MyData loaded_d;
loaded_d.load_data(source_root+"/tests/test_serialize_iterdata.bin");
std::cout << "======== print data ========" << std::endl;
loaded_d.print_data();

// // Create iteration data
// IterationMydata iter_data;

// // Add iteration data
// int niter = 5;
// for (int i=0; i<niter; i++){
//     MyData d(2, 2);

//     iter_data.add_data(i, d);
// }

// // Save data
// iter_data.save_data(source_root+"/tests/test_serialize_iterdata.bin");

// // Load data
// IterationMydata data_loaded;
// iter_data.load_data(source_root+"/tests/test_serialize_iterdata.bin");
// data_loaded.print_data();

}