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
    std::cout << "======== print saved data ========" << std::endl;
    d.print_data();

    std::string fname = source_root+"/tests/test_serialize_mydata.bin";
    // Save the tuple to a binary file
    std::ofstream ofs(fname, std::ios::binary);
    boost::archive::binary_oarchive oa(ofs);
    oa << d;
    ofs.close();

    // d.save_data(source_root+"/tests/test_serialize_mydata.bin");

    MyData loaded_d(2,2);
    try {
        std::ifstream ifs(fname, std::ios::binary);
        if (!ifs.is_open()) {
            throw std::runtime_error("Failed to open file of iteration data. \n File: " + fname);
        }

        boost::archive::binary_iarchive ia(ifs);
        ia >> loaded_d;

        ifs.close();

    } catch (const boost::archive::archive_exception& e) {
        std::cerr << "Boost archive exception: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
    }

    // loaded_d.load_data(source_root+"/tests/test_serialize_mydata.bin");
    std::cout << "======== print loaded data ========" << std::endl;
    loaded_d.print_data();

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
    std::cout << "======== print saved iteration data ========" << std::endl;
    iter_data.print_data();

    std::string file_name = source_root+"/tests/test_serialize_iterdata.bin";
    // Save data
    iter_data.save_data(file_name);

    // Load data
    IterationMydata data_loaded;

    // try {
    //     std::ifstream ifs(file_name, std::ios::binary);
    //     if (!ifs.is_open()) {
    //         throw std::runtime_error("Failed to open file of iteration data. \n File: " + file_name);
    //     }

    //     boost::archive::binary_iarchive ia(ifs);
    //     ia >> data_loaded;

    //     ifs.close();

    // } catch (const boost::archive::archive_exception& e) {
    //     std::cerr << "Boost archive exception: " << e.what() << std::endl;
    // } catch (const std::exception& e) {
    //     std::cerr << "Standard exception: " << e.what() << std::endl;
    // }

    data_loaded.load_data(source_root+"/tests/test_serialize_iterdata.bin");
    data_loaded.print_data();

}