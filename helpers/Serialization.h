/**
 * @file Serialization.h
 * @author: Hongzhe Yu
 * @date 02/03/2024
 * @brief: Serilization save and load of arbitary data.
*/

// #include <boost/archive/binary_oarchive.hpp>
// #include <boost/archive/binary_iarchive.hpp>


// namespace gvi{
    
// template <class DataType>
// void serialize(const std::string& fname, const DataType& data){
//     std::ofstream ofs(fname, std::ios::binary);
//     boost::archive::binary_oarchive oa(ofs);
//     oa << data;
//     ofs.close();
// }


// template <class DataType>
// void deserialize(const std::string& fname, DataType& data){
//     try {
//         std::ifstream ifs(fname, std::ios::binary);
//         if (!ifs.is_open()) {
//             throw std::runtime_error("Failed to open file of iteration data. \n File: " + fname);
//         }

//         boost::archive::binary_iarchive ia(ifs);
//         ia >> data;
        
//         ifs.close();

//     } catch (const boost::archive::archive_exception& e) {
//         std::cerr << "Boost archive exception: " << e.what() << std::endl;
//     } catch (const std::exception& e) {
//         std::cerr << "Standard exception: " << e.what() << std::endl;
//     }

// }

// }
