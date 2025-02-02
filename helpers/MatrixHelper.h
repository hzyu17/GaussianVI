/**
 * @file MatrixHelper.h
 * @author Hongzhe Yu (hyu419@gatech.edu)
 * @brief Define some special matrix and vector classes.
 * @version 0.1
 * @date 2023-03-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#pragma once

#ifndef MATRIX_HELPER_H
#define MATRIX_HELPER_H

#include "helpers/CommonDefinitions.h"
#include <Eigen/Dense>
// #include <boost/archive/binary_oarchive.hpp>
// #include <boost/archive/binary_iarchive.hpp>

using namespace Eigen;

namespace gvi {

class Matrix3D : public MatrixXd{
public:
    Matrix3D(){}
    Matrix3D(int row, int col, int nt):_row{row}, _col{col}, _nt{nt}, MatrixXd(row*col, nt) {}    
    Matrix3D(const Matrix3D& mat):_row{mat._row}, _col{mat._col}, _nt{mat._nt},  Eigen::MatrixXd(mat) {}
    Matrix3D(const MatrixXd & mat): MatrixXd(mat){}

    // Overloaded assignment operator that takes an Eigen::MatrixXd as input
    Matrix3D& operator=(const Eigen::MatrixXd& other) {
        Eigen::MatrixXd::operator=(other); // call base class assignment operator
        // additional custom logic for MyMatrix
        return *this;
    }

public:
    int _row, _col, _nt;
    
};


class MatrixIO{
    public:
        
        MatrixIO(){}

        template <typename T>
        void saveData(const std::string& fileName, const T& matrix, bool verbose=true) const{
            if (verbose){
                std::cout << "Saving data to: " << fileName << std::endl;
            }
            std::ofstream file(fileName);
            if (file.is_open()){
                file << matrix.format(CSVFormat);
                file.close();
            }
        }

        /**
         * @brief read a sdf map from csv file, which can be the output of some gpmp2 functions.
         * modified from an online code piece.
         * @param filename 
         * @return Matrix 
         */
        /// https://stackoverflow.com/questions/34247057/how-to-read-csv-file-and-assign-to-eigen-matrix

        Eigen::MatrixXd load_csv (const std::string & path) {
            std::ifstream indata;
            indata.open(path);
            if (indata.peek() == std::ifstream::traits_type::eof()){
                throw std::runtime_error(std::string("File dose not exist ...: ") + path);
            }
            
            std::string line;
            std::vector<double> values;
            uint rows = 0;
            while (std::getline(indata, line)) {
                std::stringstream lineStream(line);
                std::string cell;
                while (std::getline(lineStream, cell, ',')) {
                    values.push_back(std::stod(cell));
                }
                rows++;
            }
            return Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
        }

    };


struct GVIBlock{
public:
    GVIBlock(){}
    GVIBlock(int start_row, int start_col, int nrows, int ncols):
    _start_row(start_row),
    _start_col(start_col),
    _nrows(nrows),
    _ncols(ncols){}

    int row(){ return _start_row;}

    int nrows(){ return _nrows;}

    int col(){ return _start_col;}

    int ncols(){ return _ncols;}

    int _start_row;
    int _start_col;
    int _nrows;
    int _ncols;
    
}; // struct GVIBlock

class TrajectoryBlock{

public:
    TrajectoryBlock(){}

    TrajectoryBlock(int state_dim, int num_states, int start_index, int block_length):
    _state_dim(state_dim),
    _num_states(num_states),
    _start_index(start_index),
    _block_length(block_length){
        _block = GVIBlock{_start_index*state_dim, _start_index*state_dim, _block_length, _block_length};
    };

    SpMat extract(const SpMat & m){
        return m.middleRows(_block.row(), _block.nrows()).middleCols(_block.col(), _block.ncols());
    }

    Eigen::VectorXd extract_vector(const Eigen::VectorXd & vec){
        return vec.block(_block.row(), 0, _block.nrows(), 1);
    }

    void fill(Eigen::MatrixXd & block, SpMat & matrix){
        Eigen::MatrixXd mat_full{matrix};
        mat_full.block(_block.row(), _block.col(), _block.nrows(), _block.ncols()) = block;
        matrix = mat_full.sparseView();
    }

    void fill_vector(Eigen::VectorXd & vec, const Eigen::VectorXd & vec_block){
        vec.setZero();
        vec.block(_block.row(), 0, _block.nrows(), 1) = vec_block;
    }

    void print(){
        std::cout << "(starting index, block length): " << "(" << _start_index << ", " << _block_length << ")" << std::endl;
    }

private:
    int _state_dim;
    int _num_states;
    int _start_index, _block_length;
    GVIBlock _block = GVIBlock();

}; // class TrajectoryBlock


} // namespace gvi


using namespace gvi;

// // Define the serialization of iteration data type.
// namespace boost {
//    namespace serialization {        
//         template<class Archive>
//         void save(Archive& ar, 
//                   const Matrix3D& mat3d, 
//                   const unsigned int version) 
//         {   
//             int row = mat3d._row;
//             int col = mat3d._col;
//             int nt = mat3d._nt;

//             MatrixXd mat{mat3d};
//             ar & row;
//             ar & col;
//             ar & nt;
//             ar & mat;
//         }

//         template<class Archive>
//         void load(Archive& ar, Matrix3D& mat3d, const unsigned int version) {
            
//             int row = 0;
//             int col = 0;
//             int nt = 0;
            

//             ar & row;
//             ar & col;
//             ar & nt;

//             MatrixXd mat(row*col, nt);

//             ar & mat;

//             mat3d = mat;

//         }

//         // The serialization is split into save and load here.
//         template<class Archive>
//         inline void serialize(
//             Archive & ar, 
//             Matrix3D& mat3d, 
//             const unsigned int version)
//         {
//             split_free(ar, mat3d, version);
//         }

//     }
// }

#endif // MATRIX_HELPER_H