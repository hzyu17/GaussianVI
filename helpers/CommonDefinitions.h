/**
 * Commonly used definitions.
*/

#pragma once

#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

#include <vector>
#include <iostream>
#include <Eigen/SparseCholesky>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

#define STRING(x) #x
#define XSTRING(x) STRING(x)
std::string source_root{XSTRING(SOURCE_ROOT)};

namespace gvi{
    
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::SparseVector<double> SpVec; 
typedef Eigen::Triplet<double> Trip;
typedef Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>> SparseLDLT;
typedef std::pair <Eigen::VectorXd, Eigen::MatrixXd> Message;
typedef double (*CostFunctionPtr)(const double* vec_x, int dim);


//https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

} // namespace gvi


#endif // COMMON_DEFINITIONS_H