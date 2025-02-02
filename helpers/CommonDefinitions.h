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
static std::string source_root{XSTRING(SOURCE_ROOT)};

namespace gvi{
    
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::SparseVector<double> SpVec; 
typedef Eigen::Triplet<double> Trip;
typedef Eigen::SimplicialLDLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>> SparseLDLT;
typedef std::pair <Eigen::VectorXd, Eigen::MatrixXd> Message;


//https://eigen.tuxfamily.org/dox/structEigen_1_1IOFormat.html
static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");

Eigen::MatrixXd sqrtm(const Eigen::MatrixXd& mat){
    assert(mat.rows() == mat.cols());

    Eigen::RealSchur<Eigen::MatrixXd> schur(mat.rows());
    schur.compute(mat);

    const Eigen::MatrixXd& T = schur.matrixT();
    const Eigen::MatrixXd& U = schur.matrixU();

    Eigen::MatrixXd T_sqrt = T;
    
    int n = T.rows();
    for (int i = 0; i < n;) {
        if (i == n - 1 || T(i + 1, i) == 0) {
            // Diagonal block is 1x1
            T_sqrt(i, i) = std::sqrt(T(i, i));
            i++;
        } else {
            // Diagonal block is 2x2
            Eigen::Matrix2d block = T.block<2, 2>(i, i);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(block);
            Eigen::Matrix2d D_sqrt = es.eigenvalues().cwiseSqrt().asDiagonal();
            Eigen::Matrix2d S = es.eigenvectors();
            Eigen::Matrix2d S_inv = S.inverse();
            T_sqrt.block<2, 2>(i, i) = S * D_sqrt * S_inv;
            i += 2;
        }
    }

    Eigen::MatrixXd mat_sqrt = U * T_sqrt * U.transpose();
    return mat_sqrt;
}

} // namespace gvi


#endif // COMMON_DEFINITIONS_H