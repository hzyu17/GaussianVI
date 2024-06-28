#include "helpers/MatrixMultiplication.h"
#include <Eigen/Dense>

int main (void){
    // Define dimensions
    int rows = 4;
    int cols = 6;
    int vec_num = 5;

    // Generate the matrices randomly
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(rows, cols);
    Eigen::MatrixXd vectorMatrix(cols, vec_num);
    Eigen::MatrixXd result(rows, vec_num);

    // Randomly generate the vector of VectorXd
    std::vector<Eigen::VectorXd> vectors(vec_num, Eigen::VectorXd(cols));
    for (int i = 0; i < vec_num; ++i) {
        vectors[i] = Eigen::VectorXd::Random(cols);
    }

    // Convert vector of vectors to a matrix
    for (int i = 0; i < vec_num; ++i) {
        vectorMatrix.col(i) = vectors[i];
    }

    // Allocate memory for arrays to store matrix and vector data
    double* matrix_array = new double[matrix.size()];
    double* vectorMatrix_array = new double[vectorMatrix.size()];
    double* result_array = new double[result.size()];

    // Assign the value in eigen matrix into array
    Eigen::Map<Eigen::MatrixXd>(matrix_array, matrix.transpose().rows(), matrix.transpose().cols()) = matrix.transpose();
    Eigen::Map<Eigen::MatrixXd>(vectorMatrix_array, vectorMatrix.transpose().rows(), vectorMatrix.transpose().cols()) = vectorMatrix.transpose();

    MatrixMul(matrix_array, vectorMatrix_array, result_array, rows, cols, vec_num);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> result_transform(result_array, rows, vec_num);
    // result = result_transform;
    result = matrix * vectorMatrix;

    std::cout << "Result:" << std::endl << result << std::endl;
    std::cout << std::endl;

    
    std::cout << "Result1:" << std::endl << result_transform << std::endl;
    std::cout << std::endl;
    
    return 0;
}