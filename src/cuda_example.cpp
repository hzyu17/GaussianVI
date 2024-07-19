#include "helpers/CudaOperation.h"
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

    // MatrixMul(matrix.data(), vectorMatrix.data(), result.data(), rows, cols, vec_num);

    // std::cout << "Result_cuda:" << std::endl << result << std::endl;
    // std::cout << std::endl;

    // result = matrix * vectorMatrix;

    // std::cout << "Result:" << std::endl << result << std::endl;
    // std::cout << std::endl;
    
    // return 0;
}