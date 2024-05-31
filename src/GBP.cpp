#include <iostream>
#include <utility>
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

typedef pair<VectorXd, MatrixXd> Message;


Message calculate_factor_message(const Message &input_message, int target, const Message &factor_potential, int dim) {
    Message message;
    VectorXd target_vector(1);
    target_vector(0) = target;
    message.first = target_vector;

    int index_variable = -1;
    int index_target = -1;

    for (int i = 0; i < factor_potential.first.size(); i++) {
        if (factor_potential.first(i) == input_message.first(0)) {
            index_variable = i;
        }
        if (factor_potential.first(i) == target) {
            index_target = i;
        }
    }

    MatrixXd lam = factor_potential.second;
    
    lam.block(dim * index_variable, dim * index_variable, dim, dim) += input_message.second;
    
    if (index_target != 0) {
        lam.block(0, 0, dim, lam.cols()).swap(lam.block(index_target * dim, 0, dim, lam.cols()));
        lam.block(0, 0, lam.rows(), dim).swap(lam.block(0, index_target * dim, lam.rows(), dim));
    }

    MatrixXd lam_inverse = lam.block(dim, dim, lam.rows() - dim, lam.cols() - dim).inverse();
    MatrixXd lam_message = lam.block(0, 0, dim, dim) - lam.block(0, dim, dim, lam.cols() - dim) * lam_inverse * lam.block(dim, 0, lam.rows() - dim, dim);

    message.second = lam_message;
    return message;
}


Message calculate_variable_message(const Message &input_message, const Message &prior_message) {
    return {prior_message.first, input_message.second + prior_message.second};
}


MatrixXd generate_lambda(int dim, int num_states) {
    int size = dim * num_states;
    MatrixXd lam = MatrixXd::Zero(size, size);
    for (int i = 0; i < num_states; i++) {
        for (int j = max(0, i); j < min(num_states, i + 2); j++) {
            MatrixXd A = MatrixXd::Random(dim, dim);
            if (i == j){
                MatrixXd AA = A + A.transpose();
                A = AA / 2.0;
            }
            A = (A.array() * 100).round() / 100.0; // Create a random symmetric positive definite matrix
            lam.block(i * dim, j * dim, dim, dim) = A;
            if (i != j) {
                lam.block(j * dim, i * dim, dim, dim) = A.transpose();
            }
        }
    }
    return lam;
}

pair<vector<MatrixXd>, vector<MatrixXd>> Gaussian_Belief_Propagation(int dim, int num_states, const MatrixXd &Lambda) {
    vector<Message> factor(2 * num_states - 1);
    vector<Message> joint_factor(num_states - 1);
    Message variable_message;

    for (int i = 0; i < 2 * num_states - 1; ++i) {
        int var = i / 2;
        if (i % 2 == 0) {
            VectorXd variable(1);
            variable << var;
            MatrixXd lam = Lambda.block(dim * var, dim * var, dim, dim);
            factor[i] = {variable, lam};
        } else {
            VectorXd variable(2);
            variable << var, var + 1;
            MatrixXd lam = MatrixXd::Zero(2 * dim, 2 * dim);
            lam.block(0, dim, dim, dim) = Lambda.block(dim * var, dim * (var + 1), dim, dim);
            lam.block(dim, 0, dim, dim) = Lambda.block(dim * (var + 1), dim * var, dim, dim);
            joint_factor[var] = {variable, Lambda.block(dim * var, dim * var, 2 * dim, 2 * dim)};
            factor[i] = {variable, lam};
        }
    }

    vector<Message> factor_message(num_states);
    vector<Message> factor_message1(num_states);
    factor_message[0] = {VectorXd::Zero(1), MatrixXd::Zero(dim, dim)};
    factor_message1.back() = {VectorXd::Zero(1), MatrixXd::Zero(dim, dim)};
    factor_message1.back().first(0) = num_states - 1; 

    for (int i = 0; i < num_states - 1; i++) {
        variable_message = calculate_variable_message(factor_message[i], factor[2 * i]);
        factor_message[i + 1] = calculate_factor_message(variable_message, i + 1, factor[2 * i + 1], dim);
        int index = num_states - 1 - i;
        variable_message = calculate_variable_message(factor_message1[index], factor[2 * index]);
        factor_message1[index - 1] = calculate_factor_message(variable_message, index - 1, factor[2 * index - 1], dim);
    }

    vector<MatrixXd> sigma(num_states);
    vector<MatrixXd> sigma_joint(num_states - 1);
    MatrixXd identity = MatrixXd::Identity(dim, dim);
    MatrixXd identity_joint = MatrixXd::Identity(2 * dim, 2 * dim);

    for (int i = 0; i < num_states; ++i) {
        MatrixXd lam = factor_message[i].second + factor_message1[i].second + factor[2 * i].second;
        MatrixXd variance = lam.inverse();
        sigma[i] = variance;
    }

    for (int i = 0; i < num_states - 1; ++i) {
        MatrixXd lam_joint = joint_factor[i].second;
        lam_joint.block(0, 0, dim, dim) += factor_message[i].second;
        lam_joint.block(dim, dim, dim, dim) += factor_message1[i + 1].second;
        MatrixXd variance_joint = lam_joint.inverse();
        sigma_joint[i] = variance_joint;
    }

    return {sigma, sigma_joint};
}



int main(){
    int dim = 14;
    int num_states = 20;
    MatrixXd lambda;
    lambda = generate_lambda(dim, num_states);
    MatrixXd covariance = lambda.inverse();
    // cout << "Lambda:" << endl << lambda << endl << endl;
    // cout << "Covariance:" << endl << covariance << endl << endl;

    auto [sigma, sigma_joint] = Gaussian_Belief_Propagation(dim, num_states, lambda);
    
    double max_error1, max_error2;

    for (int i=0; i < num_states; i++){
        max_error1 = max(max_error1, (sigma[i]-covariance.block(dim*i, dim*i, dim, dim)).cwiseAbs().maxCoeff());
        // cout << "Covariance" << i << ":" << endl << sigma[i] << endl;
    }

    for (int i=0; i < num_states-1; i++){
        max_error2 = max(max_error2, (sigma_joint[i] - covariance.block(dim*i, dim*i, 2*dim, 2*dim)).cwiseAbs().maxCoeff());
        // cout << "Covariance" << i << "," << i+1 << ":" << endl << sigma_joint[i] << endl;
    }

    cout << "Max error of variance: " << max_error1 << endl;
    cout << "Max error of covariance: " << max_error2 << endl;
}