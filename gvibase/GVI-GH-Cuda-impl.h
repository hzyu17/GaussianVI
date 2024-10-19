
#pragma once

#ifndef GVI_GH_IMPL_H
#define GVI_GH_IMPL_H

using namespace Eigen;

#include <stdexcept>
#include <optional>
#include <omp.h>
#include "helpers/CudaOperation.h"

#define STRING(x) #x
#define XSTRING(x) STRING(x)

namespace gvi{

template <typename Factor>
void GVIGH<Factor>::switch_to_high_temperature(){
    std::cout << "Switching to high temperature.." << std::endl;
    #pragma omp parallel for
    for (auto& i_factor : _vec_factors) {
        i_factor->factor_switch_to_high_temperature();
    }
    this->_temperature = this->_high_temperature;
    // this->initilize_precision_matrix();
}

template <typename Factor>
void GVIGH<Factor>::classify_factors(){
    for (auto& i_factor : _vec_factors) {
        if (i_factor->linear_factor())
            _vec_linear_factors.push_back(i_factor);
        else if (!i_factor->linear_factor())
            _vec_nonlinear_factors.push_back(i_factor);
    }
}

/**
 * @brief optimize with backtracking
 */ 
template <typename Factor>
void GVIGH<Factor>::optimize(std::optional<bool> verbose)
{
    // default verbose
    bool is_verbose = verbose.value_or(true);
    bool is_lowtemp = true;
    bool converged = false;

    _vec_nonlinear_factors[0]->cuda_init();

    for (int i_iter = 0; i_iter < _niters; i_iter++)
    {   

        if (converged){
            break;
        }

        // ============= High temperature phase =============
        if (i_iter == _niters_lowtemp && is_lowtemp){
            this->switch_to_high_temperature();
            is_lowtemp = false;
        }

        // ============= Cost at current iteration =============
        // double cost_iter = this->cost_value(); // -> Base::cost_value(this->_mu, this->_precision);

        if (is_verbose){
            std::cout << "========= iteration " << i_iter << " ========= " << std::endl;
        }

        // ============= Collect factor costs =============
        // VectorXd fact_costs_iter = this->factor_cost_vector();

        // Timer timer;
        // timer.start();

        // std::cout << std::fixed << std::setprecision(16);
        auto result_cuda = factor_cost_vector_cuda();

        double cost_iter = std::get<0>(result_cuda);
        VectorXd fact_costs_iter = std::get<1>(result_cuda);
        VectorXd dmu = std::get<2>(result_cuda);
        SpMat dprecision = std::get<3>(result_cuda);

        if (is_verbose){
            std::cout << "--- cost_iter ---" << std::endl << cost_iter << std::endl;
            // std::cout << "Factor Costs:" << fact_costs_iter.transpose() << std::endl;
            std::cout << "--- dmu ---" << std::endl << dmu.norm() << std::endl;
            std::cout << "--- dprecision ---" << std::endl << dprecision.norm() << std::endl;
        }

        _res_recorder.update_data(_mu, _covariance, _precision, cost_iter, fact_costs_iter);

        // gradients
        // std::tuple<VectorXd, SpMat> gradients = compute_gradients(); //Used calculate partial V here

        // VectorXd dmu = std::get<0>(gradients);
        // SpMat dprecision = std::get<1>(gradients);
        
        int cnt = 0;
        int B = 1;
        double step_size = _step_size_base;

        // backtracking 
        while (true)
        {   
            // new step size
            // step_size = pow(_step_size_base, B);
            step_size = step_size * 0.75;

            auto onestep_res = onestep_linesearch(step_size, dmu, dprecision);

            double new_cost = std::get<0>(onestep_res);
            VectorXd new_mu = std::get<1>(onestep_res);
            auto new_precision = std::get<2>(onestep_res);

            // accept new cost and update mu and precision matrix
            if (new_cost < cost_iter){
                // update mean and covariance
                this->update_proposal(_alpha * new_mu + (1-_alpha) * this->_mu, _alpha * new_precision + (1-_alpha) * this->_precision); // Update using EMA
                // this->update_proposal(new_mu, new_precision); // Update the mu of GVIGH
                // std::cout << "back tracking time: "<< cnt << std::endl;
                break;
            }else{ 
                // shrinking the step size
                B += 1;
                cnt += 1;
            }

            if (cnt > _niters_backtrack)
            {
                if (is_verbose){
                    std::cout << "Reached the maximum backtracking steps." << std::endl;
                }

                if (is_lowtemp){
                    this->switch_to_high_temperature();
                    is_lowtemp = false;
                }else{
                    converged = true;
                }

                // update_proposal(new_mu, new_precision);
                break;
            }                
        }
    }

    _vec_nonlinear_factors[0]->cuda_free();

    std::cout << "=========== Saving Data ===========" << std::endl;
    save_data(is_verbose);

}

template <typename Factor>
std::tuple<double, VectorXd, VectorXd, SpMat> GVIGH<Factor>::factor_cost_vector_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    int n_nonlinear = _vec_nonlinear_factors.size();

    VectorXd fac_costs(_nfactors);
    VectorXd nonlinear_fac_cost(n_nonlinear);
    fac_costs.setZero();
    nonlinear_fac_cost.setZero();

    SpMat joint_cov = inverse_GBP(joint_precision);

    std::vector<MatrixXd> sigmapts_vec(n_nonlinear);
    std::vector<VectorXd> mean_vec(n_nonlinear);

    omp_set_num_threads(20); 

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = _vec_nonlinear_factors[i];
        opt_k->cuda_matrices(_mu, _covariance, sigmapts_vec, mean_vec); 
    }

    int sigma_rows = sigmapts_vec[0].rows();
    int sigma_cols = sigmapts_vec[0].cols();
    int mean_size = mean_vec[0].size();

    MatrixXd sigmapts_mat(sigma_rows, sigmapts_vec.size()*sigma_cols);
    MatrixXd mean_mat(mean_size, mean_vec.size());

    VectorXd E_phi_mat(n_nonlinear);
    VectorXd dmu_mat(sigma_cols * n_nonlinear);
    MatrixXd ddmu_mat(sigma_cols, sigma_cols * n_nonlinear);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        sigmapts_mat.block(0, i * sigma_cols, sigma_rows, sigma_cols) = sigmapts_vec[i];
        mean_mat.col(i) = mean_vec[i];
    }

    // Compute the cost of the nonlinear factors
    _vec_nonlinear_factors[0]->costIntegration(sigmapts_mat, nonlinear_fac_cost, sigma_cols);

    _vec_nonlinear_factors[0]->dmuIntegration(sigmapts_mat, mean_mat, E_phi_mat, dmu_mat, ddmu_mat, sigma_cols);
    E_phi_mat = nonlinear_fac_cost;

    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature; 

    int cnt = 0;

    // Use a private counter for each thread to avoid race conditions
    int thread_cnt = 0;

    // #pragma omp for
    for (int i = 0; i < _vec_factors.size(); ++i)
    {
        auto &opt_k = _vec_factors[i];
        if (opt_k->linear_factor())
            fac_costs(thread_cnt) = opt_k->fact_cost_value(_mu, _covariance); 
        else
            fac_costs(thread_cnt) = nonlinear_fac_cost(opt_k->_start_index - 1);
        thread_cnt += 1;
    }

    #pragma omp critical
    {
        cnt += thread_cnt; // Safely update the global counter
    }

    double value = fac_costs.sum();
    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    double cost = value + vec_D.array().log().sum() / 2;

    // std::cout << "nonlinear_fac_cost = "<< nonlinear_fac_cost.sum() << std::endl; 
    // std::cout << "fac_costs = " << fac_costs.sum() << std::endl; 
    // std::cout << "Entropy = " << vec_D.array().log().sum() / 2 << std::endl << std::endl; 
    
    // std::cout << "Norm of E_phi_mat in derivatives = " << E_phi_mat.norm() << "  Sum = " << E_phi_mat.cwiseAbs().sum() << std::endl;
    // std::cout << "Norm of dmu_mat in derivatives = " << dmu_mat.norm() << "  Sum = " << dmu_mat.cwiseAbs().sum() << std::endl;
    // std::cout << "Norm of ddmu_mat in derivatives = " << ddmu_mat.norm() << "  Sum = " << ddmu_mat.cwiseAbs().sum() << std::endl << std::endl;

    _Vdmu.setZero();
    _Vddmu.setZero();

    VectorXd Vdmu_sum(_dim);
    SpMat Vddmu_sum(_dim, _dim);
    Vdmu_sum.setZero();
    Vddmu_sum.setZero();

    #pragma omp parallel 
    {
        // Thread-local storage to avoid race conditions
        VectorXd Vdmu_private(Vdmu_sum.size());
        SpMat Vddmu_private(Vddmu_sum.rows(), Vddmu_sum.cols());
        Vdmu_private.setZero();
        Vddmu_private.setZero();

        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &opt_k : _vec_factors) {
            if (opt_k->linear_factor()){
                opt_k->calculate_partial_V();
                
            }
            else{
                int index = opt_k->index()-1;
                MatrixXd ddmu_i = ddmu_mat.block(0, index*sigma_cols, sigma_cols, sigma_cols);;
                VectorXd dmu_i = dmu_mat.segment(index*sigma_cols, sigma_cols);

                opt_k->calculate_partial_V(ddmu_i, dmu_i, E_phi_mat(index));
            }

            Vdmu_private += opt_k->local2joint_dmu();
            Vddmu_private += opt_k->local2joint_dprecision();
        }

        #pragma omp critical
        {
            Vdmu_sum += Vdmu_private;
            Vddmu_sum += Vddmu_private;
        }
    }

    _Vdmu = Vdmu_sum;
    _Vddmu = Vddmu_sum;

    // std::cout << "Norm of _Vdmu = " << _Vdmu.norm()  << "  Sum of _Vdmu = " << _Vdmu.cwiseAbs().sum() << std::endl;
    // std::cout << "Norm of _Vddmu = " << _Vddmu.norm()  << "  Sum of _Vddmu = " << _Vddmu.cwiseAbs().sum() << std::endl << std::endl;

    SpMat dprecision(_dim, _dim);
    dprecision.setZero();
    dprecision = _Vddmu - _precision;

    Eigen::ConjugateGradient<SpMat, Eigen::Upper> solver;
    VectorXd dmu(_dim);
    dmu.setZero();
    dmu = solver.compute(_Vddmu).solve(-_Vdmu);

    // std::cout << "Norm of dmu = " << dmu.norm()  << "  Sum of dmu = " << dmu.cwiseAbs().sum() << std::endl;
    // std::cout << "Norm of dprecision = " << dprecision.norm()  << "  Sum of dprecision = " << dprecision.cwiseAbs().sum() << std::endl << std::endl;

    return std::make_tuple(cost, fac_costs, dmu, dprecision);
}


template <typename Factor>
std::tuple<VectorXd, VectorXd, SpMat> GVIGH<Factor>::time_test_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    _vec_nonlinear_factors[0]->cuda_init();
    auto result_cuda = factor_cost_vector_cuda();
    _vec_nonlinear_factors[0]->cuda_free();

}

template <typename Factor>
inline void GVIGH<Factor>::set_precision(const SpMat &new_precision)
{
    _precision = new_precision;
    // // sparse inverse
    // inverse_inplace();
    _covariance = inverse_GBP(_precision);

    #pragma omp parallel
    {
        #pragma omp for nowait // Nowait allows threads to continue without waiting at the end of the loop
        for (auto &factor : _vec_factors)
        {
            factor->update_precision_from_joint(_covariance);
        }
    }
}

/**
 * @brief Compute the costs of all factors for a given mean and cov.
 */
template <typename Factor>
VectorXd GVIGH<Factor>::factor_cost_vector(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    VectorXd fac_costs(_nfactors);
    fac_costs.setZero();
    int cnt = 0;
    SpMat joint_cov = inverse_GBP(joint_precision);

    // Use a private counter for each thread to avoid race conditions
    int thread_cnt = 0;

    #pragma omp for
    for (int i = 0; i < _vec_factors.size(); ++i)
    {
        auto &opt_k = _vec_factors[i];
        fac_costs(thread_cnt) = opt_k->fact_cost_value(fill_joint_mean, joint_cov); 
        thread_cnt += 1;
    }

    #pragma omp critical
    {
        cnt += thread_cnt; // Safely update the global counter
    }
    
    return fac_costs;
}

template <typename Factor>
double GVIGH<Factor>::cost_value_cuda(const VectorXd& fill_joint_mean, SpMat& joint_precision)
{
    int n_nonlinear = _vec_nonlinear_factors.size();
    VectorXd nonlinear_fac_cost(n_nonlinear);
    nonlinear_fac_cost.setZero();

    SpMat joint_cov = inverse_GBP(joint_precision);

    std::vector<MatrixXd> sigmapts_vec(n_nonlinear);
    std::vector<VectorXd> mean_vec(n_nonlinear);

    #pragma omp parallel for
    for (int i = 0; i < n_nonlinear; i++)
    {
        auto &opt_k = _vec_nonlinear_factors[i];
        opt_k->cuda_matrices(fill_joint_mean, joint_cov, sigmapts_vec, mean_vec); 
    }

    int sigma_rows = sigmapts_vec[0].rows();
    int sigma_cols = sigmapts_vec[0].cols();
    int mean_size = mean_vec[0].size();

    MatrixXd sigmapts_mat(sigma_rows, sigmapts_vec.size()*sigma_cols);

    for (int i = 0; i < n_nonlinear; i++)
        sigmapts_mat.block(0, i * sigma_cols, sigma_rows, sigma_cols) = sigmapts_vec[i];

    // Compute the cost of the nonlinear factors
    _vec_nonlinear_factors[0]->newCostIntegration(sigmapts_mat, nonlinear_fac_cost, sigma_cols);

    nonlinear_fac_cost = nonlinear_fac_cost / this ->_temperature;

    double value = 0.0;

    #pragma omp parallel for reduction(+:value)
    for (int i = 0; i < _vec_linear_factors.size(); ++i)
    {
        auto &opt_k = _vec_linear_factors[i];
        value += opt_k->fact_cost_value(fill_joint_mean, joint_cov); 
    }

    value += nonlinear_fac_cost.sum();

    SparseLDLT ldlt(joint_precision);
    VectorXd vec_D = ldlt.vectorD();

    return value + vec_D.array().log().sum() / 2;
}

/**
 * @brief Compute the total cost function value given a state.
 */
template <typename Factor>
double GVIGH<Factor>::cost_value(const VectorXd &mean, SpMat &Precision)
{
    SpMat Cov = inverse_GBP(Precision);

    double value = 0.0;

    // sum up the factor costs
    #pragma omp parallel for reduction(+:value)
    for (int i = 0; i < _vec_factors.size(); ++i)
    {
        // Access the current element - ensure this is safe in a parallel context
        auto &opt_k = _vec_factors[i];
        value += opt_k->fact_cost_value(mean, Cov); 
    }

    SparseLDLT ldlt(Precision);
    VectorXd vec_D = ldlt.vectorD();

    // std::cout << "entropy cost" << std::endl << vec_D.array().log().sum() / 2 << std::endl;
    return value + vec_D.array().log().sum() / 2;
}


/**
 * @brief Compute the covariances using Gaussian Belief Propagation.
 */
template <typename Factor>
SpMat GVIGH<Factor>::inverse_GBP(const SpMat &Precision)
{
    std::vector<Message> factors(2*_num_states-1);
    std::vector<Message> joint_factors(_num_states-1);
    Message variable_message;
    MatrixXd covariance(_dim, _dim);
    covariance.setZero();

    // Extract the factors from the precision matrix
    // The variable in factors are 0, {0,1}, 1, {1,2}, 2, ..., {_num_states-1,_num_states}, _num_states
    for (int i = 0; i < 2*_num_states-1; i++) {
        int var = i / 2;
        if (i % 2 == 0) {
            VectorXd variable(1);
            variable << var;
            MatrixXd lambda = Precision.block(_dim_state * var, _dim_state * var, _dim_state, _dim_state);
            factors[i] = {variable, lambda};
        }
        else {
            VectorXd variable(2);
            variable << var, var + 1;
            MatrixXd lambda = MatrixXd::Zero(2 * _dim_state, 2 * _dim_state);
            lambda.block(0, _dim_state, _dim_state, _dim_state) = Precision.block(_dim_state * var, _dim_state * (var + 1), _dim_state, _dim_state);
            lambda.block(_dim_state, 0, _dim_state, _dim_state) = Precision.block(_dim_state * (var + 1), _dim_state * var, _dim_state, _dim_state);
            factors[i] = {variable, lambda};
            joint_factors[var] = {variable, Precision.block(_dim_state * var, _dim_state * var, 2 * _dim_state, 2 * _dim_state)};
        }
    }
    
    // Initialize message
    std::vector<Message> forward_messages(_num_states);
    std::vector<Message> backward_messages(_num_states);
    for (int i = 0; i < _num_states; i++) {
        forward_messages[i].second.setZero();
        backward_messages[i].second.setZero();
    }
    forward_messages[0] = {VectorXd::Zero(1), MatrixXd::Zero(_dim_state, _dim_state)};
    backward_messages.back() = {VectorXd::Constant(1, _num_states-1), MatrixXd::Zero(_dim_state, _dim_state)};

    // Calculate messages between factors and variables
    for (int i = 0; i < _num_states - 1; i++) {
        variable_message = calculate_variable_message(forward_messages[i], factors[2 * i]);
        forward_messages[i + 1] = calculate_factor_message(variable_message, i + 1, factors[2 * i + 1]);
        int index = _num_states - 1 - i;
        variable_message = calculate_variable_message(backward_messages[index], factors[2 * index]);
        backward_messages[index - 1] = calculate_factor_message(variable_message, index - 1, factors[2 * index - 1]);
    }

    if (_num_states == 1){
        MatrixXd lambda = forward_messages[0].second + backward_messages[0].second + factors[0].second;
        MatrixXd variance = lambda.inverse();
        covariance.block(0, 0, _dim_state, _dim_state) = variance;
    }

    // #pragma omp parallel for
    for (int i = 0; i < _num_states - 1; ++i) {
        MatrixXd lambda_joint = joint_factors[i].second;
        lambda_joint.block(0, 0, _dim_state, _dim_state) += forward_messages[i].second;
        lambda_joint.block(_dim_state, _dim_state, _dim_state, _dim_state) += backward_messages[i + 1].second;
        MatrixXd variance_joint = lambda_joint.inverse();
        covariance.block(i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state) = variance_joint;
    }

    return covariance.sparseView();
}

/**
 * @brief Compute the message of factors in GBP.
 */
template <typename Factor>
Message GVIGH<Factor>::calculate_factor_message(const Message &input_message, int target, const Message &factor_potential) {
    Message message;
    message.first = VectorXd::Constant(1, target);

    int index_variable = -1;
    int index_target = -1;

    // Identify indices of the variable and target in the factor potential
    for (int i = 0; i < factor_potential.first.size(); i++) {
        if (factor_potential.first(i) == input_message.first(0)) {
            index_variable = i;
        }
        if (factor_potential.first(i) == target) {
            index_target = i;
        }
    }

    MatrixXd lambda = factor_potential.second;
    lambda.block(_dim_state * index_variable, _dim_state * index_variable, _dim_state, _dim_state) += input_message.second;
    
    // Reorganize the lambda matrix to bring the target to the top
    if (index_target != 0) {
        lambda.block(0, 0, _dim_state, lambda.cols()).swap(lambda.block(index_target * _dim_state, 0, _dim_state, lambda.cols()));
        lambda.block(0, 0, lambda.rows(), _dim_state).swap(lambda.block(0, index_target * _dim_state, lambda.rows(), _dim_state));
    }

    MatrixXd lam_inverse = lambda.bottomRightCorner(lambda.rows() - _dim_state, lambda.cols() - _dim_state).inverse();
    MatrixXd lam_message = lambda.topLeftCorner(_dim_state, _dim_state) - lambda.topRightCorner(_dim_state, lambda.cols() - _dim_state) * lam_inverse * lambda.bottomLeftCorner(lambda.rows() - _dim_state, _dim_state);

    message.second = lam_message;
    return message;
}

}

// MatrixXd cov_error(cov_GBP.rows(), cov_GBP.cols());
// cov_error.setZero();

// for (int i = 0; i < _num_states - 1; ++i) {
//     cov_error.block(i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state) = _covariance.block(i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state) - 
//                                                                                 cov_GBP.block(i*_dim_state, i*_dim_state, 2*_dim_state, 2*_dim_state);
// }

// std::cout << "Norm of joint_cov = " << cov_GBP.norm() << "  Sum = " << cov_GBP.cwiseAbs().sum() << std::endl;
// std::cout << "Norm of cov_error = " << cov_error.norm() << "  Sum = " << cov_error.cwiseAbs().sum() << std::endl;


#endif // GVI_GH_IMPL_H